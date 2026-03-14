use polars::prelude::*;
use serde::Serialize;

use crate::error::ProfilingError;
use crate::util::extract_f64_values;

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct SurrogateTestResult {
    pub statistic_real: f64,
    pub surrogate_mean: f64,
    pub surrogate_std: f64,
    pub z_score: f64,
    pub is_nonlinear: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct BdsTestResult {
    pub statistic: f64,
    pub p_value: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct DependenceComparison {
    pub autocorrelations: Vec<f64>,
    pub mutual_informations: Vec<f64>,
    pub nonlinear_dominance: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct DelayEmbedding {
    pub optimal_delay: usize,
    pub embedding_dimension: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryProfile {
    pub partial_autocorrelations: Vec<f64>,
    pub active_information_storage: f64,
    pub delay_mutual_informations: Vec<f64>,
    pub memory_length: usize,
}

// ---------------------------------------------------------------------------
// 1. Surrogate data test
// ---------------------------------------------------------------------------

/// Surrogate data test for nonlinearity.
///
/// Fits an AR(1) model, generates surrogates by simulating with shuffled
/// residuals, and compares the time-reversal asymmetry statistic.
/// A `z_score` with absolute value > 2 indicates nonlinear structure.
pub fn surrogate_test(
    df: &DataFrame,
    column: &str,
    num_surrogates: usize,
) -> Result<SurrogateTestResult, ProfilingError> {
    let vals = extract_f64_values(df, column)?;
    let n = vals.len();
    if n < 10 {
        return Err(ProfilingError::InsufficientData(10));
    }

    let stat_real = time_reversal_asymmetry(&vals);

    // Fit AR(1): x_t = phi * x_{t-1} + e_t
    let mean = vals.iter().sum::<f64>() / n as f64;
    let centered: Vec<f64> = vals.iter().map(|x| x - mean).collect();
    let (phi, residuals) = fit_ar1(&centered);

    // Generate surrogates
    let mut surrogate_stats = Vec::with_capacity(num_surrogates);
    let mut seed: u64 = 12345;
    for _ in 0..num_surrogates {
        let shuffled_res = shuffle_with_lcg(&residuals, &mut seed);
        let surrogate = simulate_ar1(phi, mean, &shuffled_res);
        surrogate_stats.push(time_reversal_asymmetry(&surrogate));
    }

    let s_mean = surrogate_stats.iter().sum::<f64>() / surrogate_stats.len() as f64;
    let s_std = {
        let var = surrogate_stats
            .iter()
            .map(|x| (x - s_mean).powi(2))
            .sum::<f64>()
            / (surrogate_stats.len() - 1) as f64;
        var.sqrt()
    };
    let z_score = if s_std > 0.0 {
        (stat_real - s_mean) / s_std
    } else {
        0.0
    };

    Ok(SurrogateTestResult {
        statistic_real: stat_real,
        surrogate_mean: s_mean,
        surrogate_std: s_std,
        z_score,
        is_nonlinear: z_score.abs() > 2.0,
    })
}

/// Time-reversal asymmetry: T = mean(x_t² · x_{t−1} − x_t · x_{t−1}²).
fn time_reversal_asymmetry(x: &[f64]) -> f64 {
    if x.len() < 2 {
        return 0.0;
    }
    let sum: f64 = x
        .windows(2)
        .map(|w| w[1] * w[1] * w[0] - w[1] * w[0] * w[0])
        .sum();
    sum / (x.len() - 1) as f64
}

/// Fit AR(1) via Yule-Walker, return (phi, residuals).
fn fit_ar1(centered: &[f64]) -> (f64, Vec<f64>) {
    let n = centered.len();
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 1..n {
        num += centered[i] * centered[i - 1];
        den += centered[i - 1] * centered[i - 1];
    }
    let phi = if den > 0.0 { num / den } else { 0.0 };

    let residuals: Vec<f64> = (1..n)
        .map(|i| centered[i] - phi * centered[i - 1])
        .collect();

    (phi, residuals)
}

/// Simulate AR(1): x_t = mean + phi · (x_{t−1} − mean) + e_t.
fn simulate_ar1(phi: f64, mean: f64, residuals: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(residuals.len());
    out.push(mean + residuals[0]);
    for i in 1..residuals.len() {
        out.push(mean + phi * (out[i - 1] - mean) + residuals[i]);
    }
    out
}

/// Fisher-Yates shuffle using LCG PRNG.
fn shuffle_with_lcg(data: &[f64], seed: &mut u64) -> Vec<f64> {
    let mut out = data.to_vec();
    for i in (1..out.len()).rev() {
        *seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let j = (*seed as usize) % (i + 1);
        out.swap(i, j);
    }
    out
}

// ---------------------------------------------------------------------------
// 2. BDS test
// ---------------------------------------------------------------------------

/// BDS test for nonlinear serial dependence.
///
/// Tests H₀: the series is i.i.d. A significant result (`p_value < 0.05`)
/// indicates nonlinear dependence. `epsilon` is typically set to
/// 0.5–2.0 × the standard deviation of the series.
pub fn bds_test(
    df: &DataFrame,
    column: &str,
    embedding_dim: usize,
    epsilon: f64,
) -> Result<BdsTestResult, ProfilingError> {
    let vals = extract_f64_values(df, column)?;
    let n = vals.len();
    if n < 20 {
        return Err(ProfilingError::InsufficientData(20));
    }
    if embedding_dim < 2 {
        return Err(ProfilingError::InsufficientData(2));
    }

    let c1 = correlation_integral(&vals, 1, epsilon);
    let cm = correlation_integral(&vals, embedding_dim, epsilon);
    let k = triple_correlation(&vals, epsilon);

    let m = embedding_dim as f64;

    // BDS variance under H₀ (Brock–Dechert–Scheinkman formula)
    let sum_mid: f64 = (1..embedding_dim)
        .map(|j| k.powf(m - j as f64) * c1.powf(2.0 * j as f64))
        .sum();
    let v = 4.0
        * (k.powf(m) + 2.0 * sum_mid + (m - 1.0).powi(2) * c1.powf(2.0 * m)
            - m.powi(2) * k * c1.powf(2.0 * (m - 1.0)));

    let sigma = v.max(0.0).sqrt();
    let n_m = (n - embedding_dim + 1) as f64;
    let bds = if sigma > 0.0 {
        n_m.sqrt() * (cm - c1.powf(m)) / sigma
    } else {
        0.0
    };

    let p_value = 2.0 * (1.0 - normal_cdf(bds.abs()));

    Ok(BdsTestResult {
        statistic: bds,
        p_value,
    })
}

/// Correlation integral C_m(ε) using sup-norm (L∞).
fn correlation_integral(x: &[f64], m: usize, eps: f64) -> f64 {
    let n_vec = x.len() - (m - 1);
    if n_vec < 2 {
        return 0.0;
    }
    let mut count = 0u64;
    let total = (n_vec * (n_vec - 1)) / 2;
    for i in 0..n_vec {
        for j in (i + 1)..n_vec {
            let mut within = true;
            for k in 0..m {
                if (x[i + k] - x[j + k]).abs() >= eps {
                    within = false;
                    break;
                }
            }
            if within {
                count += 1;
            }
        }
    }
    count as f64 / total as f64
}

/// Triple correlation K: mean over j of h_j², where h_j = fraction of points
/// within ε of x_j.
fn triple_correlation(x: &[f64], eps: f64) -> f64 {
    let n = x.len();
    if n < 3 {
        return 0.0;
    }
    let nf = n as f64;
    let mut sum = 0.0;
    for j in 0..n {
        let c = x.iter().filter(|&&xi| (xi - x[j]).abs() < eps).count() as f64 / nf;
        sum += c * c;
    }
    sum / nf
}

// ---------------------------------------------------------------------------
// 3. Lyapunov exponent
// ---------------------------------------------------------------------------

/// Estimate the largest Lyapunov exponent (Rosenstein's method).
///
/// A positive value indicates chaos, which is favourable for reservoir
/// computing. `embedding_dim` is typically 2–10 and `tau` is the delay
/// (often 1.or from `delay_embedding`).
pub fn lyapunov_exponent(
    df: &DataFrame,
    column: &str,
    embedding_dim: usize,
    tau: usize,
) -> Result<f64, ProfilingError> {
    let vals = extract_f64_values(df, column)?;
    let n = vals.len();
    let n_vec = n.saturating_sub((embedding_dim - 1) * tau);
    if n_vec < 20 {
        return Err(ProfilingError::InsufficientData(20));
    }

    // Build embedded vectors
    let embedded: Vec<Vec<f64>> = (0..n_vec)
        .map(|i| (0..embedding_dim).map(|k| vals[i + k * tau]).collect())
        .collect();

    let min_sep = (embedding_dim - 1) * tau + 1;
    let max_steps = n_vec / 4;

    let mut divergences: Vec<Vec<f64>> = vec![Vec::new(); max_steps];

    for i in 0..n_vec {
        // Find nearest neighbour (excluding temporal neighbours)
        let mut best_dist = f64::MAX;
        let mut best_j = 0;
        for j in 0..n_vec {
            if (i as isize - j as isize).unsigned_abs() < min_sep {
                continue;
            }
            let d = euclidean_dist(&embedded[i], &embedded[j]);
            if d < best_dist && d > 0.0 {
                best_dist = d;
                best_j = j;
            }
        }
        if best_dist == f64::MAX {
            continue;
        }

        // Track divergence
        for step in 0..max_steps {
            let ii = i + step;
            let jj = best_j + step;
            if ii >= n_vec || jj >= n_vec {
                break;
            }
            let d = euclidean_dist(&embedded[ii], &embedded[jj]);
            if d > 0.0 {
                divergences[step].push(d.ln());
            }
        }
    }

    // Average log-divergence at each step
    let avg: Vec<f64> = divergences
        .iter()
        .map(|d| {
            if d.is_empty() {
                f64::NAN
            } else {
                d.iter().sum::<f64>() / d.len() as f64
            }
        })
        .collect();

    // Linear fit over the first ~10 % of steps
    let fit_len = (max_steps / 10).max(5).min(avg.len());
    let valid: Vec<(f64, f64)> = avg[..fit_len]
        .iter()
        .enumerate()
        .filter(|(_, y)| !y.is_nan())
        .map(|(i, y)| (i as f64, *y))
        .collect();
    if valid.len() < 2 {
        return Err(ProfilingError::InsufficientData(2));
    }

    Ok(linear_slope(&valid))
}

fn euclidean_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn linear_slope(pts: &[(f64, f64)]) -> f64 {
    let n = pts.len() as f64;
    let sx: f64 = pts.iter().map(|(x, _)| x).sum();
    let sy: f64 = pts.iter().map(|(_, y)| y).sum();
    let sxy: f64 = pts.iter().map(|(x, y)| x * y).sum();
    let sx2: f64 = pts.iter().map(|(x, _)| x * x).sum();
    let den = n * sx2 - sx * sx;
    if den == 0.0 {
        0.0
    } else {
        (n * sxy - sx * sy) / den
    }
}

// ---------------------------------------------------------------------------
// 4. Mutual information vs autocorrelation
// ---------------------------------------------------------------------------

/// Compare linear (autocorrelation) and nonlinear (mutual information)
/// dependence at lags 1..=`max_lag`.
///
/// `nonlinear_dominance` is `true` when the MI at any lag exceeds
/// the Gaussian-equivalent MI by more than 50 %, indicating structure
/// that a linear model cannot capture.
pub fn dependence_comparison(
    df: &DataFrame,
    column: &str,
    max_lag: usize,
) -> Result<DependenceComparison, ProfilingError> {
    let vals = extract_f64_values(df, column)?;
    if vals.len() < max_lag + 10 {
        return Err(ProfilingError::InsufficientData(max_lag + 10));
    }

    let mean = vals.iter().sum::<f64>() / vals.len() as f64;
    let var: f64 = vals.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / vals.len() as f64;

    let mut autocorrelations = Vec::with_capacity(max_lag);
    let mut mutual_informations = Vec::with_capacity(max_lag);

    for lag in 1..=max_lag {
        let acf = if var > 0.0 {
            let cov: f64 = vals[..vals.len() - lag]
                .iter()
                .zip(vals[lag..].iter())
                .map(|(a, b)| (a - mean) * (b - mean))
                .sum::<f64>()
                / vals.len() as f64;
            cov / var
        } else {
            0.0
        };
        autocorrelations.push(acf);

        let mi = binned_mutual_information(&vals[..vals.len() - lag], &vals[lag..]);
        mutual_informations.push(mi);
    }

    // MI for a Gaussian with correlation r: -0.5 ln(1 - r²)
    let nonlinear_dominance = autocorrelations
        .iter()
        .zip(mutual_informations.iter())
        .any(|(r, mi)| {
            let gaussian_mi = if r.abs() < 1.0 {
                -0.5 * (1.0 - r * r).ln()
            } else {
                f64::INFINITY
            };
            *mi > gaussian_mi * 1.5
        });

    Ok(DependenceComparison {
        autocorrelations,
        mutual_informations,
        nonlinear_dominance,
    })
}

/// Binned mutual information estimator using √N bins.
fn binned_mutual_information(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    let num_bins = ((n as f64).sqrt().ceil() as usize).max(2);

    let x_min = x.iter().cloned().fold(f64::INFINITY, f64::min);
    let x_max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let y_min = y.iter().cloned().fold(f64::INFINITY, f64::min);
    let y_max = y.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let x_range = x_max - x_min;
    let y_range = y_max - y_min;
    if x_range == 0.0 || y_range == 0.0 {
        return 0.0;
    }

    let mut joint = vec![vec![0u32; num_bins]; num_bins];
    let mut marginal_x = vec![0u32; num_bins];
    let mut marginal_y = vec![0u32; num_bins];

    for i in 0..n {
        let bx = ((x[i] - x_min) / x_range * (num_bins - 1) as f64).round() as usize;
        let by = ((y[i] - y_min) / y_range * (num_bins - 1) as f64).round() as usize;
        let bx = bx.min(num_bins - 1);
        let by = by.min(num_bins - 1);
        joint[bx][by] += 1;
        marginal_x[bx] += 1;
        marginal_y[by] += 1;
    }

    let nf = n as f64;
    let mut mi = 0.0;
    for i in 0..num_bins {
        for j in 0..num_bins {
            if joint[i][j] > 0 {
                let p_xy = joint[i][j] as f64 / nf;
                let p_x = marginal_x[i] as f64 / nf;
                let p_y = marginal_y[j] as f64 / nf;
                mi += p_xy * (p_xy / (p_x * p_y)).ln();
            }
        }
    }

    mi.max(0.0)
}

// ---------------------------------------------------------------------------
// 5. Delay embedding (Takens' reconstruction)
// ---------------------------------------------------------------------------

/// Estimate optimal delay and embedding dimension for Takens' reconstruction.
///
/// * Optimal delay — first minimum of average mutual information.
/// * Embedding dimension — false nearest neighbours (FNN) method.
pub fn delay_embedding(
    df: &DataFrame,
    column: &str,
    max_dim: usize,
) -> Result<DelayEmbedding, ProfilingError> {
    let vals = extract_f64_values(df, column)?;
    if vals.len() < 50 {
        return Err(ProfilingError::InsufficientData(50));
    }

    // Optimal delay: first minimum of AMI
    let max_tau = (vals.len() / 4).min(50);
    let mut ami_values = Vec::with_capacity(max_tau);
    for tau in 1..=max_tau {
        let mi = binned_mutual_information(&vals[..vals.len() - tau], &vals[tau..]);
        ami_values.push(mi);
    }

    let optimal_delay = first_local_minimum(&ami_values).unwrap_or(1);

    // Embedding dimension via FNN
    let tau = optimal_delay;
    let mut best_dim = 1;
    let fnn_threshold = 15.0;
    for dim in 1..=max_dim {
        let n_check = vals.len().saturating_sub(dim * tau);
        if n_check < 20 {
            break;
        }
        let fnn = false_nearest_neighbours(&vals, dim, tau, fnn_threshold);
        if fnn < 0.01 {
            best_dim = dim;
            break;
        }
        best_dim = dim;
    }

    Ok(DelayEmbedding {
        optimal_delay,
        embedding_dimension: best_dim,
    })
}

fn first_local_minimum(values: &[f64]) -> Option<usize> {
    for i in 1..values.len().saturating_sub(1) {
        if values[i] < values[i - 1] && values[i] <= values[i + 1] {
            return Some(i + 1); // 1-indexed lag
        }
    }
    None
}

/// Fraction of false nearest neighbours at dimension `dim`.
fn false_nearest_neighbours(x: &[f64], dim: usize, tau: usize, threshold: f64) -> f64 {
    // Need dim+1 embedding to check the extra coordinate
    let n_check = x.len().saturating_sub(dim * tau);
    if n_check < 10 {
        return 0.0;
    }

    let embedded: Vec<Vec<f64>> = (0..n_check)
        .map(|i| (0..dim).map(|k| x[i + k * tau]).collect())
        .collect();

    let sample_size = n_check.min(200);
    let step = (n_check / sample_size).max(1);

    let mut fnn_count = 0;
    let mut total = 0;

    for idx in (0..n_check).step_by(step) {
        let mut best_dist = f64::MAX;
        let mut best_j = 0;
        for j in 0..n_check {
            if j == idx {
                continue;
            }
            let d = euclidean_dist(&embedded[idx], &embedded[j]);
            if d < best_dist && d > 0.0 {
                best_dist = d;
                best_j = j;
            }
        }
        if best_dist == f64::MAX {
            continue;
        }
        let extra_idx = idx + dim * tau;
        let extra_j = best_j + dim * tau;
        if extra_idx >= x.len() || extra_j >= x.len() {
            continue;
        }
        if (x[extra_idx] - x[extra_j]).abs() / best_dist > threshold {
            fnn_count += 1;
        }
        total += 1;
    }

    if total == 0 {
        0.0
    } else {
        fnn_count as f64 / total as f64
    }
}

// ---------------------------------------------------------------------------
// 6. Memory capacity
// ---------------------------------------------------------------------------

/// Temporal memory profile: PACF, active information storage, delay MI.
///
/// * `memory_length` — last lag where |PACF| exceeds the 2/√N significance
///   threshold, indicating how deep the temporal memory extends.
pub fn memory_profile(
    df: &DataFrame,
    column: &str,
    max_lag: usize,
) -> Result<MemoryProfile, ProfilingError> {
    let vals = extract_f64_values(df, column)?;
    if vals.len() < max_lag + 10 {
        return Err(ProfilingError::InsufficientData(max_lag + 10));
    }

    let pacf = durbin_levinson(&vals, max_lag);

    let dmi: Vec<f64> = (1..=max_lag)
        .map(|lag| binned_mutual_information(&vals[..vals.len() - lag], &vals[lag..]))
        .collect();

    let ais = dmi.iter().sum::<f64>();

    let threshold = 2.0 / (vals.len() as f64).sqrt();
    let memory_length = pacf
        .iter()
        .rposition(|p| p.abs() > threshold)
        .map(|i| i + 1)
        .unwrap_or(0);

    Ok(MemoryProfile {
        partial_autocorrelations: pacf,
        active_information_storage: ais,
        delay_mutual_informations: dmi,
        memory_length,
    })
}

/// Durbin-Levinson recursion for partial autocorrelation.
fn durbin_levinson(x: &[f64], max_lag: usize) -> Vec<f64> {
    let n = x.len();
    let mean = x.iter().sum::<f64>() / n as f64;
    let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
    if var == 0.0 {
        return vec![0.0; max_lag];
    }

    let acf: Vec<f64> = (1..=max_lag)
        .map(|lag| {
            let cov: f64 = x[..n - lag]
                .iter()
                .zip(x[lag..].iter())
                .map(|(a, b)| (a - mean) * (b - mean))
                .sum::<f64>()
                / n as f64;
            cov / var
        })
        .collect();

    let mut pacf = Vec::with_capacity(max_lag);
    let mut phi = vec![0.0; max_lag];
    let mut phi_prev = vec![0.0; max_lag];

    phi[0] = acf[0];
    pacf.push(phi[0]);

    for k in 1..max_lag {
        let mut num = acf[k];
        for j in 0..k {
            num -= phi[j] * acf[k - 1 - j];
        }
        let mut den = 1.0;
        for j in 0..k {
            den -= phi[j] * acf[j];
        }

        phi_prev[..max_lag].copy_from_slice(&phi[..max_lag]);

        let new_phi = if den.abs() > 1e-15 { num / den } else { 0.0 };
        phi[k] = new_phi;
        for j in 0..k {
            phi[j] = phi_prev[j] - new_phi * phi_prev[k - 1 - j];
        }
        pacf.push(new_phi);
    }

    pacf
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Standard normal CDF (Abramowitz & Stegun rational approximation).
fn normal_cdf(x: f64) -> f64 {
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs() / std::f64::consts::SQRT_2;
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    0.5 * (1.0 + sign * y)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Logistic map at r = 3.9 (chaotic regime).
    fn logistic_map(n: usize) -> DataFrame {
        let mut x = vec![0.0f64; n];
        x[0] = 0.1;
        let r = 3.9;
        for i in 1..n {
            x[i] = r * x[i - 1] * (1.0 - x[i - 1]);
        }
        df! { "x" => &x }.unwrap()
    }

    /// AR(1) with φ = 0.5 and small noise.
    fn linear_series(n: usize) -> DataFrame {
        let mut x = vec![0.0f64; n];
        let mut seed: u64 = 42;
        for i in 1..n {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let noise = (seed as f64 / u64::MAX as f64 - 0.5) * 0.1;
            x[i] = 0.5 * x[i - 1] + noise;
        }
        df! { "x" => &x }.unwrap()
    }

    /// Pure sine wave.
    fn sine_wave(n: usize) -> DataFrame {
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        df! { "x" => &x }.unwrap()
    }

    // -- Surrogate test --

    #[test]
    fn test_surrogate_logistic_detects_nonlinearity() {
        let df = logistic_map(1000);
        let result = surrogate_test(&df, "x", 100).unwrap();
        assert!(result.z_score.abs() > 2.0);
        assert!(result.is_nonlinear);
    }

    // -- BDS test --

    #[test]
    fn test_bds_logistic() {
        let df = logistic_map(500);
        let result = bds_test(&df, "x", 3, 0.5).unwrap();
        assert!(result.p_value < 0.05);
    }

    #[test]
    fn test_bds_returns_valid() {
        let df = linear_series(500);
        let result = bds_test(&df, "x", 2, 0.05).unwrap();
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
        assert!(result.statistic.is_finite());
    }

    // -- Lyapunov exponent --

    #[test]
    fn test_lyapunov_logistic_positive() {
        let df = logistic_map(2000);
        let l = lyapunov_exponent(&df, "x", 3, 1).unwrap();
        assert!(l > 0.0, "Lyapunov exponent should be positive for logistic map, got {l}");
    }

    // -- Dependence comparison --

    #[test]
    fn test_dependence_comparison_lengths() {
        let df = logistic_map(1000);
        let result = dependence_comparison(&df, "x", 10).unwrap();
        assert_eq!(result.autocorrelations.len(), 10);
        assert_eq!(result.mutual_informations.len(), 10);
    }

    #[test]
    fn test_dependence_mi_positive() {
        let df = logistic_map(1000);
        let result = dependence_comparison(&df, "x", 5).unwrap();
        for mi in &result.mutual_informations {
            assert!(*mi >= 0.0);
        }
    }

    // -- Delay embedding --

    #[test]
    fn test_delay_embedding_sine() {
        let df = sine_wave(500);
        let result = delay_embedding(&df, "x", 10).unwrap();
        assert!(result.optimal_delay >= 1);
        assert!(result.embedding_dimension >= 1);
    }

    // -- Memory profile --

    #[test]
    fn test_memory_profile_ar() {
        let df = linear_series(1000);
        let result = memory_profile(&df, "x", 10).unwrap();
        assert_eq!(result.partial_autocorrelations.len(), 10);
        assert_eq!(result.delay_mutual_informations.len(), 10);
        // AR(1) with phi=0.5 should have significant PACF at lag 1
        assert!(result.partial_autocorrelations[0].abs() > 0.1);
        assert!(result.memory_length >= 1);
    }

    #[test]
    fn test_memory_ais_positive() {
        let df = logistic_map(1000);
        let result = memory_profile(&df, "x", 10).unwrap();
        assert!(result.active_information_storage > 0.0);
    }

    // -- Helpers --

    #[test]
    fn test_normal_cdf_values() {
        assert!((normal_cdf(0.0) - 0.5).abs() < 1e-6);
        assert!((normal_cdf(1.96) - 0.975).abs() < 1e-3);
        assert!(normal_cdf(-3.0) < 0.01);
    }
}
