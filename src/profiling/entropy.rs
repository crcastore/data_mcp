use std::collections::HashMap;

use polars::prelude::*;

use crate::error::ProfilingError;

/// Shannon entropy in nats (natural logarithm).
///
/// For numeric (float/int) columns the values are binned using the
/// Freedman–Diaconis rule so that entropy reflects the shape of the
/// distribution rather than just the number of unique values.
/// For categorical/string columns exact value counts are used.
pub fn entropy(df: &DataFrame, column: &str) -> Result<f64, ProfilingError> {
    let col = df
        .column(column)
        .map_err(|_| ProfilingError::ColumnNotFound(column.to_string()))?;
    let series = col.as_materialized_series();
    let n = series.len();
    if n == 0 {
        return Err(ProfilingError::EmptyDataset);
    }

    if series.dtype().is_numeric() {
        entropy_numeric(series, n)
    } else {
        entropy_categorical(series, n)
    }
}

/// Entropy for numeric columns — differential entropy estimate via histogram.
///
/// Uses the Freedman–Diaconis rule for bin width.  The result is
/// `H = -Σ p_i ln(p_i / w)` where `w` is the bin width, which
/// approximates the differential entropy of the continuous distribution.
fn entropy_numeric(series: &Series, _n: usize) -> Result<f64, ProfilingError> {
    let cast = series.cast(&DataType::Float64)?;
    let ca = cast.f64()?;

    // Collect non-null values.
    let mut vals: Vec<f64> = ca.into_no_null_iter().collect();
    if vals.is_empty() {
        return Err(ProfilingError::EmptyDataset);
    }

    vals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let min = vals[0];
    let max = vals[vals.len() - 1];

    // Single-value column → entropy 0.
    if (max - min).abs() < f64::EPSILON {
        return Ok(0.0);
    }

    // Freedman–Diaconis bin width: 2 × IQR × n^(-1/3), with a floor of
    // Sturges' rule (ceil(log2(n)) + 1) bins.
    let q1 = vals[vals.len() / 4];
    let q3 = vals[3 * vals.len() / 4];
    let iqr = q3 - q1;

    let num_bins = if iqr > f64::EPSILON {
        let bin_width = 2.0 * iqr * (vals.len() as f64).powf(-1.0 / 3.0);
        let fd_bins = ((max - min) / bin_width).ceil() as usize;
        fd_bins.max(2)
    } else {
        // IQR ≈ 0 → fall back to Sturges' rule.
        ((vals.len() as f64).log2().ceil() as usize) + 1
    };

    let bin_width = (max - min) / num_bins as f64;

    let mut counts = vec![0usize; num_bins];
    for &v in &vals {
        let idx = ((v - min) / bin_width).floor() as usize;
        // Last value lands exactly on max → put it in the last bin.
        let idx = idx.min(num_bins - 1);
        counts[idx] += 1;
    }

    let nf = vals.len() as f64;
    // Differential entropy estimate: -Σ p_i * ln(p_i / w)
    let h: f64 = counts
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / nf;
            -p * (p / bin_width).ln()
        })
        .sum();

    Ok(h)
}

/// Entropy for categorical columns using exact value counts.
fn entropy_categorical(series: &Series, n: usize) -> Result<f64, ProfilingError> {
    let str_col = series.cast(&DataType::String)?;
    let ca = str_col.str()?;

    let mut counts: HashMap<&str, usize> = HashMap::new();
    for val in ca.into_no_null_iter() {
        *counts.entry(val).or_default() += 1;
    }

    let nf = n as f64;
    let h = counts
        .values()
        .map(|&c| {
            let p = c as f64 / nf;
            -p * p.ln()
        })
        .sum::<f64>();

    Ok(h)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entropy_uniform_categorical() {
        // 4 equally frequent values → H = ln(4) ≈ 1.386
        let df = df! { "c" => &["a", "b", "c", "d"] }.unwrap();
        let h = entropy(&df, "c").unwrap();
        assert!((h - 4.0f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_entropy_single_value_categorical() {
        let df = df! { "c" => &["x", "x", "x"] }.unwrap();
        assert!((entropy(&df, "c").unwrap()).abs() < 1e-10);
    }

    #[test]
    fn test_entropy_numeric_categorical_path() {
        // Integer column with 3 equally frequent values — uses binning.
        let df = df! { "n" => &[1i64, 2, 3, 1, 2, 3] }.unwrap();
        let h = entropy(&df, "n").unwrap();
        // Binned entropy should be positive (exact value depends on bin count).
        assert!(h > 0.0);
    }

    #[test]
    fn test_entropy_constant_numeric() {
        // All same value → entropy 0.
        let df = df! { "n" => &[5.0f64, 5.0, 5.0, 5.0] }.unwrap();
        assert!((entropy(&df, "n").unwrap()).abs() < 1e-10);
    }

    #[test]
    fn test_entropy_numeric_differs_by_distribution() {
        // Concentrated: all 500 values in a narrow band [49.9, 50.1].
        let concentrated: Vec<f64> = (0..500).map(|i| 49.9 + (i as f64) * 0.0004).collect();
        // Spread: 500 values evenly across [0, 100].
        let spread: Vec<f64> = (0..500).map(|i| i as f64 * 0.2).collect();

        let df_conc = df! { "v" => &concentrated }.unwrap();
        let df_spread = df! { "v" => &spread }.unwrap();
        let h_conc = entropy(&df_conc, "v").unwrap();
        let h_spread = entropy(&df_spread, "v").unwrap();
        // Differential entropy can be negative for narrow distributions.
        // But the spread distribution should always have higher entropy.
        assert!(h_spread > h_conc,
            "spread entropy {} should exceed concentrated entropy {}", h_spread, h_conc);
    }

    #[test]
    fn test_entropy_not_just_log_n() {
        // 1000 unique floats should NOT give entropy = ln(1000).
        let vals: Vec<f64> = (0..1000).map(|i| i as f64 * 0.001).collect();
        let df = df! { "v" => &vals }.unwrap();
        let h = entropy(&df, "v").unwrap();
        let log_n = (1000.0f64).ln();
        assert!((h - log_n).abs() > 0.1,
            "entropy {} should not equal ln(1000) = {}", h, log_n);
    }
}
