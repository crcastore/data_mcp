use std::collections::HashMap;

use crate::error::ProfilingError;

/// Entropy for numeric columns — differential entropy estimate via histogram.
///
/// Uses the Freedman–Diaconis rule for bin width. The result is
/// `H = -Σ p_i ln(p_i / w)` where `w` is the bin width, which
/// approximates the differential entropy of the continuous distribution.
pub fn entropy_numeric(vals: &[f64]) -> Result<f64, ProfilingError> {
    if vals.is_empty() {
        return Err(ProfilingError::EmptyDataset);
    }

    let mut sorted = vals.to_vec();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let min = sorted[0];
    let max = sorted[sorted.len() - 1];

    // Single-value column → entropy 0.
    if (max - min).abs() < f64::EPSILON {
        return Ok(0.0);
    }

    // Freedman–Diaconis bin width: 2 × IQR × n^(-1/3), with a floor of
    // Sturges' rule (ceil(log2(n)) + 1) bins.
    let q1 = sorted[sorted.len() / 4];
    let q3 = sorted[3 * sorted.len() / 4];
    let iqr = q3 - q1;

    let num_bins = if iqr > f64::EPSILON {
        let bin_width = 2.0 * iqr * (sorted.len() as f64).powf(-1.0 / 3.0);
        let fd_bins = ((max - min) / bin_width).ceil() as usize;
        fd_bins.max(2)
    } else {
        // IQR ≈ 0 → fall back to Sturges' rule.
        ((sorted.len() as f64).log2().ceil() as usize) + 1
    };

    let bin_width = (max - min) / num_bins as f64;

    let mut counts = vec![0usize; num_bins];
    for &v in &sorted {
        let idx = ((v - min) / bin_width).floor() as usize;
        let idx = idx.min(num_bins - 1);
        counts[idx] += 1;
    }

    let nf = sorted.len() as f64;
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
pub fn entropy_categorical(vals: &[String]) -> Result<f64, ProfilingError> {
    let n = vals.len();
    if n == 0 {
        return Err(ProfilingError::EmptyDataset);
    }

    let mut counts: HashMap<&str, usize> = HashMap::new();
    for val in vals {
        *counts.entry(val.as_str()).or_default() += 1;
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
        let vals: Vec<String> = vec!["a", "b", "c", "d"]
            .into_iter()
            .map(String::from)
            .collect();
        let h = entropy_categorical(&vals).unwrap();
        assert!((h - 4.0f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_entropy_single_value_categorical() {
        let vals: Vec<String> = vec!["x", "x", "x"]
            .into_iter()
            .map(String::from)
            .collect();
        assert!(entropy_categorical(&vals).unwrap().abs() < 1e-10);
    }

    #[test]
    fn test_entropy_numeric_binned() {
        let vals: Vec<f64> = vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0];
        let h = entropy_numeric(&vals).unwrap();
        assert!(h > 0.0);
    }

    #[test]
    fn test_entropy_constant_numeric() {
        let vals = vec![5.0, 5.0, 5.0, 5.0];
        assert!(entropy_numeric(&vals).unwrap().abs() < 1e-10);
    }

    #[test]
    fn test_entropy_numeric_differs_by_distribution() {
        let concentrated: Vec<f64> = (0..500).map(|i| 49.9 + (i as f64) * 0.0004).collect();
        let spread: Vec<f64> = (0..500).map(|i| i as f64 * 0.2).collect();
        let h_conc = entropy_numeric(&concentrated).unwrap();
        let h_spread = entropy_numeric(&spread).unwrap();
        assert!(
            h_spread > h_conc,
            "spread entropy {} should exceed concentrated entropy {}",
            h_spread,
            h_conc
        );
    }

    #[test]
    fn test_entropy_not_just_log_n() {
        let vals: Vec<f64> = (0..1000).map(|i| i as f64 * 0.001).collect();
        let h = entropy_numeric(&vals).unwrap();
        let log_n = (1000.0f64).ln();
        assert!(
            (h - log_n).abs() > 0.1,
            "entropy {} should not equal ln(1000) = {}",
            h,
            log_n
        );
    }
}
