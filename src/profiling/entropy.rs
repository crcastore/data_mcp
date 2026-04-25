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

    let mut buf = vals.to_vec();
    let cmp = |a: &f64, b: &f64| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal);

    // O(n) min/max via scan.
    let min = buf.iter().copied().reduce(f64::min).unwrap();
    let max = buf.iter().copied().reduce(f64::max).unwrap();

    // Single-value column → entropy 0.
    if (max - min).abs() < f64::EPSILON {
        return Ok(0.0);
    }

    // O(n) quickselect for Q1 and Q3 (Freedman–Diaconis bin width).
    let n = buf.len();
    let q1_idx = n / 4;
    let q3_idx = 3 * n / 4;
    buf.select_nth_unstable_by(q1_idx, cmp);
    let q1 = buf[q1_idx];
    buf[q1_idx..].select_nth_unstable_by(q3_idx - q1_idx, cmp);
    let q3 = buf[q3_idx];
    let iqr = q3 - q1;

    let num_bins = if iqr > f64::EPSILON {
        let bin_width = 2.0 * iqr * (n as f64).powf(-1.0 / 3.0);
        let fd_bins = ((max - min) / bin_width).ceil() as usize;
        fd_bins.max(2)
    } else {
        // IQR ≈ 0 → fall back to Sturges' rule.
        ((n as f64).log2().ceil() as usize) + 1
    };

    let bin_width = (max - min) / num_bins as f64;

    let mut counts = vec![0usize; num_bins];
    for &v in vals {
        let idx = ((v - min) / bin_width).floor() as usize;
        let idx = idx.min(num_bins - 1);
        counts[idx] += 1;
    }

    let nf = n as f64;
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

#[cfg(test)]
mod tests {
    use super::*;

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
