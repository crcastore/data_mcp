use serde::Serialize;

use crate::error::ProfilingError;

#[derive(Debug, Clone, Serialize)]
pub struct Quantiles {
    pub min: f64,
    pub q25: f64,
    pub q50: f64,
    pub q75: f64,
    pub max: f64,
}

/// Min, Q25, median, Q75, max via linear interpolation (numpy default).
pub fn quantiles(vals: &[f64]) -> Result<Quantiles, ProfilingError> {
    if vals.is_empty() {
        return Err(ProfilingError::EmptyDataset);
    }
    let mut sorted = vals.to_vec();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Ok(quantiles_from_sorted(&sorted))
}

/// Build Quantiles from an already-sorted slice.
pub fn quantiles_from_sorted(sorted: &[f64]) -> Quantiles {
    Quantiles {
        min: sorted[0],
        q25: lerp_percentile(sorted, 0.25),
        q50: lerp_percentile(sorted, 0.50),
        q75: lerp_percentile(sorted, 0.75),
        max: sorted[sorted.len() - 1],
    }
}

/// Linear-interpolation percentile on a *sorted* slice.
fn lerp_percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.len() == 1 {
        return sorted[0];
    }
    let idx = p * (sorted.len() - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    let frac = idx - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantiles() {
        let q = quantiles(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        assert!((q.min - 1.0).abs() < 1e-10);
        assert!((q.q50 - 3.0).abs() < 1e-10);
        assert!((q.max - 5.0).abs() < 1e-10);
        assert!(q.q25 >= q.min && q.q25 <= q.q50);
        assert!(q.q75 >= q.q50 && q.q75 <= q.max);
    }
}
