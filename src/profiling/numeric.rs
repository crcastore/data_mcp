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

/// O(n) quantiles using `select_nth_unstable` (quickselect) instead of a full sort.
/// Computes min, q25, median, q75, max with linear interpolation matching numpy's default.
pub fn quantiles_select(vals: &[f64]) -> Quantiles {
    let n = vals.len();
    if n == 1 {
        let v = vals[0];
        return Quantiles { min: v, q25: v, q50: v, q75: v, max: v };
    }

    let mut buf = vals.to_vec();
    let cmp = |a: &f64, b: &f64| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal);

    // Collect the unique indices we need, in sorted order.
    let percentiles = [0.25, 0.50, 0.75];
    let mut indices: Vec<usize> = Vec::with_capacity(8);
    for &p in &percentiles {
        let idx = p * (n - 1) as f64;
        indices.push(idx.floor() as usize);
        indices.push(idx.ceil() as usize);
    }
    indices.sort_unstable();
    indices.dedup();

    // Partition around each index in ascending order.
    // After select_nth_unstable_by(k), buf[..k] are all <= buf[k] and buf[k+1..] are all >= buf[k].
    // So we can narrow the slice for subsequent selects.
    let mut left = 0;
    for &k in &indices {
        if k >= left {
            buf[left..].select_nth_unstable_by(k - left, cmp);
        }
        left = k + 1;
    }

    // min/max via simple scans — O(n) and avoids needing those positions partitioned.
    let min = buf.iter().copied().reduce(f64::min).unwrap();
    let max = buf.iter().copied().reduce(f64::max).unwrap();

    let select_lerp = |p: f64| -> f64 {
        let idx = p * (n - 1) as f64;
        let lo = idx.floor() as usize;
        let hi = idx.ceil() as usize;
        let frac = idx - lo as f64;
        buf[lo] * (1.0 - frac) + buf[hi] * frac
    };

    Quantiles {
        min,
        q25: select_lerp(0.25),
        q50: select_lerp(0.50),
        q75: select_lerp(0.75),
        max,
    }
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

    #[test]
    fn test_quantiles_select_matches_sorted() {
        let data: Vec<f64> = (0..1000).map(|i| (i as f64) * 0.7 - 200.0).collect();
        let q_sort = quantiles(&data).unwrap();
        let q_sel = quantiles_select(&data);
        assert!((q_sort.min - q_sel.min).abs() < 1e-10);
        assert!((q_sort.q25 - q_sel.q25).abs() < 1e-10);
        assert!((q_sort.q50 - q_sel.q50).abs() < 1e-10);
        assert!((q_sort.q75 - q_sel.q75).abs() < 1e-10);
        assert!((q_sort.max - q_sel.max).abs() < 1e-10);
    }
}
