use polars::prelude::*;
use serde::Serialize;

use crate::error::ProfilingError;
use crate::util::extract_f64_values;

/// Arithmetic mean.
pub fn mean(df: &DataFrame, column: &str) -> Result<f64, ProfilingError> {
    let vals = extract_f64_values(df, column)?;
    if vals.is_empty() {
        return Err(ProfilingError::EmptyDataset);
    }
    Ok(vals.iter().sum::<f64>() / vals.len() as f64)
}

/// Sample variance (ddof = 1).
pub fn variance(df: &DataFrame, column: &str) -> Result<f64, ProfilingError> {
    let vals = extract_f64_values(df, column)?;
    let n = vals.len();
    if n < 2 {
        return Err(ProfilingError::InsufficientData(2));
    }
    let mean = vals.iter().sum::<f64>() / n as f64;
    let var = vals.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    Ok(var)
}

#[derive(Debug, Clone, Serialize)]
pub struct Quantiles {
    pub min: f64,
    pub q25: f64,
    pub q50: f64,
    pub q75: f64,
    pub max: f64,
}

/// Min, Q25, median, Q75, max via linear interpolation (numpy default).
pub fn quantiles(df: &DataFrame, column: &str) -> Result<Quantiles, ProfilingError> {
    let mut vals = extract_f64_values(df, column)?;
    if vals.is_empty() {
        return Err(ProfilingError::EmptyDataset);
    }
    vals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Ok(Quantiles {
        min: vals[0],
        q25: lerp_percentile(&vals, 0.25),
        q50: lerp_percentile(&vals, 0.50),
        q75: lerp_percentile(&vals, 0.75),
        max: vals[vals.len() - 1],
    })
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

    fn df_nums() -> DataFrame {
        df! { "x" => &[1.0f64, 2.0, 3.0, 4.0, 5.0] }.unwrap()
    }

    #[test]
    fn test_mean() {
        assert!((mean(&df_nums(), "x").unwrap() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_variance() {
        // sample var of [1,2,3,4,5] = 10/4 = 2.5
        assert!((variance(&df_nums(), "x").unwrap() - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_quantiles() {
        let q = quantiles(&df_nums(), "x").unwrap();
        assert!((q.min - 1.0).abs() < 1e-10);
        assert!((q.q50 - 3.0).abs() < 1e-10);
        assert!((q.max - 5.0).abs() < 1e-10);
        assert!(q.q25 >= q.min && q.q25 <= q.q50);
        assert!(q.q75 >= q.q50 && q.q75 <= q.max);
    }

    #[test]
    fn test_not_numeric() {
        let df = df! { "s" => &["a", "b"] }.unwrap();
        assert!(mean(&df, "s").is_err());
    }
}
