use polars::prelude::*;

use crate::error::ProfilingError;
use crate::util::extract_f64_values;

/// Adjusted Fisher–Pearson sample skewness.
///
/// G₁ = √(n(n−1)) / (n−2) · m₃ / m₂^(3/2)
///
/// where m₂, m₃ are the biased central moments.
pub fn skewness(df: &DataFrame, column: &str) -> Result<f64, ProfilingError> {
    let vals = extract_f64_values(df, column)?;
    let n = vals.len();
    if n < 3 {
        return Err(ProfilingError::InsufficientData(3));
    }
    let nf = n as f64;
    let mean = vals.iter().sum::<f64>() / nf;
    let m2: f64 = vals.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / nf;
    if m2 == 0.0 {
        return Ok(0.0);
    }
    let m3: f64 = vals.iter().map(|x| (x - mean).powi(3)).sum::<f64>() / nf;
    let g1 = m3 / m2.powf(1.5);
    let adj = (nf * (nf - 1.0)).sqrt() / (nf - 2.0);
    Ok(adj * g1)
}

/// Skewness from pre-extracted values and a pre-computed mean.
pub fn skewness_from_vals(vals: &[f64], mean: f64) -> Result<f64, ProfilingError> {
    let n = vals.len();
    if n < 3 {
        return Err(ProfilingError::InsufficientData(3));
    }
    let nf = n as f64;
    let m2: f64 = vals.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / nf;
    if m2 == 0.0 {
        return Ok(0.0);
    }
    let m3: f64 = vals.iter().map(|x| (x - mean).powi(3)).sum::<f64>() / nf;
    let g1 = m3 / m2.powf(1.5);
    let adj = (nf * (nf - 1.0)).sqrt() / (nf - 2.0);
    Ok(adj * g1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symmetric() {
        let df = df! { "x" => &[1.0f64, 2.0, 3.0, 4.0, 5.0] }.unwrap();
        assert!(skewness(&df, "x").unwrap().abs() < 1e-10);
    }

    #[test]
    fn test_right_skewed() {
        let df = df! { "x" => &[1.0f64, 1.0, 1.0, 1.0, 10.0] }.unwrap();
        assert!(skewness(&df, "x").unwrap() > 0.0);
    }

    #[test]
    fn test_left_skewed() {
        let df = df! { "x" => &[10.0f64, 9.0, 9.0, 9.0, 1.0] }.unwrap();
        assert!(skewness(&df, "x").unwrap() < 0.0);
    }

    #[test]
    fn test_constant_column() {
        let df = df! { "x" => &[5.0f64, 5.0, 5.0, 5.0] }.unwrap();
        assert!((skewness(&df, "x").unwrap()).abs() < 1e-10);
    }
}
