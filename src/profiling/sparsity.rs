use crate::error::ProfilingError;

/// Fraction of zero values in a numeric column (0.0 – 1.0).
pub fn sparsity_numeric(vals: &[f64]) -> Result<f64, ProfilingError> {
    if vals.is_empty() {
        return Err(ProfilingError::EmptyDataset);
    }
    let zero_count = vals.iter().filter(|&&v| v == 0.0).count();
    Ok(zero_count as f64 / vals.len() as f64)
}

/// Fraction of empty-string values in a string column (0.0 – 1.0).
pub fn sparsity_string(vals: &[String]) -> Result<f64, ProfilingError> {
    if vals.is_empty() {
        return Err(ProfilingError::EmptyDataset);
    }
    let empty_count = vals.iter().filter(|v| v.is_empty()).count();
    Ok(empty_count as f64 / vals.len() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparsity_numeric() {
        assert!((sparsity_numeric(&[0.0, 0.0, 1.0, 2.0, 0.0]).unwrap() - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_sparsity_string() {
        let vals: Vec<String> = vec!["", "hello", "", "world"]
            .into_iter()
            .map(String::from)
            .collect();
        assert!((sparsity_string(&vals).unwrap() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_sparsity_dense() {
        assert!(sparsity_numeric(&[1.0, 2.0, 3.0]).unwrap().abs() < 1e-10);
    }
}
