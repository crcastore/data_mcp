use polars::prelude::*;

use crate::error::ProfilingError;

/// Fraction of zero-like + null values in the column (0.0 – 1.0).
///
/// For numeric columns, "zero-like" means `== 0`.
/// For string columns, "zero-like" means empty string `""`.
pub fn sparsity(df: &DataFrame, column: &str) -> Result<f64, ProfilingError> {
    let col = df
        .column(column)
        .map_err(|_| ProfilingError::ColumnNotFound(column.to_string()))?;
    let total = col.len();
    if total == 0 {
        return Err(ProfilingError::EmptyDataset);
    }
    let null_count = col.null_count();

    let zero_count = if col.dtype().is_numeric() {
        let casted = col.cast(&DataType::Float64)?;
        let series = casted.as_materialized_series();
        let ca = series.f64()?;
        ca.iter().flatten().filter(|v| *v == 0.0 || v.is_nan()).count()
    } else {
        let casted = col.cast(&DataType::String)?;
        let series = casted.as_materialized_series();
        let ca = series.str()?;
        ca.iter().flatten().filter(|v| v.is_empty()).count()
    };

    Ok((null_count + zero_count) as f64 / total as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparsity_numeric() {
        let df = df! { "x" => &[0.0f64, 0.0, 1.0, 2.0, 0.0] }.unwrap();
        assert!((sparsity(&df, "x").unwrap() - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_sparsity_with_nulls() {
        let df = df! { "x" => &[Some(0.0f64), None, Some(1.0), None] }.unwrap();
        // 1 zero + 2 nulls out of 4 → 0.75
        assert!((sparsity(&df, "x").unwrap() - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_sparsity_string() {
        let df = df! { "s" => &["", "hello", "", "world"] }.unwrap();
        assert!((sparsity(&df, "s").unwrap() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_sparsity_dense() {
        let df = df! { "x" => &[1.0f64, 2.0, 3.0] }.unwrap();
        assert!((sparsity(&df, "x").unwrap()).abs() < 1e-10);
    }

    #[test]
    fn test_sparsity_with_nan() {
        let df = df! { "x" => &[Some(f64::NAN), Some(0.0f64), Some(1.0), None] }.unwrap();
        // NaN(1) + zero(1) + null(1) out of 4 → 0.75
        assert!((sparsity(&df, "x").unwrap() - 0.75).abs() < 1e-10);
    }
}
