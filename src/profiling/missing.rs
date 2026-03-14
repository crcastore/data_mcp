use polars::prelude::*;

use crate::error::ProfilingError;

/// Fraction of null values in the column (0.0 – 1.0).
pub fn missing_rate(df: &DataFrame, column: &str) -> Result<f64, ProfilingError> {
    let col = df
        .column(column)
        .map_err(|_| ProfilingError::ColumnNotFound(column.to_string()))?;
    let len = col.len();
    if len == 0 {
        return Err(ProfilingError::EmptyDataset);
    }
    Ok(col.null_count() as f64 / len as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_missing_rate_with_nulls() {
        let df = df! {
            "a" => &[Some(1i64), None, Some(3), None, Some(5)],
        }
        .unwrap();
        let rate = missing_rate(&df, "a").unwrap();
        assert!((rate - 0.4).abs() < 1e-10);
    }

    #[test]
    fn test_missing_rate_no_nulls() {
        let df = df! { "a" => &[1i64, 2, 3] }.unwrap();
        assert!((missing_rate(&df, "a").unwrap()).abs() < 1e-10);
    }

    #[test]
    fn test_missing_rate_column_not_found() {
        let df = df! { "a" => &[1i64] }.unwrap();
        assert!(missing_rate(&df, "z").is_err());
    }
}
