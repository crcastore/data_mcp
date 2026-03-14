use polars::prelude::*;

use crate::error::ProfilingError;

/// Count of distinct non-null values.
pub fn unique_count(df: &DataFrame, column: &str) -> Result<usize, ProfilingError> {
    let col = df
        .column(column)
        .map_err(|_| ProfilingError::ColumnNotFound(column.to_string()))?;
    let n = col.n_unique()?;
    // n_unique counts null as a distinct value when present; subtract it.
    if col.null_count() > 0 {
        Ok(n.saturating_sub(1))
    } else {
        Ok(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unique_count() {
        let df = df! { "c" => &["a", "b", "a", "c", "b"] }.unwrap();
        assert_eq!(unique_count(&df, "c").unwrap(), 3);
    }

    #[test]
    fn test_unique_count_with_nulls() {
        let df = df! { "c" => &[Some("a"), None, Some("a"), Some("b")] }.unwrap();
        assert_eq!(unique_count(&df, "c").unwrap(), 2);
    }

    #[test]
    fn test_unique_count_numeric() {
        let df = df! { "n" => &[1i64, 2, 2, 3, 3, 3] }.unwrap();
        assert_eq!(unique_count(&df, "n").unwrap(), 3);
    }
}
