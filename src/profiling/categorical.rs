use polars::prelude::*;

use crate::error::ProfilingError;

/// Count of distinct values.
pub fn unique_count(df: &DataFrame, column: &str) -> Result<usize, ProfilingError> {
    let col = df
        .column(column)
        .map_err(|_| ProfilingError::ColumnNotFound(column.to_string()))?;
    Ok(col.n_unique()?)
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
    fn test_unique_count_numeric() {
        let df = df! { "n" => &[1i64, 2, 2, 3, 3, 3] }.unwrap();
        assert_eq!(unique_count(&df, "n").unwrap(), 3);
    }
}
