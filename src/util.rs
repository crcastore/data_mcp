use polars::prelude::*;

use crate::error::ProfilingError;

/// Extract f64 values from a column (assumes Float64).
pub fn extract_f64_values(df: &DataFrame, column: &str) -> Result<Vec<f64>, ProfilingError> {
    let col = df
        .column(column)
        .map_err(|_| ProfilingError::ColumnNotFound(column.to_string()))?;
    let series = col.as_materialized_series();
    let ca = series.f64()?;
    Ok(ca.into_no_null_iter().collect())
}

/// Names of all numeric (Float64) columns in the dataframe.
pub fn numeric_column_names(df: &DataFrame) -> Vec<String> {
    df.columns()
        .iter()
        .filter(|c| c.dtype() == &DataType::Float64)
        .map(|c| c.name().to_string())
        .collect()
}
