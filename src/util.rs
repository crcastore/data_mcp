use polars::prelude::*;

use crate::error::ProfilingError;

/// Extract non-null f64 values from a numeric column.
pub fn extract_f64_values(df: &DataFrame, column: &str) -> Result<Vec<f64>, ProfilingError> {
    let col = df
        .column(column)
        .map_err(|_| ProfilingError::ColumnNotFound(column.to_string()))?;
    if !col.dtype().is_numeric() {
        return Err(ProfilingError::NotNumeric(column.to_string()));
    }
    let casted = col.cast(&DataType::Float64)?;
    let series = casted.as_materialized_series();
    let ca = series.f64()?;
    Ok(ca.iter().flatten().collect())
}

/// Names of all numeric columns in the dataframe.
pub fn numeric_column_names(df: &DataFrame) -> Vec<String> {
    df.columns()
        .iter()
        .filter(|c| c.dtype().is_numeric())
        .map(|c| c.name().to_string())
        .collect()
}
