use thiserror::Error;

#[derive(Debug, Error)]
pub enum ProfilingError {
    #[error("column not found: {0}")]
    ColumnNotFound(String),
    #[error("column '{0}' is not numeric")]
    NotNumeric(String),
    #[error("dataset is empty")]
    EmptyDataset,
    #[error("no numeric columns found")]
    NoNumericColumns,
    #[error("not enough data points (need at least {0})")]
    InsufficientData(usize),
    #[error("{0}")]
    Csv(#[from] csv::Error),
}
