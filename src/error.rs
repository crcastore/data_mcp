use thiserror::Error;

#[derive(Debug, Error)]
pub enum ProfilingError {
    #[error("column not found: {0}")]
    ColumnNotFound(String),
    #[error("dataset is empty")]
    EmptyDataset,
    #[error("not enough data points (need at least {0})")]
    InsufficientData(usize),
    #[error("not enough columns for this operation (need at least 2)")]
    NotEnoughColumns,
    #[error("invalid prediction task configuration: {0}")]
    InvalidPredictionTask(String),
    #[error("duckdb error: {0}")]
    DuckDb(#[from] duckdb::Error),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}
