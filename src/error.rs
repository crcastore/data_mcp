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
    #[error("{0}")]
    Csv(#[from] csv::Error),
    #[error("parquet error: {0}")]
    Parquet(#[from] parquet::errors::ParquetError),
    #[error("arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),
    #[error("duckdb conversion failed: {0}")]
    DuckDb(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}
