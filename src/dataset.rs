use polars::prelude::*;
use std::path::Path;

use crate::error::ProfilingError;
use crate::profiling;

pub struct Dataset {
    df: DataFrame,
}

impl Dataset {
    pub fn from_csv<P: AsRef<Path>>(path: P) -> Result<Self, ProfilingError> {
        let df = CsvReadOptions::default()
            .try_into_reader_with_file_path(Some(path.as_ref().to_path_buf()))?
            .finish()?;
        Ok(Self { df })
    }

    pub fn new(df: DataFrame) -> Self {
        Self { df }
    }

    pub fn dataframe(&self) -> &DataFrame {
        &self.df
    }

    // --- Shape ---

    pub fn row_count(&self) -> usize {
        profiling::shape::row_count(&self.df)
    }

    pub fn column_count(&self) -> usize {
        profiling::shape::column_count(&self.df)
    }

    pub fn column_types(&self) -> Vec<(String, String)> {
        profiling::shape::column_types(&self.df)
    }

    // --- Missing ---

    pub fn missing_rate(&self, column: &str) -> Result<f64, ProfilingError> {
        profiling::missing::missing_rate(&self.df, column)
    }

    // --- Numeric ---

    pub fn mean(&self, column: &str) -> Result<f64, ProfilingError> {
        profiling::numeric::mean(&self.df, column)
    }

    pub fn variance(&self, column: &str) -> Result<f64, ProfilingError> {
        profiling::numeric::variance(&self.df, column)
    }

    pub fn quantiles(&self, column: &str) -> Result<profiling::numeric::Quantiles, ProfilingError> {
        profiling::numeric::quantiles(&self.df, column)
    }

    // --- Distribution ---

    pub fn skewness(&self, column: &str) -> Result<f64, ProfilingError> {
        profiling::distribution::skewness(&self.df, column)
    }

    // --- Categorical ---

    pub fn unique_count(&self, column: &str) -> Result<usize, ProfilingError> {
        profiling::categorical::unique_count(&self.df, column)
    }

    // --- Entropy ---

    pub fn entropy(&self, column: &str) -> Result<f64, ProfilingError> {
        profiling::entropy::entropy(&self.df, column)
    }

    // --- Correlation ---

    pub fn correlation_matrix(
        &self,
    ) -> Result<profiling::correlation::CorrelationMatrix, ProfilingError> {
        profiling::correlation::correlation_matrix(&self.df)
    }

    // --- Sparsity ---

    pub fn sparsity(&self, column: &str) -> Result<f64, ProfilingError> {
        profiling::sparsity::sparsity(&self.df, column)
    }

    // --- Reservoir Computing ---

    pub fn surrogate_test(
        &self,
        column: &str,
        num_surrogates: usize,
    ) -> Result<profiling::reservoir::SurrogateTestResult, ProfilingError> {
        profiling::reservoir::surrogate_test(&self.df, column, num_surrogates)
    }

    pub fn bds_test(
        &self,
        column: &str,
        embedding_dim: usize,
        epsilon: f64,
    ) -> Result<profiling::reservoir::BdsTestResult, ProfilingError> {
        profiling::reservoir::bds_test(&self.df, column, embedding_dim, epsilon)
    }

    pub fn lyapunov_exponent(
        &self,
        column: &str,
        embedding_dim: usize,
        tau: usize,
    ) -> Result<f64, ProfilingError> {
        profiling::reservoir::lyapunov_exponent(&self.df, column, embedding_dim, tau)
    }

    pub fn dependence_comparison(
        &self,
        column: &str,
        max_lag: usize,
    ) -> Result<profiling::reservoir::DependenceComparison, ProfilingError> {
        profiling::reservoir::dependence_comparison(&self.df, column, max_lag)
    }

    pub fn delay_embedding(
        &self,
        column: &str,
        max_dim: usize,
    ) -> Result<profiling::reservoir::DelayEmbedding, ProfilingError> {
        profiling::reservoir::delay_embedding(&self.df, column, max_dim)
    }

    pub fn memory_profile(
        &self,
        column: &str,
        max_lag: usize,
    ) -> Result<profiling::reservoir::MemoryProfile, ProfilingError> {
        profiling::reservoir::memory_profile(&self.df, column, max_lag)
    }
}
