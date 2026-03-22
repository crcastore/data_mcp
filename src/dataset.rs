use polars::prelude::*;
use std::collections::HashMap;
use std::path::Path;

use crate::error::ProfilingError;
use crate::profiling;
use crate::profiling::correlation::CorrelationMatrix;
use crate::profiling::numeric::Quantiles;

/// Cast every numeric column to Float64 at load time — all data assumed numeric.
fn cast_all_to_f64(df: DataFrame) -> Result<DataFrame, ProfilingError> {
    let height = df.height();
    let cols: Vec<Column> = df
        .columns()
        .iter()
        .map(|c: &Column| {
            if c.dtype().is_numeric() {
                c.cast(&DataType::Float64).unwrap_or_else(|_| c.clone())
            } else {
                c.clone()
            }
        })
        .collect();
    Ok(DataFrame::new(height, cols)?)
}

/// Precomputed statistics for all numeric (Float64) columns.
struct PrecomputedStats {
    /// Column name → extracted f64 values (no nulls).
    column_data: HashMap<String, Vec<f64>>,
    /// Column name → arithmetic mean (via Polars Series::mean).
    means: HashMap<String, f64>,
    /// Column name → sample variance, ddof=1 (via Polars Series::var).
    variances: HashMap<String, f64>,
    /// Pearson correlation matrix (via rayon-parallel BLAS-style computation).
    correlation: CorrelationMatrix,
}

/// Precompute all stats using Polars built-in SIMD-optimised aggregations.
fn precompute(df: &DataFrame) -> PrecomputedStats {
    let mut column_data: HashMap<String, Vec<f64>> = HashMap::new();
    let mut means: HashMap<String, f64> = HashMap::new();
    let mut variances: HashMap<String, f64> = HashMap::new();

    for col in df.columns() {
        if col.dtype() != &DataType::Float64 {
            continue;
        }
        let name = col.name().to_string();
        let series = col.as_materialized_series();
        let ca = series.f64().unwrap();

        // Polars SIMD-optimised per-column stats
        means.insert(name.clone(), series.mean().unwrap_or(0.0));
        variances.insert(name.clone(), series.var(1).unwrap_or(0.0));

        // Cache raw values for downstream use (skewness, quantiles, reservoir)
        column_data.insert(name, ca.into_no_null_iter().collect());
    }

    // Rayon-parallel Pearson correlation matrix over all numeric columns
    let correlation = profiling::correlation::correlation_matrix(df)
        .unwrap_or_else(|_| CorrelationMatrix {
            columns: vec![],
            matrix: vec![],
        });

    PrecomputedStats {
        column_data,
        means,
        variances,
        correlation,
    }
}

pub struct Dataset {
    df: DataFrame,
    stats: PrecomputedStats,
}

impl Dataset {
    pub fn from_csv<P: AsRef<Path>>(path: P) -> Result<Self, ProfilingError> {
        let df = CsvReadOptions::default()
            .try_into_reader_with_file_path(Some(path.as_ref().to_path_buf()))?
            .finish()?;
        let df = cast_all_to_f64(df)?;
        let stats = precompute(&df);
        Ok(Self { df, stats })
    }

    pub fn new(df: DataFrame) -> Self {
        let df = cast_all_to_f64(df).expect("failed to cast columns to f64");
        let stats = precompute(&df);
        Self { df, stats }
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

    // --- Numeric (cached) ---

    pub fn mean(&self, column: &str) -> Result<f64, ProfilingError> {
        self.stats
            .means
            .get(column)
            .copied()
            .ok_or_else(|| ProfilingError::ColumnNotFound(column.to_string()))
    }

    pub fn variance(&self, column: &str) -> Result<f64, ProfilingError> {
        self.stats
            .variances
            .get(column)
            .copied()
            .ok_or_else(|| ProfilingError::ColumnNotFound(column.to_string()))
    }

    pub fn quantiles(&self, column: &str) -> Result<Quantiles, ProfilingError> {
        let vals = self
            .stats
            .column_data
            .get(column)
            .ok_or_else(|| ProfilingError::ColumnNotFound(column.to_string()))?;
        if vals.is_empty() {
            return Err(ProfilingError::EmptyDataset);
        }
        let mut sorted = vals.clone();
        sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        Ok(profiling::numeric::quantiles_from_sorted(&sorted))
    }

    // --- Distribution (uses cached mean) ---

    pub fn skewness(&self, column: &str) -> Result<f64, ProfilingError> {
        let vals = self
            .stats
            .column_data
            .get(column)
            .ok_or_else(|| ProfilingError::ColumnNotFound(column.to_string()))?;
        let mean = self.stats.means[column];
        profiling::distribution::skewness_from_vals(vals, mean)
    }

    // --- Categorical ---

    pub fn unique_count(&self, column: &str) -> Result<usize, ProfilingError> {
        profiling::categorical::unique_count(&self.df, column)
    }

    // --- Entropy ---

    pub fn entropy(&self, column: &str) -> Result<f64, ProfilingError> {
        profiling::entropy::entropy(&self.df, column)
    }

    // --- Correlation (cached) ---

    pub fn correlation_matrix(&self) -> Result<CorrelationMatrix, ProfilingError> {
        Ok(self.stats.correlation.clone())
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
