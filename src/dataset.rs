use polars::prelude::*;
use std::collections::HashMap;
use std::path::Path;

use ndarray::{Array2, Axis};
use ndarray_stats::CorrelationExt;

use crate::error::ProfilingError;
use crate::profiling;
use crate::profiling::correlation::CorrelationMatrix;
use crate::profiling::numeric::Quantiles;

/// Cast every numeric column to Float64 at load time.
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
    /// Column name → arithmetic mean.
    means: HashMap<String, f64>,
    /// Column name → sample variance (ddof=1).
    variances: HashMap<String, f64>,
    /// Pearson correlation matrix across all numeric columns.
    correlation: CorrelationMatrix,
}

/// Build precomputed stats using ndarray matrix algebra.
fn precompute(df: &DataFrame) -> PrecomputedStats {
    let numeric_names: Vec<String> = df
        .columns()
        .iter()
        .filter(|c| c.dtype() == &DataType::Float64)
        .map(|c| c.name().to_string())
        .collect();

    let n = df.height();
    let m = numeric_names.len();

    // ── Step 1: Extract columns into HashMap and build an n×m ndarray ──
    let mut column_data: HashMap<String, Vec<f64>> = HashMap::with_capacity(m);
    let mut data = Array2::<f64>::zeros((n, m));

    for (j, name) in numeric_names.iter().enumerate() {
        let col = df.column(name).unwrap();
        let series = col.as_materialized_series();
        let ca = series.f64().unwrap();
        let vals: Vec<f64> = ca.into_no_null_iter().collect();
        for (k, &v) in vals.iter().enumerate() {
            data[[k, j]] = v;
        }
        column_data.insert(name.clone(), vals);
    }

    // ── Step 2: Means via ndarray (μ = mean along axis 0) ──
    let means_arr = data.mean_axis(Axis(0)).unwrap();
    let mut means: HashMap<String, f64> = HashMap::with_capacity(m);
    for (j, name) in numeric_names.iter().enumerate() {
        means.insert(name.clone(), means_arr[j]);
    }

    // ── Step 3: Covariance matrix via ndarray-stats  Σ = Cᵀ·C / (n−1) ──
    //   .cov() requires n > ddof and m >= 1. For m < 2 we skip it and
    //   compute variance directly from the single column.
    let (variances, correlation) = if m >= 2 && n > 1 {
        let cov = data.cov(1.0).unwrap(); // ddof=1 → sample covariance

        let mut variances: HashMap<String, f64> = HashMap::with_capacity(m);
        for (j, name) in numeric_names.iter().enumerate() {
            variances.insert(name.clone(), cov[[j, j]]);
        }

        // Correlation: r_ij = Σ_ij / √(Σ_ii · Σ_jj)
        let mut matrix = vec![vec![0.0f64; m]; m];
        for i in 0..m {
            matrix[i][i] = 1.0;
            for j in (i + 1)..m {
                let denom = (cov[[i, i]] * cov[[j, j]]).sqrt();
                let r = if denom == 0.0 { 0.0 } else { cov[[i, j]] / denom };
                matrix[i][j] = r;
                matrix[j][i] = r;
            }
        }

        let correlation = CorrelationMatrix {
            columns: numeric_names.clone(),
            matrix,
        };
        (variances, correlation)
    } else {
        // Single column or empty — compute variance manually.
        let mut variances: HashMap<String, f64> = HashMap::with_capacity(m);
        for (j, name) in numeric_names.iter().enumerate() {
            let mu = means_arr[j];
            let vals = &column_data[name];
            if vals.len() > 1 {
                let var = vals.iter().map(|x| (x - mu).powi(2)).sum::<f64>()
                    / (vals.len() as f64 - 1.0);
                variances.insert(name.clone(), var);
            } else {
                variances.insert(name.clone(), 0.0);
            }
        }
        let matrix = if m == 1 { vec![vec![1.0]] } else { vec![] };
        let correlation = CorrelationMatrix {
            columns: numeric_names.clone(),
            matrix,
        };
        (variances, correlation)
    };

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
