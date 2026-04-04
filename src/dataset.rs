use std::collections::{HashMap, HashSet};
use std::path::Path;

use ndarray::Array2;
use ndarray_stats::CorrelationExt;

use crate::error::ProfilingError;
use crate::profiling;
use crate::profiling::correlation::CorrelationMatrix;
use crate::profiling::numeric::Quantiles;

/// Column data — either numeric (f64) or string.
#[derive(Debug, Clone)]
pub enum ColumnData {
    Numeric(Vec<f64>),
    String(Vec<String>),
}

/// Precomputed statistics for all numeric columns.
struct PrecomputedStats {
    /// Column name → arithmetic mean.
    means: HashMap<String, f64>,
    /// Column name → sample variance (ddof=1).
    variances: HashMap<String, f64>,
    /// Full covariance matrix.
    covariance: CovarianceMatrix,
    /// Pearson correlation matrix.
    correlation: CorrelationMatrix,
}

/// Raw covariance matrix stored alongside column names.
#[derive(Debug, Clone, serde::Serialize)]
pub struct CovarianceMatrix {
    pub columns: Vec<String>,
    /// Flat row-major m×m matrix.
    pub matrix: Vec<f64>,
}

impl CovarianceMatrix {
    pub fn get(&self, i: usize, j: usize) -> f64 {
        let m = self.columns.len();
        self.matrix[i * m + j]
    }
}

/// Precompute mean, covariance, correlation, and variance using ndarray-stats.
fn precompute(
    numeric_names: &[String],
    columns: &HashMap<String, ColumnData>,
    nrows: usize,
) -> PrecomputedStats {
    let m = numeric_names.len();

    let mut means: HashMap<String, f64> = HashMap::with_capacity(m);

    // Build an n×m Array2 for ndarray-stats covariance.
    let mut matrix = Array2::<f64>::zeros((nrows, m));
    for (j, name) in numeric_names.iter().enumerate() {
        if let Some(ColumnData::Numeric(vals)) = columns.get(name) {
            for (i, &v) in vals.iter().enumerate() {
                matrix[[i, j]] = v;
            }
            let mean = matrix.column(j).mean().unwrap_or(0.0);
            means.insert(name.clone(), mean);
        }
    }

    // Covariance matrix via ndarray-stats (ddof=1).
    // Variances are read from the diagonal.
    let (cov_flat, corr_matrix, variances) = if m >= 2 && nrows > 1 {
        let cov_mat = matrix.t().cov(1.0).unwrap_or_else(|_| Array2::zeros((m, m)));

        let mut flat = vec![0.0f64; m * m];
        for i in 0..m {
            for j in 0..m {
                flat[i * m + j] = cov_mat[[i, j]];
            }
        }

        // Variances from the diagonal.
        let mut vars: HashMap<String, f64> = HashMap::with_capacity(m);
        for (j, name) in numeric_names.iter().enumerate() {
            vars.insert(name.clone(), cov_mat[[j, j]]);
        }

        // Correlation from covariance.
        let mut corr = vec![vec![0.0f64; m]; m];
        for i in 0..m {
            corr[i][i] = 1.0;
            for j in (i + 1)..m {
                let denom = (cov_mat[[i, i]] * cov_mat[[j, j]]).sqrt();
                let r = if denom == 0.0 {
                    0.0
                } else {
                    cov_mat[[i, j]] / denom
                };
                corr[i][j] = r;
                corr[j][i] = r;
            }
        }
        (flat, corr, vars)
    } else if m == 1 {
        // Single column — compute variance directly.
        let var = if nrows > 1 {
            let mean = means[&numeric_names[0]];
            matrix.column(0).mapv(|x| (x - mean).powi(2)).sum() / (nrows - 1) as f64
        } else {
            0.0
        };
        let mut vars = HashMap::new();
        vars.insert(numeric_names[0].clone(), var);
        (vec![var], vec![vec![1.0]], vars)
    } else {
        (vec![], vec![], HashMap::new())
    };

    let covariance = CovarianceMatrix {
        columns: numeric_names.to_vec(),
        matrix: cov_flat,
    };

    let correlation = CorrelationMatrix {
        columns: numeric_names.to_vec(),
        matrix: corr_matrix,
    };

    PrecomputedStats {
        means,
        variances,
        covariance,
        correlation,
    }
}

pub struct Dataset {
    /// Column name → data (preserves insertion order via `col_order`).
    columns: HashMap<String, ColumnData>,
    /// Column names in original order.
    col_order: Vec<String>,
    /// Names of numeric columns only, in order.
    #[allow(dead_code)]
    numeric_names: Vec<String>,
    /// Number of rows.
    nrows: usize,
    /// Precomputed stats.
    stats: PrecomputedStats,
}

impl Dataset {
    pub fn from_csv<P: AsRef<Path>>(path: P) -> Result<Self, ProfilingError> {
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_path(path)?;

        let headers: Vec<String> = reader
            .headers()?
            .iter()
            .map(|h| h.to_string())
            .collect();

        let ncols = headers.len();

        // Accumulate raw string values per column.
        let mut raw_cols: Vec<Vec<String>> = vec![Vec::new(); ncols];
        for result in reader.records() {
            let record = result?;
            for (j, field) in record.iter().enumerate() {
                if j < ncols {
                    raw_cols[j].push(field.to_string());
                }
            }
        }

        let nrows = raw_cols.first().map_or(0, |c| c.len());

        // Determine column types: if all non-empty values parse as f64, it's numeric.
        let mut columns = HashMap::with_capacity(ncols);
        let mut numeric_names = Vec::new();

        for (j, name) in headers.iter().enumerate() {
            let raw = &raw_cols[j];
            let parsed: Vec<Option<f64>> = raw
                .iter()
                .map(|s| {
                    let trimmed = s.trim();
                    if trimmed.is_empty() {
                        None
                    } else {
                        trimmed.parse::<f64>().ok()
                    }
                })
                .collect();

            let non_empty_count = parsed.iter().filter(|v| v.is_some()).count();
            let all_numeric = non_empty_count > 0
                && parsed
                    .iter()
                    .zip(raw.iter())
                    .all(|(p, s)| p.is_some() || s.trim().is_empty());

            if all_numeric {
                let vals: Vec<f64> = parsed.iter().map(|v| v.unwrap_or(0.0)).collect();
                columns.insert(name.clone(), ColumnData::Numeric(vals));
                numeric_names.push(name.clone());
            } else {
                columns.insert(name.clone(), ColumnData::String(raw.clone()));
            }
        }

        let stats = precompute(&numeric_names, &columns, nrows);

        Ok(Self {
            columns,
            col_order: headers,
            numeric_names,
            nrows,
            stats,
        })
    }

    /// Build a Dataset from pre-built column data (for tests / benches).
    pub fn from_columns(col_order: Vec<String>, columns: HashMap<String, ColumnData>) -> Self {
        let nrows = columns
            .values()
            .next()
            .map(|c| match c {
                ColumnData::Numeric(v) => v.len(),
                ColumnData::String(v) => v.len(),
            })
            .unwrap_or(0);

        let numeric_names: Vec<String> = col_order
            .iter()
            .filter(|n| matches!(columns.get(*n), Some(ColumnData::Numeric(_))))
            .cloned()
            .collect();

        let stats = precompute(&numeric_names, &columns, nrows);

        Self {
            columns,
            col_order,
            numeric_names,
            nrows,
            stats,
        }
    }

    // --- Shape ---

    pub fn row_count(&self) -> usize {
        self.nrows
    }

    pub fn column_count(&self) -> usize {
        self.col_order.len()
    }

    pub fn column_types(&self) -> Vec<(String, String)> {
        self.col_order
            .iter()
            .map(|name| {
                let dtype = match self.columns.get(name) {
                    Some(ColumnData::Numeric(_)) => "f64",
                    Some(ColumnData::String(_)) => "str",
                    None => "unknown",
                };
                (name.clone(), dtype.to_string())
            })
            .collect()
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
        let vals = self.get_numeric(column)?;
        if vals.is_empty() {
            return Err(ProfilingError::EmptyDataset);
        }
        let mut sorted = vals.to_vec();
        sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        Ok(profiling::numeric::quantiles_from_sorted(&sorted))
    }

    // --- Distribution (uses cached mean) ---

    pub fn skewness(&self, column: &str) -> Result<f64, ProfilingError> {
        let vals = self.get_numeric(column)?;
        let mean = self.stats.means[column];
        profiling::distribution::skewness_from_vals(vals, mean)
    }

    // --- Categorical ---

    pub fn unique_count(&self, column: &str) -> Result<usize, ProfilingError> {
        match self.columns.get(column) {
            Some(ColumnData::Numeric(vals)) => {
                let set: HashSet<u64> = vals.iter().map(|v| v.to_bits()).collect();
                Ok(set.len())
            }
            Some(ColumnData::String(vals)) => {
                let set: HashSet<&str> = vals.iter().map(|s| s.as_str()).collect();
                Ok(set.len())
            }
            None => Err(ProfilingError::ColumnNotFound(column.to_string())),
        }
    }

    // --- Entropy ---

    pub fn entropy(&self, column: &str) -> Result<f64, ProfilingError> {
        match self.columns.get(column) {
            Some(ColumnData::Numeric(vals)) => profiling::entropy::entropy_numeric(vals),
            Some(ColumnData::String(vals)) => profiling::entropy::entropy_categorical(vals),
            None => Err(ProfilingError::ColumnNotFound(column.to_string())),
        }
    }

    // --- Covariance (cached) ---

    pub fn covariance_matrix(&self) -> Result<CovarianceMatrix, ProfilingError> {
        Ok(self.stats.covariance.clone())
    }

    // --- Correlation (cached) ---

    pub fn correlation_matrix(&self) -> Result<CorrelationMatrix, ProfilingError> {
        Ok(self.stats.correlation.clone())
    }

    // --- Sparsity ---

    pub fn sparsity(&self, column: &str) -> Result<f64, ProfilingError> {
        match self.columns.get(column) {
            Some(ColumnData::Numeric(vals)) => {
                if vals.is_empty() {
                    return Err(ProfilingError::EmptyDataset);
                }
                let zero_count = vals.iter().filter(|&&v| v == 0.0).count();
                Ok(zero_count as f64 / vals.len() as f64)
            }
            Some(ColumnData::String(vals)) => {
                if vals.is_empty() {
                    return Err(ProfilingError::EmptyDataset);
                }
                let empty_count = vals.iter().filter(|v| v.is_empty()).count();
                Ok(empty_count as f64 / vals.len() as f64)
            }
            None => Err(ProfilingError::ColumnNotFound(column.to_string())),
        }
    }

    // --- Reservoir Computing ---

    pub fn surrogate_test(
        &self,
        column: &str,
        num_surrogates: usize,
    ) -> Result<profiling::reservoir::SurrogateTestResult, ProfilingError> {
        let vals = self.get_numeric(column)?;
        profiling::reservoir::surrogate_test(vals, num_surrogates)
    }

    pub fn bds_test(
        &self,
        column: &str,
        embedding_dim: usize,
        epsilon: f64,
    ) -> Result<profiling::reservoir::BdsTestResult, ProfilingError> {
        let vals = self.get_numeric(column)?;
        profiling::reservoir::bds_test(vals, embedding_dim, epsilon)
    }

    pub fn lyapunov_exponent(
        &self,
        column: &str,
        embedding_dim: usize,
        tau: usize,
    ) -> Result<f64, ProfilingError> {
        let vals = self.get_numeric(column)?;
        profiling::reservoir::lyapunov_exponent(vals, embedding_dim, tau)
    }

    pub fn dependence_comparison(
        &self,
        column: &str,
        max_lag: usize,
    ) -> Result<profiling::reservoir::DependenceComparison, ProfilingError> {
        let vals = self.get_numeric(column)?;
        profiling::reservoir::dependence_comparison(vals, max_lag)
    }

    pub fn delay_embedding(
        &self,
        column: &str,
        max_dim: usize,
    ) -> Result<profiling::reservoir::DelayEmbedding, ProfilingError> {
        let vals = self.get_numeric(column)?;
        profiling::reservoir::delay_embedding(vals, max_dim)
    }

    pub fn memory_profile(
        &self,
        column: &str,
        max_lag: usize,
    ) -> Result<profiling::reservoir::MemoryProfile, ProfilingError> {
        let vals = self.get_numeric(column)?;
        profiling::reservoir::memory_profile(vals, max_lag)
    }

    // --- Helpers ---

    fn get_numeric(&self, column: &str) -> Result<&[f64], ProfilingError> {
        match self.columns.get(column) {
            Some(ColumnData::Numeric(vals)) => Ok(vals),
            Some(ColumnData::String(_)) => Err(ProfilingError::NotNumeric(column.to_string())),
            None => Err(ProfilingError::ColumnNotFound(column.to_string())),
        }
    }
}
