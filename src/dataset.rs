use std::collections::{HashMap, HashSet};
use std::path::Path;

use faer::Mat;
use ndarray::Array2;
use ndarray_stats::CorrelationExt;

use crate::error::ProfilingError;
use crate::profiling;
use crate::profiling::correlation::CorrelationMatrix;
use crate::profiling::numeric::Quantiles;

/// Precomputed statistics for all columns.
struct PrecomputedStats {
    /// Column name → arithmetic mean.
    means: HashMap<String, f64>,
    /// Column name → sample variance (ddof=1).
    variances: HashMap<String, f64>,
    /// Full covariance matrix.
    covariance: CovarianceMatrix,
    /// Pearson correlation matrix.
    correlation: CorrelationMatrix,
    /// SVD of the covariance matrix.
    svd: Option<SvdDecomposition>,
}

/// SVD decomposition of the covariance matrix (A = U S Vᵀ).
#[derive(Debug, Clone, serde::Serialize)]
pub struct SvdDecomposition {
    /// Column names corresponding to the matrix axes.
    pub columns: Vec<String>,
    /// Singular values in descending order (length m).
    pub singular_values: Vec<f64>,
    /// Left singular vectors U, flat row-major m×m.
    pub u: Vec<f64>,
    /// Right singular vectors Vᵀ, flat row-major m×m.
    pub vt: Vec<f64>,
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
    col_order: &[String],
    columns: &HashMap<String, Vec<f64>>,
    nrows: usize,
) -> PrecomputedStats {
    let m = col_order.len();

    let mut means: HashMap<String, f64> = HashMap::with_capacity(m);

    // Build an n×m Array2 for ndarray-stats covariance.
    let mut matrix = Array2::<f64>::zeros((nrows, m));
    for (j, name) in col_order.iter().enumerate() {
        if let Some(vals) = columns.get(name) {
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
        for (j, name) in col_order.iter().enumerate() {
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
            let mean = means[&col_order[0]];
            matrix.column(0).mapv(|x| (x - mean).powi(2)).sum() / (nrows - 1) as f64
        } else {
            0.0
        };
        let mut vars = HashMap::new();
        vars.insert(col_order[0].clone(), var);
        (vec![var], vec![vec![1.0]], vars)
    } else {
        (vec![], vec![], HashMap::new())
    };

    let covariance = CovarianceMatrix {
        columns: col_order.to_vec(),
        matrix: cov_flat,
    };

    let correlation = CorrelationMatrix {
        columns: col_order.to_vec(),
        matrix: corr_matrix,
    };

    // SVD of the covariance matrix via faer.
    let svd = if m >= 2 {
        let cov_faer = Mat::from_fn(m, m, |i, j| covariance.get(i, j));
        match cov_faer.svd() {
            Ok(decomp) => {
                let u_mat = decomp.U();
                let vt_mat = decomp.V().transpose();
                let s_diag = decomp.S();

                let mut u_flat = vec![0.0f64; m * m];
                let mut vt_flat = vec![0.0f64; m * m];
                let mut singular_values = vec![0.0f64; m];

                for i in 0..m {
                    singular_values[i] = s_diag[i];
                    for j in 0..m {
                        u_flat[i * m + j] = u_mat[(i, j)];
                        vt_flat[i * m + j] = vt_mat[(i, j)];
                    }
                }

                Some(SvdDecomposition {
                    columns: col_order.to_vec(),
                    singular_values,
                    u: u_flat,
                    vt: vt_flat,
                })
            }
            Err(_) => None,
        }
    } else {
        None
    };

    PrecomputedStats {
        means,
        variances,
        covariance,
        correlation,
        svd,
    }
}

pub struct Dataset {
    /// Column name → f64 data.
    columns: HashMap<String, Vec<f64>>,
    /// Column names in original order.
    col_order: Vec<String>,
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

        // Parse all columns as f64. Non-parseable / empty values become 0.0.
        let mut columns = HashMap::with_capacity(ncols);

        for (j, name) in headers.iter().enumerate() {
            let raw = &raw_cols[j];
            let vals: Vec<f64> = raw
                .iter()
                .map(|s| s.trim().parse::<f64>().unwrap_or(0.0))
                .collect();
            columns.insert(name.clone(), vals);
        }

        let stats = precompute(&headers, &columns, nrows);

        Ok(Self {
            columns,
            col_order: headers,
            nrows,
            stats,
        })
    }

    /// Build a Dataset from pre-built column data (for tests / benches).
    pub fn from_columns(col_order: Vec<String>, columns: HashMap<String, Vec<f64>>) -> Self {
        let nrows = columns.values().next().map_or(0, |v| v.len());

        let stats = precompute(&col_order, &columns, nrows);

        Self {
            columns,
            col_order,
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
            .map(|name| (name.clone(), "f64".to_string()))
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
        let vals = self.get_column(column)?;
        if vals.is_empty() {
            return Err(ProfilingError::EmptyDataset);
        }
        let mut sorted = vals.to_vec();
        sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        Ok(profiling::numeric::quantiles_from_sorted(&sorted))
    }

    // --- Distribution (uses cached mean) ---

    pub fn skewness(&self, column: &str) -> Result<f64, ProfilingError> {
        let vals = self.get_column(column)?;
        let mean = self.stats.means[column];
        profiling::distribution::skewness_from_vals(vals, mean)
    }

    // --- Unique count ---

    pub fn unique_count(&self, column: &str) -> Result<usize, ProfilingError> {
        let vals = self.get_column(column)?;
        let set: HashSet<u64> = vals.iter().map(|v| v.to_bits()).collect();
        Ok(set.len())
    }

    // --- Entropy ---

    pub fn entropy(&self, column: &str) -> Result<f64, ProfilingError> {
        let vals = self.get_column(column)?;
        profiling::entropy::entropy_numeric(vals)
    }

    // --- Covariance (cached) ---

    pub fn covariance_matrix(&self) -> Result<CovarianceMatrix, ProfilingError> {
        Ok(self.stats.covariance.clone())
    }

    // --- Correlation (cached) ---

    pub fn correlation_matrix(&self) -> Result<CorrelationMatrix, ProfilingError> {
        Ok(self.stats.correlation.clone())
    }

    // --- SVD (cached) ---

    pub fn svd(&self) -> Result<&SvdDecomposition, ProfilingError> {
        self.stats
            .svd
            .as_ref()
            .ok_or(ProfilingError::NotEnoughColumns)
    }

    // --- Sparsity ---

    pub fn sparsity(&self, column: &str) -> Result<f64, ProfilingError> {
        let vals = self.get_column(column)?;
        if vals.is_empty() {
            return Err(ProfilingError::EmptyDataset);
        }
        let zero_count = vals.iter().filter(|&&v| v == 0.0).count();
        Ok(zero_count as f64 / vals.len() as f64)
    }

    // --- Reservoir Computing ---

    pub fn surrogate_test(
        &self,
        column: &str,
        num_surrogates: usize,
    ) -> Result<profiling::reservoir::SurrogateTestResult, ProfilingError> {
        let vals = self.get_column(column)?;
        profiling::reservoir::surrogate_test(vals, num_surrogates)
    }

    pub fn bds_test(
        &self,
        column: &str,
        embedding_dim: usize,
        epsilon: f64,
    ) -> Result<profiling::reservoir::BdsTestResult, ProfilingError> {
        let vals = self.get_column(column)?;
        profiling::reservoir::bds_test(vals, embedding_dim, epsilon)
    }

    pub fn lyapunov_exponent(
        &self,
        column: &str,
        embedding_dim: usize,
        tau: usize,
    ) -> Result<f64, ProfilingError> {
        let vals = self.get_column(column)?;
        profiling::reservoir::lyapunov_exponent(vals, embedding_dim, tau)
    }

    pub fn dependence_comparison(
        &self,
        column: &str,
        max_lag: usize,
    ) -> Result<profiling::reservoir::DependenceComparison, ProfilingError> {
        let vals = self.get_column(column)?;
        profiling::reservoir::dependence_comparison(vals, max_lag)
    }

    pub fn delay_embedding(
        &self,
        column: &str,
        max_dim: usize,
    ) -> Result<profiling::reservoir::DelayEmbedding, ProfilingError> {
        let vals = self.get_column(column)?;
        profiling::reservoir::delay_embedding(vals, max_dim)
    }

    pub fn memory_profile(
        &self,
        column: &str,
        max_lag: usize,
    ) -> Result<profiling::reservoir::MemoryProfile, ProfilingError> {
        let vals = self.get_column(column)?;
        profiling::reservoir::memory_profile(vals, max_lag)
    }

    // --- Helpers ---

    fn get_column(&self, column: &str) -> Result<&[f64], ProfilingError> {
        self.columns
            .get(column)
            .map(|v| v.as_slice())
            .ok_or_else(|| ProfilingError::ColumnNotFound(column.to_string()))
    }
}
