use std::collections::HashMap;
use std::path::Path;
use std::process::Command;

use arrow::array::{Array, AsArray};
use arrow::datatypes::Float64Type;
use faer::Mat;
use ndarray::{Array2, Axis};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

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
    /// Column name → skewness.
    skewness: HashMap<String, f64>,
    /// Column name → entropy.
    entropy: HashMap<String, f64>,
    /// Column name → quantiles.
    quantiles: HashMap<String, Quantiles>,
    /// Column name → sparsity.
    sparsity: HashMap<String, f64>,
    /// Full covariance matrix.
    covariance: CovarianceMatrix,
    /// Pearson correlation matrix.
    correlation: CorrelationMatrix,
    /// Eigendecomposition of the covariance matrix.
    eigen: Option<EigenDecomposition>,
}

/// Eigendecomposition of the covariance matrix (A = Q Λ Qᵀ).
#[derive(Debug, Clone, serde::Serialize)]
pub struct EigenDecomposition {
    /// Column names corresponding to the matrix axes.
    pub columns: Vec<String>,
    /// Eigenvalues in descending order (length m).
    pub eigenvalues: Vec<f64>,
    /// Eigenvectors as rows, flat row-major m×m (each row is a principal axis).
    pub eigenvectors: Vec<f64>,
}

/// PCA result derived from the eigendecomposition of the covariance matrix.
#[derive(Debug, Clone, serde::Serialize)]
pub struct PcaResult {
    /// Original column names.
    pub columns: Vec<String>,
    /// Number of components returned.
    pub n_components: usize,
    /// Eigenvalues (variance explained by each component).
    pub explained_variance: Vec<f64>,
    /// Fraction of total variance explained by each component.
    pub explained_variance_ratio: Vec<f64>,
    /// Cumulative explained variance ratio.
    pub cumulative_variance_ratio: Vec<f64>,
    /// Principal component loadings, flat row-major n_components × m.
    /// Each row is a principal component (eigenvector).
    pub components: Vec<f64>,
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

    // Build an n×m Array2.
    let mut matrix = Array2::<f64>::zeros((nrows, m));
    for (j, name) in col_order.iter().enumerate() {
        if let Some(vals) = columns.get(name) {
            for (i, &v) in vals.iter().enumerate() {
                matrix[[i, j]] = v;
            }
        }
    }

    // Mean per column.
    let mean_row = matrix.mean_axis(Axis(0)).unwrap();
    let mut means: HashMap<String, f64> = HashMap::with_capacity(m);
    for (j, name) in col_order.iter().enumerate() {
        means.insert(name.clone(), mean_row[j]);
    }

    // Per-column stats: skewness, entropy, quantiles, sparsity.
    // Compute all in one loop to avoid repeated HashMap lookups.
    let mut skewness_map: HashMap<String, f64> = HashMap::with_capacity(m);
    let mut entropy_map: HashMap<String, f64> = HashMap::with_capacity(m);
    let mut quantiles_map: HashMap<String, Quantiles> = HashMap::with_capacity(m);
    let mut sparsity_map: HashMap<String, f64> = HashMap::with_capacity(m);

    for name in col_order.iter() {
        if let Some(vals) = columns.get(name) {
            let mean = means[name];

            // Skewness
            if let Ok(s) = profiling::distribution::skewness_from_vals(vals, mean) {
                skewness_map.insert(name.clone(), s);
            }

            // Entropy
            if let Ok(e) = profiling::entropy::entropy_numeric(vals) {
                entropy_map.insert(name.clone(), e);
            }

            // Quantiles
            if !vals.is_empty() {
                quantiles_map.insert(name.clone(), profiling::numeric::quantiles_select(vals));
            }

            // Sparsity
            if !vals.is_empty() {
                let zero_count = vals.iter().filter(|&&v| v == 0.0).count();
                sparsity_map.insert(name.clone(), zero_count as f64 / vals.len() as f64);
            }
        }
    }

    // Covariance via BLAS-accelerated matrix multiply: Cᵀ·C / (n-1).
    let (cov_flat, corr_matrix, variances) = if m >= 2 && nrows > 1 {
        let centered = &matrix - &mean_row;
        let n = nrows as f64;
        let cov_mat = centered.t().dot(&centered) / (n - 1.0);

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

    // Eigendecomposition of the symmetric covariance matrix via faer.
    let eigen = if m >= 2 {
        let cov_faer = Mat::from_fn(m, m, |i, j| covariance.get(i, j));
        match cov_faer.self_adjoint_eigen(faer::Side::Lower) {
            Ok(decomp) => {
                let u_mat = decomp.U();
                let s_diag = decomp.S();

                // Eigenvalues are in nondecreasing order; reverse to descending.
                let mut eigenvalues = vec![0.0f64; m];
                let mut vt_flat = vec![0.0f64; m * m];

                for i in 0..m {
                    let ri = m - 1 - i; // reversed index
                    eigenvalues[i] = s_diag.column_vector()[ri].max(0.0);
                    for j in 0..m {
                        // Eigenvectors are columns of U; store as rows of Vᵀ (reversed order).
                        vt_flat[i * m + j] = u_mat[(j, ri)];
                    }
                }

                Some(EigenDecomposition {
                    columns: col_order.to_vec(),
                    eigenvalues,
                    eigenvectors: vt_flat,
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
        skewness: skewness_map,
        entropy: entropy_map,
        quantiles: quantiles_map,
        sparsity: sparsity_map,
        covariance,
        correlation,
        eigen,
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
        let csv_path = path.as_ref();
        let parquet_path = csv_path.with_extension("parquet");

        // Use duckdb CLI to convert CSV → Parquet.
        let query = format!(
            "COPY (SELECT * FROM read_csv_auto('{}')) TO '{}' (FORMAT PARQUET);",
            csv_path.display(),
            parquet_path.display(),
        );
        let output = Command::new("duckdb")
            .args(["-c", &query])
            .output()?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(ProfilingError::DuckDb(stderr.into_owned()));
        }

        // Read the Parquet file via arrow.
        let file = std::fs::File::open(&parquet_path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let reader = builder.build()?;

        let mut headers: Vec<String> = Vec::new();
        let mut columns: HashMap<String, Vec<f64>> = HashMap::new();
        let mut headers_set = false;

        for batch in reader {
            let batch = batch?;
            if !headers_set {
                headers = batch
                    .schema()
                    .fields()
                    .iter()
                    .map(|f| f.name().clone())
                    .collect();
                for name in &headers {
                    columns.insert(name.clone(), Vec::new());
                }
                headers_set = true;
            }

            for (j, name) in headers.iter().enumerate() {
                let col = batch.column(j);
                // Try to downcast to Float64; fall back to casting.
                let f64_arr = col
                    .as_primitive_opt::<Float64Type>()
                    .cloned()
                    .or_else(|| {
                        arrow::compute::cast(col, &arrow::datatypes::DataType::Float64)
                            .ok()
                            .and_then(|a| a.as_primitive_opt::<Float64Type>().cloned())
                    });

                if let Some(arr) = f64_arr {
                    let vals = columns.get_mut(name).unwrap();
                    for i in 0..arr.len() {
                        vals.push(if arr.is_null(i) { 0.0 } else { arr.value(i) });
                    }
                } else {
                    // Non-numeric column: push zeros.
                    let vals = columns.get_mut(name).unwrap();
                    vals.resize(vals.len() + batch.num_rows(), 0.0);
                }
            }
        }

        // Clean up temporary parquet file.
        let _ = std::fs::remove_file(&parquet_path);

        let nrows = columns.values().next().map_or(0, |v| v.len());
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
        self.stats
            .quantiles
            .get(column)
            .cloned()
            .ok_or_else(|| ProfilingError::ColumnNotFound(column.to_string()))
    }

    // --- Distribution (cached) ---

    pub fn skewness(&self, column: &str) -> Result<f64, ProfilingError> {
        self.stats
            .skewness
            .get(column)
            .copied()
            .ok_or_else(|| ProfilingError::ColumnNotFound(column.to_string()))
    }

    // --- Entropy (cached) ---

    pub fn entropy(&self, column: &str) -> Result<f64, ProfilingError> {
        self.stats
            .entropy
            .get(column)
            .copied()
            .ok_or_else(|| ProfilingError::ColumnNotFound(column.to_string()))
    }

    // --- Covariance (cached) ---

    pub fn covariance_matrix(&self) -> Result<CovarianceMatrix, ProfilingError> {
        Ok(self.stats.covariance.clone())
    }

    // --- Correlation (cached) ---

    pub fn correlation_matrix(&self) -> Result<CorrelationMatrix, ProfilingError> {
        Ok(self.stats.correlation.clone())
    }

    // --- Eigendecomposition (cached) ---

    pub fn eigen(&self) -> Result<&EigenDecomposition, ProfilingError> {
        self.stats
            .eigen
            .as_ref()
            .ok_or(ProfilingError::NotEnoughColumns)
    }

    // --- PCA (derived from cached eigendecomposition) ---

    pub fn pca(&self, n_components: Option<usize>) -> Result<PcaResult, ProfilingError> {
        let eigen = self.eigen()?;
        let m = eigen.columns.len();
        let k = n_components.unwrap_or(m).min(m);

        let total_var: f64 = eigen.eigenvalues.iter().sum();
        let explained_variance: Vec<f64> = eigen.eigenvalues[..k].to_vec();
        let explained_variance_ratio: Vec<f64> = explained_variance
            .iter()
            .map(|&v| if total_var > 0.0 { v / total_var } else { 0.0 })
            .collect();

        let mut cumulative_variance_ratio = Vec::with_capacity(k);
        let mut cumsum = 0.0;
        for &r in &explained_variance_ratio {
            cumsum += r;
            cumulative_variance_ratio.push(cumsum);
        }

        // Each row of eigenvectors is a principal component (eigenvector of covariance matrix).
        let mut components = Vec::with_capacity(k * m);
        for i in 0..k {
            for j in 0..m {
                components.push(eigen.eigenvectors[i * m + j]);
            }
        }

        Ok(PcaResult {
            columns: eigen.columns.clone(),
            n_components: k,
            explained_variance,
            explained_variance_ratio,
            cumulative_variance_ratio,
            components,
        })
    }

    // --- Sparsity ---

    pub fn sparsity(&self, column: &str) -> Result<f64, ProfilingError> {
        self.stats
            .sparsity
            .get(column)
            .copied()
            .ok_or_else(|| ProfilingError::ColumnNotFound(column.to_string()))
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
