use std::collections::HashMap;
use std::path::Path;

use duckdb::Connection;
use faer::Mat;

use crate::error::ProfilingError;
use crate::profiling;
use crate::profiling::correlation::CorrelationMatrix;
use crate::profiling::numeric::Quantiles;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Supervised prediction task type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum PredictionType {
    Regression,
    BinaryClassification,
    MultiCategoryClassification,
}

/// Eigendecomposition of the correlation matrix (A = Q Λ Qᵀ).
#[derive(Debug, Clone, serde::Serialize)]
pub struct EigenDecomposition {
    pub columns: Vec<String>,
    /// Eigenvalues in descending order (length m).
    pub eigenvalues: Vec<f64>,
    /// Eigenvectors as rows, flat row-major m×m (each row is a principal axis).
    pub eigenvectors: Vec<f64>,
}

/// PCA result derived from the eigendecomposition of the correlation matrix.
#[derive(Debug, Clone, serde::Serialize)]
pub struct PcaResult {
    pub columns: Vec<String>,
    pub n_components: usize,
    pub explained_variance: Vec<f64>,
    pub explained_variance_ratio: Vec<f64>,
    pub cumulative_variance_ratio: Vec<f64>,
    /// Principal component loadings, flat row-major n_components × m.
    pub components: Vec<f64>,
}

/// Data split for supervised ML: X design matrix and y target vector.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SupervisedLearningData {
    pub prediction_type: PredictionType,
    pub feature_columns: Vec<String>,
    pub target_column: String,
    pub nrows: usize,
    pub nfeatures: usize,
    /// Design matrix values in flat row-major layout (nrows × nfeatures).
    pub x: Vec<f64>,
    /// Target/predictor vector (length nrows).
    pub y: Vec<f64>,
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

// ---------------------------------------------------------------------------
// Internal stats cache
// ---------------------------------------------------------------------------

struct PrecomputedStats {
    covariance: CovarianceMatrix,
    eigen: Option<EigenDecomposition>,
}

// ---------------------------------------------------------------------------
// Dataset
// ---------------------------------------------------------------------------

pub struct Dataset {
    /// Raw column values (used for on-demand per-column stats).
    columns: HashMap<String, Vec<f64>>,
    /// In-memory DuckDB copy used for SQL-native on-demand stats.
    conn: Connection,
    col_order: Vec<String>,
    nrows: usize,
    stats: PrecomputedStats,
}

// ---------------------------------------------------------------------------
// Helpers shared by both construction paths
// ---------------------------------------------------------------------------

/// SQL-quote a column name.
fn q(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

/// Eigendecomposition of a symmetric m×m correlation matrix (flat row-major).
/// Computed from a covariance matrix. Returns eigenvalues/eigenvectors in descending eigenvalue order.
fn compute_eigen_from_correlation(cov: &[f64], col_order: &[String]) -> Option<EigenDecomposition> {
    let m = col_order.len();
    if m < 2 {
        return None;
    }

    // Convert covariance to correlation
    let mut corr_flat = vec![0.0f64; m * m];
    for i in 0..m {
        corr_flat[i * m + i] = 1.0;
        for j in (i + 1)..m {
            let denom = (cov[i * m + i] * cov[j * m + j]).sqrt();
            let r = if denom == 0.0 { 0.0 } else { cov[i * m + j] / denom };
            corr_flat[i * m + j] = r;
            corr_flat[j * m + i] = r;
        }
    }

    // Eigendecompose correlation matrix
    let corr_faer = Mat::from_fn(m, m, |i, j| corr_flat[i * m + j]);
    let decomp = corr_faer.self_adjoint_eigen(faer::Side::Lower).ok()?;
    let u = decomp.U();
    let s = decomp.S();
    // faer returns eigenvalues in nondecreasing order; reverse to descending.
    let mut eigenvalues = vec![0.0f64; m];
    let mut eigenvectors = vec![0.0f64; m * m];
    for i in 0..m {
        let ri = m - 1 - i;
        eigenvalues[i] = s.column_vector()[ri].max(0.0);
        for j in 0..m {
            eigenvectors[i * m + j] = u[(j, ri)];
        }
    }
    Some(EigenDecomposition { columns: col_order.to_vec(), eigenvalues, eigenvectors })
}

/// Pearson correlation matrix derived from a flat row-major covariance matrix.
fn correlation_from_cov(cov: &[f64], col_order: &[String]) -> CorrelationMatrix {
    let m = col_order.len();
    let mut matrix = vec![vec![0.0f64; m]; m];
    for i in 0..m {
        matrix[i][i] = 1.0;
        for j in (i + 1)..m {
            let denom = (cov[i * m + i] * cov[j * m + j]).sqrt();
            let r = if denom == 0.0 { 0.0 } else { cov[i * m + j] / denom };
            matrix[i][j] = r;
            matrix[j][i] = r;
        }
    }
    CorrelationMatrix { columns: col_order.to_vec(), matrix }
}

// ---------------------------------------------------------------------------
// from_csv — DuckDB ingestion path
// ---------------------------------------------------------------------------

/// Returns the names of numeric columns from table `data`, in schema order.
fn numeric_columns(conn: &Connection) -> Result<Vec<String>, ProfilingError> {
    let mut stmt = conn.prepare(
        "SELECT column_name, data_type FROM information_schema.columns \
         WHERE table_name = 'data' ORDER BY ordinal_position",
    )?;
    let pairs = stmt.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
    })?;
    Ok(pairs
        .filter_map(|r| r.ok())
        .filter(|(_, t)| {
            let u = t.to_uppercase();
            u.contains("INT") || u.contains("DOUBLE") || u.contains("FLOAT")
                || u.contains("REAL") || u.contains("DECIMAL") || u.contains("NUMERIC")
        })
        .map(|(n, _)| n)
        .collect())
}

/// Single-scan query: upper-triangle covariance via COVAR_SAMP, mirrored to full m×m flat matrix.
fn query_covariance(
    conn: &Connection,
    col_order: &[String],
) -> Result<Vec<f64>, ProfilingError> {
    let m = col_order.len();
    let mut parts = Vec::with_capacity(m * (m + 1) / 2);
    let mut positions: Vec<(usize, usize)> = Vec::with_capacity(m * (m + 1) / 2);
    for i in 0..m {
        for j in i..m {
            parts.push(format!(
                "COVAR_SAMP({}, {})",
                q(&col_order[i]),
                q(&col_order[j])
            ));
            positions.push((i, j));
        }
    }
    let sql = format!("SELECT {} FROM data", parts.join(", "));

    conn.query_row(&sql, [], |row| {
        let mut flat = vec![0.0f64; m * m];
        for (idx, &(i, j)) in positions.iter().enumerate() {
            let v: f64 = row.get::<_, Option<f64>>(idx)?.unwrap_or(0.0);
            flat[i * m + j] = v;
            flat[j * m + i] = v;
        }
        Ok(flat)
    })
    .map_err(Into::into)
}

/// Fetch all numeric column values in one query (needed for entropy).
fn fetch_raw_columns(
    conn: &Connection,
    col_order: &[String],
    nrows: usize,
) -> Result<HashMap<String, Vec<f64>>, ProfilingError> {
    if col_order.is_empty() {
        return Ok(HashMap::new());
    }
    let select = col_order
        .iter()
        .map(|n| format!("{}::DOUBLE", q(n)))
        .collect::<Vec<_>>()
        .join(", ");
    let sql = format!("SELECT {select} FROM data");
    let mut stmt = conn.prepare(&sql)?;
    let mut columns: HashMap<String, Vec<f64>> = col_order
        .iter()
        .map(|n| (n.clone(), Vec::with_capacity(nrows)))
        .collect();
    let mut rows = stmt.query([])?;
    while let Some(row) = rows.next()? {
        for (i, name) in col_order.iter().enumerate() {
            let v: f64 = row.get::<_, Option<f64>>(i)?.unwrap_or(0.0);
            columns.get_mut(name).unwrap().push(v);
        }
    }
    Ok(columns)
}

/// Load a HashMap of columns into an in-memory DuckDB table.
fn load_columns_to_duckdb(
    conn: &Connection,
    col_order: &[String],
    columns: &HashMap<String, Vec<f64>>,
    nrows: usize,
) -> Result<(), ProfilingError> {
    if col_order.is_empty() || nrows == 0 {
        return Ok(());
    }
    // Create table with one DOUBLE column per key.
    let col_defs = col_order
        .iter()
        .map(|n| format!("{} DOUBLE", q(n)))
        .collect::<Vec<_>>()
        .join(", ");
    conn.execute_batch(&format!("CREATE TABLE data ({col_defs})"))?;

    // Insert all rows via a prepared statement.
    let placeholders = col_order.iter().map(|_| "?").collect::<Vec<_>>().join(", ");
    let insert_sql = format!("INSERT INTO data VALUES ({placeholders})");
    let mut stmt = conn.prepare(&insert_sql)?;
    for i in 0..nrows {
        let params: Vec<duckdb::types::Value> = col_order
            .iter()
            .map(|n| duckdb::types::Value::Double(columns[n][i]))
            .collect();
        stmt.execute(duckdb::params_from_iter(params.iter()))?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Dataset implementation
// ---------------------------------------------------------------------------

impl Dataset {
    /// Load a CSV using DuckDB and precompute covariance/PCA from SQL aggregates.
    pub fn from_csv<P: AsRef<Path>>(path: P) -> Result<Self, ProfilingError> {
        let conn = Connection::open_in_memory()?;
        let path_str = path.as_ref().display().to_string().replace('\'', "''");
        conn.execute_batch(&format!(
            "CREATE TABLE data AS SELECT * FROM read_csv_auto('{path_str}')"
        ))?;

        let col_order = numeric_columns(&conn)?;
        let nrows: usize = conn.query_row(
            "SELECT COUNT(*) FROM data",
            [],
            |r| r.get::<_, i64>(0),
        )? as usize;

        if col_order.is_empty() || nrows == 0 {
            return Ok(Self {
                columns: HashMap::new(),
                conn,
                col_order: vec![],
                nrows,
                stats: PrecomputedStats {
                    covariance: CovarianceMatrix { columns: vec![], matrix: vec![] },
                    eigen: None,
                },
            });
        }

        let columns = fetch_raw_columns(&conn, &col_order, nrows)?;
        let cov_flat = query_covariance(&conn, &col_order)?;

        let covariance = CovarianceMatrix { columns: col_order.clone(), matrix: cov_flat.clone() };
        let eigen = compute_eigen_from_correlation(&cov_flat, &col_order);

        Ok(Self {
            columns,
            conn,
            col_order,
            nrows,
            stats: PrecomputedStats {
                covariance,
                eigen,
            },
        })
    }

    /// Build a Dataset from pre-built column data (for tests and benchmarks).
    /// Uses the same hybrid path as from_csv for consistent results.
    pub fn from_columns(col_order: Vec<String>, columns: HashMap<String, Vec<f64>>) -> Self {
        let nrows = columns.values().next().map_or(0, |v| v.len());

        let conn = Connection::open_in_memory().expect("DuckDB in-memory open failed");
        load_columns_to_duckdb(&conn, &col_order, &columns, nrows)
            .expect("failed to load columns into DuckDB");

        let cov_flat = query_covariance(&conn, &col_order)
            .expect("DuckDB covariance failed");

        let covariance = CovarianceMatrix { columns: col_order.clone(), matrix: cov_flat.clone() };
        let eigen = compute_eigen_from_correlation(&cov_flat, &col_order);

        Self {
            columns,
            conn,
            col_order,
            nrows,
            stats: PrecomputedStats {
                covariance,
                eigen,
            },
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

    // --- Numeric (on demand) ---

    pub fn mean(&self, column: &str) -> Result<f64, ProfilingError> {
        self.get_column(column)?;
        let c = q(column);
        let sql = format!("SELECT AVG({c}) FROM data");
        let v: Option<f64> = self.conn.query_row(&sql, [], |r| r.get(0))?;
        Ok(v.unwrap_or(0.0))
    }

    pub fn variance(&self, column: &str) -> Result<f64, ProfilingError> {
        self.get_column(column)?;
        let c = q(column);
        let sql = format!("SELECT VAR_SAMP({c}) FROM data");
        let v: Option<f64> = self.conn.query_row(&sql, [], |r| r.get(0))?;
        Ok(v.unwrap_or(0.0))
    }

    pub fn quantiles(&self, column: &str) -> Result<Quantiles, ProfilingError> {
        self.get_column(column)?;
        let c = q(column);
        let sql = format!(
            "SELECT MIN({c}), QUANTILE_CONT({c}, 0.25), QUANTILE_CONT({c}, 0.5), QUANTILE_CONT({c}, 0.75), MAX({c}) FROM data"
        );
        self.conn.query_row(&sql, [], |row| {
            let min: f64 = row.get::<_, Option<f64>>(0)?.unwrap_or(0.0);
            let q25: f64 = row.get::<_, Option<f64>>(1)?.unwrap_or(min);
            let q50: f64 = row.get::<_, Option<f64>>(2)?.unwrap_or(min);
            let q75: f64 = row.get::<_, Option<f64>>(3)?.unwrap_or(min);
            let max: f64 = row.get::<_, Option<f64>>(4)?.unwrap_or(min);
            Ok(Quantiles { min, q25, q50, q75, max })
        }).map_err(Into::into)
    }

    // --- Distribution (on demand) ---

    pub fn skewness(&self, column: &str) -> Result<f64, ProfilingError> {
        self.get_column(column)?;
        let c = q(column);
        let sql = format!("SELECT SKEWNESS({c}) FROM data");
        let v: Option<f64> = self.conn.query_row(&sql, [], |r| r.get(0))?;
        Ok(v.unwrap_or(0.0))
    }

    // --- Entropy (on demand) ---

    pub fn entropy(&self, column: &str) -> Result<f64, ProfilingError> {
        let vals = self.get_column(column)?;
        profiling::entropy::entropy_numeric(vals)
    }

    // --- Covariance (cached) ---

    pub fn covariance_matrix(&self) -> Result<CovarianceMatrix, ProfilingError> {
        Ok(self.stats.covariance.clone())
    }

    // --- Correlation (derived from cached covariance) ---

    pub fn correlation_matrix(&self) -> Result<CorrelationMatrix, ProfilingError> {
        Ok(correlation_from_cov(&self.stats.covariance.matrix, &self.col_order))
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
        self.get_column(column)?;
        if self.nrows == 0 {
            return Err(ProfilingError::EmptyDataset);
        }
        let c = q(column);
        let sql = format!("SELECT SUM(CASE WHEN {c} = 0.0 THEN 1 ELSE 0 END)::DOUBLE / {} FROM data", self.nrows);
        let v: Option<f64> = self.conn.query_row(&sql, [], |r| r.get(0))?;
        Ok(v.unwrap_or(0.0))
    }

    // --- ML helpers ---

    /// Build supervised-learning inputs from a target column.
    ///
    /// Returns the X design matrix (row-major) using all columns except
    /// `target_column`, and y as the values of `target_column`.
    pub fn design_matrix_and_target(
        &self,
        target_column: &str,
        prediction_type: PredictionType,
    ) -> Result<SupervisedLearningData, ProfilingError> {
        let y = self
            .columns
            .get(target_column)
            .cloned()
            .ok_or_else(|| ProfilingError::ColumnNotFound(target_column.to_string()))?;

        if self.col_order.len() < 2 {
            return Err(ProfilingError::NotEnoughColumns);
        }

        let feature_columns: Vec<String> = self
            .col_order
            .iter()
            .filter(|name| name.as_str() != target_column)
            .cloned()
            .collect();

        if feature_columns.is_empty() {
            return Err(ProfilingError::NotEnoughColumns);
        }

        match prediction_type {
            PredictionType::Regression => {}
            PredictionType::BinaryClassification => {
                let mut labels = y.clone();
                labels.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                labels.dedup_by(|a, b| (*a - *b).abs() < f64::EPSILON);
                if labels.len() != 2 {
                    return Err(ProfilingError::InvalidPredictionTask(format!(
                        "binary classification requires exactly 2 distinct target labels, found {}",
                        labels.len()
                    )));
                }
            }
            PredictionType::MultiCategoryClassification => {
                let mut labels = y.clone();
                labels.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                labels.dedup_by(|a, b| (*a - *b).abs() < f64::EPSILON);
                if labels.len() < 3 {
                    return Err(ProfilingError::InvalidPredictionTask(format!(
                        "multi-category classification requires at least 3 distinct target labels, found {}",
                        labels.len()
                    )));
                }
            }
        }

        let nrows = self.nrows;
        let nfeatures = feature_columns.len();
        let mut x = Vec::with_capacity(nrows * nfeatures);
        for row in 0..nrows {
            for name in &feature_columns {
                x.push(self.columns[name][row]);
            }
        }

        Ok(SupervisedLearningData {
            prediction_type,
            feature_columns,
            target_column: target_column.to_string(),
            nrows,
            nfeatures,
            x,
            y,
        })
    }

    // --- Helpers ---

    fn get_column(&self, column: &str) -> Result<&[f64], ProfilingError> {
        self.columns
            .get(column)
            .map(|v| v.as_slice())
            .ok_or_else(|| ProfilingError::ColumnNotFound(column.to_string()))
    }
}
