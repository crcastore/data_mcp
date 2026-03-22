use polars::prelude::*;
use rayon::prelude::*;
use serde::Serialize;

use crate::error::ProfilingError;
use crate::util::numeric_column_names;

#[derive(Debug, Clone, Serialize)]
pub struct CorrelationMatrix {
    pub columns: Vec<String>,
    pub matrix: Vec<Vec<f64>>,
}

/// Pearson correlation matrix for all numeric columns.
///
/// Uses a BLAS-style approach:
///   - Contiguous column-major layout for cache-friendly access
///   - Tight dot-product inner loop (SIMD-friendly)
///   - Rayon-parallelised upper-triangle computation
pub fn correlation_matrix(df: &DataFrame) -> Result<CorrelationMatrix, ProfilingError> {
    let names = numeric_column_names(df);
    if names.is_empty() {
        return Err(ProfilingError::NoNumericColumns);
    }

    let n = df.height();
    let m = names.len();

    // Pack all columns into a contiguous column-major buffer:
    //   values[col * n .. (col+1) * n]
    let mut values = vec![0.0f64; m * n];

    for (j, name) in names.iter().enumerate() {
        let col = df
            .column(name)
            .map_err(|_| ProfilingError::ColumnNotFound(name.to_string()))?;
        let series = col.as_materialized_series();
        let ca = series.f64()?;
        let offset = j * n;
        for (k, v) in ca.into_no_null_iter().enumerate() {
            values[offset + k] = v;
        }
    }

    // Enumerate all (i, j) pairs in the upper triangle.
    let pairs: Vec<(usize, usize)> = (0..m)
        .flat_map(|i| ((i + 1)..m).map(move |j| (i, j)))
        .collect();

    // Compute correlations in parallel – each pair is independent.
    let correlations: Vec<(usize, usize, f64)> = pairs
        .par_iter()
        .map(|&(i, j)| {
            let r = pearson(
                &values[i * n..(i + 1) * n],
                &values[j * n..(j + 1) * n],
            );
            (i, j, r)
        })
        .collect();

    let mut matrix = vec![vec![0.0f64; m]; m];
    for i in 0..m {
        matrix[i][i] = 1.0;
    }
    for (i, j, r) in correlations {
        matrix[i][j] = r;
        matrix[j][i] = r;
    }

    Ok(CorrelationMatrix {
        columns: names,
        matrix,
    })
}

/// Pearson r via the sum-of-products formula (SIMD-friendly tight loop).
///
///   r = (n·Σxy − Σx·Σy) / √[(n·Σx² − (Σx)²)(n·Σy² − (Σy)²)]
#[inline]
fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mut sum_x = 0.0f64;
    let mut sum_y = 0.0f64;
    let mut sum_xy = 0.0f64;
    let mut sum_x2 = 0.0f64;
    let mut sum_y2 = 0.0f64;

    for k in 0..x.len() {
        let xk = x[k];
        let yk = y[k];
        sum_x += xk;
        sum_y += yk;
        sum_xy += xk * yk;
        sum_x2 += xk * xk;
        sum_y2 += yk * yk;
    }

    let num = n * sum_xy - sum_x * sum_y;
    let den_x = n * sum_x2 - sum_x * sum_x;
    let den_y = n * sum_y2 - sum_y * sum_y;
    let denom = (den_x * den_y).sqrt();

    if denom == 0.0 {
        0.0
    } else {
        num / denom
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perfect_positive() {
        let df = df! {
            "a" => &[1.0f64, 2.0, 3.0, 4.0],
            "b" => &[10.0f64, 20.0, 30.0, 40.0],
        }
        .unwrap();
        let cm = correlation_matrix(&df).unwrap();
        assert!((cm.matrix[0][1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_perfect_negative() {
        let df = df! {
            "a" => &[1.0f64, 2.0, 3.0],
            "b" => &[30.0f64, 20.0, 10.0],
        }
        .unwrap();
        let cm = correlation_matrix(&df).unwrap();
        assert!((cm.matrix[0][1] + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_diagonal_is_one() {
        let df = df! {
            "x" => &[1.0f64, 2.0, 3.0],
            "y" => &[4.0f64, 5.0, 6.0],
            "z" => &[7.0f64, 8.0, 9.0],
        }
        .unwrap();
        let cm = correlation_matrix(&df).unwrap();
        for i in 0..cm.columns.len() {
            assert!((cm.matrix[i][i] - 1.0).abs() < 1e-10);
        }
    }
}
