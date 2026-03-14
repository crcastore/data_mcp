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

/// Pearson correlation matrix for all numeric columns (pairwise complete).
///
/// Uses a BLAS-style approach:
///   - Contiguous column-major layout for cache-friendly access
///   - Branch-free mask-weighted dot products (SIMD-friendly inner loop)
///   - Rayon-parallelised upper-triangle computation
pub fn correlation_matrix(df: &DataFrame) -> Result<CorrelationMatrix, ProfilingError> {
    let names = numeric_column_names(df);
    if names.is_empty() {
        return Err(ProfilingError::NoNumericColumns);
    }

    let n = df.height();
    let m = names.len();

    // Pack all columns into two contiguous column-major buffers:
    //   values[col * n .. (col+1) * n]  – f64 value (0.0 where missing)
    //   masks [col * n .. (col+1) * n]  – 1.0 valid, 0.0 missing/NaN
    let mut values = vec![0.0f64; m * n];
    let mut masks = vec![0.0f64; m * n];

    for (j, name) in names.iter().enumerate() {
        let col = df
            .column(name)
            .map_err(|_| ProfilingError::ColumnNotFound(name.to_string()))?;
        let casted = col.cast(&DataType::Float64)?;
        let series = casted.as_materialized_series();
        let ca = series.f64()?;
        let offset = j * n;
        for (k, opt) in ca.iter().enumerate() {
            match opt {
                Some(v) if !v.is_nan() => {
                    values[offset + k] = v;
                    masks[offset + k] = 1.0;
                }
                _ => {} // stays 0.0 / 0.0
            }
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
            let r = pearson_masked(
                &values[i * n..(i + 1) * n],
                &masks[i * n..(i + 1) * n],
                &values[j * n..(j + 1) * n],
                &masks[j * n..(j + 1) * n],
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

/// Mask-weighted Pearson r in a single pass (BLAS / SIMD-friendly).
///
/// The inner loop is completely branch-free: the mask value (0 or 1)
/// acts as a multiplicative weight, so the compiler can auto-vectorize
/// the entire loop into packed SIMD instructions.
///
/// Uses the numerically stable "sum of products" formula:
///   r = (n·Σxy − Σx·Σy) / √[(n·Σx² − (Σx)²)(n·Σy² − (Σy)²)]
#[inline]
fn pearson_masked(x: &[f64], mx: &[f64], y: &[f64], my: &[f64]) -> f64 {
    let len = x.len();
    let mut count = 0.0f64;
    let mut sum_x = 0.0f64;
    let mut sum_y = 0.0f64;
    let mut sum_xy = 0.0f64;
    let mut sum_x2 = 0.0f64;
    let mut sum_y2 = 0.0f64;

    for k in 0..len {
        let w = mx[k] * my[k]; // 1.0 when both valid, 0.0 otherwise
        let xw = x[k] * w;
        let yw = y[k] * w;
        count += w;
        sum_x += xw;
        sum_y += yw;
        sum_xy += x[k] * yw;
        sum_x2 += x[k] * xw;
        sum_y2 += y[k] * yw;
    }

    if count < 2.0 {
        return f64::NAN;
    }

    let num = count * sum_xy - sum_x * sum_y;
    let den_x = count * sum_x2 - sum_x * sum_x;
    let den_y = count * sum_y2 - sum_y * sum_y;
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

    #[test]
    fn test_with_nulls() {
        let df = df! {
            "a" => &[Some(1.0f64), None, Some(3.0), Some(4.0)],
            "b" => &[Some(10.0f64), Some(20.0), None, Some(40.0)],
        }
        .unwrap();
        let cm = correlation_matrix(&df).unwrap();
        // Only rows 0 and 3 have both non-null → r = 1.0
        assert!((cm.matrix[0][1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_with_nan() {
        let df = df! {
            "a" => &[Some(1.0f64), Some(f64::NAN), Some(3.0), Some(4.0)],
            "b" => &[Some(10.0f64), Some(20.0), Some(30.0), Some(40.0)],
        }
        .unwrap();
        let cm = correlation_matrix(&df).unwrap();
        // Row 1 has NaN in "a" → treated as missing, pairwise drops it
        // Remaining pairs: (1,10),(3,30),(4,40) → r = 1.0
        assert!((cm.matrix[0][1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_all_null_column() {
        let df = df! {
            "a" => &[Option::<f64>::None, None, None],
            "b" => &[Some(1.0f64), Some(2.0), Some(3.0)],
        }
        .unwrap();
        let cm = correlation_matrix(&df).unwrap();
        // No valid pairs → NaN
        assert!(cm.matrix[0][1].is_nan());
    }
}
