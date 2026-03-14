use polars::prelude::*;
use serde::Serialize;

use crate::error::ProfilingError;
use crate::util::numeric_column_names;

#[derive(Debug, Clone, Serialize)]
pub struct CorrelationMatrix {
    pub columns: Vec<String>,
    pub matrix: Vec<Vec<f64>>,
}

/// Pearson correlation matrix for all numeric columns (pairwise complete).
pub fn correlation_matrix(df: &DataFrame) -> Result<CorrelationMatrix, ProfilingError> {
    let names = numeric_column_names(df);
    if names.is_empty() {
        return Err(ProfilingError::NoNumericColumns);
    }

    // Extract each column preserving nulls so we can do pairwise alignment.
    let data: Vec<Vec<Option<f64>>> = names
        .iter()
        .map(|n| extract_nullable(df, n))
        .collect::<Result<_, _>>()?;

    let m = names.len();
    let mut matrix = vec![vec![0.0f64; m]; m];

    for i in 0..m {
        matrix[i][i] = 1.0;
        for j in (i + 1)..m {
            let r = pearson_paired(&data[i], &data[j]);
            matrix[i][j] = r;
            matrix[j][i] = r;
        }
    }

    Ok(CorrelationMatrix {
        columns: names,
        matrix,
    })
}

/// Extract a column as Vec<Option<f64>> (keeps null positions for row-alignment).
fn extract_nullable(df: &DataFrame, column: &str) -> Result<Vec<Option<f64>>, ProfilingError> {
    let col = df
        .column(column)
        .map_err(|_| ProfilingError::ColumnNotFound(column.to_string()))?;
    let casted = col.cast(&DataType::Float64)?;
    let series = casted.as_materialized_series();
    let ca = series.f64()?;
    Ok(ca.iter().collect())
}

/// Pearson r computed only on rows where *both* values are non-null.
fn pearson_paired(x: &[Option<f64>], y: &[Option<f64>]) -> f64 {
    let pairs: Vec<(f64, f64)> = x
        .iter()
        .zip(y.iter())
        .filter_map(|(a, b)| Some((*a.as_ref()?, *b.as_ref()?)))
        .collect();

    let n = pairs.len();
    if n < 2 {
        return f64::NAN;
    }
    let nf = n as f64;
    let mean_x = pairs.iter().map(|(x, _)| x).sum::<f64>() / nf;
    let mean_y = pairs.iter().map(|(_, y)| y).sum::<f64>() / nf;

    let (mut cov, mut var_x, mut var_y) = (0.0, 0.0, 0.0);
    for &(xi, yi) in &pairs {
        let dx = xi - mean_x;
        let dy = yi - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let denom = (var_x * var_y).sqrt();
    if denom == 0.0 {
        0.0
    } else {
        cov / denom
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
}
