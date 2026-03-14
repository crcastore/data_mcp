use std::collections::HashMap;

use polars::prelude::*;

use crate::error::ProfilingError;

/// Shannon entropy in nats (natural logarithm).
pub fn entropy(df: &DataFrame, column: &str) -> Result<f64, ProfilingError> {
    let col = df
        .column(column)
        .map_err(|_| ProfilingError::ColumnNotFound(column.to_string()))?;
    let n = col.len();
    if n == 0 {
        return Err(ProfilingError::EmptyDataset);
    }

    // Cast every type to String for uniform counting.
    let str_col = col.cast(&DataType::String)?;
    let series = str_col.as_materialized_series();
    let ca = series.str()?;

    let mut counts: HashMap<&str, usize> = HashMap::new();
    for val in ca.into_no_null_iter() {
        *counts.entry(val).or_default() += 1;
    }

    let nf = n as f64;
    let h = counts
        .values()
        .map(|&c| {
            let p = c as f64 / nf;
            -p * p.ln()
        })
        .sum::<f64>();

    Ok(h)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entropy_uniform() {
        // 4 equally frequent values → H = ln(4) ≈ 1.386
        let df = df! { "c" => &["a", "b", "c", "d"] }.unwrap();
        let h = entropy(&df, "c").unwrap();
        assert!((h - 4.0f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_entropy_single_value() {
        let df = df! { "c" => &["x", "x", "x"] }.unwrap();
        assert!((entropy(&df, "c").unwrap()).abs() < 1e-10);
    }

    #[test]
    fn test_entropy_numeric() {
        let df = df! { "n" => &[1i64, 2, 3, 1, 2, 3] }.unwrap();
        // Three equally frequent values → H = ln(3)
        let h = entropy(&df, "n").unwrap();
        assert!((h - 3.0f64.ln()).abs() < 1e-10);
    }
}
