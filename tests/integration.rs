use polars::prelude::*;

use mcp::dataset::Dataset;

fn sample_ds() -> Dataset {
    let df = df! {
        "id"     => &[1i64, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "age"    => &[Some(25i64), Some(30), Some(35), Some(25), Some(40),
                      None, Some(28), Some(33), Some(45), Some(22)],
        "income" => &[Some(50000.0f64), Some(60000.0), Some(70000.0), None,
                      Some(80000.0), Some(75000.0), Some(45000.0), Some(62000.0),
                      Some(90000.0), Some(40000.0)],
        "city"   => &[Some("NYC"), Some("LA"), Some("NYC"), Some("LA"),
                      Some("NYC"), None, Some("Chicago"), Some("LA"),
                      Some("NYC"), Some("Chicago")],
        "score"  => &[85.5f64, 90.0, 0.0, 78.5, 0.0, 92.0, 0.0, 88.5, 95.0, 0.0],
    }
    .unwrap();
    Dataset::new(df)
}

#[test]
fn test_row_count() {
    assert_eq!(sample_ds().row_count(), 10);
}

#[test]
fn test_column_count() {
    assert_eq!(sample_ds().column_count(), 5);
}

#[test]
fn test_column_types_length() {
    assert_eq!(sample_ds().column_types().len(), 5);
}

#[test]
fn test_missing_rates() {
    let ds = sample_ds();
    assert!((ds.missing_rate("age").unwrap() - 0.1).abs() < 1e-10);
    assert!((ds.missing_rate("income").unwrap() - 0.1).abs() < 1e-10);
    assert!((ds.missing_rate("city").unwrap() - 0.1).abs() < 1e-10);
    assert!(ds.missing_rate("score").unwrap().abs() < 1e-10);
}

#[test]
fn test_mean_age() {
    let ds = sample_ds();
    // age non-null: [25,30,35,25,40,28,33,45,22] → sum=283, n=9
    let m = ds.mean("age").unwrap();
    assert!((m - 283.0 / 9.0).abs() < 1e-10);
}

#[test]
fn test_variance_positive() {
    assert!(sample_ds().variance("age").unwrap() > 0.0);
}

#[test]
fn test_quantiles_order() {
    let q = sample_ds().quantiles("age").unwrap();
    assert!((q.min - 22.0).abs() < 1e-10);
    assert!((q.max - 45.0).abs() < 1e-10);
    assert!(q.q25 <= q.q50);
    assert!(q.q50 <= q.q75);
}

#[test]
fn test_skewness_reasonable() {
    let s = sample_ds().skewness("age").unwrap();
    assert!(s.abs() < 3.0); // not wildly out of range
}

#[test]
fn test_unique_count_city() {
    // non-null cities: NYC, LA, Chicago → 3
    assert_eq!(sample_ds().unique_count("city").unwrap(), 3);
}

#[test]
fn test_unique_count_id() {
    assert_eq!(sample_ds().unique_count("id").unwrap(), 10);
}

#[test]
fn test_entropy_positive() {
    let h = sample_ds().entropy("city").unwrap();
    assert!(h > 0.0);
    // 3 categories → H ≤ ln(3) ≈ 1.099
    assert!(h < 3.0f64.ln() + 0.01);
}

#[test]
fn test_correlation_diagonal() {
    let cm = sample_ds().correlation_matrix().unwrap();
    for i in 0..cm.columns.len() {
        assert!(
            (cm.matrix[i][i] - 1.0).abs() < 1e-10,
            "diagonal element [{}][{}] = {} ≠ 1",
            i,
            i,
            cm.matrix[i][i]
        );
    }
}

#[test]
fn test_correlation_range() {
    let cm = sample_ds().correlation_matrix().unwrap();
    for row in &cm.matrix {
        for &val in row {
            assert!(
                val.is_nan() || (-1.0 - 1e-10..=1.0 + 1e-10).contains(&val),
                "correlation {val} out of [-1, 1]"
            );
        }
    }
}

#[test]
fn test_sparsity_score() {
    // score: [85.5, 90.0, 0.0, 78.5, 0.0, 92.0, 0.0, 88.5, 95.0, 0.0]
    // 4 zeros, 0 nulls → 4/10 = 0.4
    assert!((sample_ds().sparsity("score").unwrap() - 0.4).abs() < 1e-10);
}
