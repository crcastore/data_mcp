use std::collections::HashMap;

use mcp::dataset::{Dataset, PredictionType};

fn sample_ds() -> Dataset {
    let order = vec!["id".into(), "age".into(), "income".into(), "score".into()];
    let mut cols = HashMap::new();
    cols.insert("id".to_string(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    cols.insert("age".to_string(), vec![25.0, 30.0, 35.0, 25.0, 40.0, 31.0, 28.0, 33.0, 45.0, 22.0]);
    cols.insert("income".to_string(), vec![50000.0, 60000.0, 70000.0, 55000.0, 80000.0, 75000.0, 45000.0, 62000.0, 90000.0, 40000.0]);
    cols.insert("score".to_string(), vec![85.5, 90.0, 0.0, 78.5, 0.0, 92.0, 0.0, 88.5, 95.0, 0.0]);
    Dataset::from_columns(order, cols)
}

#[test]
fn test_row_count() {
    assert_eq!(sample_ds().row_count(), 10);
}

#[test]
fn test_column_count() {
    assert_eq!(sample_ds().column_count(), 4);
}

#[test]
fn test_column_types_length() {
    assert_eq!(sample_ds().column_types().len(), 4);
}

#[test]
fn test_mean_age() {
    let ds = sample_ds();
    // age: [25,30,35,25,40,31,28,33,45,22] → sum=314, n=10
    let m = ds.mean("age").unwrap();
    assert!((m - 314.0 / 10.0).abs() < 1e-10);
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
fn test_entropy_positive() {
    let h = sample_ds().entropy("age").unwrap();
    assert!(h > 0.0);
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
                (-1.0 - 1e-10..=1.0 + 1e-10).contains(&val),
                "correlation {val} out of [-1, 1]"
            );
        }
    }
}

#[test]
fn test_sparsity_score() {
    // score: [85.5, 90.0, 0.0, 78.5, 0.0, 92.0, 0.0, 88.5, 95.0, 0.0]
    // 4 zeros out of 10 → 0.4
    assert!((sample_ds().sparsity("score").unwrap() - 0.4).abs() < 1e-10);
}

#[test]
fn test_design_matrix_and_target_shapes_and_columns() {
    let ds = sample_ds();
    let sl = ds.design_matrix_and_target("score", PredictionType::Regression).unwrap();

    assert_eq!(sl.prediction_type, PredictionType::Regression);
    assert_eq!(sl.target_column, "score");
    assert_eq!(sl.feature_columns, vec!["id", "age", "income"]);
    assert_eq!(sl.nrows, 10);
    assert_eq!(sl.nfeatures, 3);
    assert_eq!(sl.x.len(), sl.nrows * sl.nfeatures);
    assert_eq!(sl.y.len(), sl.nrows);
}

#[test]
fn test_design_matrix_and_target_values_row_major() {
    let ds = sample_ds();
    let sl = ds.design_matrix_and_target("score", PredictionType::Regression).unwrap();

    // First row in feature order [id, age, income].
    assert!((sl.x[0] - 1.0).abs() < 1e-10);
    assert!((sl.x[1] - 25.0).abs() < 1e-10);
    assert!((sl.x[2] - 50000.0).abs() < 1e-10);

    // y should be the score column.
    assert!((sl.y[0] - 85.5).abs() < 1e-10);
    assert!((sl.y[2] - 0.0).abs() < 1e-10);
}

#[test]
fn test_design_matrix_and_target_missing_target() {
    let ds = sample_ds();
    assert!(ds
        .design_matrix_and_target("missing", PredictionType::Regression)
        .is_err());
}

#[test]
fn test_design_matrix_binary_classification_validation() {
    let order = vec!["f1".into(), "target".into()];
    let mut cols = HashMap::new();
    cols.insert("f1".to_string(), vec![1.0, 2.0, 3.0, 4.0]);
    cols.insert("target".to_string(), vec![0.0, 1.0, 0.0, 1.0]);
    let ds = Dataset::from_columns(order, cols);
    assert!(ds
        .design_matrix_and_target("target", PredictionType::BinaryClassification)
        .is_ok());
}

#[test]
fn test_design_matrix_multiclass_validation() {
    let order = vec!["f1".into(), "target".into()];
    let mut cols = HashMap::new();
    cols.insert("f1".to_string(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    cols.insert("target".to_string(), vec![0.0, 1.0, 2.0, 1.0, 2.0]);
    let ds = Dataset::from_columns(order, cols);
    assert!(ds
        .design_matrix_and_target("target", PredictionType::MultiCategoryClassification)
        .is_ok());
}

#[test]
fn test_design_matrix_binary_rejects_nonbinary_target() {
    let order = vec!["f1".into(), "target".into()];
    let mut cols = HashMap::new();
    cols.insert("f1".to_string(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    cols.insert("target".to_string(), vec![0.0, 1.0, 2.0, 1.0, 2.0]);
    let ds = Dataset::from_columns(order, cols);
    assert!(ds
        .design_matrix_and_target("target", PredictionType::BinaryClassification)
        .is_err());
}


