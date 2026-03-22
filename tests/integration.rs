use polars::prelude::*;

use mcp::dataset::Dataset;

fn sample_ds() -> Dataset {
    let df = df! {
        "id"     => &[1i64, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "age"    => &[25i64, 30, 35, 25, 40, 31, 28, 33, 45, 22],
        "income" => &[50000.0f64, 60000.0, 70000.0, 55000.0,
                      80000.0, 75000.0, 45000.0, 62000.0,
                      90000.0, 40000.0],
        "city"   => &["NYC", "LA", "NYC", "LA",
                      "NYC", "Chicago", "Chicago", "LA",
                      "NYC", "Chicago"],
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
fn test_unique_count_city() {
    // cities: NYC, LA, Chicago → 3
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

// ---------------------------------------------------------------------------
// Reservoir computing diagnostics — uses a chaotic logistic map series
// ---------------------------------------------------------------------------

fn reservoir_ds() -> Dataset {
    let n = 2000;
    let mut x = vec![0.0f64; n];
    x[0] = 0.1;
    let r = 3.9;
    for i in 1..n {
        x[i] = r * x[i - 1] * (1.0 - x[i - 1]);
    }
    let df = df! { "signal" => &x }.unwrap();
    Dataset::new(df)
}

#[test]
fn test_surrogate_test_via_dataset() {
    let ds = reservoir_ds();
    let result = ds.surrogate_test("signal", 50).unwrap();
    assert!(result.z_score.abs() > 2.0, "logistic map should be nonlinear");
    assert!(result.is_nonlinear);
}

#[test]
fn test_bds_test_via_dataset() {
    let ds = reservoir_ds();
    let result = ds.bds_test("signal", 3, 0.5).unwrap();
    assert!(result.p_value < 0.05, "logistic map should reject i.i.d. null");
    assert!(result.statistic.is_finite());
}

#[test]
fn test_lyapunov_via_dataset() {
    let ds = reservoir_ds();
    let l = ds.lyapunov_exponent("signal", 3, 1).unwrap();
    assert!(l > 0.0, "chaotic series should have positive Lyapunov exponent, got {l}");
}

#[test]
fn test_dependence_comparison_via_dataset() {
    let ds = reservoir_ds();
    let result = ds.dependence_comparison("signal", 5).unwrap();
    assert_eq!(result.autocorrelations.len(), 5);
    assert_eq!(result.mutual_informations.len(), 5);
    for mi in &result.mutual_informations {
        assert!(*mi >= 0.0);
    }
}

#[test]
fn test_delay_embedding_via_dataset() {
    let ds = reservoir_ds();
    let result = ds.delay_embedding("signal", 10).unwrap();
    assert!(result.optimal_delay >= 1);
    assert!(result.embedding_dimension >= 1 && result.embedding_dimension <= 10);
}

#[test]
fn test_memory_profile_via_dataset() {
    let ds = reservoir_ds();
    let result = ds.memory_profile("signal", 10).unwrap();
    assert_eq!(result.partial_autocorrelations.len(), 10);
    assert_eq!(result.delay_mutual_informations.len(), 10);
    assert!(result.active_information_storage > 0.0);
}
