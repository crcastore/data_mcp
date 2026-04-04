use std::collections::HashMap;

use criterion::{Criterion, black_box, criterion_group, criterion_main};

use mcp::dataset::Dataset;

/// Build a 40-column × 10 000-row Dataset (all numeric f64).
fn build_dataset() -> Dataset {
    const ROWS: usize = 10_000;
    const NUM_COLS: usize = 40;

    let mut cols = HashMap::new();
    let mut order = Vec::with_capacity(NUM_COLS);

    // Numeric columns — deterministic pseudo-random via simple LCG.
    let mut seed: u64 = 42;
    for i in 0..NUM_COLS {
        let name = format!("col_{i:02}");
        let mut vals: Vec<f64> = Vec::with_capacity(ROWS);
        for _ in 0..ROWS {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            vals.push((seed as f64) / u64::MAX as f64 * 1000.0);
        }
        order.push(name.clone());
        cols.insert(name, vals);
    }

    Dataset::from_columns(order, cols)
}

// ---------------------------------------------------------------------------
// Benchmarks — all go through Dataset methods
// ---------------------------------------------------------------------------

fn bench_row_count(c: &mut Criterion) {
    let ds = build_dataset();
    c.bench_function("Dataset::row_count", |b| {
        b.iter(|| black_box(ds.row_count()))
    });
}

fn bench_column_count(c: &mut Criterion) {
    let ds = build_dataset();
    c.bench_function("Dataset::column_count", |b| {
        b.iter(|| black_box(ds.column_count()))
    });
}

fn bench_column_types(c: &mut Criterion) {
    let ds = build_dataset();
    c.bench_function("Dataset::column_types", |b| {
        b.iter(|| black_box(ds.column_types()))
    });
}

fn bench_quantiles(c: &mut Criterion) {
    let ds = build_dataset();
    c.bench_function("Dataset::quantiles", |b| {
        b.iter(|| ds.quantiles(black_box("col_00")).unwrap())
    });
}

fn bench_unique_count(c: &mut Criterion) {
    let ds = build_dataset();
    c.bench_function("Dataset::unique_count", |b| {
        b.iter(|| ds.unique_count(black_box("col_00")).unwrap())
    });
}

fn bench_entropy(c: &mut Criterion) {
    let ds = build_dataset();
    c.bench_function("Dataset::entropy", |b| {
        b.iter(|| ds.entropy(black_box("col_00")).unwrap())
    });
}

fn bench_skewness(c: &mut Criterion) {
    let ds = build_dataset();
    c.bench_function("Dataset::skewness", |b| {
        b.iter(|| ds.skewness(black_box("col_00")).unwrap())
    });
}

fn bench_sparsity(c: &mut Criterion) {
    let ds = build_dataset();
    c.bench_function("Dataset::sparsity", |b| {
        b.iter(|| ds.sparsity(black_box("col_00")).unwrap())
    });
}

fn bench_dataset_from_columns(c: &mut Criterion) {
    // Pre-build the data; bench only from_columns cost.
    let ds = build_dataset();
    // We can't easily clone internal state, so just re-time build_dataset.
    let _ = ds;
    c.bench_function("Dataset::from_columns", |b| {
        b.iter(|| build_dataset())
    });
}

fn bench_surrogate_test(c: &mut Criterion) {
    let ds = build_dataset();
    c.bench_function("Dataset::surrogate_test", |b| {
        b.iter(|| ds.surrogate_test(black_box("col_00"), 100).unwrap())
    });
}

fn bench_bds_test(c: &mut Criterion) {
    let ds = build_dataset();
    c.bench_function("Dataset::bds_test", |b| {
        b.iter(|| ds.bds_test(black_box("col_00"), 3, 400.0).unwrap())
    });
}

fn bench_lyapunov_exponent(c: &mut Criterion) {
    let ds = build_dataset();
    c.bench_function("Dataset::lyapunov_exponent", |b| {
        b.iter(|| ds.lyapunov_exponent(black_box("col_00"), 3, 1).unwrap())
    });
}

fn bench_dependence_comparison(c: &mut Criterion) {
    let ds = build_dataset();
    c.bench_function("Dataset::dependence_comparison", |b| {
        b.iter(|| ds.dependence_comparison(black_box("col_00"), 10).unwrap())
    });
}

fn bench_delay_embedding(c: &mut Criterion) {
    let ds = build_dataset();
    c.bench_function("Dataset::delay_embedding", |b| {
        b.iter(|| ds.delay_embedding(black_box("col_00"), 10).unwrap())
    });
}

fn bench_memory_profile(c: &mut Criterion) {
    let ds = build_dataset();
    c.bench_function("Dataset::memory_profile", |b| {
        b.iter(|| ds.memory_profile(black_box("col_00"), 10).unwrap())
    });
}

criterion_group!(
    benches,
    bench_row_count,
    bench_column_count,
    bench_column_types,
    bench_quantiles,
    bench_unique_count,
    bench_entropy,
    bench_skewness,
    bench_sparsity,
    bench_dataset_from_columns,
    bench_surrogate_test,
    bench_bds_test,
    bench_lyapunov_exponent,
    bench_dependence_comparison,
    bench_delay_embedding,
    bench_memory_profile,
);
criterion_main!(benches);
