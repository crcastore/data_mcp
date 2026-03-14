use criterion::{Criterion, black_box, criterion_group, criterion_main};
use polars::prelude::*;

use mcp::dataset::Dataset;
use mcp::profiling::{
    categorical, correlation, distribution, entropy, missing, numeric, shape, sparsity,
};

/// Build a 50-column × 10 000-row DataFrame.
///
/// - Columns `num_00`..`num_39`  → random-ish Float64 (with ~5 % nulls)
/// - Columns `cat_00`..`cat_09`  → Utf8 categories     (with ~5 % nulls)
fn build_dataframe() -> DataFrame {
    const ROWS: usize = 10_000;
    const NUM_COLS: usize = 40;
    const CAT_COLS: usize = 10;

    let categories = ["alpha", "beta", "gamma", "delta", "epsilon"];

    let mut columns: Vec<Column> = Vec::with_capacity(NUM_COLS + CAT_COLS);

    // Numeric columns — deterministic pseudo-random via simple LCG.
    let mut seed: u64 = 42;
    for i in 0..NUM_COLS {
        let name = format!("num_{i:02}");
        let mut vals: Vec<Option<f64>> = Vec::with_capacity(ROWS);
        for _ in 0..ROWS {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            // ~5 % nulls
            if seed % 20 == 0 {
                vals.push(None);
            } else {
                vals.push(Some((seed as f64) / u64::MAX as f64 * 1000.0));
            }
        }
        let ca = Float64Chunked::new(name.into(), &vals);
        columns.push(ca.into_column());
    }

    // Categorical (string) columns.
    for i in 0..CAT_COLS {
        let name = format!("cat_{i:02}");
        let mut vals: Vec<Option<&str>> = Vec::with_capacity(ROWS);
        for _ in 0..ROWS {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            if seed % 20 == 0 {
                vals.push(None);
            } else {
                vals.push(Some(categories[(seed % categories.len() as u64) as usize]));
            }
        }
        let ca = StringChunked::new(name.into(), &vals);
        columns.push(ca.into_column());
    }

    DataFrame::new(ROWS, columns).expect("failed to build benchmark DataFrame")
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

fn bench_row_count(c: &mut Criterion) {
    let df = build_dataframe();
    c.bench_function("shape::row_count", |b| {
        b.iter(|| shape::row_count(black_box(&df)))
    });
}

fn bench_column_count(c: &mut Criterion) {
    let df = build_dataframe();
    c.bench_function("shape::column_count", |b| {
        b.iter(|| shape::column_count(black_box(&df)))
    });
}

fn bench_column_types(c: &mut Criterion) {
    let df = build_dataframe();
    c.bench_function("shape::column_types", |b| {
        b.iter(|| shape::column_types(black_box(&df)))
    });
}

fn bench_mean(c: &mut Criterion) {
    let df = build_dataframe();
    c.bench_function("numeric::mean", |b| {
        b.iter(|| numeric::mean(black_box(&df), "num_00").unwrap())
    });
}

fn bench_variance(c: &mut Criterion) {
    let df = build_dataframe();
    c.bench_function("numeric::variance", |b| {
        b.iter(|| numeric::variance(black_box(&df), "num_00").unwrap())
    });
}

fn bench_quantiles(c: &mut Criterion) {
    let df = build_dataframe();
    c.bench_function("numeric::quantiles", |b| {
        b.iter(|| numeric::quantiles(black_box(&df), "num_00").unwrap())
    });
}

fn bench_unique_count(c: &mut Criterion) {
    let df = build_dataframe();
    c.bench_function("categorical::unique_count", |b| {
        b.iter(|| categorical::unique_count(black_box(&df), "cat_00").unwrap())
    });
}

fn bench_missing_rate(c: &mut Criterion) {
    let df = build_dataframe();
    c.bench_function("missing::missing_rate", |b| {
        b.iter(|| missing::missing_rate(black_box(&df), "num_00").unwrap())
    });
}

fn bench_entropy(c: &mut Criterion) {
    let df = build_dataframe();
    c.bench_function("entropy::entropy", |b| {
        b.iter(|| entropy::entropy(black_box(&df), "cat_00").unwrap())
    });
}

fn bench_skewness(c: &mut Criterion) {
    let df = build_dataframe();
    c.bench_function("distribution::skewness", |b| {
        b.iter(|| distribution::skewness(black_box(&df), "num_00").unwrap())
    });
}

fn bench_correlation_matrix(c: &mut Criterion) {
    let df = build_dataframe();
    c.bench_function("correlation::correlation_matrix", |b| {
        b.iter(|| correlation::correlation_matrix(black_box(&df)).unwrap())
    });
}

fn bench_sparsity(c: &mut Criterion) {
    let df = build_dataframe();
    c.bench_function("sparsity::sparsity", |b| {
        b.iter(|| sparsity::sparsity(black_box(&df), "num_00").unwrap())
    });
}

fn bench_dataset_from_dataframe(c: &mut Criterion) {
    let df = build_dataframe();
    c.bench_function("Dataset::new", |b| {
        b.iter(|| Dataset::new(black_box(df.clone())))
    });
}

criterion_group!(
    benches,
    bench_row_count,
    bench_column_count,
    bench_column_types,
    bench_mean,
    bench_variance,
    bench_quantiles,
    bench_unique_count,
    bench_missing_rate,
    bench_entropy,
    bench_skewness,
    bench_correlation_matrix,
    bench_sparsity,
    bench_dataset_from_dataframe,
);
criterion_main!(benches);
