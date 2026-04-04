use std::collections::HashMap;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

use mcp::dataset::Dataset;

// ── helpers ─────────────────────────────────────────────────────────────────

fn make_dataset(n: usize) -> Dataset {
    let mut val = Vec::with_capacity(n);
    let mut sparse = Vec::with_capacity(n);

    for i in 0..n {
        val.push((i as f64).sin() * 100.0);
        sparse.push(if i % 10 == 0 { i as f64 } else { 0.0 });
    }

    let order = vec!["val".into(), "sparse".into()];
    let mut cols = HashMap::new();
    cols.insert("val".to_string(), val);
    cols.insert("sparse".to_string(), sparse);
    Dataset::from_columns(order, cols)
}

const SIZES: &[usize] = &[10, 1_000];

// ── benchmarks ──────────────────────────────────────────────────────────────

/// Cached O(1) lookups — short measurement, few samples.
fn bench_cached(c: &mut Criterion) {
    let mut group = c.benchmark_group("cached");
    group.measurement_time(Duration::from_millis(500));
    group.sample_size(10);

    for &n in SIZES {
        let ds = make_dataset(n);

        group.bench_with_input(BenchmarkId::new("mean", n), &ds, |b, ds| {
            b.iter(|| ds.mean("val").unwrap());
        });
        group.bench_with_input(BenchmarkId::new("variance", n), &ds, |b, ds| {
            b.iter(|| ds.variance("val").unwrap());
        });
        group.bench_with_input(BenchmarkId::new("correlation_matrix", n), &ds, |b, ds| {
            b.iter(|| ds.correlation_matrix().unwrap());
        });
        group.bench_with_input(BenchmarkId::new("row_count", n), &ds, |b, ds| {
            b.iter(|| ds.row_count());
        });
        group.bench_with_input(BenchmarkId::new("column_count", n), &ds, |b, ds| {
            b.iter(|| ds.column_count());
        });
        group.bench_with_input(BenchmarkId::new("column_types", n), &ds, |b, ds| {
            b.iter(|| ds.column_types());
        });
    }
    group.finish();
}

/// Functions that do real work per call.
fn bench_compute(c: &mut Criterion) {
    let mut group = c.benchmark_group("compute");
    group.measurement_time(Duration::from_secs(2));

    for &n in SIZES {
        let ds = make_dataset(n);

        group.bench_with_input(BenchmarkId::new("quantiles", n), &ds, |b, ds| {
            b.iter(|| ds.quantiles("val").unwrap());
        });
        group.bench_with_input(BenchmarkId::new("skewness", n), &ds, |b, ds| {
            b.iter(|| ds.skewness("val").unwrap());
        });
        group.bench_with_input(BenchmarkId::new("unique_count", n), &ds, |b, ds| {
            b.iter(|| ds.unique_count("val").unwrap());
        });
        group.bench_with_input(BenchmarkId::new("entropy", n), &ds, |b, ds| {
            b.iter(|| ds.entropy("val").unwrap());
        });
        group.bench_with_input(BenchmarkId::new("sparsity", n), &ds, |b, ds| {
            b.iter(|| ds.sparsity("sparse").unwrap());
        });
    }
    group.finish();
}

/// Benchmark Dataset construction (precompute cost).
fn bench_init(c: &mut Criterion) {
    let mut group = c.benchmark_group("init");
    group.measurement_time(Duration::from_secs(2));
    group.sample_size(10);

    for &n in SIZES {
        // Build raw data outside the loop — only measure precompute.
        let mut val = Vec::with_capacity(n);
        let mut sparse = Vec::with_capacity(n);
        for i in 0..n {
            val.push((i as f64).sin() * 100.0);
            sparse.push(if i % 10 == 0 { i as f64 } else { 0.0 });
        }

        group.bench_with_input(BenchmarkId::new("Dataset::from_columns", n), &n, |b, _| {
            b.iter(|| {
                let order = vec!["val".into(), "sparse".into()];
                let mut cols = HashMap::new();
                cols.insert("val".to_string(), val.clone());
                cols.insert("sparse".to_string(), sparse.clone());
                Dataset::from_columns(order, cols)
            });
        });
    }
    group.finish();
}

// ── harness ─────────────────────────────────────────────────────────────────

criterion_group!(benches, bench_cached, bench_compute, bench_init);
criterion_main!(benches);
