use criterion::{AxisScale, BenchmarkId, Criterion, PlotConfiguration, criterion_group, criterion_main};
use polars::prelude::*;

use mcp::dataset::Dataset;

// ── helpers ─────────────────────────────────────────────────────────────────

/// Build a Dataset with `n` rows containing:
///   - "val"    : Float64 numeric column
///   - "cat"    : Utf8 categorical column (10 distinct values)
///   - "sparse" : Float64 column that is ~90 % zeros
///   - "maybe"  : Float64 column with ~20 % nulls
fn make_dataset(n: usize) -> Dataset {
    let mut val = Vec::with_capacity(n);
    let mut cat = Vec::with_capacity(n);
    let mut sparse = Vec::with_capacity(n);
    let mut maybe: Vec<Option<f64>> = Vec::with_capacity(n);

    let cats = [
        "alpha", "beta", "gamma", "delta", "epsilon",
        "zeta", "eta", "theta", "iota", "kappa",
    ];

    for i in 0..n {
        val.push((i as f64).sin() * 100.0);
        cat.push(cats[i % cats.len()].to_string());
        sparse.push(if i % 10 == 0 { i as f64 } else { 0.0 });
        maybe.push(if i % 5 == 0 { None } else { Some(i as f64) });
    }

    let df = DataFrame::new(n, vec![
        Column::new("val".into(), &val),
        Column::new("cat".into(), &cat),
        Column::new("sparse".into(), &sparse),
        Column::new("maybe".into(), &maybe),
    ])
    .expect("failed to build benchmark DataFrame");

    Dataset::new(df)
}

const SIZES: &[usize] = &[100, 1_000, 10_000, 100_000];

fn log_plot() -> PlotConfiguration {
    PlotConfiguration::default().summary_scale(AxisScale::Logarithmic)
}

// ── benchmark groups ────────────────────────────────────────────────────────

fn bench_row_count(c: &mut Criterion) {
    let mut group = c.benchmark_group("row_count");
    group.plot_config(log_plot());
    for &n in SIZES {
        let ds = make_dataset(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &ds, |b, ds| {
            b.iter(|| ds.row_count());
        });
    }
    group.finish();
}

fn bench_column_count(c: &mut Criterion) {
    let mut group = c.benchmark_group("column_count");
    group.plot_config(log_plot());
    for &n in SIZES {
        let ds = make_dataset(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &ds, |b, ds| {
            b.iter(|| ds.column_count());
        });
    }
    group.finish();
}

fn bench_column_types(c: &mut Criterion) {
    let mut group = c.benchmark_group("column_types");
    group.plot_config(log_plot());
    for &n in SIZES {
        let ds = make_dataset(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &ds, |b, ds| {
            b.iter(|| ds.column_types());
        });
    }
    group.finish();
}

fn bench_missing_rate(c: &mut Criterion) {
    let mut group = c.benchmark_group("missing_rate");
    group.plot_config(log_plot());
    for &n in SIZES {
        let ds = make_dataset(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &ds, |b, ds| {
            b.iter(|| ds.missing_rate("maybe").unwrap());
        });
    }
    group.finish();
}

fn bench_mean(c: &mut Criterion) {
    let mut group = c.benchmark_group("mean");
    group.plot_config(log_plot());
    for &n in SIZES {
        let ds = make_dataset(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &ds, |b, ds| {
            b.iter(|| ds.mean("val").unwrap());
        });
    }
    group.finish();
}

fn bench_variance(c: &mut Criterion) {
    let mut group = c.benchmark_group("variance");
    group.plot_config(log_plot());
    for &n in SIZES {
        let ds = make_dataset(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &ds, |b, ds| {
            b.iter(|| ds.variance("val").unwrap());
        });
    }
    group.finish();
}

fn bench_quantiles(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantiles");
    group.plot_config(log_plot());
    for &n in SIZES {
        let ds = make_dataset(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &ds, |b, ds| {
            b.iter(|| ds.quantiles("val").unwrap());
        });
    }
    group.finish();
}

fn bench_skewness(c: &mut Criterion) {
    let mut group = c.benchmark_group("skewness");
    group.plot_config(log_plot());
    for &n in SIZES {
        let ds = make_dataset(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &ds, |b, ds| {
            b.iter(|| ds.skewness("val").unwrap());
        });
    }
    group.finish();
}

fn bench_unique_count(c: &mut Criterion) {
    let mut group = c.benchmark_group("unique_count");
    group.plot_config(log_plot());
    for &n in SIZES {
        let ds = make_dataset(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &ds, |b, ds| {
            b.iter(|| ds.unique_count("cat").unwrap());
        });
    }
    group.finish();
}

fn bench_entropy(c: &mut Criterion) {
    let mut group = c.benchmark_group("entropy");
    group.plot_config(log_plot());
    for &n in SIZES {
        let ds = make_dataset(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &ds, |b, ds| {
            b.iter(|| ds.entropy("cat").unwrap());
        });
    }
    group.finish();
}

fn bench_sparsity(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparsity");
    group.plot_config(log_plot());
    for &n in SIZES {
        let ds = make_dataset(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &ds, |b, ds| {
            b.iter(|| ds.sparsity("sparse").unwrap());
        });
    }
    group.finish();
}

fn bench_correlation_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("correlation_matrix");
    group.plot_config(log_plot());
    for &n in SIZES {
        let ds = make_dataset(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &ds, |b, ds| {
            b.iter(|| ds.correlation_matrix().unwrap());
        });
    }
    group.finish();
}

// ── harness ─────────────────────────────────────────────────────────────────

criterion_group!(
    benches,
    bench_row_count,
    bench_column_count,
    bench_column_types,
    bench_missing_rate,
    bench_mean,
    bench_variance,
    bench_quantiles,
    bench_skewness,
    bench_unique_count,
    bench_entropy,
    bench_sparsity,
    bench_correlation_matrix,
);
criterion_main!(benches);
