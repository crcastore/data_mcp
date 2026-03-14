# MCP Dataset Profiling Tool — Implementation Plan

## Goal

Build an MCP server (in Rust) that accepts a tabular dataset (CSV) and exposes
dataset-profiling functions as MCP **tools**. A connected LLM can call these
tools to understand the shape, quality, and statistical properties of a dataset
without writing code.

---

## Architecture Overview

```
┌─────────────┐       stdio / SSE        ┌──────────────────────┐
│  LLM Client │  ◄──── JSON-RPC ────►   │  mcp-profiler (Rust) │
│  (e.g. Claude)                         │                      │
└─────────────┘                          │  ┌────────────────┐  │
                                         │  │  Dataset Store  │  │
                                         │  │  (in-memory df) │  │
                                         │  └────────────────┘  │
                                         │  ┌────────────────┐  │
                                         │  │  Tool Registry  │  │
                                         │  │  (35 tools)     │  │
                                         │  └────────────────┘  │
                                         └──────────────────────┘
```

### Key decisions

| Decision | Choice | Rationale |
|---|---|---|
| Transport | **stdio** (primary), SSE (stretch) | stdio is simplest for local use |
| Data format | **CSV** loaded via `polars` | polars is fast, ergonomic, and has stats built-in |
| MCP SDK | `mcp-rs` or hand-rolled JSON-RPC | Evaluate `mcp-rs` crate maturity; fall back to manual impl |
| State | Single in-memory `DataFrame` | User loads a dataset once; tools query it |

---

## Crate Dependencies

| Crate | Purpose |
|---|---|
| `polars` | DataFrame, CSV parsing, statistics |
| `serde` / `serde_json` | JSON-RPC serialization |
| `tokio` | Async runtime (for stdio/SSE transport) |
| `clap` | CLI args (e.g. `--transport stdio`) |
| `statrs` | Skewness, kurtosis, entropy, mutual info |
| `log` / `env_logger` | Diagnostics |

---

## Phased Implementation

### Phase 0 — Scaffold & Transport (Day 1)

- [ ] Set up MCP JSON-RPC handler over **stdio**
  - Read newline-delimited JSON from stdin, write to stdout
  - Implement `initialize`, `tools/list`, `tools/call` methods
- [ ] Add `load_dataset(path)` tool — reads a CSV into an in-memory `DataFrame`
- [ ] Return proper MCP error codes for bad input
- [ ] Verify with a manual JSON-RPC smoke test

**Exit criteria:** can send `tools/list` and get a response; can call
`load_dataset` and have the server hold a DataFrame.

---

### Phase 1 — Minimal MVP Tools (Day 2-3)

Implement the **12 MVP tools** that give a useful first-pass profile:

| # | Tool | Input | Output |
|---|---|---|---|
| 1 | `row_count` | — | `{ "rows": 1000 }` |
| 2 | `column_count` | — | `{ "columns": 12 }` |
| 3 | `column_types` | — | `{ "columns": { "age": "Int64", ... } }` |
| 4 | `missing_rate` | `column?` | `{ "column": "age", "rate": 0.05 }` or all columns |
| 5 | `mean` | `column` | `{ "column": "age", "mean": 34.2 }` |
| 6 | `variance` | `column` | `{ "column": "age", "variance": 121.5 }` |
| 7 | `quantiles` | `column` | `{ "column": "age", "q25": 25, "q50": 34, "q75": 45 }` |
| 8 | `skewness` | `column` | `{ "column": "age", "skewness": 0.32 }` |
| 9 | `unique_count` | `column` | `{ "column": "city", "unique": 42 }` |
| 10 | `entropy` | `column` | `{ "column": "city", "entropy": 3.21 }` |
| 11 | `correlation_matrix` | — | `{ "matrix": { "age×income": 0.72, ... } }` |
| 12 | `sparsity` | `column` | `{ "column": "notes", "sparsity": 0.87 }` |

Implementation approach per tool:
1. Register tool name + JSON Schema for its parameters in `tools/list`.
2. Dispatch in `tools/call` match arm → call a function in `src/profiling/`.
3. Each function receives `&DataFrame` + params, returns `serde_json::Value`.

**Exit criteria:** all 12 tools callable via JSON-RPC and returning correct
results on a sample CSV.

---

### Phase 2 — Full Tool Suite (Day 4-6)

Add the remaining **~23 tools** grouped by category:

#### 2a — Missing Data & Duplicates
- `missing_count(column)`
- `dataset_missing_rate()`
- `duplicate_rows()`
- `duplicate_columns()`

#### 2b — Numeric Statistics
- `median(column)`
- `std_dev(column)`
- `min(column)`
- `max(column)`

#### 2c — Distribution Shape
- `kurtosis(column)`

#### 2d — Categorical Analysis
- `top_k_categories(column, k=10)`
- `category_frequency(column)`

#### 2e — Correlation (extended)
- `pearson_correlation_matrix()`
- `spearman_correlation_matrix()`
- `correlation_pairs(threshold)`

#### 2f — Outlier Detection
- `iqr_outliers(column)`
- `zscore_outliers(column, z=3.0)`
- `outlier_rate(column)`

#### 2g — Feature Analysis
- `variance_threshold_check(min_variance=0.01)`
- `mutual_information(feature, target)`
- `anova_f_score(feature, target)`
- `correlation_with_target(feature, target)`
- `detect_type(column)` — infer semantic type (numeric, categorical, boolean, datetime, text)
- `feature_range(column)`
- `range_ratio()`

#### 2h — Class Imbalance
- `class_distribution(target)`
- `imbalance_ratio(target)`

**Exit criteria:** full `tools/list` returns 35 tools; integration test
passes for each.

---

### Phase 3 — Convenience & Polish (Day 7)

- [ ] **`profile_summary()`** — meta-tool that calls the MVP set internally
  and returns a single combined JSON report (so the LLM can get a full
  picture in one call)
- [ ] Proper error messages when dataset not loaded or column not found
- [ ] Truncate large outputs (e.g. correlation matrix on 100+ columns)
- [ ] Add `--csv <path>` CLI flag to pre-load a dataset on startup
- [ ] Write a `README.md` with usage instructions

---

## Project Layout

```
mcp/
├── Cargo.toml
├── PLAN.md              ← this file
├── README.md
├── src/
│   ├── main.rs          ← CLI entry, tokio runtime
│   ├── transport/
│   │   ├── mod.rs
│   │   └── stdio.rs     ← stdin/stdout JSON-RPC loop
│   ├── protocol/
│   │   ├── mod.rs
│   │   ├── types.rs     ← MCP message structs
│   │   └── handler.rs   ← dispatch initialize / tools/list / tools/call
│   ├── state.rs         ← Arc<Mutex<Option<DataFrame>>> shared state
│   ├── tools/
│   │   ├── mod.rs       ← ToolRegistry + tool metadata
│   │   ├── load.rs      ← load_dataset
│   │   ├── shape.rs     ← row_count, column_count, column_types
│   │   ├── missing.rs   ← missing_count, missing_rate, dataset_missing_rate
│   │   ├── numeric.rs   ← mean, median, variance, std_dev, min, max, quantiles
│   │   ├── distribution.rs ← skewness, kurtosis
│   │   ├── categorical.rs  ← unique_count, top_k, category_frequency
│   │   ├── entropy.rs
│   │   ├── correlation.rs  ← pearson, spearman, pairs, with_target
│   │   ├── sparsity.rs
│   │   ├── outliers.rs     ← iqr, zscore, outlier_rate
│   │   ├── features.rs     ← variance_threshold, mutual_info, anova, detect_type, range
│   │   ├── duplicates.rs
│   │   ├── imbalance.rs
│   │   └── summary.rs      ← profile_summary meta-tool
│   └── util.rs          ← column lookup helpers, error formatting
├── tests/
│   ├── sample.csv       ← small test dataset
│   └── integration.rs   ← end-to-end JSON-RPC tests
└── .vscode/
    ├── tasks.json
    └── launch.json
```

---

## MCP Protocol Cheat-Sheet (what we implement)

```jsonc
// → Client sends
{ "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": { ... } }

// ← Server responds
{ "jsonrpc": "2.0", "id": 1, "result": {
    "protocolVersion": "2024-11-05",
    "capabilities": { "tools": {} },
    "serverInfo": { "name": "mcp-profiler", "version": "0.1.0" }
}}

// → Client sends
{ "jsonrpc": "2.0", "id": 2, "method": "tools/list" }

// ← Server responds with array of tool definitions (name, description, inputSchema)

// → Client sends
{ "jsonrpc": "2.0", "id": 3, "method": "tools/call",
  "params": { "name": "mean", "arguments": { "column": "age" } } }

// ← Server responds
{ "jsonrpc": "2.0", "id": 3, "result": {
    "content": [{ "type": "text", "text": "{\"column\":\"age\",\"mean\":34.2}" }]
}}
```

---

## Testing Strategy

| Layer | What | How |
|---|---|---|
| Unit | Each profiling function | `#[cfg(test)]` with hand-built DataFrames |
| Integration | Full JSON-RPC round-trips | Spawn process, pipe JSON via stdin, assert stdout |
| Manual | Real-world CSV | Load a Kaggle dataset, ask Claude to profile it |

---

## Open Questions

1. **Which MCP Rust SDK?** — Evaluate `mcp-server` / `rmcp` crates. If none
   are mature enough, hand-roll the thin JSON-RPC layer (~200 LOC).
2. **Large datasets** — Should we stream the CSV or require it fits in memory?
   Start with in-memory; add streaming later if needed.
3. **Multiple datasets** — Support loading more than one CSV and referencing
   them by name? Defer to Phase 3+ unless trivial.

---

## Next Step

**Start Phase 0:** scaffold the MCP server, get `initialize` and `tools/list`
working over stdio, and implement `load_dataset`.
