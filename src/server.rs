use std::io::{self, BufRead, Write};

use serde::Deserialize;
use serde_json::{Value, json};

use crate::dataset::Dataset;

// ---------------------------------------------------------------------------
// JSON-RPC types
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct JsonRpcRequest {
    #[allow(dead_code)]
    jsonrpc: String,
    id: Option<Value>,
    method: String,
    params: Option<Value>,
}

// ---------------------------------------------------------------------------
// MCP Server
// ---------------------------------------------------------------------------

pub struct McpServer {
    dataset: Option<Dataset>,
}

impl McpServer {
    pub fn new() -> Self {
        Self { dataset: None }
    }

    /// Run the MCP server, reading JSON-RPC messages from stdin and writing
    /// responses to stdout (newline-delimited JSON).
    pub fn run(&mut self) -> io::Result<()> {
        let stdin = io::stdin();
        let stdout = io::stdout();
        let mut out = stdout.lock();

        for line in stdin.lock().lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }

            let req: JsonRpcRequest = match serde_json::from_str(&line) {
                Ok(r) => r,
                Err(e) => {
                    let resp = json!({
                        "jsonrpc": "2.0",
                        "id": null,
                        "error": { "code": -32700, "message": format!("Parse error: {e}") }
                    });
                    writeln!(out, "{}", resp)?;
                    out.flush()?;
                    continue;
                }
            };

            // Notifications (no id) — acknowledge silently.
            if req.id.is_none() {
                continue;
            }

            let id = req.id.unwrap();
            let resp = match self.dispatch(&req.method, req.params.as_ref()) {
                Ok(val) => json!({ "jsonrpc": "2.0", "id": id, "result": val }),
                Err(msg) => json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "error": { "code": -32601, "message": msg }
                }),
            };

            writeln!(out, "{}", resp)?;
            out.flush()?;
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Dispatch
    // -----------------------------------------------------------------------

    fn dispatch(&mut self, method: &str, params: Option<&Value>) -> Result<Value, String> {
        match method {
            "initialize" => Ok(json!({
                "protocolVersion": "2024-11-05",
                "capabilities": { "tools": {} },
                "serverInfo": { "name": "data-profiler", "version": "0.1.0" }
            })),

            "tools/list" => Ok(json!({ "tools": tools_schema() })),

            "tools/call" => {
                let params = params.ok_or("Missing params")?;
                let name = params.get("name").and_then(|v| v.as_str())
                    .ok_or("Missing tool name")?;
                let args = params.get("arguments").cloned().unwrap_or(json!({}));

                match self.call_tool(name, &args) {
                    Ok(text) => Ok(json!({
                        "content": [{ "type": "text", "text": text }]
                    })),
                    Err(e) => Ok(json!({
                        "content": [{ "type": "text", "text": e }],
                        "isError": true
                    })),
                }
            }

            _ => Err(format!("Method not found: {method}")),
        }
    }

    // -----------------------------------------------------------------------
    // Tool dispatch
    // -----------------------------------------------------------------------

    fn call_tool(&mut self, name: &str, args: &Value) -> Result<String, String> {
        // load_dataset needs &mut self — handle separately.
        if name == "load_dataset" {
            let path = get_str(args, "path")?;
            let ds = Dataset::from_csv(path).map_err(|e| e.to_string())?;
            let rows = ds.row_count();
            let cols = ds.column_count();
            self.dataset = Some(ds);
            return Ok(json!({
                "status": "loaded",
                "path": path,
                "rows": rows,
                "columns": cols
            }).to_string());
        }

        // Everything else requires a loaded dataset.
        let ds = self.dataset.as_ref()
            .ok_or("No dataset loaded. Call load_dataset first.")?;

        match name {
            // --- Shape ---
            "row_count" => Ok(json!({"rows": ds.row_count()}).to_string()),

            "column_count" => Ok(json!({"columns": ds.column_count()}).to_string()),

            "column_types" => {
                let types: Vec<Value> = ds.column_types().into_iter()
                    .map(|(n, t)| json!({"name": n, "type": t}))
                    .collect();
                Ok(json!({"columns": types}).to_string())
            }

            // --- Numeric ---
            "mean" => {
                let c = get_str(args, "column")?;
                let v = ds.mean(c).map_err(|e| e.to_string())?;
                Ok(json!({"column": c, "mean": v}).to_string())
            }

            "variance" => {
                let c = get_str(args, "column")?;
                let v = ds.variance(c).map_err(|e| e.to_string())?;
                Ok(json!({"column": c, "variance": v}).to_string())
            }

            "quantiles" => {
                let c = get_str(args, "column")?;
                let q = ds.quantiles(c).map_err(|e| e.to_string())?;
                Ok(json!({
                    "column": c,
                    "min": q.min, "q25": q.q25, "median": q.q50,
                    "q75": q.q75, "max": q.max
                }).to_string())
            }

            // --- Distribution ---
            "skewness" => {
                let c = get_str(args, "column")?;
                let v = ds.skewness(c).map_err(|e| e.to_string())?;
                Ok(json!({"column": c, "skewness": v}).to_string())
            }

            // --- Categorical ---
            "unique_count" => {
                let c = get_str(args, "column")?;
                let v = ds.unique_count(c).map_err(|e| e.to_string())?;
                Ok(json!({"column": c, "unique_count": v}).to_string())
            }

            // --- Entropy ---
            "entropy" => {
                let c = get_str(args, "column")?;
                let v = ds.entropy(c).map_err(|e| e.to_string())?;
                Ok(json!({"column": c, "entropy": v}).to_string())
            }

            // --- Covariance ---
            "covariance_matrix" => {
                let cm = ds.covariance_matrix().map_err(|e| e.to_string())?;
                Ok(serde_json::to_string(&cm).unwrap())
            }

            // --- Correlation ---
            "correlation_matrix" => {
                let cm = ds.correlation_matrix().map_err(|e| e.to_string())?;
                Ok(serde_json::to_string(&cm).unwrap())
            }

            // --- Sparsity ---
            "sparsity" => {
                let c = get_str(args, "column")?;
                let v = ds.sparsity(c).map_err(|e| e.to_string())?;
                Ok(json!({"column": c, "sparsity": v}).to_string())
            }

            // --- PCA ---
            "pca" => {
                let n = args.get("n_components").and_then(|v| v.as_u64()).map(|v| v as usize);
                let r = ds.pca(n).map_err(|e| e.to_string())?;
                Ok(serde_json::to_string(&r).unwrap())
            }

            // --- Reservoir: Surrogate test ---
            "surrogate_test" => {
                let c = get_str(args, "column")?;
                let n = get_u64_or(args, "num_surrogates", 100) as usize;
                let r = ds.surrogate_test(c, n).map_err(|e| e.to_string())?;
                Ok(serde_json::to_string(&r).unwrap())
            }

            // --- Reservoir: BDS test ---
            "bds_test" => {
                let c = get_str(args, "column")?;
                let dim = get_u64(args, "embedding_dim")? as usize;
                let eps = get_f64(args, "epsilon")?;
                let r = ds.bds_test(c, dim, eps).map_err(|e| e.to_string())?;
                Ok(serde_json::to_string(&r).unwrap())
            }

            // --- Reservoir: Lyapunov exponent ---
            "lyapunov_exponent" => {
                let c = get_str(args, "column")?;
                let dim = get_u64_or(args, "embedding_dim", 3) as usize;
                let tau = get_u64_or(args, "tau", 1) as usize;
                let v = ds.lyapunov_exponent(c, dim, tau).map_err(|e| e.to_string())?;
                Ok(json!({"column": c, "lyapunov_exponent": v}).to_string())
            }

            // --- Reservoir: Dependence comparison ---
            "dependence_comparison" => {
                let c = get_str(args, "column")?;
                let max_lag = get_u64_or(args, "max_lag", 10) as usize;
                let r = ds.dependence_comparison(c, max_lag).map_err(|e| e.to_string())?;
                Ok(serde_json::to_string(&r).unwrap())
            }

            // --- Reservoir: Delay embedding ---
            "delay_embedding" => {
                let c = get_str(args, "column")?;
                let max_dim = get_u64_or(args, "max_dim", 10) as usize;
                let r = ds.delay_embedding(c, max_dim).map_err(|e| e.to_string())?;
                Ok(serde_json::to_string(&r).unwrap())
            }

            // --- Reservoir: Memory profile ---
            "memory_profile" => {
                let c = get_str(args, "column")?;
                let max_lag = get_u64_or(args, "max_lag", 10) as usize;
                let r = ds.memory_profile(c, max_lag).map_err(|e| e.to_string())?;
                Ok(serde_json::to_string(&r).unwrap())
            }

            _ => Err(format!("Unknown tool: {name}")),
        }
    }
}

// ---------------------------------------------------------------------------
// Parameter helpers
// ---------------------------------------------------------------------------

fn get_str<'a>(args: &'a Value, key: &str) -> Result<&'a str, String> {
    args.get(key)
        .and_then(|v| v.as_str())
        .ok_or_else(|| format!("Missing '{key}' parameter"))
}

fn get_u64(args: &Value, key: &str) -> Result<u64, String> {
    args.get(key)
        .and_then(|v| v.as_u64())
        .ok_or_else(|| format!("Missing '{key}' parameter"))
}

fn get_f64(args: &Value, key: &str) -> Result<f64, String> {
    args.get(key)
        .and_then(|v| v.as_f64())
        .ok_or_else(|| format!("Missing '{key}' parameter"))
}

fn get_u64_or(args: &Value, key: &str, default: u64) -> u64 {
    args.get(key).and_then(|v| v.as_u64()).unwrap_or(default)
}

// ---------------------------------------------------------------------------
// Tool definitions (JSON Schema for MCP tools/list)
// ---------------------------------------------------------------------------

fn tools_schema() -> Value {
    json!([
        {
            "name": "load_dataset",
            "description": "Load a CSV file into memory for profiling. Must be called before any other tool.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Absolute path to the CSV file" }
                },
                "required": ["path"]
            }
        },
        {
            "name": "row_count",
            "description": "Return the number of rows in the loaded dataset.",
            "inputSchema": { "type": "object", "properties": {} }
        },
        {
            "name": "column_count",
            "description": "Return the number of columns in the loaded dataset.",
            "inputSchema": { "type": "object", "properties": {} }
        },
        {
            "name": "column_types",
            "description": "Return the name and data type (f64) of every column.",
            "inputSchema": { "type": "object", "properties": {} }
        },
        {
            "name": "mean",
            "description": "Arithmetic mean of a numeric column.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "column": { "type": "string", "description": "Column name" }
                },
                "required": ["column"]
            }
        },
        {
            "name": "variance",
            "description": "Sample variance (ddof=1) of a numeric column.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "column": { "type": "string", "description": "Column name" }
                },
                "required": ["column"]
            }
        },
        {
            "name": "quantiles",
            "description": "Min, 25th, 50th (median), 75th percentile, and max of a numeric column.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "column": { "type": "string", "description": "Column name" }
                },
                "required": ["column"]
            }
        },
        {
            "name": "skewness",
            "description": "Adjusted Fisher-Pearson skewness of a numeric column.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "column": { "type": "string", "description": "Column name" }
                },
                "required": ["column"]
            }
        },
        {
            "name": "unique_count",
            "description": "Number of distinct values in a numeric column.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "column": { "type": "string", "description": "Column name" }
                },
                "required": ["column"]
            }
        },
        {
            "name": "entropy",
            "description": "Shannon entropy (in nats) of a column's value distribution.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "column": { "type": "string", "description": "Column name" }
                },
                "required": ["column"]
            }
        },
        {
            "name": "covariance_matrix",
            "description": "Full covariance matrix (ddof=1) for all numeric columns. Returns column names and a flat row-major m×m matrix.",
            "inputSchema": { "type": "object", "properties": {} }
        },
        {
            "name": "correlation_matrix",
            "description": "Pearson correlation matrix for all numeric columns.",
            "inputSchema": { "type": "object", "properties": {} }
        },
        {
            "name": "sparsity",
            "description": "Fraction of zero values in a column.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "column": { "type": "string", "description": "Column name" }
                },
                "required": ["column"]
            }
        },
        {
            "name": "pca",
            "description": "Principal Component Analysis derived from the precomputed SVD of the covariance matrix. Returns eigenvalues, explained variance ratios, cumulative variance ratios, and principal component loadings.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "n_components": { "type": "integer", "description": "Number of principal components to return (default: all)" }
                }
            }
        },
        {
            "name": "surrogate_test",
            "description": "Surrogate data test for nonlinearity. Compares the time-reversal asymmetry of the real series against AR(1) surrogates. A |z_score| > 2 indicates nonlinear structure, useful for assessing reservoir computing suitability.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "column": { "type": "string", "description": "Numeric column name" },
                    "num_surrogates": { "type": "integer", "description": "Number of surrogate series to generate (default: 100)", "default": 100 }
                },
                "required": ["column"]
            }
        },
        {
            "name": "bds_test",
            "description": "BDS test for nonlinear serial dependence. Tests H₀: the series is i.i.d. A p_value < 0.05 indicates nonlinear temporal dynamics.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "column": { "type": "string", "description": "Numeric column name" },
                    "embedding_dim": { "type": "integer", "description": "Embedding dimension (≥ 2)" },
                    "epsilon": { "type": "number", "description": "Distance threshold (typically 0.5–2× the series std dev)" }
                },
                "required": ["column", "embedding_dim", "epsilon"]
            }
        },
        {
            "name": "lyapunov_exponent",
            "description": "Estimate the largest Lyapunov exponent (Rosenstein's method). Positive values indicate chaotic dynamics, favourable for reservoir computing.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "column": { "type": "string", "description": "Numeric column name" },
                    "embedding_dim": { "type": "integer", "description": "Embedding dimension (default: 3)", "default": 3 },
                    "tau": { "type": "integer", "description": "Time delay (default: 1)", "default": 1 }
                },
                "required": ["column"]
            }
        },
        {
            "name": "dependence_comparison",
            "description": "Compare linear (autocorrelation) and nonlinear (mutual information) dependence across lags. Flags nonlinear_dominance when MI substantially exceeds the Gaussian-equivalent MI.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "column": { "type": "string", "description": "Numeric column name" },
                    "max_lag": { "type": "integer", "description": "Maximum lag to compute (default: 10)", "default": 10 }
                },
                "required": ["column"]
            }
        },
        {
            "name": "delay_embedding",
            "description": "Estimate optimal delay (via AMI first minimum) and embedding dimension (via false nearest neighbours) for Takens' time-delay reconstruction.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "column": { "type": "string", "description": "Numeric column name" },
                    "max_dim": { "type": "integer", "description": "Maximum dimension to test (default: 10)", "default": 10 }
                },
                "required": ["column"]
            }
        },
        {
            "name": "memory_profile",
            "description": "Temporal memory profile: partial autocorrelations (PACF), delay mutual information, active information storage, and estimated memory length.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "column": { "type": "string", "description": "Numeric column name" },
                    "max_lag": { "type": "integer", "description": "Maximum lag to compute (default: 10)", "default": 10 }
                },
                "required": ["column"]
            }
        }
    ])
}
