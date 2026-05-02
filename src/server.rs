use std::io::{self, BufRead, Write};

use serde::Deserialize;
use serde_json::{Value, json};

use crate::dataset::{Dataset, PredictionType};

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
            let target_column = args.get("target_column").and_then(|v| v.as_str()).map(|s| s.to_string());
            let ds = Dataset::from_csv(path, target_column).map_err(|e| e.to_string())?;
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

            // --- PCA Projection ---
            "project_onto_pca" => {
                let n = args.get("n_components").and_then(|v| v.as_u64()).map(|v| v as usize);
                let r = ds.project_onto_pca(n).map_err(|e| e.to_string())?;
                Ok(serde_json::to_string(&r).unwrap())
            }

            // --- PCA Reconstruction ---
            "reconstruct_from_pca" => {
                let n = args.get("n_components").and_then(|v| v.as_u64()).map(|v| v as usize);
                let r = ds.reconstruct_from_pca(n).map_err(|e| e.to_string())?;
                Ok(serde_json::to_string(&r).unwrap())
            }

            // --- Supervised ML split ---
            "design_matrix_and_target" => {
                let target_column = get_str(args, "target_column")?;
                let prediction_type = get_prediction_type(args)?;
                let r = ds
                    .design_matrix_and_target(target_column, prediction_type)
                    .map_err(|e| e.to_string())?;
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

fn get_prediction_type(args: &Value) -> Result<PredictionType, String> {
    let raw = get_str(args, "prediction_type")?;
    match raw {
        "regression" => Ok(PredictionType::Regression),
        "binary_classification" => Ok(PredictionType::BinaryClassification),
        "multi_category" => Ok(PredictionType::MultiCategoryClassification),
        _ => Err(format!(
            "Invalid 'prediction_type': {raw}. Expected one of: regression, binary_classification, multi_category"
        )),
    }
}

// ---------------------------------------------------------------------------
// Tool definitions (JSON Schema for MCP tools/list)
// ---------------------------------------------------------------------------

fn tools_schema() -> Value {
    json!([
        {
            "name": "load_dataset",
            "description": "Load a CSV file into memory for profiling. Must be called before any other tool. Optionally specify a target_column to exclude from profiling and PCA (used for supervised ML tasks).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Absolute path to the CSV file" },
                    "target_column": { "type": "string", "description": "Optional target column name to exclude from profiling/PCA" }
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
            "description": "Principal Component Analysis derived from the eigendecomposition of the correlation matrix. Returns eigenvalues, explained variance ratios, cumulative variance ratios, and principal component loadings.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "n_components": { "type": "integer", "description": "Number of principal components to return (default: all)" }
                }
            }
        },
        {
            "name": "project_onto_pca",
            "description": "Project data onto the first n_components principal components, transforming the dataset into PCA coordinate space. Returns the projected data with component names and explained variance.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "n_components": { "type": "integer", "description": "Number of principal components to project onto (default: all)" }
                }
            }
        },
        {
            "name": "reconstruct_from_pca",
            "description": "Reconstruct the original features from the first n_components principal components. Returns an approximation of the original data in feature space.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "n_components": { "type": "integer", "description": "Number of principal components to use for reconstruction (default: all)" }
                }
            }
        },
        {
            "name": "design_matrix_and_target",
            "description": "Build supervised-learning inputs by selecting one target column and using the remaining columns as the X design matrix.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "target_column": { "type": "string", "description": "Column to use as the y target vector" },
                    "prediction_type": {
                        "type": "string",
                        "description": "Prediction task type",
                        "enum": ["regression", "binary_classification", "multi_category"]
                    }
                },
                "required": ["target_column", "prediction_type"]
            }
        }
    ])
}
