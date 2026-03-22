use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct CorrelationMatrix {
    pub columns: Vec<String>,
    pub matrix: Vec<Vec<f64>>,
}
