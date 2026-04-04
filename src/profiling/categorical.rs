use std::collections::HashSet;

/// Count of distinct values in a numeric column.
pub fn unique_count_numeric(vals: &[f64]) -> usize {
    let set: HashSet<u64> = vals.iter().map(|v| v.to_bits()).collect();
    set.len()
}

/// Count of distinct values in a string column.
pub fn unique_count_string(vals: &[String]) -> usize {
    let set: HashSet<&str> = vals.iter().map(|s| s.as_str()).collect();
    set.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unique_count_string() {
        let vals: Vec<String> = vec!["a", "b", "a", "c", "b"]
            .into_iter()
            .map(String::from)
            .collect();
        assert_eq!(unique_count_string(&vals), 3);
    }

    #[test]
    fn test_unique_count_numeric() {
        let vals = vec![1.0, 2.0, 2.0, 3.0, 3.0, 3.0];
        assert_eq!(unique_count_numeric(&vals), 3);
    }
}
