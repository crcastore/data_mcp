use std::collections::HashSet;

/// Count of distinct values in a numeric column.
pub fn unique_count(vals: &[f64]) -> usize {
    let set: HashSet<u64> = vals.iter().map(|v| v.to_bits()).collect();
    set.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unique_count_numeric() {
        let vals = vec![1.0, 2.0, 2.0, 3.0, 3.0, 3.0];
        assert_eq!(unique_count(&vals), 3);
    }
}
