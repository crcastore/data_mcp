use polars::prelude::*;

pub fn row_count(df: &DataFrame) -> usize {
    df.height()
}

pub fn column_count(df: &DataFrame) -> usize {
    df.width()
}

pub fn column_types(df: &DataFrame) -> Vec<(String, String)> {
    df.columns()
        .iter()
        .map(|c: &Column| (c.name().to_string(), format!("{}", c.dtype())))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> DataFrame {
        df! {
            "a" => &[1i64, 2, 3],
            "b" => &[1.0f64, 2.0, 3.0],
            "c" => &["x", "y", "z"],
        }
        .unwrap()
    }

    #[test]
    fn test_row_count() {
        assert_eq!(row_count(&sample()), 3);
    }

    #[test]
    fn test_column_count() {
        assert_eq!(column_count(&sample()), 3);
    }

    #[test]
    fn test_column_types() {
        let types = column_types(&sample());
        assert_eq!(types.len(), 3);
        assert_eq!(types[0].0, "a");
        assert_eq!(types[1].0, "b");
        assert_eq!(types[2].0, "c");
    }
}
