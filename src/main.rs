use std::env;

use mcp::dataset::Dataset;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: mcp <csv-file>");
        std::process::exit(1);
    }

    let ds = Dataset::from_csv(&args[1])?;
    println!("=== Dataset Profile ===");
    println!("Rows:    {}", ds.row_count());
    println!("Columns: {}", ds.column_count());
    println!();

    println!("Column Types:");
    for (name, dtype) in ds.column_types() {
        println!("  {name}: {dtype}");
    }
    println!();

    for (name, _dtype) in ds.column_types() {
        println!("--- {name} ---");
        println!("  Missing rate:  {:.4}", ds.missing_rate(&name)?);
        println!("  Unique values: {}", ds.unique_count(&name)?);
        println!("  Entropy:       {:.4}", ds.entropy(&name)?);
        println!("  Sparsity:      {:.4}", ds.sparsity(&name)?);

        if let Ok(m) = ds.mean(&name) {
            println!("  Mean:          {:.4}", m);
            if let Ok(v) = ds.variance(&name) {
                println!("  Variance:      {:.4}", v);
            }
            if let Ok(s) = ds.skewness(&name) {
                println!("  Skewness:      {:.4}", s);
            }
            if let Ok(q) = ds.quantiles(&name) {
                println!(
                    "  Quantiles:     min={:.2}  q25={:.2}  median={:.2}  q75={:.2}  max={:.2}",
                    q.min, q.q25, q.q50, q.q75, q.max
                );
            }
        }
        println!();
    }

    if let Ok(corr) = ds.correlation_matrix() {
        println!("=== Correlation Matrix ===");
        print!("{:>14}", "");
        for name in &corr.columns {
            print!("{:>14}", name);
        }
        println!();
        for (i, name) in corr.columns.iter().enumerate() {
            print!("{:>14}", name);
            for j in 0..corr.columns.len() {
                print!("{:>14.4}", corr.matrix[i][j]);
            }
            println!();
        }
    }

    Ok(())
}
