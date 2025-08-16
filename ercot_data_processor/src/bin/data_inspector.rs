use anyhow::Result;
use polars::prelude::*;
use std::env;
use std::path::PathBuf;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    
    if args.len() != 2 {
        eprintln!("Usage: {} <parquet_file>", args[0]);
        eprintln!("Example: {} /path/to/2024.parquet", args[0]);
        std::process::exit(1);
    }
    
    let file_path = PathBuf::from(&args[1]);
    
    if !file_path.exists() {
        eprintln!("Error: File does not exist: {}", file_path.display());
        std::process::exit(1);
    }
    
    println!("🔍 Inspecting Parquet File: {}", file_path.display());
    println!("{}", "=".repeat(80));
    
    // Load the parquet file
    let df = LazyFrame::scan_parquet(&file_path, ScanArgsParquet::default())?
        .collect()?;
    
    // Basic info
    println!("📊 Basic Information:");
    println!("  Rows: {}", df.height());
    println!("  Columns: {}", df.width());
    println!("  Column names: {:?}", df.get_column_names());
    
    // Show schema
    println!("\n📋 Schema:");
    for (name, dtype) in df.schema().iter() {
        println!("  {}: {}", name, dtype);
    }
    
    // Sample data
    println!("\n📄 Sample Data (first 10 rows):");
    let sample = df.head(Some(10));
    println!("{}", sample);
    
    // Unique counts for key columns
    if df.get_column_names().contains(&"SettlementPointName") {
        let unique_points = df.column("SettlementPointName")?.unique()?.len();
        println!("\n📍 Unique Settlement Points: {}", unique_points);
        
        // Show some settlement point names
        let sample_points = df.column("SettlementPointName")?
            .unique()?
            .head(Some(10));
        println!("  Sample points: {:?}", sample_points.utf8()?.into_no_null_iter().collect::<Vec<_>>());
    }
    
    if df.get_column_names().contains(&"SettlementPoint") {
        let unique_points = df.column("SettlementPoint")?.unique()?.len();
        println!("\n📍 Unique Settlement Points: {}", unique_points);
    }
    
    if df.get_column_names().contains(&"AncillaryType") {
        let unique_types = df.column("AncillaryType")?.unique()?.len();
        println!("\n🔧 Unique Ancillary Types: {}", unique_types);
        
        let sample_types = df.column("AncillaryType")?
            .unique()?;
        println!("  Types: {:?}", sample_types.utf8()?.into_no_null_iter().collect::<Vec<_>>());
    }
    
    // Date range analysis
    if df.get_column_names().contains(&"datetime") {
        let datetime_col = df.column("datetime")?;
        if let Ok(int_col) = datetime_col.cast(&DataType::Int64) {
            let int_series = int_col.i64()?;
            if let (Some(min_ts), Some(max_ts)) = (int_series.min(), int_series.max()) {
                use chrono::DateTime;
                let min_dt = DateTime::from_timestamp_millis(min_ts).unwrap();
                let max_dt = DateTime::from_timestamp_millis(max_ts).unwrap();
                println!("\n📅 Date Range:");
                println!("  From: {}", min_dt.format("%Y-%m-%d %H:%M:%S UTC"));
                println!("  To:   {}", max_dt.format("%Y-%m-%d %H:%M:%S UTC"));
                
                // Count unique timestamps
                let unique_timestamps = int_series.n_unique()?;
                println!("  Unique timestamps: {}", unique_timestamps);
            }
        }
    }
    
    // Check for nulls
    println!("\n🔍 Null Value Analysis:");
    for col_name in df.get_column_names() {
        let col = df.column(col_name)?;
        let null_count = col.null_count();
        if null_count > 0 {
            println!("  {}: {} nulls ({:.2}%)", col_name, null_count, 
                (null_count as f64 / df.height() as f64) * 100.0);
        }
    }
    
    Ok(())
}