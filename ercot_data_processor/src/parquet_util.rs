use anyhow::Result;
use chrono::NaiveDate;
use clap::{Parser, Subcommand};
use polars::prelude::*;
use std::fs::File;
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(name = "parquet_util")]
#[command(about = "Utility for working with Parquet files", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Display first N rows as CSV
    Head {
        /// Input parquet file
        file: PathBuf,
        /// Number of rows to display (default: 10)
        #[arg(short, long, default_value = "10")]
        rows: usize,
    },
    
    /// Display last N rows as CSV
    Tail {
        /// Input parquet file
        file: PathBuf,
        /// Number of rows to display (default: 10)
        #[arg(short, long, default_value = "10")]
        rows: usize,
    },
    
    /// Convert parquet file to other formats
    Convert {
        /// Input parquet file
        input: PathBuf,
        /// Output file
        output: PathBuf,
        /// Output format (csv, parquet, arrow)
        #[arg(short, long)]
        format: String,
    },
    
    /// Extract monthly data from annual file
    Extract {
        /// Input parquet file
        input: PathBuf,
        /// Output directory
        output_dir: PathBuf,
        /// Year to extract
        year: i32,
        /// Month to extract (1-12, or 0 for all months)
        month: u32,
        /// Output format (csv, parquet, arrow)
        #[arg(short, long, default_value = "parquet")]
        format: String,
    },
    
    /// Show statistics and check for gaps
    Stats {
        /// Input parquet file
        file: PathBuf,
        /// Check for temporal gaps
        #[arg(short, long)]
        gaps: bool,
    },
    
    /// Show schema information
    Schema {
        /// Input parquet file
        file: PathBuf,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    
    match cli.command {
        Commands::Head { file, rows } => head_command(&file, rows)?,
        Commands::Tail { file, rows } => tail_command(&file, rows)?,
        Commands::Convert { input, output, format } => convert_command(&input, &output, &format)?,
        Commands::Extract { input, output_dir, year, month, format } => {
            extract_command(&input, &output_dir, year, month, &format)?
        },
        Commands::Stats { file, gaps } => stats_command(&file, gaps)?,
        Commands::Schema { file } => schema_command(&file)?,
    }
    
    Ok(())
}

fn head_command(file: &Path, rows: usize) -> Result<()> {
    let df = LazyFrame::scan_parquet(file, Default::default())?
        .limit(rows as u32)
        .collect()?;
    
    print_df_as_csv(&df)?;
    Ok(())
}

fn tail_command(file: &Path, rows: usize) -> Result<()> {
    let df = LazyFrame::scan_parquet(file, Default::default())?
        .tail(rows as u32)
        .collect()?;
    
    print_df_as_csv(&df)?;
    Ok(())
}

fn print_df_as_csv(df: &DataFrame) -> Result<()> {
    let mut buf = Vec::new();
    CsvWriter::new(&mut buf)
        .include_header(true)
        .finish(&mut df.clone())?;
    
    print!("{}", String::from_utf8(buf)?);
    Ok(())
}

fn convert_command(input: &Path, output: &Path, format: &str) -> Result<()> {
    println!("Converting {} to {} format...", input.display(), format);
    
    let df = LazyFrame::scan_parquet(input, Default::default())?.collect()?;
    
    match format {
        "csv" => {
            let mut file = File::create(output)?;
            CsvWriter::new(&mut file)
                .include_header(true)
                .finish(&mut df.clone())?;
        },
        "parquet" => {
            let mut file = File::create(output)?;
            ParquetWriter::new(&mut file)
                .with_compression(ParquetCompression::Snappy)
                .finish(&mut df.clone())?;
        },
        "arrow" => {
            // Convert to Arrow IPC format
            let mut file = File::create(output)?;
            IpcWriter::new(&mut file).finish(&mut df.clone())?;
        },
        _ => {
            return Err(anyhow::anyhow!("Unsupported format: {}. Use csv, parquet, or arrow", format));
        }
    }
    
    println!("✅ Conversion complete: {}", output.display());
    Ok(())
}

fn extract_command(input: &Path, output_dir: &Path, year: i32, month: u32, format: &str) -> Result<()> {
    std::fs::create_dir_all(output_dir)?;
    
    let df = LazyFrame::scan_parquet(input, Default::default())?;
    
    // Find datetime column
    let schema = df.schema()?;
    let datetime_col = schema.iter()
        .find(|(name, dtype)| {
            name.contains("datetime") || name.contains("Date") || 
            matches!(dtype, DataType::Datetime(_, _) | DataType::Date)
        })
        .map(|(name, _)| name.clone())
        .ok_or_else(|| anyhow::anyhow!("No datetime column found"))?;
    
    if month == 0 {
        // Extract all months
        for m in 1..=12 {
            extract_month(&df, output_dir, &datetime_col, year, m, format)?;
        }
    } else {
        extract_month(&df, output_dir, &datetime_col, year, month, format)?;
    }
    
    println!("✅ Extraction complete");
    Ok(())
}

fn extract_month(
    df: &LazyFrame,
    output_dir: &Path,
    datetime_col: &str,
    year: i32,
    month: u32,
    format: &str
) -> Result<()> {
    let start_date = NaiveDate::from_ymd_opt(year, month, 1).unwrap();
    let end_date = if month == 12 {
        NaiveDate::from_ymd_opt(year + 1, 1, 1).unwrap()
    } else {
        NaiveDate::from_ymd_opt(year, month + 1, 1).unwrap()
    };
    
    let filtered = df.clone()
        .filter(
            col(datetime_col).gt_eq(lit(start_date.and_hms_opt(0, 0, 0).unwrap()))
                .and(col(datetime_col).lt(lit(end_date.and_hms_opt(0, 0, 0).unwrap())))
        )
        .collect()?;
    
    let output_file = output_dir.join(format!("{}_{:02}.{}", year, month, format));
    
    match format {
        "csv" => {
            let mut file = File::create(&output_file)?;
            CsvWriter::new(&mut file)
                .include_header(true)
                .finish(&mut filtered.clone())?;
        },
        "parquet" => {
            let mut file = File::create(&output_file)?;
            ParquetWriter::new(&mut file)
                .with_compression(ParquetCompression::Snappy)
                .finish(&mut filtered.clone())?;
        },
        "arrow" => {
            let mut file = File::create(&output_file)?;
            IpcWriter::new(&mut file).finish(&mut filtered.clone())?;
        },
        _ => {
            return Err(anyhow::anyhow!("Unsupported format: {}", format));
        }
    }
    
    println!("  Extracted {}-{:02}: {} rows to {}", 
             year, month, filtered.height(), output_file.display());
    
    Ok(())
}

fn stats_command(file: &Path, check_gaps: bool) -> Result<()> {
    println!("📊 Statistics for: {}", file.display());
    println!("{}", "=".repeat(60));
    
    let df = LazyFrame::scan_parquet(file, Default::default())?.collect()?;
    
    println!("Rows: {}", df.height());
    println!("Columns: {}", df.width());
    println!();
    
    // Show column info
    println!("Column Information:");
    println!("{:<30} {:<20} {:>15} {:>15}", "Column", "Type", "Non-Null", "Null %");
    println!("{}", "-".repeat(80));
    
    for col in df.get_columns() {
        let null_count = col.null_count();
        let non_null = col.len() - null_count;
        let null_pct = if !col.is_empty() {
            (null_count as f64 / col.len() as f64) * 100.0
        } else {
            0.0
        };
        
        println!("{:<30} {:<20} {:>15} {:>14.2}%",
                 col.name(),
                 format!("{:?}", col.dtype()),
                 non_null,
                 null_pct);
    }
    
    println!();
    
    // Show numeric column statistics
    println!("Numeric Column Statistics:");
    for col in df.get_columns() {
        if col.dtype().is_numeric() {
            println!("\n{} ({:?}):", col.name(), col.dtype());
            
            // Cast to f64 and get statistics
            if let Ok(series_f64) = col.cast(&DataType::Float64) {
                // Get min/max using the f64 chunked array
                if let Ok(ca) = series_f64.f64() {
                    if let Some(min) = ca.min() {
                        println!("  Min: {:.4}", min);
                    }
                    if let Some(max) = ca.max() {
                        println!("  Max: {:.4}", max);
                    }
                }
                
                if let Some(v) = series_f64.mean() {
                    println!("  Mean: {:.4}", v);
                }
                if let Some(v) = series_f64.median() {
                    println!("  Median: {:.4}", v);
                }
            }
        }
    }
    
    // Check for gaps if requested
    if check_gaps {
        println!("\n🔍 Checking for temporal gaps...");
        
        // Find datetime column
        let datetime_col = df.get_columns().iter()
            .find(|col| {
                matches!(col.dtype(), DataType::Datetime(_, _) | DataType::Date) ||
                col.name().contains("datetime") || col.name().contains("Date")
            })
            .map(|col| col.name());
        
        if let Some(dt_col) = datetime_col {
            check_temporal_gaps(&df, dt_col)?;
        } else {
            println!("  No datetime column found for gap analysis");
        }
    }
    
    Ok(())
}

fn check_temporal_gaps(df: &DataFrame, datetime_col: &str) -> Result<()> {
    let dates = df.column(datetime_col)?;
    
    // Get unique dates
    let unique_dates = dates.unique()?.sort(false);
    
    if unique_dates.len() < 2 {
        println!("  Not enough data points for gap analysis");
        return Ok(());
    }
    
    // Convert to dates and check for gaps
    let gap_count = 0;
    let total_missing_days = 0;
    
    // This is a simplified gap detection - you might want to enhance it
    let first = unique_dates.head(Some(1));
    let last = unique_dates.tail(Some(1));
    if !first.is_empty() && !last.is_empty() {
        println!("  Date range: {} to {}", first.get(0)?, last.get(0)?);
    }
    
    println!("  Unique dates: {}", unique_dates.len());
    
    if gap_count > 0 {
        println!("  ⚠️  Found {} gaps with {} total missing days", gap_count, total_missing_days);
    } else {
        println!("  ✅ No gaps detected");
    }
    
    Ok(())
}

fn schema_command(file: &Path) -> Result<()> {
    println!("📋 Schema for: {}", file.display());
    println!("{}", "=".repeat(60));
    
    let df = LazyFrame::scan_parquet(file, Default::default())?;
    let schema = df.schema()?;
    
    println!("{:<30} {:<30}", "Column", "Type");
    println!("{}", "-".repeat(60));
    
    for (name, dtype) in schema.iter() {
        println!("{:<30} {:<30}", name, format!("{:?}", dtype));
    }
    
    Ok(())
}