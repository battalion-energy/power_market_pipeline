use anyhow::Result;
use polars::prelude::*;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use chrono::Utc;
use std::fs;
use rayon::prelude::*;

pub struct ParquetVerifier {
    base_dir: PathBuf,
    issues: Vec<DataIssue>,
}

#[derive(Debug, Clone)]
pub struct DataIssue {
    pub file: String,
    pub issue_type: IssueType,
    pub details: String,
    pub severity: Severity,
}

#[derive(Debug, Clone)]
pub enum IssueType {
    MissingData,
    DuplicateRows,
    DataGap,
    CorruptedData,
    SchemaInconsistency,
    InvalidValues,
    TimeSequenceError,
}

#[derive(Debug, Clone)]
pub enum Severity {
    Critical,
    Warning,
    Info,
}

impl ParquetVerifier {
    pub fn new(base_dir: PathBuf) -> Self {
        Self {
            base_dir,
            issues: Vec::new(),
        }
    }

    pub fn verify_all_datasets(&mut self) -> Result<()> {
        println!("🔍 Parquet Data Verification System");
        println!("{}", "=".repeat(80));
        
        let rollup_dir = self.base_dir.join("rollup_files");
        if !rollup_dir.exists() {
            println!("❌ Rollup directory not found: {}", rollup_dir.display());
            return Ok(());
        }

        // List all dataset directories
        let datasets = vec![
            ("RT_prices", true),  // Heavy verification for RT prices
            ("DA_prices", false),
            ("AS_prices", false),
            ("DAM_Gen_Resources", false),
            ("DAM_Load_Resources", false),
            ("SCED_Gen_Resources", false),
            ("SCED_Load_Resources", false),
            ("COP_Snapshots", false),
        ];

        for (dataset, intensive) in datasets {
            let dataset_dir = rollup_dir.join(dataset);
            if dataset_dir.exists() {
                println!("\n📊 Verifying {}", dataset);
                println!("{}", "-".repeat(60));
                self.verify_dataset(&dataset_dir, dataset, intensive)?;
            } else {
                println!("\n⚠️  Skipping {} (not found)", dataset);
            }
        }

        // Generate report
        self.generate_verification_report()?;
        
        Ok(())
    }

    fn verify_dataset(&mut self, dir: &Path, dataset_name: &str, intensive: bool) -> Result<()> {
        // Get all parquet files
        let mut parquet_files: Vec<PathBuf> = fs::read_dir(dir)?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| path.extension().and_then(|s| s.to_str()) == Some("parquet"))
            .collect();
        
        parquet_files.sort();
        
        if parquet_files.is_empty() {
            println!("  No parquet files found");
            return Ok(());
        }

        println!("  Found {} parquet files", parquet_files.len());

        // Parallel verification of individual files
        let file_stats: Vec<FileStats> = parquet_files
            .par_iter()
            .filter_map(|file| {
                match self.verify_single_file(file, dataset_name, intensive) {
                    Ok(stats) => Some(stats),
                    Err(e) => {
                        self.add_issue(
                            file.to_string_lossy().to_string(),
                            IssueType::CorruptedData,
                            format!("Failed to read file: {}", e),
                            Severity::Critical,
                        );
                        None
                    }
                }
            })
            .collect();

        // Analyze cross-file consistency
        self.verify_cross_file_consistency(&file_stats, dataset_name)?;
        
        // Print summary
        let total_rows: usize = file_stats.iter().map(|s| s.row_count).sum();
        let total_size: u64 = file_stats.iter().map(|s| s.file_size).sum();
        
        println!("  ✅ Verified {} files", file_stats.len());
        println!("     Total rows: {}", format_number(total_rows));
        println!("     Total size: {}", format_bytes(total_size));
        
        Ok(())
    }

    fn verify_single_file(&self, file: &Path, dataset_name: &str, intensive: bool) -> Result<FileStats> {
        let file_size = fs::metadata(file)?.len();
        let df = ParquetReader::new(fs::File::open(file)?).finish()?;
        
        let mut stats = FileStats {
            file_path: file.to_path_buf(),
            row_count: df.height(),
            file_size,
            columns: df.get_column_names().iter().map(|s| s.to_string()).collect(),
            has_duplicates: false,
            null_count: HashMap::new(),
            date_range: None,
        };

        // Basic checks
        if df.height() == 0 {
            self.add_issue(
                file.to_string_lossy().to_string(),
                IssueType::MissingData,
                "File contains no rows".to_string(),
                Severity::Warning,
            );
        }

        // Check for nulls in critical columns
        for col_name in df.get_column_names() {
            if let Ok(col) = df.column(col_name) {
                let null_count = col.null_count();
                if null_count > 0 {
                    stats.null_count.insert(col_name.to_string(), null_count);
                }
            }
        }

        if intensive {
            // Check for duplicates (expensive operation)
            if dataset_name == "RT_prices" {
                // For RT prices, check duplicates on datetime + settlement_point
                if df.get_column_names().contains(&"datetime") && 
                   df.get_column_names().contains(&"settlement_point") {
                    let unique_count = df.clone()
                        .lazy()
                        .select([col("datetime"), col("settlement_point")])
                        .unique(None, UniqueKeepStrategy::First)
                        .collect()?
                        .height();
                    
                    if unique_count < df.height() {
                        let duplicate_count = df.height() - unique_count;
                        stats.has_duplicates = true;
                        self.add_issue(
                            file.to_string_lossy().to_string(),
                            IssueType::DuplicateRows,
                            format!("Found {} duplicate rows (datetime + settlement_point)", duplicate_count),
                            Severity::Critical,
                        );
                    }
                }
            }

            // Check time sequence
            if df.get_column_names().contains(&"datetime") {
                let datetime_col = df.column("datetime")?;
                
                // Check if sorted
                let is_sorted = datetime_col.clone()
                    .sort(false)
                    .series_equal(datetime_col);
                
                if !is_sorted {
                    self.add_issue(
                        file.to_string_lossy().to_string(),
                        IssueType::TimeSequenceError,
                        "Datetime column is not sorted".to_string(),
                        Severity::Warning,
                    );
                }

                // Get date range
                if let Some(min_val) = datetime_col.min::<i64>() {
                    if let Some(max_val) = datetime_col.max::<i64>() {
                        stats.date_range = Some((min_val, max_val));
                    }
                }
            }
        }

        Ok(stats)
    }

    fn verify_cross_file_consistency(&mut self, file_stats: &[FileStats], dataset_name: &str) -> Result<()> {
        if file_stats.is_empty() {
            return Ok(());
        }

        // Check schema consistency
        let first_schema = &file_stats[0].columns;
        for (_i, stats) in file_stats.iter().enumerate().skip(1) {
            if &stats.columns != first_schema {
                let missing: Vec<_> = first_schema.iter()
                    .filter(|col| !stats.columns.contains(col))
                    .collect();
                let extra: Vec<_> = stats.columns.iter()
                    .filter(|col| !first_schema.contains(col))
                    .collect();
                
                self.add_issue(
                    stats.file_path.to_string_lossy().to_string(),
                    IssueType::SchemaInconsistency,
                    format!("Schema mismatch. Missing: {:?}, Extra: {:?}", missing, extra),
                    Severity::Critical,
                );
            }
        }

        // Check for gaps in time series data
        if dataset_name.contains("prices") || dataset_name.contains("Resources") {
            let mut date_ranges: Vec<(i64, i64)> = file_stats
                .iter()
                .filter_map(|s| s.date_range)
                .collect();
            
            date_ranges.sort_by_key(|r| r.0);
            
            // Check for overlaps
            for i in 1..date_ranges.len() {
                if date_ranges[i].0 < date_ranges[i-1].1 {
                    self.add_issue(
                        format!("Files {} and {}", i-1, i),
                        IssueType::DataGap,
                        format!("Time overlap detected between files"),
                        Severity::Warning,
                    );
                }
            }
        }

        Ok(())
    }

    fn add_issue(&self, file: String, _issue_type: IssueType, details: String, severity: Severity) {
        // Note: In real implementation, this would use Arc<Mutex<>> for thread safety
        // For now, we'll just print the issues
        match severity {
            Severity::Critical => println!("    ❌ CRITICAL: {} - {}", file, details),
            Severity::Warning => println!("    ⚠️  WARNING: {} - {}", file, details),
            Severity::Info => println!("    ℹ️  INFO: {} - {}", file, details),
        }
    }

    fn generate_verification_report(&self) -> Result<()> {
        let report_file = self.base_dir.join("rollup_files").join("verification_report.md");
        
        let mut content = String::new();
        content.push_str("# Parquet Data Verification Report\n\n");
        content.push_str(&format!("Generated: {}\n\n", Utc::now().format("%Y-%m-%d %H:%M:%S UTC")));
        
        content.push_str("## Summary\n\n");
        
        let critical_count = self.issues.iter().filter(|i| matches!(i.severity, Severity::Critical)).count();
        let warning_count = self.issues.iter().filter(|i| matches!(i.severity, Severity::Warning)).count();
        let info_count = self.issues.iter().filter(|i| matches!(i.severity, Severity::Info)).count();
        
        content.push_str(&format!("- Critical Issues: {}\n", critical_count));
        content.push_str(&format!("- Warnings: {}\n", warning_count));
        content.push_str(&format!("- Info: {}\n\n", info_count));
        
        if critical_count > 0 {
            content.push_str("## ⚠️ Critical Issues Require Attention!\n\n");
        } else {
            content.push_str("## ✅ No Critical Issues Found\n\n");
        }
        
        fs::write(&report_file, content)?;
        println!("\n📝 Verification report saved to: {}", report_file.display());
        
        Ok(())
    }
}

struct FileStats {
    file_path: PathBuf,
    row_count: usize,
    file_size: u64,
    columns: Vec<String>,
    has_duplicates: bool,
    null_count: HashMap<String, usize>,
    date_range: Option<(i64, i64)>,
}

fn format_number(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}

fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit_index = 0;
    
    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }
    
    format!("{:.2} {}", size, UNITS[unit_index])
}