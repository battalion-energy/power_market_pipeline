use anyhow::Result;
use polars::prelude::*;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::fs::OpenOptions;
use std::io::Write;
use chrono::Local;

use crate::schema_detector::{SchemaRegistry, read_csv_with_schema_registry};

struct ColumnMismatchInfo {
    column: String,
    expected_type: String,
    actual_value: String,
}

pub struct ValidatedAnnualProcessor {
    base_dir: PathBuf,
    schema_registry: SchemaRegistry,
    type_mismatch_log: Arc<Mutex<std::fs::File>>,
}

impl ValidatedAnnualProcessor {
    pub fn new(base_dir: PathBuf) -> Result<Self> {
        let registry_path = base_dir.join("ercot_schema_registry.json");
        let schema_registry = SchemaRegistry::load_from_file(&registry_path)?;
        println!("✅ Loaded schema registry with {} patterns", schema_registry.schemas.len());
        
        // Create type mismatch log file
        let log_path = base_dir.join("type_mismatch_log.txt");
        let log_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)?;
        
        // Write header
        let mut log = log_file.try_clone()?;
        writeln!(log, "\n{} === Type Mismatch Log Started ===", Local::now().format("%Y-%m-%d %H:%M:%S"))?;
        
        Ok(Self {
            base_dir,
            schema_registry,
            type_mismatch_log: Arc::new(Mutex::new(log_file)),
        })
    }
    
    fn log_type_mismatch(&self, file: &Path, error: &str) {
        if let Ok(mut log) = self.type_mismatch_log.lock() {
            let _ = writeln!(log, "[{}] File: {}", 
                Local::now().format("%H:%M:%S"), 
                file.display());
            let _ = writeln!(log, "  Error: {}", error);
            
            // Try to extract detailed column info from error
            if error.contains("dtype") {
                if let Some(col_info) = Self::extract_column_info(error) {
                    let _ = writeln!(log, "  Column: {}", col_info.column);
                    let _ = writeln!(log, "  Expected Type: {}", col_info.expected_type);
                    let _ = writeln!(log, "  Actual Value: {}", col_info.actual_value);
                    let _ = writeln!(log, "  Suggested Fix: Add '{}' to Float64 overrides", col_info.column);
                }
            }
            let _ = writeln!(log, "");
            let _ = log.flush();
        }
    }
    
    fn extract_column_info(error: &str) -> Option<ColumnMismatchInfo> {
        // Parse error like: "unable to vstack, dtypes for column "RRSFFR" don't match: `i64` and `f64`"
        // Or: "Could not parse `"AEEC"` as dtype `f64` at column 'SettlementPointPrice'"
        
        if error.contains("don't match") {
            // Format: dtypes for column "X" don't match: `type1` and `type2`
            let parts: Vec<&str> = error.split("\"").collect();
            if parts.len() > 1 {
                let column = parts[1].to_string();
                let types_part = error.split("match:").nth(1)?;
                let types: Vec<&str> = types_part.split("and").collect();
                if types.len() == 2 {
                    return Some(ColumnMismatchInfo {
                        column,
                        expected_type: types[1].trim().trim_matches('`').to_string(),
                        actual_value: types[0].trim().trim_matches('`').to_string(),
                    });
                }
            }
        } else if error.contains("Could not parse") {
            // Format: Could not parse `"VALUE"` as dtype `type` at column 'COLUMN'
            let value = error.split("`\"").nth(1)?.split("\"").next()?.to_string();
            let dtype = error.split("dtype `").nth(1)?.split("`").next()?.to_string();
            let column = error.split("column '").nth(1)?.split("'").next()?.to_string();
            
            return Some(ColumnMismatchInfo {
                column,
                expected_type: dtype,
                actual_value: value,
            });
        }
        
        None
    }
    
    pub fn process_all_with_validation(&self) -> Result<()> {
        println!("🚀 Enhanced Annual Processor with Schema Validation");
        println!("{}", "=".repeat(80));
        
        // Create output directories
        let rollup_dir = self.base_dir.join("rollup_files");
        std::fs::create_dir_all(&rollup_dir)?;
        
        // Process each data type using schema registry for type safety
        self.process_cop_files(&rollup_dir)?;
        self.process_sced_files(&rollup_dir)?;
        self.process_dam_files(&rollup_dir)?;
        
        println!("\n✅ All processing complete with schema validation!");
        Ok(())
    }
    
    fn extract_year_from_60day_filename(file: &Path) -> Option<i32> {
        let filename = file.file_name()?.to_str()?;
        
        // Format 1: 60d_COP_Adjustment_Period_Snapshot-DD-MMM-YY.csv (after Dec 13, 2022)
        // or other 60d files like 60d_SCED_Gen_Resource_Data-DD-MMM-YY.csv
        if let Some(idx) = filename.rfind('-') {
            let year_str = &filename[idx+1..].replace(".csv", "");
            if year_str.len() == 2 {
                // Convert 2-digit year to 4-digit
                let year_num: i32 = year_str.parse().ok()?;
                let full_year = if year_num >= 50 {
                    1900 + year_num
                } else {
                    2000 + year_num
                };
                return Some(full_year);
            }
        }
        
        // Format 2: CompleteCOP_MMDDYYYY.csv (before Dec 13, 2022)
        if filename.starts_with("CompleteCOP_") && filename.ends_with(".csv") {
            // Extract MMDDYYYY part
            let date_part = filename.replace("CompleteCOP_", "").replace(".csv", "");
            if date_part.len() == 8 {
                // Last 4 digits are the year
                let year_str = &date_part[4..8];
                if let Ok(year) = year_str.parse::<i32>() {
                    return Some(year);
                }
            }
        }
        
        None
    }
    
    fn process_cop_files(&self, output_dir: &Path) -> Result<()> {
        println!("\n📊 Processing 60-Day COP Adjustment Period Snapshot files...");
        
        let cop_dir = self.base_dir.join("60-Day_COP_Adjustment_Period_Snapshot/csv");
        if !cop_dir.exists() {
            println!("  ⚠️ COP directory not found");
            return Ok(());
        }
        
        let cop_output = output_dir.join("COP_60day");
        std::fs::create_dir_all(&cop_output)?;
        
        let pattern = cop_dir.join("*.csv");
        let files: Vec<PathBuf> = glob::glob(pattern.to_str().unwrap())?
            .filter_map(Result::ok)
            .collect();
            
        println!("  Found {} COP files total", files.len());
        
        // Group files by year
        let mut files_by_year: BTreeMap<i32, Vec<PathBuf>> = BTreeMap::new();
        for file in files {
            if let Some(year) = Self::extract_year_from_60day_filename(&file) {
                files_by_year.entry(year).or_default().push(file);
            }
        }
        
        let mut total_errors = 0;
        let mut total_rows = 0;
        
        for (year, year_files) in files_by_year {
            println!("  Processing year {}: {} files", year, year_files.len());
            
            let pb = ProgressBar::new(year_files.len() as u64);
            pb.set_style(ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
                .unwrap());
            
            let batch_size = 50;
            let mut all_dfs = Vec::new();
            let year_errors = Arc::new(AtomicUsize::new(0));
            
            for batch in year_files.chunks(batch_size) {
                let errors_clone = Arc::clone(&year_errors);
                let batch_dfs: Vec<DataFrame> = batch
                    .par_iter()
                    .filter_map(|file| {
                        pb.inc(1);
                        match read_csv_with_schema_registry(file, &self.schema_registry) {
                            Ok(df) => Some(df),
                            Err(e) => {
                                errors_clone.fetch_add(1, Ordering::Relaxed);
                                let error_str = format!("{}", e);
                                eprintln!("    Error reading {}: {}", file.file_name().unwrap().to_str().unwrap(), &error_str);
                                self.log_type_mismatch(file, &error_str);
                                None
                            }
                        }
                    })
                    .collect();
                
                all_dfs.extend(batch_dfs);
            }
            
            let year_errors = year_errors.load(Ordering::Relaxed);
            
            pb.finish_with_message("done");
            
            if !all_dfs.is_empty() {
                // Combine all dataframes for the year
                let combined_df = match self.combine_dataframes(all_dfs) {
                    Ok(df) => df,
                    Err(e) => {
                        let error_str = format!("Year {} combine error: {}", year, e);
                        eprintln!("    ❌ {}", error_str);
                        self.log_type_mismatch(&cop_output.join(format!("{}.parquet", year)), &error_str);
                        continue;
                    }
                };
                let output_file = cop_output.join(format!("{}.parquet", year));
                
                // Write to parquet
                let mut file = std::fs::File::create(&output_file)?;
                ParquetWriter::new(&mut file).finish(&mut combined_df.clone())?;
                
                let rows = combined_df.height();
                total_rows += rows;
                println!("    ✅ Saved {} rows to {}.parquet", rows, year);
            }
            
            if year_errors > 0 {
                println!("    ⚠️  {} files had errors", year_errors);
                total_errors += year_errors;
            }
        }
        
        if total_errors == 0 {
            println!("  🎉 Complete: {} total rows, NO TYPE ERRORS!", total_rows);
        } else {
            println!("  ⚠️  Complete: {} rows saved, {} files had errors", total_rows, total_errors);
        }
        
        Ok(())
    }
    
    fn process_sced_files(&self, output_dir: &Path) -> Result<()> {
        println!("\n📊 Processing 60-Day SCED Gen Resource Data files...");
        
        let sced_dir = self.base_dir.join("60-Day_SCED_Disclosure_Reports/csv");
        if !sced_dir.exists() {
            println!("  ⚠️ SCED directory not found");
            return Ok(());
        }
        
        let sced_output = output_dir.join("SCED_Gen_Resources");
        std::fs::create_dir_all(&sced_output)?;
        
        // Only process Gen_Resource_Data files
        let pattern = sced_dir.join("60d_SCED_Gen_Resource_Data*.csv");
        let files: Vec<PathBuf> = glob::glob(pattern.to_str().unwrap())?
            .filter_map(Result::ok)
            .collect();
            
        println!("  Found {} SCED files total", files.len());
        
        // Group files by year
        let mut files_by_year: BTreeMap<i32, Vec<PathBuf>> = BTreeMap::new();
        for file in files {
            if let Some(year) = Self::extract_year_from_60day_filename(&file) {
                files_by_year.entry(year).or_default().push(file);
            }
        }
        
        let mut total_errors = 0;
        let mut total_rows = 0;
        
        for (year, year_files) in files_by_year {
            println!("  Processing year {}: {} files", year, year_files.len());
            
            let pb = ProgressBar::new(year_files.len() as u64);
            pb.set_style(ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
                .unwrap());
            
            let batch_size = 100;
            let mut all_dfs = Vec::new();
            let year_errors = Arc::new(AtomicUsize::new(0));
            
            for batch in year_files.chunks(batch_size) {
                let errors_clone = Arc::clone(&year_errors);
                let batch_dfs: Vec<DataFrame> = batch
                    .par_iter()
                    .filter_map(|file| {
                        pb.inc(1);
                        match read_csv_with_schema_registry(file, &self.schema_registry) {
                            Ok(df) => Some(df),
                            Err(e) => {
                                errors_clone.fetch_add(1, Ordering::Relaxed);
                                let error_str = format!("{}", e);
                                eprintln!("    Error reading {}: {}", file.file_name().unwrap().to_str().unwrap(), &error_str);
                                self.log_type_mismatch(file, &error_str);
                                None
                            }
                        }
                    })
                    .collect();
                
                all_dfs.extend(batch_dfs);
            }
            
            let year_errors = year_errors.load(Ordering::Relaxed);
            
            pb.finish_with_message("done");
            
            if !all_dfs.is_empty() {
                // Combine all dataframes for the year
                let combined_df = match self.combine_dataframes(all_dfs) {
                    Ok(df) => df,
                    Err(e) => {
                        let error_str = format!("Year {} combine error: {}", year, e);
                        eprintln!("    ❌ {}", error_str);
                        let log_file = PathBuf::from(format!("{}_combine_error.log", year));
                        self.log_type_mismatch(&log_file, &error_str);
                        continue;
                    }
                };
                let output_file = sced_output.join(format!("{}.parquet", year));
                
                // Write to parquet
                let mut file = std::fs::File::create(&output_file)?;
                ParquetWriter::new(&mut file).finish(&mut combined_df.clone())?;
                
                let rows = combined_df.height();
                total_rows += rows;
                println!("    ✅ Saved {} rows to {}.parquet", rows, year);
            }
            
            if year_errors > 0 {
                println!("    ⚠️  {} files had errors", year_errors);
                total_errors += year_errors;
            }
        }
        
        if total_errors == 0 {
            println!("  🎉 Complete: {} total rows, NO TYPE ERRORS!", total_rows);
        } else {
            println!("  ⚠️  Complete: {} rows saved, {} files had errors", total_rows, total_errors);
        }
        
        Ok(())
    }
    
    fn process_dam_files(&self, output_dir: &Path) -> Result<()> {
        println!("\n📊 Processing 60-Day DAM Gen Resource Data files...");
        
        let dam_dir = self.base_dir.join("60-Day_DAM_Disclosure_Reports/csv");
        if !dam_dir.exists() {
            println!("  ⚠️ DAM directory not found");
            return Ok(());
        }
        
        let dam_output = output_dir.join("DAM_Gen_Resources");
        std::fs::create_dir_all(&dam_output)?;
        
        // Only process Gen_Resource_Data files for now
        let pattern = dam_dir.join("60d_DAM_Gen_Resource_Data*.csv");
        let files: Vec<PathBuf> = glob::glob(pattern.to_str().unwrap())?
            .filter_map(Result::ok)
            .collect();
            
        println!("  Found {} DAM files total", files.len());
        
        // Group files by year
        let mut files_by_year: BTreeMap<i32, Vec<PathBuf>> = BTreeMap::new();
        for file in files {
            if let Some(year) = Self::extract_year_from_60day_filename(&file) {
                files_by_year.entry(year).or_default().push(file);
            }
        }
        
        let mut total_errors = 0;
        let mut total_rows = 0;
        
        for (year, year_files) in files_by_year {
            println!("  Processing year {}: {} files", year, year_files.len());
            
            let pb = ProgressBar::new(year_files.len() as u64);
            pb.set_style(ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
                .unwrap());
            
            let batch_size = 100;
            let mut all_dfs = Vec::new();
            let year_errors = Arc::new(AtomicUsize::new(0));
            
            for batch in year_files.chunks(batch_size) {
                let errors_clone = Arc::clone(&year_errors);
                let batch_dfs: Vec<DataFrame> = batch
                    .par_iter()
                    .filter_map(|file| {
                        pb.inc(1);
                        match read_csv_with_schema_registry(file, &self.schema_registry) {
                            Ok(df) => Some(df),
                            Err(e) => {
                                errors_clone.fetch_add(1, Ordering::Relaxed);
                                let error_str = format!("{}", e);
                                eprintln!("    Error reading {}: {}", file.file_name().unwrap().to_str().unwrap(), &error_str);
                                self.log_type_mismatch(file, &error_str);
                                None
                            }
                        }
                    })
                    .collect();
                
                all_dfs.extend(batch_dfs);
            }
            
            let year_errors = year_errors.load(Ordering::Relaxed);
            
            pb.finish_with_message("done");
            
            if !all_dfs.is_empty() {
                // Combine all dataframes for the year
                let combined_df = match self.combine_dataframes(all_dfs) {
                    Ok(df) => df,
                    Err(e) => {
                        let error_str = format!("Year {} combine error: {}", year, e);
                        eprintln!("    ❌ {}", error_str);
                        let log_file = PathBuf::from(format!("{}_combine_error.log", year));
                        self.log_type_mismatch(&log_file, &error_str);
                        continue;
                    }
                };
                let output_file = dam_output.join(format!("{}.parquet", year));
                
                // Write to parquet
                let mut file = std::fs::File::create(&output_file)?;
                ParquetWriter::new(&mut file).finish(&mut combined_df.clone())?;
                
                let rows = combined_df.height();
                total_rows += rows;
                println!("    ✅ Saved {} rows to {}.parquet", rows, year);
            }
            
            if year_errors > 0 {
                println!("    ⚠️  {} files had errors", year_errors);
                total_errors += year_errors;
            }
        }
        
        if total_errors == 0 {
            println!("  🎉 Complete: {} total rows, NO TYPE ERRORS!", total_rows);
        } else {
            println!("  ⚠️  Complete: {} rows saved, {} files had errors", total_rows, total_errors);
        }
        
        Ok(())
    }
    
    fn combine_dataframes(&self, dfs: Vec<DataFrame>) -> Result<DataFrame> {
        if dfs.is_empty() {
            return Err(anyhow::anyhow!("No dataframes to combine"));
        }
        
        // First, find all unique columns across all dataframes and their types
        let mut all_columns = std::collections::HashSet::new();
        let mut column_types: std::collections::HashMap<String, DataType> = std::collections::HashMap::new();
        
        for df in &dfs {
            for col in df.get_column_names() {
                all_columns.insert(col.to_string());
                // Store the first non-null type we see for each column
                if !column_types.contains_key(col) {
                    if let Ok(series) = df.column(col) {
                        column_types.insert(col.to_string(), series.dtype().clone());
                    }
                }
            }
        }
        
        // Convert to sorted vec for consistent ordering
        let mut all_columns: Vec<String> = all_columns.into_iter().collect();
        all_columns.sort();
        
        // Normalize all dataframes to have the same columns
        let mut normalized_dfs = Vec::new();
        for df in dfs {
            let mut df = df.lazy();
            
            // Add missing columns as nulls with proper type
            let current_df = df.clone().collect()?;
            for col_name in &all_columns {
                if !current_df.get_column_names().contains(&col_name.as_str()) {
                    // Get the type from our column_types map
                    let target_dtype = column_types.get(col_name)
                        .cloned()
                        .unwrap_or(DataType::Utf8);
                    
                    // Create null column with the correct type
                    let null_col = match target_dtype {
                        DataType::Float64 => lit(NULL).cast(DataType::Float64),
                        DataType::Int64 => lit(NULL).cast(DataType::Int64),
                        DataType::Utf8 => lit(NULL).cast(DataType::Utf8),
                        DataType::Date => lit(NULL).cast(DataType::Date),
                        DataType::Datetime(tu, tz) => lit(NULL).cast(DataType::Datetime(tu, tz)),
                        _ => lit(NULL).cast(DataType::Utf8),
                    };
                    
                    df = df.with_column(null_col.alias(col_name));
                }
            }
            
            // Select columns in consistent order
            let cols: Vec<Expr> = all_columns.iter().map(|c| col(c)).collect();
            df = df.select(&cols);
            
            normalized_dfs.push(df.collect()?);
        }
        
        // Now combine all normalized dataframes
        let mut combined = normalized_dfs[0].clone();
        for df in normalized_dfs.into_iter().skip(1) {
            combined = combined.vstack(&df)?;
        }
        
        Ok(combined)
    }
}