use anyhow::Result;
use chrono::{DateTime, Datelike, Duration, NaiveDate, Utc};
use polars::prelude::*;
use rayon::prelude::*;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use indicatif::{ProgressBar, ProgressStyle};

#[derive(Debug, Clone)]
pub struct DataGap {
    start_date: NaiveDate,
    end_date: NaiveDate,
    missing_days: i64,
}

#[derive(Debug)]
pub struct ProcessingStats {
    total_files: usize,
    processed_files: usize,
    #[allow(dead_code)]
    failed_files: usize,
    total_rows: usize,
    years_covered: Vec<i32>,
    gaps: Vec<DataGap>,
}

pub struct EnhancedAnnualProcessor {
    base_dir: PathBuf,
    output_dir: PathBuf,
    processing_stats: Arc<Mutex<HashMap<String, ProcessingStats>>>,
}

impl EnhancedAnnualProcessor {
    pub fn new(base_dir: PathBuf) -> Self {
        let output_dir = base_dir.join("rollup_files");
        Self {
            base_dir,
            output_dir,
            processing_stats: Arc::new(Mutex::new(HashMap::new())),
        }
    }
    
    pub fn process_all_data(&self) -> Result<()> {
        let start_time = std::time::Instant::now();
        
        println!("🚀 Enhanced Annual Parquet Processor");
        println!("📁 Base directory: {}", self.base_dir.display());
        println!("📂 Output directory: {}", self.output_dir.display());
        println!("{}", "=".repeat(80));
        
        // Create output directory
        fs::create_dir_all(&self.output_dir)?;
        
        // Process each data type - RT prices last since they're the most voluminous
        let processors = vec![
            ("DA_prices", "DAM_Settlement_Point_Prices", ProcessorType::DayAheadPrice),
            ("AS_prices", "DAM_Clearing_Prices_for_Capacity", ProcessorType::AncillaryService),
            ("DAM_Gen_Resources", "60-Day_DAM_Disclosure_Reports", ProcessorType::DAMGenResource),
            ("SCED_Gen_Resources", "60-Day_SCED_Disclosure_Reports", ProcessorType::SCEDGenResource),
            ("COP_Snapshots", "60-Day_COP_Adjustment_Period_Snapshot", ProcessorType::COPSnapshot),
            ("RT_prices", "Settlement_Point_Prices_at_Resource_Nodes,_Hubs_and_Load_Zones", ProcessorType::RealTimePrice),
        ];
        
        for (output_name, source_dir, processor_type) in processors {
            println!("\n📊 Processing: {}", output_name);
            println!("{}", "-".repeat(60));
            
            let source_path = self.base_dir.join(source_dir);
            if !source_path.exists() {
                println!("⚠️  Source directory not found: {}", source_path.display());
                continue;
            }
            
            let output_path = self.output_dir.join(output_name);
            fs::create_dir_all(&output_path)?;
            
            match processor_type {
                ProcessorType::RealTimePrice => self.process_rt_prices(&source_path, &output_path)?,
                ProcessorType::DayAheadPrice => self.process_da_prices(&source_path, &output_path)?,
                ProcessorType::AncillaryService => self.process_as_prices(&source_path, &output_path)?,
                ProcessorType::DAMGenResource => self.process_dam_gen_resources(&source_path, &output_path)?,
                ProcessorType::SCEDGenResource => self.process_sced_gen_resources(&source_path, &output_path)?,
                ProcessorType::COPSnapshot => self.process_cop_snapshots(&source_path, &output_path)?,
            }
        }
        
        // Generate overall status report
        self.generate_status_report()?;
        
        let elapsed = start_time.elapsed();
        println!("\n✅ Processing complete!");
        println!("⏱️  Total time: {:?}", elapsed);
        
        Ok(())
    }
    
    fn process_rt_prices(&self, source_dir: &Path, output_dir: &Path) -> Result<()> {
        let csv_dir = source_dir.join("csv");
        if !csv_dir.exists() {
            let csv_dir = source_dir; // Try without csv subdirectory
            if !csv_dir.exists() {
                println!("⚠️  No CSV directory found");
                return Ok(());
            }
        }
        
        // Get all CSV files
        let pattern = csv_dir.join("*.csv");
        let files = glob::glob(pattern.to_str().unwrap())?
            .filter_map(Result::ok)
            .collect::<Vec<_>>();
        
        println!("  Found {} RT price files", files.len());
        
        // Group by year and process
        let mut files_by_year: BTreeMap<i32, Vec<PathBuf>> = BTreeMap::new();
        for file in files {
            if let Some(year) = extract_year_from_filename(&file) {
                files_by_year.entry(year).or_default().push(file);
            }
        }
        
        for (year, year_files) in files_by_year {
            println!("  Processing year {}: {} files", year, year_files.len());
            
            let pb = ProgressBar::new(year_files.len() as u64);
            pb.set_style(ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
                .unwrap());
            
            // Process in parallel batches
            let batch_size = 100;
            let mut all_dfs = Vec::new();
            
            for batch in year_files.chunks(batch_size) {
                let batch_dfs: Vec<DataFrame> = batch
                    .par_iter()
                    .filter_map(|file| {
                        pb.inc(1);
                        match self.read_rt_price_file(file) {
                            Ok(df) => Some(df),
                            Err(e) => {
                                eprintln!("    Error reading {}: {}", file.display(), e);
                                None
                            }
                        }
                    })
                    .collect();
                
                all_dfs.extend(batch_dfs);
            }
            
            pb.finish_with_message("done");
            
            if !all_dfs.is_empty() {
                // Combine all dataframes
                let combined_df = self.combine_dataframes(all_dfs)?;
                
                // Check for gaps
                let gaps = self.detect_gaps(&combined_df, "datetime")?;
                
                // Save to parquet
                let output_file = output_dir.join(format!("{}.parquet", year));
                let mut file = std::fs::File::create(&output_file)?;
                ParquetWriter::new(&mut file).finish(&mut combined_df.clone())?;
                
                println!("    ✅ Saved {} rows to {}", combined_df.height(), output_file.display());
                
                // Save gaps report
                self.save_gaps_report(output_dir, year, &gaps)?;
                
                // Update stats
                self.update_stats("RT_prices", year, year_files.len(), combined_df.height(), gaps);
            }
        }
        
        // Save schema
        self.save_schema(output_dir, "RT_prices", r#"{
  "description": "Real-time settlement point prices at 15-minute intervals",
  "columns": {
    "datetime": {"type": "Datetime", "description": "Timestamp of the price interval"},
    "DeliveryDate": {"type": "Date", "description": "Operating day"},
    "DeliveryHour": {"type": "Float64", "description": "Hour of the day (1-24)"},
    "DeliveryInterval": {"type": "Float64", "description": "15-minute interval (1-4)"},
    "SettlementPointName": {"type": "Utf8", "description": "Settlement point name"},
    "SettlementPointType": {"type": "Utf8", "description": "Type: RN, HB, LZ"},
    "SettlementPointPrice": {"type": "Float64", "description": "Price in $/MWh"},
    "DSTFlag": {"type": "Utf8", "description": "Daylight Saving Time flag"}
  }
}"#)?;
        
        Ok(())
    }
    
    fn read_rt_price_file(&self, file: &Path) -> Result<DataFrame> {
        // Read CSV with Float64 for price column, keep dates as strings
        let schema_overrides = Schema::from_iter([
            Field::new("SettlementPointPrice", DataType::Float64),
            Field::new("DeliveryDate", DataType::Utf8),
        ]);
        
        let df = CsvReader::from_path(file)?
            .has_header(true)
            .with_try_parse_dates(false)  // Don't auto-parse dates
            .with_dtypes(Some(Arc::new(schema_overrides)))
            .finish()?;
        
        // Normalize the dataframe to handle schema evolution (DST flag added in 2011)
        let mut df = crate::schema_normalizer::normalize_rt_prices(df)?;
        
        // Parse dates and create proper datetime column
        let delivery_dates = df.column("DeliveryDate")?;
        let delivery_hours = df.column("DeliveryHour")?.cast(&DataType::Int32)?;
        let delivery_intervals = df.column("DeliveryInterval")?.cast(&DataType::Int32)?;
        
        let hours = delivery_hours.i32()?
            .apply(|v| if v.unwrap_or(0) == 24 { Some(0) } else { v.map(|x| x - 1) });
        
        let minutes = delivery_intervals.i32()?
            .apply(|i| i.map(|v| (v - 1) * 15));
        
        let mut datetimes = Vec::new();
        for i in 0..df.height() {
            if let Some(date_str) = delivery_dates.utf8()?.get(i) {
                if let Ok(date) = NaiveDate::parse_from_str(date_str, "%m/%d/%Y") {
                    let hour = hours.get(i).unwrap_or(0) as u32;
                    let minute = minutes.get(i).unwrap_or(0) as u32;
                    let mut datetime = date.and_hms_opt(hour, minute, 0).unwrap();
                    
                    // Handle hour 24
                    if delivery_hours.i32()?.get(i) == Some(24) {
                        datetime += Duration::days(1);
                    }
                    
                    datetimes.push(Some(datetime.and_utc().timestamp_millis()));
                } else {
                    datetimes.push(None);
                }
            } else {
                datetimes.push(None);
            }
        }
        
        let datetime_series = Series::new("datetime", datetimes);
        df.with_column(datetime_series)?;
        
        // Ensure price is Float64
        df.with_column(df.column("SettlementPointPrice")?.cast(&DataType::Float64)?)?;
        
        Ok(df)
    }
    
    fn process_da_prices(&self, source_dir: &Path, output_dir: &Path) -> Result<()> {
        let csv_dir = source_dir.join("csv");
        let csv_dir = if csv_dir.exists() { csv_dir } else { source_dir.to_path_buf() };
        
        // Pattern: cdr.00012331.*.DAMSPNP4190.csv
        let pattern = csv_dir.join("*.csv");
        let files = glob::glob(pattern.to_str().unwrap())?
            .filter_map(Result::ok)
            .filter(|p| {
                let name = p.file_name().unwrap().to_str().unwrap();
                name.contains("DAMSPNP4190") || name.contains("00012331")
            })
            .collect::<Vec<_>>();
        
        println!("  Found {} DA price files", files.len());
        
        // Group by year and process
        let mut files_by_year: BTreeMap<i32, Vec<PathBuf>> = BTreeMap::new();
        for file in files {
            if let Some(year) = extract_year_from_filename(&file) {
                files_by_year.entry(year).or_default().push(file);
            }
        }
        
        for (year, year_files) in files_by_year {
            println!("  Processing year {}: {} files", year, year_files.len());
            
            let pb = ProgressBar::new(year_files.len() as u64);
            pb.set_style(ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
                .unwrap());
            
            let batch_size = 100;
            let mut all_dfs = Vec::new();
            
            for batch in year_files.chunks(batch_size) {
                let batch_dfs: Vec<DataFrame> = batch
                    .par_iter()
                    .filter_map(|file| {
                        pb.inc(1);
                        match self.read_da_price_file(file) {
                            Ok(df) => Some(df),
                            Err(e) => {
                                eprintln!("    Error reading {}: {}", file.display(), e);
                                None
                            }
                        }
                    })
                    .collect();
                
                all_dfs.extend(batch_dfs);
            }
            
            pb.finish_with_message("done");
            
            if !all_dfs.is_empty() {
                let combined_df = self.combine_dataframes(all_dfs)?;
                let gaps = self.detect_gaps(&combined_df, "datetime")?;
                
                let output_file = output_dir.join(format!("{}.parquet", year));
                let mut file = std::fs::File::create(&output_file)?;
                ParquetWriter::new(&mut file).finish(&mut combined_df.clone())?;
                
                println!("    ✅ Saved {} rows to {}", combined_df.height(), output_file.display());
                
                self.save_gaps_report(output_dir, year, &gaps)?;
                self.update_stats("DA_prices", year, year_files.len(), combined_df.height(), gaps);
            }
        }
        
        self.save_schema(output_dir, "DA_prices", r#"{
  "description": "Day-ahead market settlement point prices (hourly)",
  "columns": {
    "datetime": {"type": "Datetime", "description": "Timestamp of the price hour"},
    "DeliveryDate": {"type": "Date", "description": "Operating day"},
    "HourEnding": {"type": "Utf8", "description": "Hour ending time"},
    "SettlementPoint": {"type": "Utf8", "description": "Settlement point name"},
    "SettlementPointPrice": {"type": "Float64", "description": "Price in $/MWh"},
    "DSTFlag": {"type": "Utf8", "description": "Daylight Saving Time flag"}
  }
}"#)?;
        
        Ok(())
    }
    
    fn read_da_price_file(&self, file: &Path) -> Result<DataFrame> {
        // Read CSV with Float64 for price column
        let schema_overrides = Schema::from_iter([
            Field::new("SettlementPointPrice", DataType::Float64),
        ]);
        
        let df = CsvReader::from_path(file)?
            .has_header(true)
            .with_try_parse_dates(true)
            .with_dtypes(Some(Arc::new(schema_overrides)))
            .finish()?;
        
        // Normalize the dataframe to handle schema evolution (DST flag added in 2011)
        let df = crate::schema_normalizer::normalize_dam_prices(df)?;
        
        // Parse HourEnding and create datetime, ensure price is Float64
        let df = df.lazy()
            .with_column(
                col("DeliveryDate").cast(DataType::Date)
            )
            .with_column(
                col("SettlementPointPrice").cast(DataType::Float64)  // Critical: Force Float64
            )
            .with_column(
                col("HourEnding").cast(DataType::Float64).alias("hour")
            )
            .with_column(
                (col("DeliveryDate").cast(DataType::Datetime(TimeUnit::Milliseconds, None)) +
                 duration_hours(col("hour") - lit(1)))
                .alias("datetime")
            )
            .collect()?;
        
        Ok(df)
    }
    
    fn process_as_prices(&self, source_dir: &Path, output_dir: &Path) -> Result<()> {
        let csv_dir = source_dir.join("csv");
        let csv_dir = if csv_dir.exists() { csv_dir } else { source_dir.to_path_buf() };
        
        // Pattern: cdr.00012329.*.DAMCPCNP4188.csv
        let pattern = csv_dir.join("*.csv");
        let files = glob::glob(pattern.to_str().unwrap())?
            .filter_map(Result::ok)
            .filter(|p| {
                let name = p.file_name().unwrap().to_str().unwrap();
                name.contains("DAMCPCNP4188") || name.contains("00012329")
            })
            .collect::<Vec<_>>();
        
        println!("  Found {} AS price files", files.len());
        
        let mut files_by_year: BTreeMap<i32, Vec<PathBuf>> = BTreeMap::new();
        for file in files {
            if let Some(year) = extract_year_from_filename(&file) {
                files_by_year.entry(year).or_default().push(file);
            }
        }
        
        for (year, year_files) in files_by_year {
            println!("  Processing year {}: {} files", year, year_files.len());
            
            let pb = ProgressBar::new(year_files.len() as u64);
            pb.set_style(ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
                .unwrap());
            
            let batch_size = 100;
            let mut all_dfs = Vec::new();
            
            for batch in year_files.chunks(batch_size) {
                let batch_dfs: Vec<DataFrame> = batch
                    .par_iter()
                    .filter_map(|file| {
                        pb.inc(1);
                        match self.read_as_price_file(file) {
                            Ok(df) => Some(df),
                            Err(e) => {
                                eprintln!("    Error reading {}: {}", file.display(), e);
                                None
                            }
                        }
                    })
                    .collect();
                
                all_dfs.extend(batch_dfs);
            }
            
            pb.finish_with_message("done");
            
            if !all_dfs.is_empty() {
                let combined_df = self.combine_dataframes(all_dfs)?;
                let gaps = self.detect_gaps(&combined_df, "datetime")?;
                
                let output_file = output_dir.join(format!("{}.parquet", year));
                let mut file = std::fs::File::create(&output_file)?;
                ParquetWriter::new(&mut file).finish(&mut combined_df.clone())?;
                
                println!("    ✅ Saved {} rows to {}", combined_df.height(), output_file.display());
                
                self.save_gaps_report(output_dir, year, &gaps)?;
                self.update_stats("AS_prices", year, year_files.len(), combined_df.height(), gaps);
            }
        }
        
        self.save_schema(output_dir, "AS_prices", r#"{
  "description": "Day-ahead ancillary services clearing prices",
  "columns": {
    "datetime": {"type": "Datetime", "description": "Timestamp of the price hour"},
    "DeliveryDate": {"type": "Date", "description": "Operating day"},
    "HourEnding": {"type": "Utf8", "description": "Hour ending time"},
    "AncillaryType": {"type": "Utf8", "description": "Type: REGUP, REGDN, RRS, ECRS, NSPIN"},
    "MCPC": {"type": "Float64", "description": "Market Clearing Price for Capacity in $/MW"},
    "DSTFlag": {"type": "Utf8", "description": "Daylight Saving Time flag"}
  }
}"#)?;
        
        Ok(())
    }
    
    fn read_as_price_file(&self, file: &Path) -> Result<DataFrame> {
        // Read CSV with Float64 for price column
        let schema_overrides = Schema::from_iter([
            Field::new("MCPC", DataType::Float64),
        ]);
        
        let df = CsvReader::from_path(file)?
            .has_header(true)
            .with_try_parse_dates(true)
            .with_dtypes(Some(Arc::new(schema_overrides)))
            .finish()?;
        
        // Normalize the dataframe to handle schema evolution (DST flag added in 2011)
        let df = crate::schema_normalizer::normalize_as_prices(df)?;
        
        let df = df.lazy()
            .with_column(
                col("DeliveryDate").cast(DataType::Date)
            )
            .with_column(
                col("MCPC").cast(DataType::Float64)  // Critical: Force Float64
            )
            .with_column(
                col("HourEnding").cast(DataType::Float64).alias("hour")
            )
            .with_column(
                (col("DeliveryDate").cast(DataType::Datetime(TimeUnit::Milliseconds, None)) +
                 duration_hours(col("hour") - lit(1)))
                .alias("datetime")
            )
            .collect()?;
        
        Ok(df)
    }
    
    fn process_dam_gen_resources(&self, source_dir: &Path, output_dir: &Path) -> Result<()> {
        let csv_dir = source_dir.join("csv");
        let csv_dir = if csv_dir.exists() { csv_dir } else { source_dir.to_path_buf() };
        
        // Pattern: 60d_DAM_Gen_Resource_Data-DD-MMM-YY.csv
        let pattern = csv_dir.join("60d_DAM_Gen_Resource_Data-*.csv");
        let files = glob::glob(pattern.to_str().unwrap())?
            .filter_map(Result::ok)
            .collect::<Vec<_>>();
        
        println!("  Found {} DAM Gen Resource files", files.len());
        
        // Group by year
        let mut files_by_year: BTreeMap<i32, Vec<PathBuf>> = BTreeMap::new();
        for file in files {
            if let Some(year) = extract_year_from_60day_filename(&file) {
                files_by_year.entry(year).or_default().push(file);
            }
        }
        
        for (year, year_files) in files_by_year {
            println!("  Processing year {}: {} files", year, year_files.len());
            
            let pb = ProgressBar::new(year_files.len() as u64);
            pb.set_style(ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
                .unwrap());
            
            let mut all_dfs = Vec::new();
            
            for file in &year_files {
                pb.inc(1);
                match self.read_dam_gen_file(file) {
                    Ok(df) => all_dfs.push(df),
                    Err(e) => eprintln!("    Error reading {}: {}", file.display(), e),
                }
            }
            
            pb.finish_with_message("done");
            
            if !all_dfs.is_empty() {
                let combined_df = self.combine_dataframes(all_dfs)?;
                let gaps = self.detect_gaps(&combined_df, "datetime")?;
                
                let output_file = output_dir.join(format!("{}.parquet", year));
                let mut file = std::fs::File::create(&output_file)?;
                ParquetWriter::new(&mut file).finish(&mut combined_df.clone())?;
                
                println!("    ✅ Saved {} rows to {}", combined_df.height(), output_file.display());
                
                self.save_gaps_report(output_dir, year, &gaps)?;
                self.update_stats("DAM_Gen_Resources", year, year_files.len(), combined_df.height(), gaps);
            }
        }
        
        self.save_schema(output_dir, "DAM_Gen_Resources", r#"{
  "description": "60-day DAM generation resource data including BESS",
  "columns": {
    "datetime": {"type": "Datetime", "description": "Timestamp"},
    "DeliveryDate": {"type": "Date", "description": "Operating day"},
    "HourEnding": {"type": "Utf8", "description": "Hour ending"},
    "ResourceName": {"type": "Utf8", "description": "Resource name"},
    "ResourceType": {"type": "Utf8", "description": "Type (PWRSTR for BESS)"},
    "AwardedQuantity": {"type": "Float64", "description": "DAM energy award MW"},
    "SettlementPointName": {"type": "Utf8", "description": "Settlement point"},
    "EnergySettlementPointPrice": {"type": "Float64", "description": "Price $/MWh"}
  }
}"#)?;
        
        Ok(())
    }
    
    fn read_dam_gen_file(&self, file: &Path) -> Result<DataFrame> {
        // Force critical numeric columns to be Float64
        let mut schema_overrides = Schema::new();
        schema_overrides.with_column("LSL".into(), DataType::Float64);
        schema_overrides.with_column("HSL".into(), DataType::Float64);
        schema_overrides.with_column("Awarded Quantity".into(), DataType::Float64);
        schema_overrides.with_column("Energy Settlement Point Price".into(), DataType::Float64);
        schema_overrides.with_column("RegUp Awarded".into(), DataType::Float64);
        schema_overrides.with_column("RegDown Awarded".into(), DataType::Float64);
        schema_overrides.with_column("RRS Awarded".into(), DataType::Float64);
        schema_overrides.with_column("RRSPFR Awarded".into(), DataType::Float64);
        schema_overrides.with_column("RRSFFR Awarded".into(), DataType::Float64);
        schema_overrides.with_column("RRSUFR Awarded".into(), DataType::Float64);
        schema_overrides.with_column("ECRS Awarded".into(), DataType::Float64);
        schema_overrides.with_column("ECRSSD Awarded".into(), DataType::Float64);
        schema_overrides.with_column("NonSpin Awarded".into(), DataType::Float64);
        
        // Add MCPC columns - treat as Utf8 to handle mixed string/float values
        schema_overrides.with_column("ECRS MCPC".into(), DataType::Utf8);
        schema_overrides.with_column("RegUp MCPC".into(), DataType::Float64);
        schema_overrides.with_column("RegDown MCPC".into(), DataType::Float64);
        schema_overrides.with_column("RRS MCPC".into(), DataType::Float64);
        schema_overrides.with_column("NonSpin MCPC".into(), DataType::Float64);
        
        let df = CsvReader::from_path(file)?
            .has_header(true)
            .with_dtypes(Some(Arc::new(schema_overrides)))
            .infer_schema(None)  // Don't infer - use our explicit schema
            .finish()?;
        
        // Get available columns
        let columns = df.get_column_names();
        let mut select_cols = vec![
            col("Delivery Date").alias("DeliveryDate"),
            col("Hour Ending").alias("HourEnding"),
        ];
        
        // Add columns with consistent schema - use default values for missing columns
        // This ensures all DataFrames have the same columns for vstack
        
        // Required fields
        if columns.contains(&"Resource Name") {
            select_cols.push(col("Resource Name").alias("ResourceName"));
        } else {
            select_cols.push(lit("").alias("ResourceName"));
        }
        
        if columns.contains(&"Resource Type") {
            select_cols.push(col("Resource Type").alias("ResourceType"));
        } else {
            select_cols.push(lit("").alias("ResourceType"));
        }
        
        if columns.contains(&"Settlement Point Name") {
            select_cols.push(col("Settlement Point Name").alias("SettlementPointName"));
        } else {
            select_cols.push(lit("").alias("SettlementPointName"));
        }
        
        if columns.contains(&"LSL") {
            select_cols.push(col("LSL").cast(DataType::Float64));
        } else {
            select_cols.push(lit(0.0).alias("LSL"));
        }
        
        if columns.contains(&"HSL") {
            select_cols.push(col("HSL").cast(DataType::Float64));
        } else {
            select_cols.push(lit(0.0).alias("HSL"));
        }
        
        if columns.contains(&"Awarded Quantity") {
            select_cols.push(col("Awarded Quantity").cast(DataType::Float64).alias("AwardedQuantity"));
        } else {
            select_cols.push(lit(0.0).alias("AwardedQuantity"));
        }
        
        if columns.contains(&"Energy Settlement Point Price") {
            select_cols.push(col("Energy Settlement Point Price").cast(DataType::Float64).alias("EnergySettlementPointPrice"));
        } else {
            select_cols.push(lit(0.0).alias("EnergySettlementPointPrice"));
        }
        
        // AS awards - normalize column names
        if columns.contains(&"RegUp Awarded") {
            select_cols.push(col("RegUp Awarded").cast(DataType::Float64).alias("RegUpAwarded"));
        } else {
            select_cols.push(lit(0.0).alias("RegUpAwarded"));
        }
        
        if columns.contains(&"RegDown Awarded") {
            select_cols.push(col("RegDown Awarded").cast(DataType::Float64).alias("RegDownAwarded"));
        } else {
            select_cols.push(lit(0.0).alias("RegDownAwarded"));
        }
        
        // Handle RRS schema evolution - old format had "RRS Awarded", new has separate columns
        if columns.contains(&"RRS Awarded") {
            // Old format - single RRS column, map to RRSAwarded
            select_cols.push(col("RRS Awarded").cast(DataType::Float64).alias("RRSAwarded"));
        } else {
            select_cols.push(lit(0.0).alias("RRSAwarded"));
        }
        
        // New RRS columns (added ~2022)
        if columns.contains(&"RRSPFR Awarded") {
            select_cols.push(col("RRSPFR Awarded").cast(DataType::Float64).alias("RRSPFRAwarded"));
        } else {
            select_cols.push(lit(0.0).alias("RRSPFRAwarded"));
        }
        
        if columns.contains(&"RRSFFR Awarded") {
            select_cols.push(col("RRSFFR Awarded").cast(DataType::Float64).alias("RRSFFRAwarded"));
        } else {
            select_cols.push(lit(0.0).alias("RRSFFRAwarded"));
        }
        
        if columns.contains(&"RRSUFR Awarded") {
            select_cols.push(col("RRSUFR Awarded").cast(DataType::Float64).alias("RRSUFRAwarded"));
        } else {
            select_cols.push(lit(0.0).alias("RRSUFRAwarded"));
        }
        
        // ECRS columns
        if columns.contains(&"ECRS Awarded") {
            select_cols.push(col("ECRS Awarded").cast(DataType::Float64).alias("ECRSAwarded"));
        } else {
            select_cols.push(lit(0.0).alias("ECRSAwarded"));
        }
        
        if columns.contains(&"ECRSSD Awarded") {
            select_cols.push(col("ECRSSD Awarded").cast(DataType::Float64).alias("ECRSSDAwarded"));
        } else {
            select_cols.push(lit(0.0).alias("ECRSSDAwarded"));
        }
        
        if columns.contains(&"NonSpin Awarded") {
            select_cols.push(col("NonSpin Awarded").cast(DataType::Float64).alias("NonSpinAwarded"));
        } else {
            select_cols.push(lit(0.0).alias("NonSpinAwarded"));
        }
        
        // Build dataframe with available columns
        let df = df.lazy()
            .select(select_cols)
            .with_column(
                col("DeliveryDate").cast(DataType::Date)
            )
            .with_column(
                col("HourEnding").cast(DataType::Float64).alias("hour")
            )
            .with_column(
                (col("DeliveryDate").cast(DataType::Datetime(TimeUnit::Milliseconds, None)) +
                 duration_hours(col("hour") - lit(1)))
                .alias("datetime")
            )
            .collect()?;
        
        Ok(df)
    }
    
    fn process_sced_gen_resources(&self, source_dir: &Path, output_dir: &Path) -> Result<()> {
        let csv_dir = source_dir.join("csv");
        let csv_dir = if csv_dir.exists() { csv_dir } else { source_dir.to_path_buf() };
        
        // Pattern: 60d_SCED_Gen_Resource_Data-DD-MMM-YY.csv
        let pattern = csv_dir.join("60d_SCED_Gen_Resource_Data-*.csv");
        let files = glob::glob(pattern.to_str().unwrap())?
            .filter_map(Result::ok)
            .collect::<Vec<_>>();
        
        println!("  Found {} SCED Gen Resource files", files.len());
        
        // These files are very large, process year by year
        let mut files_by_year: BTreeMap<i32, Vec<PathBuf>> = BTreeMap::new();
        for file in files {
            if let Some(year) = extract_year_from_60day_filename(&file) {
                files_by_year.entry(year).or_default().push(file);
            }
        }
        
        for (year, year_files) in files_by_year {
            println!("  Processing year {}: {} files", year, year_files.len());
            
            // Process SCED files one at a time due to size
            let mut all_dfs = Vec::new();
            
            for file in &year_files {
                // Reading file...
                match self.read_sced_gen_file(file) {
                    Ok(df) => {
                        // Loaded rows
                        all_dfs.push(df);
                    },
                    Err(e) => eprintln!("    Error reading {}: {}", file.display(), e),
                }
            }
            
            if !all_dfs.is_empty() {
                // Normalize all DataFrames to have the same columns before combining
                let normalized_dfs = self.normalize_sced_dataframes(all_dfs)?;
                let combined_df = self.combine_dataframes(normalized_dfs)?;
                
                let output_file = output_dir.join(format!("{}.parquet", year));
                let mut file = std::fs::File::create(&output_file)?;
                ParquetWriter::new(&mut file).finish(&mut combined_df.clone())?;
                
                println!("    ✅ Saved {} rows to {}", combined_df.height(), output_file.display());
                
                // SCED files are 5-minute, gap detection would be complex
                self.update_stats("SCED_Gen_Resources", year, year_files.len(), combined_df.height(), vec![]);
            }
        }
        
        Ok(())
    }
    
    fn normalize_sced_dataframes(&self, dfs: Vec<DataFrame>) -> Result<Vec<DataFrame>> {
        if dfs.is_empty() {
            return Ok(vec![]);
        }
        
        // Find all unique columns across all DataFrames
        let mut all_columns = std::collections::HashSet::new();
        for df in &dfs {
            for col in df.get_column_names() {
                all_columns.insert(col.to_string());
            }
        }
        
        // Sort columns for consistent ordering
        let mut all_columns: Vec<String> = all_columns.into_iter().collect();
        all_columns.sort();
        
        println!("    Normalizing to {} columns", all_columns.len());
        
        // Normalize each DataFrame to have all columns
        let mut normalized = Vec::new();
        for df in dfs {
            let mut select_exprs = Vec::new();
            let existing_cols = df.get_column_names();
            
            for col_name in &all_columns {
                if existing_cols.contains(&col_name.as_str()) {
                    // Column exists, use it
                    select_exprs.push(col(col_name));
                } else {
                    // Column doesn't exist, create with null values
                    // Determine appropriate default based on column name
                    if col_name.contains("AS_") || col_name.contains("Price") || 
                       col_name.contains("MW") || col_name.contains("Point") {
                        // Numeric column - use 0.0
                        select_exprs.push(lit(0.0f64).alias(col_name));
                    } else {
                        // String column - use empty string
                        select_exprs.push(lit("").alias(col_name));
                    }
                }
            }
            
            let normalized_df = df.lazy()
                .select(select_exprs)
                .collect()?;
                
            normalized.push(normalized_df);
        }
        
        Ok(normalized)
    }
    
    fn read_sced_gen_file(&self, file: &Path) -> Result<DataFrame> {
        // Force critical numeric columns to be Float64 to prevent i64 inference
        let mut schema_overrides = Schema::new();
        schema_overrides.with_column("LSL".into(), DataType::Float64);
        schema_overrides.with_column("HSL".into(), DataType::Float64);
        schema_overrides.with_column("Base Point".into(), DataType::Float64);
        schema_overrides.with_column("Telemetered Net Output".into(), DataType::Float64);
        schema_overrides.with_column("Ancillary Service RRS".into(), DataType::Float64);
        schema_overrides.with_column("Ancillary Service RRSPFR".into(), DataType::Float64);
        schema_overrides.with_column("Ancillary Service RRSFFR".into(), DataType::Float64);
        schema_overrides.with_column("Ancillary Service RRSUFR".into(), DataType::Float64);
        schema_overrides.with_column("Ancillary Service Reg-Up".into(), DataType::Float64);
        schema_overrides.with_column("Ancillary Service Reg-Down".into(), DataType::Float64);
        schema_overrides.with_column("Ancillary Service REGUP".into(), DataType::Float64);
        schema_overrides.with_column("Ancillary Service REGDN".into(), DataType::Float64);
        schema_overrides.with_column("Ancillary Service Non-Spin".into(), DataType::Float64);
        schema_overrides.with_column("Ancillary Service NSRS".into(), DataType::Float64);  // Non-Spinning Reserve Service
        schema_overrides.with_column("Ancillary Service NSPIN".into(), DataType::Float64);  // Another variant
        schema_overrides.with_column("Ancillary Service ECRS".into(), DataType::Float64);
        schema_overrides.with_column("Ancillary Service ECRSSD".into(), DataType::Float64);
        
        // Add SCED curve price columns - these are ALWAYS Float64!
        // SCED1 curve points - up to 50 points to be absolutely sure!
        for i in 1..=50 {
            schema_overrides.with_column(format!("SCED1 Curve-MW{}", i).into(), DataType::Float64);
            schema_overrides.with_column(format!("SCED1 Curve-Price{}", i).into(), DataType::Float64);
        }
        
        // SCED2 curve points - up to 50 points to be absolutely sure!
        for i in 1..=50 {
            schema_overrides.with_column(format!("SCED2 Curve-MW{}", i).into(), DataType::Float64);
            schema_overrides.with_column(format!("SCED2 Curve-Price{}", i).into(), DataType::Float64);
        }
        
        // Output Schedule columns
        schema_overrides.with_column("Output Schedule".into(), DataType::Float64);
        schema_overrides.with_column("Output Schedule 2".into(), DataType::Float64);
        
        let df = CsvReader::from_path(file)?
            .has_header(true)
            .with_dtypes(Some(Arc::new(schema_overrides)))
            .infer_schema(None)  // Don't infer - use our explicit schema
            .finish()?;
        
        // Get available columns
        let columns = df.get_column_names();
        
        // Build select list with required columns
        let mut select_cols = vec![
            col("SCED Time Stamp").alias("SCEDTimeStamp"),
            col("Resource Name").alias("ResourceName"),
            col("Resource Type").alias("ResourceType"),
            col("Base Point").cast(DataType::Float64).alias("BasePoint"),
            col("HSL").cast(DataType::Float64),
            col("LSL").cast(DataType::Float64),
        ];
        
        // Add optional columns if they exist
        if columns.contains(&"Telemetered Net Output") {
            select_cols.push(col("Telemetered Net Output").cast(DataType::Float64).alias("TelemeteredNetOutput"));
        }
        
        // Add AS columns if they exist (these were added over time)
        // RRS was split into three categories after 2022
        if columns.contains(&"Ancillary Service RRS") {
            select_cols.push(col("Ancillary Service RRS").cast(DataType::Float64).alias("AS_RRS"));
        }
        if columns.contains(&"Ancillary Service RRSPFR") {
            select_cols.push(col("Ancillary Service RRSPFR").cast(DataType::Float64).alias("AS_RRSPFR"));
        }
        if columns.contains(&"Ancillary Service RRSFFR") {
            select_cols.push(col("Ancillary Service RRSFFR").cast(DataType::Float64).alias("AS_RRSFFR"));
        }
        if columns.contains(&"Ancillary Service RRSUFR") {
            select_cols.push(col("Ancillary Service RRSUFR").cast(DataType::Float64).alias("AS_RRSUFR"));
        }
        if columns.contains(&"Ancillary Service Reg-Up") {
            select_cols.push(col("Ancillary Service Reg-Up").cast(DataType::Float64).alias("AS_RegUp"));
        }
        if columns.contains(&"Ancillary Service Reg-Down") {
            select_cols.push(col("Ancillary Service Reg-Down").cast(DataType::Float64).alias("AS_RegDown"));
        }
        if columns.contains(&"Ancillary Service Non-Spin") {
            select_cols.push(col("Ancillary Service Non-Spin").cast(DataType::Float64).alias("AS_NonSpin"));
        }
        if columns.contains(&"Ancillary Service NSRS") {
            select_cols.push(col("Ancillary Service NSRS").cast(DataType::Float64).alias("AS_NSRS"));
        }
        if columns.contains(&"Ancillary Service ECRS") {
            select_cols.push(col("Ancillary Service ECRS").cast(DataType::Float64).alias("AS_ECRS"));
        }
        if columns.contains(&"Ancillary Service ECRSSD") {
            select_cols.push(col("Ancillary Service ECRSSD").cast(DataType::Float64).alias("AS_ECRSSD"));
        }
        
        // Apply selection and add datetime column
        let df = df.lazy()
            .select(select_cols)
            .with_column(
                col("SCEDTimeStamp").cast(DataType::Utf8).alias("datetime")
            )
            .collect()?;
        
        Ok(df)
    }
    
    fn process_cop_snapshots(&self, source_dir: &Path, output_dir: &Path) -> Result<()> {
        let csv_dir = source_dir.join("csv");
        let csv_dir = if csv_dir.exists() { csv_dir } else { source_dir.to_path_buf() };
        
        // Pattern 1: 60d_COP_Adjustment_Period_Snapshot-DD-MMM-YY.csv (after Dec 13, 2022)
        let pattern1 = csv_dir.join("60d_COP_Adjustment_Period_Snapshot-*.csv");
        let files1 = glob::glob(pattern1.to_str().unwrap())?
            .filter_map(Result::ok)
            .collect::<Vec<_>>();
        
        // Pattern 2: CompleteCOP_MMDDYYYY.csv (before Dec 13, 2022)
        let pattern2 = csv_dir.join("CompleteCOP_*.csv");
        let files2 = glob::glob(pattern2.to_str().unwrap())?
            .filter_map(Result::ok)
            .collect::<Vec<_>>();
        
        let mut files = files1;
        files.extend(files2);
        
        println!("  Found {} COP Snapshot files total", files.len());
        
        let mut files_by_year: BTreeMap<i32, Vec<PathBuf>> = BTreeMap::new();
        for file in files {
            if let Some(year) = extract_year_from_60day_filename(&file) {
                files_by_year.entry(year).or_default().push(file);
            }
        }
        
        for (year, year_files) in files_by_year {
            println!("  Processing year {}: {} files", year, year_files.len());
            
            let mut all_dfs = Vec::new();
            
            for file in &year_files {
                match self.read_cop_file(file) {
                    Ok(df) => all_dfs.push(df),
                    Err(e) => eprintln!("    Error reading {}: {}", file.display(), e),
                }
            }
            
            if !all_dfs.is_empty() {
                let combined_df = self.combine_dataframes(all_dfs)?;
                
                let output_file = output_dir.join(format!("{}.parquet", year));
                let mut file = std::fs::File::create(&output_file)?;
                ParquetWriter::new(&mut file).finish(&mut combined_df.clone())?;
                
                println!("    ✅ Saved {} rows to {}", combined_df.height(), output_file.display());
                
                self.update_stats("COP_Snapshots", year, year_files.len(), combined_df.height(), vec![]);
            }
        }
        
        Ok(())
    }
    
    fn read_cop_file(&self, file: &Path) -> Result<DataFrame> {
        // Force ALL columns to have explicit types to prevent inference errors
        let mut schema_overrides = Schema::new();
        
        // String columns that must NOT be parsed as numbers
        schema_overrides.with_column("Delivery Date".into(), DataType::Utf8);
        schema_overrides.with_column("QSE Name".into(), DataType::Utf8);
        schema_overrides.with_column("Resource Name".into(), DataType::Utf8);
        schema_overrides.with_column("Hour Ending".into(), DataType::Utf8);
        schema_overrides.with_column("Status".into(), DataType::Utf8);
        
        // Numeric columns - force to Float64
        schema_overrides.with_column("High Sustained Limit".into(), DataType::Float64);
        schema_overrides.with_column("Low Sustained Limit".into(), DataType::Float64);
        schema_overrides.with_column("High Emergency Limit".into(), DataType::Float64);
        schema_overrides.with_column("Low Emergency Limit".into(), DataType::Float64);
        schema_overrides.with_column("Reg Up".into(), DataType::Float64);
        schema_overrides.with_column("Reg Down".into(), DataType::Float64);
        schema_overrides.with_column("RRSPFR".into(), DataType::Float64);
        schema_overrides.with_column("RRSFFR".into(), DataType::Float64);
        schema_overrides.with_column("RRSUFR".into(), DataType::Float64);
        schema_overrides.with_column("NSPIN".into(), DataType::Float64);
        schema_overrides.with_column("ECRS".into(), DataType::Float64);
        
        // SOC columns (may not exist in all files)
        schema_overrides.with_column("Minimum SOC".into(), DataType::Float64);
        schema_overrides.with_column("Maximum SOC".into(), DataType::Float64);
        schema_overrides.with_column("Hour Beginning Planned SOC".into(), DataType::Float64);
        
        let df = CsvReader::from_path(file)?
            .has_header(true)
            .with_dtypes(Some(Arc::new(schema_overrides)))
            .infer_schema(None)  // Don't infer - use our explicit schema
            .finish()?;
        
        // Check which columns exist in the file
        let columns = df.get_column_names();
        let has_soc_columns = columns.contains(&"Minimum SOC");
        
        // Build select expressions based on available columns
        let mut select_exprs = vec![
            col("Delivery Date").alias("DeliveryDate"),
            col("Hour Ending").alias("HourEnding"),
            col("Resource Name").alias("ResourceName"),
            col("Status"),
            col("High Sustained Limit").alias("HSL"),
            col("Low Sustained Limit").alias("LSL"),
        ];
        
        // Add SOC columns if they exist, otherwise use defaults
        if has_soc_columns {
            select_exprs.push(col("Minimum SOC").alias("MinSOC"));
            select_exprs.push(col("Maximum SOC").alias("MaxSOC"));
            select_exprs.push(col("Hour Beginning Planned SOC").alias("PlannedSOC"));
        } else {
            // Files without SOC columns - use null Float64 values for type consistency
            select_exprs.push(lit(NULL).cast(DataType::Float64).alias("MinSOC"));
            select_exprs.push(lit(NULL).cast(DataType::Float64).alias("MaxSOC"));
            select_exprs.push(lit(NULL).cast(DataType::Float64).alias("PlannedSOC"));
        }
        
        // Select relevant columns for BESS SOC tracking
        let mut df = df.lazy()
            .select(select_exprs)
            .with_column(
                col("DeliveryDate").cast(DataType::Date)
            )
            .collect()?;
        
        // Convert HourEnding from "01:00" format to numeric hour
        let hour_values: Vec<f64> = df.column("HourEnding")?
            .utf8()?
            .into_iter()
            .map(|opt_val| {
                opt_val.map(|val| {
                    // Extract hour from "01:00" format or parse as-is if numeric
                    if val.contains(':') {
                        val.split(':').next()
                            .and_then(|h| h.parse::<f64>().ok())
                            .unwrap_or(0.0)
                    } else {
                        val.parse::<f64>().unwrap_or(0.0)
                    }
                }).unwrap_or(0.0)
            })
            .collect();
        
        let hour_series = Series::new("hour", hour_values);
        df.with_column(hour_series)?;
        
        // Now add datetime column
        let df = df.lazy()
            .with_column(
                (col("DeliveryDate").cast(DataType::Datetime(TimeUnit::Milliseconds, None)) +
                 duration_hours(col("hour") - lit(1)))
                .alias("datetime")
            )
            .collect()?;
        
        Ok(df)
    }
    
    fn combine_dataframes(&self, dfs: Vec<DataFrame>) -> Result<DataFrame> {
        if dfs.is_empty() {
            return Err(anyhow::anyhow!("No dataframes to combine"));
        }
        
        let mut combined = dfs[0].clone();
        for df in dfs.into_iter().skip(1) {
            combined = combined.vstack(&df)?;
        }
        
        // Sort by datetime if it exists
        if combined.get_column_names().contains(&"datetime") {
            combined = combined.lazy()
                .sort("datetime", Default::default())
                .collect()?;
        }
        
        Ok(combined)
    }
    
    fn detect_gaps(&self, df: &DataFrame, datetime_col: &str) -> Result<Vec<DataGap>> {
        let mut gaps = Vec::new();
        
        if !df.get_column_names().contains(&datetime_col) {
            return Ok(gaps);
        }
        
        // Get datetime column and convert i64 timestamps to dates
        let datetime_col_data = df.column(datetime_col)?;
        
        let dates = if let Ok(dt) = datetime_col_data.datetime() {
            // Already datetime type
            dt.as_datetime_iter()
                .filter_map(|dt| dt.map(|d| NaiveDate::from_ymd_opt(
                    d.year(), d.month(), d.day()
                ).unwrap()))
                .collect::<HashSet<_>>()
        } else if let Ok(timestamps) = datetime_col_data.i64() {
            // i64 timestamps in milliseconds
            timestamps.into_iter()
                .filter_map(|ts| ts.map(|t| {
                    let dt = DateTime::<Utc>::from_timestamp_millis(t).unwrap();
                    NaiveDate::from_ymd_opt(dt.year(), dt.month(), dt.day()).unwrap()
                }))
                .collect::<HashSet<_>>()
        } else {
            // Can't parse datetime column, skip gap detection
            return Ok(gaps);
        };
        
        if dates.is_empty() {
            return Ok(gaps);
        }
        
        let mut sorted_dates: Vec<_> = dates.into_iter().collect();
        sorted_dates.sort();
        
        let mut prev_date = sorted_dates[0];
        for &date in &sorted_dates[1..] {
            let days_diff = (date - prev_date).num_days();
            if days_diff > 1 {
                gaps.push(DataGap {
                    start_date: prev_date + Duration::days(1),
                    end_date: date - Duration::days(1),
                    missing_days: days_diff - 1,
                });
            }
            prev_date = date;
        }
        
        Ok(gaps)
    }
    
    fn save_gaps_report(&self, output_dir: &Path, year: i32, gaps: &[DataGap]) -> Result<()> {
        // Save Markdown report
        let report_file = output_dir.join("gaps_report.md");
        let mut content = String::new();
        
        if report_file.exists() {
            content = fs::read_to_string(&report_file)?;
        } else {
            content.push_str("# Data Gaps Report\n\n");
        }
        
        content.push_str(&format!("\n## Year {}\n", year));
        
        if gaps.is_empty() {
            content.push_str("✅ No gaps detected\n");
        } else {
            content.push_str(&format!("⚠️ {} gaps detected:\n\n", gaps.len()));
            for gap in gaps {
                content.push_str(&format!(
                    "- {} to {} ({} days missing)\n",
                    gap.start_date, gap.end_date, gap.missing_days
                ));
            }
        }
        
        fs::write(report_file, &content)?;
        
        // Also save to CSV for easier analysis
        let csv_file = output_dir.join("gaps_report.csv");
        let mut csv_content = String::new();
        
        // Write or append CSV header
        if !csv_file.exists() {
            csv_content.push_str("dataset,year,start_date,end_date,missing_days\n");
        } else {
            csv_content = fs::read_to_string(&csv_file)?;
        }
        
        // Add gaps data
        let dataset_name = output_dir.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");
        
        for gap in gaps {
            csv_content.push_str(&format!(
                "{},{},{},{},{}\n",
                dataset_name, year, gap.start_date, gap.end_date, gap.missing_days
            ));
        }
        
        // If no gaps, still add a record to show we checked
        if gaps.is_empty() {
            csv_content.push_str(&format!(
                "{},{},,,0\n",
                dataset_name, year
            ));
        }
        
        fs::write(csv_file, &csv_content)?;
        
        Ok(())
    }
    
    fn save_schema(&self, output_dir: &Path, _name: &str, schema_json: &str) -> Result<()> {
        let schema_file = output_dir.join("schema.json");
        fs::write(schema_file, schema_json)?;
        Ok(())
    }
    
    fn update_stats(&self, dataset: &str, year: i32, files: usize, rows: usize, gaps: Vec<DataGap>) {
        let mut stats = self.processing_stats.lock().unwrap();
        let entry = stats.entry(dataset.to_string()).or_insert_with(|| ProcessingStats {
            total_files: 0,
            processed_files: 0,
            failed_files: 0,
            total_rows: 0,
            years_covered: Vec::new(),
            gaps: Vec::new(),
        });
        
        entry.total_files += files;
        entry.processed_files += files;
        entry.total_rows += rows;
        if !entry.years_covered.contains(&year) {
            entry.years_covered.push(year);
            entry.years_covered.sort();
        }
        entry.gaps.extend(gaps);
    }
    
    fn generate_status_report(&self) -> Result<()> {
        let stats = self.processing_stats.lock().unwrap();
        let report_file = self.output_dir.join("processing_status_report.md");
        
        let mut content = String::new();
        content.push_str("# ERCOT Annual Parquet Processing Status Report\n\n");
        content.push_str(&format!("Generated: {}\n\n", Utc::now().format("%Y-%m-%d %H:%M:%S UTC")));
        
        content.push_str("## Summary\n\n");
        content.push_str("| Dataset | Files | Rows | Years | Gaps |\n");
        content.push_str("|---------|-------|------|-------|------|\n");
        
        for (dataset, stat) in stats.iter() {
            let gaps_str = if stat.gaps.is_empty() { 
                "None".to_string() 
            } else { 
                format!("{} gaps", stat.gaps.len()) 
            };
            content.push_str(&format!(
                "| {} | {} | {} | {} | {} |\n",
                dataset,
                stat.processed_files,
                stat.total_rows,
                format!("{:?}", stat.years_covered),
                gaps_str
            ));
        }
        
        content.push_str("\n## Data Quality Notes\n\n");
        content.push_str("- All price columns enforced as Float64 to prevent type mismatches\n");
        content.push_str("- Datetime columns created from delivery date and hour/interval\n");
        content.push_str("- Files processed in parallel for performance\n");
        content.push_str("- Gap detection performed on temporal data\n");
        
        fs::write(report_file, &content)?;
        
        // Also print to stdout
        println!("\n{}", "=".repeat(80));
        println!("{}", content);
        
        Ok(())
    }
}

#[derive(Debug)]
enum ProcessorType {
    RealTimePrice,
    DayAheadPrice,
    AncillaryService,
    DAMGenResource,
    SCEDGenResource,
    COPSnapshot,
}

fn extract_year_from_filename(path: &Path) -> Option<i32> {
    let name = path.file_name()?.to_str()?;
    
    // Look for YYYYMMDD pattern in the filename
    // Example: cdr.00012301.0000000000000000.20230104.001704.SPPHLZNP6905_20230104_0015.csv
    for part in name.split('.') {
        if part.len() >= 8 {
            // Check if starts with 20 and parse first 4 chars as year
            if part.starts_with("20") {
                if let Ok(year) = part[0..4].parse::<i32>() {
                    if (2000..=2100).contains(&year) {
                        return Some(year);
                    }
                }
            }
        }
    }
    
    None
}

fn extract_year_from_60day_filename(path: &Path) -> Option<i32> {
    let name = path.file_name()?.to_str()?;
    
    // Pattern 1: DD-MMM-YY (e.g., 07-JAN-25) for 60d_COP files
    if let Some(captures) = regex::Regex::new(r"(\d{2})-([A-Z]{3})-(\d{2})").unwrap().captures(name) {
        let year_str = captures.get(3)?.as_str();
        let year: i32 = year_str.parse().ok()?;
        // Convert 2-digit year to 4-digit
        let full_year = if year < 50 { 2000 + year } else { 1900 + year };
        return Some(full_year);
    }
    
    // Pattern 2: CompleteCOP_MMDDYYYY.csv (e.g., CompleteCOP_10132022.csv)
    if let Some(captures) = regex::Regex::new(r"CompleteCOP_(\d{2})(\d{2})(\d{4})").unwrap().captures(name) {
        let year_str = captures.get(3)?.as_str();
        let year: i32 = year_str.parse().ok()?;
        return Some(year);
    }
    
    None
}

// Helper functions for duration
fn duration_hours(hours: Expr) -> Expr {
    hours * lit(3600000)  // Convert hours to milliseconds
}