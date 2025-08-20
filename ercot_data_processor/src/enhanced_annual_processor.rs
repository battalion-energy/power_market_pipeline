use anyhow::Result;
use chrono::{DateTime, Datelike, Duration, NaiveDate, Utc};
use polars::prelude::*;
use rayon::prelude::*;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{self, File};
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
    selected_dataset: Option<String>,
}

impl EnhancedAnnualProcessor {
    pub fn new(base_dir: PathBuf) -> Self {
        let output_dir = base_dir.join("rollup_files");
        Self {
            base_dir,
            output_dir,
            processing_stats: Arc::new(Mutex::new(HashMap::new())),
            selected_dataset: None,
        }
    }
    
    pub fn with_dataset(mut self, dataset: String) -> Self {
        self.selected_dataset = Some(dataset);
        self
    }
    
    pub fn process_all_data(&self) -> Result<()> {
        let start_time = std::time::Instant::now();
        
        println!("🚀 Enhanced Annual Parquet Processor");
        println!("📁 Base directory: {}", self.base_dir.display());
        println!("📂 Output directory: {}", self.output_dir.display());
        if let Some(ref dataset) = self.selected_dataset {
            println!("🎯 Selected dataset: {}", dataset);
        }
        println!("{}", "=".repeat(80));
        
        // Create output directory
        fs::create_dir_all(&self.output_dir)?;
        
        // Process each data type - RT prices last since they're the most voluminous
        let mut processors = vec![
            ("DA_prices", "DAM_Settlement_Point_Prices", ProcessorType::DayAheadPrice),
            ("AS_prices", "DAM_Clearing_Prices_for_Capacity", ProcessorType::AncillaryService),
            ("DAM_Gen_Resources", "60-Day_DAM_Disclosure_Reports", ProcessorType::DAMGenResource),
            ("DAM_Load_Resources", "60-Day_DAM_Disclosure_Reports", ProcessorType::DAMLoadResource),
            ("DAM_Energy_Bid_Awards", "60-Day_DAM_Disclosure_Reports", ProcessorType::DAMEnergyBidAwards),
            ("SCED_Gen_Resources", "60-Day_SCED_Disclosure_Reports", ProcessorType::SCEDGenResource),
            ("SCED_Load_Resources", "60-Day_SCED_Disclosure_Reports", ProcessorType::SCEDLoadResource),
            ("COP_Snapshots", "60-Day_COP_Adjustment_Period_Snapshot", ProcessorType::COPSnapshot),
            ("RT_prices", "Settlement_Point_Prices_at_Resource_Nodes,_Hubs_and_Load_Zones", ProcessorType::RealTimePrice),
        ];
        
        // Filter to selected dataset if specified
        if let Some(ref dataset) = self.selected_dataset {
            processors.retain(|(name, _, _)| name == dataset);
            if processors.is_empty() {
                println!("❌ Unknown dataset: {}", dataset);
                println!("Available datasets:");
                println!("  - DA_prices (Day-Ahead Settlement Point Prices)");
                println!("  - AS_prices (Ancillary Services Clearing Prices)");
                println!("  - DAM_Gen_Resources (60-Day DAM Generation Resources)");
                println!("  - DAM_Load_Resources (60-Day DAM Load Resources)");
                println!("  - DAM_Energy_Bid_Awards (60-Day DAM Energy Bid Awards - BESS charging!)");
                println!("  - SCED_Gen_Resources (60-Day SCED Generation Resources)");
                println!("  - SCED_Load_Resources (60-Day SCED Load Resources)");
                println!("  - COP_Snapshots (60-Day COP Adjustment Period Snapshots)");
                println!("  - RT_prices (Real-Time Settlement Point Prices)");
                return Ok(());
            }
        }
        
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
                ProcessorType::DAMLoadResource => self.process_dam_load_resources(&source_path, &output_path)?,
                ProcessorType::DAMEnergyBidAwards => self.process_dam_energy_bid_awards(&source_path, &output_path)?,
                ProcessorType::SCEDGenResource => self.process_sced_gen_resources(&source_path, &output_path)?,
                ProcessorType::SCEDLoadResource => self.process_sced_load_resources(&source_path, &output_path)?,
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
            
            // Process in parallel batches and combine incrementally
            let batch_size = 1000;  // Larger batches for better parallelism
            let mut combined_batches = Vec::new();
            
            for (batch_idx, batch) in year_files.chunks(batch_size).enumerate() {
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
                
                // Combine this batch's dataframes
                if !batch_dfs.is_empty() {
                    if batch_idx % 10 == 9 {
                        pb.set_message(format!("combining batch {}", batch_idx + 1));
                    }
                    match self.combine_dataframes(batch_dfs) {
                        Ok(batch_df) => combined_batches.push(batch_df),
                        Err(e) => eprintln!("    Error combining batch {}: {}", batch_idx, e),
                    }
                }
            }
            
            pb.finish_with_message("done");
            
            if !combined_batches.is_empty() {
                println!("    Performing final combination of {} batch results...", combined_batches.len());
                // Final combination of all batch results
                let combined_df = self.combine_dataframes(combined_batches)?;
                
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
            Field::new("DeliveryDate", DataType::String),
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
            if let Some(date_str) = delivery_dates.str()?.get(i) {
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
                println!("    Debug: Combining {} dataframes", all_dfs.len());
                let total_rows_before: usize = all_dfs.iter().map(|df| df.height()).sum();
                println!("    Debug: Total rows before combining: {}", total_rows_before);
                
                let combined_df = self.combine_dataframes(all_dfs)?;
                println!("    Debug: Rows after combining: {}", combined_df.height());
                
                // Check unique dates in the combined dataframe
                if combined_df.get_column_names().contains(&"DeliveryDate") {
                    let delivery_dates = combined_df.column("DeliveryDate")?;
                    let unique_dates = delivery_dates.unique()?;
                    println!("    Debug: Unique DeliveryDate values: {}", unique_dates.len());
                }
                
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
        // Read CSV with Float64 for price column and string for dates initially
        let schema_overrides = Schema::from_iter([
            Field::new("SettlementPointPrice", DataType::Float64),
            Field::new("DeliveryDate", DataType::String),  // Read as string first
            Field::new("HourEnding", DataType::String),     // Read as string first
        ]);
        
        let df = CsvReader::from_path(file)?
            .has_header(true)
            .with_try_parse_dates(false)  // Don't auto-parse dates
            .with_dtypes(Some(Arc::new(schema_overrides)))
            .finish()?;
        
        // Normalize the dataframe to handle schema evolution (DST flag added in 2011)
        let df = crate::schema_normalizer::normalize_dam_prices(df)?;
        
        // Process the dataframe - first collect then parse dates
        let mut df = df.lazy()
            .with_column(
                col("SettlementPointPrice").cast(DataType::Float64)  // Critical: Force Float64
            )
            .with_column(
                col("HourEnding").alias("hour")
            )
            .collect()?;
        
        // Now parse the dates from MM/DD/YYYY string to Date type
        let mut parsed_dates: Vec<Option<i32>> = Vec::new();
        let mut date_strings: Vec<Option<String>> = Vec::new();
        
        // Unix epoch date for calculating days since epoch
        let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
        
        // Extract dates and process them
        {
            let delivery_dates = df.column("DeliveryDate")?;
            if let Ok(dates_str) = delivery_dates.str() {
                for i in 0..dates_str.len() {
                    if let Some(date_str) = dates_str.get(i) {
                        if let Ok(date) = NaiveDate::parse_from_str(date_str, "%m/%d/%Y") {
                            // Calculate days since Unix epoch (1970-01-01)
                            let days_since_epoch = (date - epoch).num_days() as i32;
                            parsed_dates.push(Some(days_since_epoch));
                            
                            // Format as ISO date with timezone indicator
                            // ERCOT operates in Central Time (CT)
                            let formatted = format!("{}T00:00:00-06:00", date.format("%Y-%m-%d"));
                            date_strings.push(Some(formatted));
                        } else {
                            parsed_dates.push(None);
                            date_strings.push(None);
                        }
                    } else {
                        parsed_dates.push(None);
                        date_strings.push(None);
                    }
                }
            }
        }
        
        // Create new Date column and replace the string one
        let date_series = Series::new("DeliveryDate", parsed_dates);
        df.with_column(date_series.cast(&DataType::Date)?)?;
        
        // Add string representation with timezone
        let date_str_series = Series::new("DeliveryDateStr", date_strings);
        df.with_column(date_str_series)?;
        
        // Add datetime column as unix timestamp (milliseconds since epoch)
        // This will be computed from DeliveryDate + HourEnding
        let df = df.lazy()
            .with_column(
                lit(NULL).cast(DataType::Int64).alias("datetime_ms")
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
        // Read CSV with Float64 for price column and string for dates initially
        let schema_overrides = Schema::from_iter([
            Field::new("MCPC", DataType::Float64),
            Field::new("DeliveryDate", DataType::String),  // Read as string first
            Field::new("HourEnding", DataType::String),     // Read as string first
        ]);
        
        let df = CsvReader::from_path(file)?
            .has_header(true)
            .with_try_parse_dates(false)  // Don't auto-parse dates
            .with_dtypes(Some(Arc::new(schema_overrides)))
            .finish()?;
        
        // Normalize the dataframe to handle schema evolution (DST flag added in 2011)
        let df = crate::schema_normalizer::normalize_as_prices(df)?;
        
        // Process the dataframe - first collect then parse dates
        let mut df = df.lazy()
            .with_column(
                col("MCPC").cast(DataType::Float64)  // Critical: Force Float64
            )
            .with_column(
                col("HourEnding").alias("hour")
            )
            .collect()?;
        
        // Now parse the dates from MM/DD/YYYY string to Date type
        let mut parsed_dates: Vec<Option<i32>> = Vec::new();
        let mut date_strings: Vec<Option<String>> = Vec::new();
        
        // Unix epoch date for calculating days since epoch
        let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
        
        // Extract dates and process them
        {
            let delivery_dates = df.column("DeliveryDate")?;
            if let Ok(dates_str) = delivery_dates.str() {
                for i in 0..dates_str.len() {
                    if let Some(date_str) = dates_str.get(i) {
                        if let Ok(date) = NaiveDate::parse_from_str(date_str, "%m/%d/%Y") {
                            // Calculate days since Unix epoch (1970-01-01)
                            let days_since_epoch = (date - epoch).num_days() as i32;
                            parsed_dates.push(Some(days_since_epoch));
                            
                            // Format as ISO date with timezone indicator
                            // ERCOT operates in Central Time (CT)
                            let formatted = format!("{}T00:00:00-06:00", date.format("%Y-%m-%d"));
                            date_strings.push(Some(formatted));
                        } else {
                            parsed_dates.push(None);
                            date_strings.push(None);
                        }
                    } else {
                        parsed_dates.push(None);
                        date_strings.push(None);
                    }
                }
            }
        }
        
        // Create new Date column and replace the string one
        let date_series = Series::new("DeliveryDate", parsed_dates);
        df.with_column(date_series.cast(&DataType::Date)?)?;
        
        // Add string representation with timezone
        let date_str_series = Series::new("DeliveryDateStr", date_strings);
        df.with_column(date_str_series)?;
        
        // Add datetime column as unix timestamp (milliseconds since epoch)
        let df = df.lazy()
            .with_column(
                lit(NULL).cast(DataType::Int64).alias("datetime_ms")
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
        schema_overrides.with_column("ECRS MCPC".into(), DataType::String);
        schema_overrides.with_column("RegUp MCPC".into(), DataType::Float64);
        schema_overrides.with_column("RegDown MCPC".into(), DataType::Float64);
        schema_overrides.with_column("RRS MCPC".into(), DataType::Float64);
        schema_overrides.with_column("NonSpin MCPC".into(), DataType::Float64);
        
        // Keep Delivery Date as Utf8 to parse it properly
        schema_overrides.with_column("Delivery Date".into(), DataType::String);
        schema_overrides.with_column("Hour Ending".into(), DataType::String);
        
        let df = CsvReader::from_path(file)?
            .has_header(true)
            .with_dtypes(Some(Arc::new(schema_overrides)))
            .infer_schema(None)  // Don't infer - use our explicit schema
            .finish()?;
        
        // Get available columns
        let columns = df.get_column_names();
        let mut select_cols = vec![];
        
        // Add date columns first - these are critical
        if columns.contains(&"Delivery Date") {
            select_cols.push(col("Delivery Date").alias("DeliveryDate"));
        } else {
            select_cols.push(lit("").alias("DeliveryDate"));
        }
        
        if columns.contains(&"Hour Ending") {
            select_cols.push(col("Hour Ending").alias("HourEnding"));
        } else {
            select_cols.push(lit("1").alias("HourEnding"));
        }
        
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
        
        // Add QSE submitted bid curve points (up to 10 pairs for DAM)
        for i in 1..=10 {
            let mw_col = format!("QSE submitted Curve-MW{}", i);
            let price_col = format!("QSE submitted Curve-Price{}", i);
            
            if columns.contains(&mw_col.as_str()) {
                select_cols.push(col(&mw_col).cast(DataType::Float64));
            }
            if columns.contains(&price_col.as_str()) {
                select_cols.push(col(&price_col).cast(DataType::Float64));
            }
        }
        
        // Build dataframe with available columns
        let df = df.lazy()
            .select(select_cols)
            .with_column(
                // Parse HourEnding as integer (it's a string like "1", "2", etc.)
                col("HourEnding").cast(DataType::Int32).alias("hour")
            )
            .collect()?;
        
        // Now create a proper datetime from date string and hour
        // The DeliveryDate is in MM/DD/YYYY format
        let mut df = df.lazy()
            .with_column(
                // Cast DeliveryDate string to proper date
                col("DeliveryDate")
                    .cast(DataType::String)
                    .alias("DeliveryDate")
            )
            .collect()?;
        
        // Parse dates from MM/DD/YYYY format to Date32
        let mut parsed_dates = Vec::new();
        let mut date_strings = Vec::new();
        let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
        
        // Extract dates and process them
        {
            let delivery_dates = df.column("DeliveryDate")?;
            if let Ok(dates_str) = delivery_dates.str() {
                for i in 0..dates_str.len() {
                    if let Some(date_str) = dates_str.get(i) {
                        if let Ok(date) = NaiveDate::parse_from_str(date_str, "%m/%d/%Y") {
                            // Calculate days since Unix epoch (1970-01-01)
                            let days_since_epoch = (date - epoch).num_days() as i32;
                            parsed_dates.push(Some(days_since_epoch));
                            
                            // Format as ISO date with timezone indicator
                            let formatted = format!("{}T00:00:00-06:00", date.format("%Y-%m-%d"));
                            date_strings.push(Some(formatted));
                        } else {
                            parsed_dates.push(None);
                            date_strings.push(None);
                        }
                    } else {
                        parsed_dates.push(None);
                        date_strings.push(None);
                    }
                }
            }
        }
        
        // Create new Date column and replace the string one
        let date_series = Series::new("DeliveryDate", parsed_dates);
        df.with_column(date_series.cast(&DataType::Date)?)?;
        
        // Add string representation with timezone
        let date_str_series = Series::new("DeliveryDateStr", date_strings);
        df.with_column(date_str_series)?;
        
        // Create datetime column by combining date and hour
        let df = df.lazy()
            .with_column(
                // Hour Ending 1 = 00:00-01:00, so we subtract 1 hour
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
        
        // Add SCED1 bid curve points (up to 35 pairs)
        for i in 1..=35 {
            let mw_col = format!("SCED1 Curve-MW{}", i);
            let price_col = format!("SCED1 Curve-Price{}", i);
            
            if columns.contains(&mw_col.as_str()) {
                select_cols.push(col(&mw_col).cast(DataType::Float64));
            }
            if columns.contains(&price_col.as_str()) {
                select_cols.push(col(&price_col).cast(DataType::Float64));
            }
        }
        
        // Add SCED2 bid curve points (up to 35 pairs)
        for i in 1..=35 {
            let mw_col = format!("SCED2 Curve-MW{}", i);
            let price_col = format!("SCED2 Curve-Price{}", i);
            
            if columns.contains(&mw_col.as_str()) {
                select_cols.push(col(&mw_col).cast(DataType::Float64));
            }
            if columns.contains(&price_col.as_str()) {
                select_cols.push(col(&price_col).cast(DataType::Float64));
            }
        }
        
        // Apply selection and add datetime column
        let df = df.lazy()
            .select(select_cols)
            .with_column(
                col("SCEDTimeStamp").cast(DataType::String).alias("datetime")
            )
            .collect()?;
        
        Ok(df)
    }
    
    fn process_sced_load_resources(&self, source_dir: &Path, output_dir: &Path) -> Result<()> {
        let csv_dir = source_dir.join("csv");
        let csv_dir = if csv_dir.exists() { csv_dir } else { source_dir.to_path_buf() };
        
        // Pattern: 60d_Load_Resource_Data_in_SCED-DD-MMM-YY.csv
        let pattern = csv_dir.join("60d_Load_Resource_Data_in_SCED-*.csv");
        let files = glob::glob(pattern.to_str().unwrap())?
            .filter_map(Result::ok)
            .collect::<Vec<_>>();
        
        println!("  Found {} SCED Load Resource files", files.len());
        
        // Group by year
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
                match self.read_sced_load_file(file) {
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
                
                self.update_stats("SCED_Load_Resources", year, year_files.len(), combined_df.height(), vec![]);
            }
        }
        
        Ok(())
    }
    
    fn read_sced_load_file(&self, file: &Path) -> Result<DataFrame> {
        // Force critical numeric columns to be Float64
        let mut schema_overrides = Schema::new();
        schema_overrides.with_column("Max Power Consumption".into(), DataType::Float64);
        schema_overrides.with_column("LDL".into(), DataType::Float64);
        schema_overrides.with_column("HDL".into(), DataType::Float64);
        schema_overrides.with_column("Base Point".into(), DataType::Float64);
        schema_overrides.with_column("Telemetered Load".into(), DataType::Float64);
        schema_overrides.with_column("AS LRS".into(), DataType::Float64);
        schema_overrides.with_column("HASL".into(), DataType::Float64);
        schema_overrides.with_column("LASL".into(), DataType::Float64);
        
        // Add bid curve columns
        for i in 1..=10 {
            schema_overrides.with_column(format!("SCED Bid to Buy Curve-MW{}", i).into(), DataType::Float64);
            schema_overrides.with_column(format!("SCED Bid to Buy Curve-Price{}", i).into(), DataType::Float64);
        }
        
        // Define null values - empty strings should be treated as null for numeric columns
        let null_values = NullValues::AllColumns(vec!["".to_string(), "NA".to_string(), "N/A".to_string()]);
        
        let df = CsvReader::from_path(file)?
            .has_header(true)
            .with_dtypes(Some(Arc::new(schema_overrides)))
            .infer_schema(None)  // Don't infer - we have explicit schema
            .with_null_values(Some(null_values))  // Treat empty strings as null
            .finish()?;
        
        // Get available columns
        let columns = df.get_column_names();
        
        // Build select list - include ALL columns that exist
        let mut select_cols = vec![];
        
        // Always include key columns if they exist
        if columns.contains(&"SCED Time Stamp") {
            select_cols.push(col("SCED Time Stamp").alias("SCEDTimeStamp"));
        }
        if columns.contains(&"Load Resource Name") {
            select_cols.push(col("Load Resource Name").alias("LoadResourceName"));
        }
        if columns.contains(&"Settlement Point Name") {
            select_cols.push(col("Settlement Point Name").alias("SettlementPointName"));
        }
        
        // Add numeric columns
        if columns.contains(&"Max Power Consumption") {
            select_cols.push(col("Max Power Consumption").cast(DataType::Float64).alias("MaxPowerConsumption"));
        }
        if columns.contains(&"LDL") {
            select_cols.push(col("LDL").cast(DataType::Float64));
        }
        if columns.contains(&"HDL") {
            select_cols.push(col("HDL").cast(DataType::Float64));
        }
        if columns.contains(&"Base Point") {
            select_cols.push(col("Base Point").cast(DataType::Float64).alias("BasePoint"));
        }
        if columns.contains(&"Telemetered Load") {
            select_cols.push(col("Telemetered Load").cast(DataType::Float64).alias("TelemeteredLoad"));
        }
        if columns.contains(&"AS LRS") {
            select_cols.push(col("AS LRS").cast(DataType::Float64).alias("AS_LRS"));
        }
        if columns.contains(&"HASL") {
            select_cols.push(col("HASL").cast(DataType::Float64));
        }
        if columns.contains(&"LASL") {
            select_cols.push(col("LASL").cast(DataType::Float64));
        }
        
        // Add SCED Bid to Buy curve points (up to 10 pairs)
        for i in 1..=10 {
            let mw_col = format!("SCED Bid to Buy Curve-MW{}", i);
            let price_col = format!("SCED Bid to Buy Curve-Price{}", i);
            
            if columns.contains(&mw_col.as_str()) {
                select_cols.push(col(&mw_col).cast(DataType::Float64));
            }
            if columns.contains(&price_col.as_str()) {
                select_cols.push(col(&price_col).cast(DataType::Float64));
            }
        }
        
        // Apply selection and add datetime column
        let df = df.lazy()
            .select(select_cols)
            .with_column(
                col("SCEDTimeStamp").cast(DataType::String).alias("datetime")
            )
            .collect()?;
        
        Ok(df)
    }
    
    fn process_dam_load_resources(&self, source_dir: &Path, output_dir: &Path) -> Result<()> {
        let csv_dir = source_dir.join("csv");
        let csv_dir = if csv_dir.exists() { csv_dir } else { source_dir.to_path_buf() };
        
        // Pattern: 60d_DAM_Load_Resource_Data-DD-MMM-YY.csv
        let pattern = csv_dir.join("60d_DAM_Load_Resource_Data-*.csv");
        let files = glob::glob(pattern.to_str().unwrap())?
            .filter_map(Result::ok)
            .collect::<Vec<_>>();
        
        println!("  Found {} DAM Load Resource files", files.len());
        
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
                match self.read_dam_load_file(file) {
                    Ok(df) => all_dfs.push(df),
                    Err(e) => eprintln!("    Error reading {}: {}", file.display(), e),
                }
            }
            
            pb.finish_with_message("done");
            
            if !all_dfs.is_empty() {
                // Normalize all DataFrames to have the same columns before combining
                let normalized_dfs = self.normalize_dam_load_dataframes(all_dfs)?;
                let combined_df = self.combine_dataframes(normalized_dfs)?;
                let gaps = self.detect_gaps(&combined_df, "datetime")?;
                
                let output_file = output_dir.join(format!("{}.parquet", year));
                let mut file = std::fs::File::create(&output_file)?;
                ParquetWriter::new(&mut file).finish(&mut combined_df.clone())?;
                
                println!("    ✅ Saved {} rows to {}", combined_df.height(), output_file.display());
                
                self.save_gaps_report(output_dir, year, &gaps)?;
                self.update_stats("DAM_Load_Resources", year, year_files.len(), combined_df.height(), gaps);
            }
        }
        
        Ok(())
    }
    
    fn process_dam_energy_bid_awards(&self, source_dir: &Path, output_dir: &Path) -> Result<()> {
        let csv_dir = source_dir.join("csv");
        let csv_dir = if csv_dir.exists() { csv_dir } else { source_dir.to_path_buf() };
        
        // Pattern: 60d_DAM_EnergyBidAwards-DD-MMM-YY.csv
        let pattern = csv_dir.join("60d_DAM_EnergyBidAwards-*.csv");
        let files = glob::glob(pattern.to_str().unwrap())?
            .filter_map(Result::ok)
            .collect::<Vec<_>>();
        
        println!("  Found {} DAM Energy Bid Award files", files.len());
        
        // Group by year
        let mut files_by_year: BTreeMap<i32, Vec<PathBuf>> = BTreeMap::new();
        for file in files {
            if let Some(year) = extract_year_from_60day_filename(&file) {
                files_by_year.entry(year).or_insert_with(Vec::new).push(file);
            }
        }
        
        // Process each year
        for (year, year_files) in files_by_year {
            println!("  Processing {} ({} files)...", year, year_files.len());
            
            let mut all_dfs = Vec::new();
            for file in &year_files {
                match self.read_dam_energy_bid_awards_file(file) {
                    Ok(df) => all_dfs.push(df),
                    Err(e) => println!("    ⚠️  Error reading {}: {}", file.display(), e),
                }
            }
            
            if !all_dfs.is_empty() {
                // Combine all dataframes
                let combined_df = if all_dfs.len() == 1 {
                    all_dfs.into_iter().next().unwrap()
                } else {
                    concat(
                        all_dfs.iter().map(|df| df.clone().lazy()).collect::<Vec<_>>().as_slice(),
                        UnionArgs::default(),
                    )?.collect()?
                };
                
                // Save as parquet
                let output_file = output_dir.join(format!("{}.parquet", year));
                let mut file = File::create(&output_file)?;
                ParquetWriter::new(&mut file)
                    .with_compression(ParquetCompression::Snappy)
                    .finish(&mut combined_df.clone())?;
                
                println!("    ✅ Saved {} rows to {}", combined_df.height(), output_file.display());
                
                self.update_stats("DAM_Energy_Bid_Awards", year, year_files.len(), combined_df.height(), vec![]);
            }
        }
        
        Ok(())
    }
    
    fn read_dam_energy_bid_awards_file(&self, file: &Path) -> Result<DataFrame> {
        // Force critical columns to be Float64
        let mut schema_overrides = Schema::new();
        
        // Key columns for BESS charging analysis
        schema_overrides.with_column("Energy Only Bid Award in MW".into(), DataType::Float64);
        schema_overrides.with_column("Settlement Point Price".into(), DataType::Float64);
        
        // Read CSV
        let df = CsvReader::from_path(file)?
            .has_header(true)
            .with_dtypes(Some(Arc::new(schema_overrides)))
            .infer_schema(None)
            .finish()?;
        
        // Parse dates and standardize columns
        let df = df.lazy()
            .with_column(
                // Parse Delivery Date - handle as string to date conversion
                col("Delivery Date")
                    .cast(DataType::String)
                    .alias("DeliveryDate")
            )
            .with_column(
                // Simple hour to datetime conversion
                col("Hour Ending").cast(DataType::Int32).alias("hour")
            )
            .with_column(
                // Rename for consistency
                col("Energy Only Bid Award in MW").alias("EnergyBidAwardMW")
            )
            .with_column(
                col("Settlement Point").alias("SettlementPoint")
            )
            .with_column(
                col("Settlement Point Price").alias("SettlementPointPrice")
            )
            .collect()?;
        
        Ok(df)
    }
    
    fn read_dam_load_file(&self, file: &Path) -> Result<DataFrame> {
        // Force critical numeric columns to be Float64
        let mut schema_overrides = Schema::new();
        
        // Load resource specific columns
        schema_overrides.with_column("Max Power Consumption for Load Resource".into(), DataType::Float64);
        schema_overrides.with_column("Low Power Consumption for Load Resource".into(), DataType::Float64);
        
        // Award columns - ALL must be Float64
        schema_overrides.with_column("Energy Bid Award".into(), DataType::Float64);
        schema_overrides.with_column("AS Physical Responsive Awards".into(), DataType::Float64);
        schema_overrides.with_column("RRS Awarded".into(), DataType::Float64);
        schema_overrides.with_column("RRSPFR Awarded".into(), DataType::Float64);
        schema_overrides.with_column("RRSFFR Awarded".into(), DataType::Float64);
        schema_overrides.with_column("RRSUFR Awarded".into(), DataType::Float64);
        schema_overrides.with_column("RegUp Awarded".into(), DataType::Float64);
        schema_overrides.with_column("RegDown Awarded".into(), DataType::Float64);
        schema_overrides.with_column("NonSpin Awarded".into(), DataType::Float64);
        schema_overrides.with_column("ECRS Awarded".into(), DataType::Float64);
        schema_overrides.with_column("ECRSSD Awarded".into(), DataType::Float64);
        schema_overrides.with_column("ECRSMD Awarded".into(), DataType::Float64);
        
        // MCPC columns - ALL must be Float64
        schema_overrides.with_column("RRS MCPC".into(), DataType::Float64);
        schema_overrides.with_column("RRSPFR MCPC".into(), DataType::Float64);
        schema_overrides.with_column("RRSFFR MCPC".into(), DataType::Float64);
        schema_overrides.with_column("RRSUFR MCPC".into(), DataType::Float64);
        schema_overrides.with_column("RegUp MCPC".into(), DataType::Float64);
        schema_overrides.with_column("RegDown MCPC".into(), DataType::Float64);
        schema_overrides.with_column("NonSpin MCPC".into(), DataType::Float64);
        schema_overrides.with_column("ECRS MCPC".into(), DataType::Float64);
        
        // Read CSV with explicit schema
        let df = CsvReader::from_path(file)?
            .has_header(true)
            .with_dtypes(Some(Arc::new(schema_overrides)))
            .infer_schema(None)  // Don't infer - use our explicit schema
            .finish()?;
        
        // Build dataframe with ALL available columns
        let df = df.lazy()
            .with_column(
                // Parse HourEnding as integer
                col("Hour Ending").cast(DataType::Int32).alias("hour")
            )
            .collect()?;
        
        // First rename Delivery Date to DeliveryDate
        let mut df = df.lazy()
            .with_column(
                col("Delivery Date").alias("DeliveryDate")
            )
            .collect()?;
        
        // Parse dates from MM/DD/YYYY format to Date32
        let mut parsed_dates = Vec::new();
        let mut date_strings = Vec::new();
        let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
        
        // Extract dates and process them
        {
            let delivery_dates = df.column("DeliveryDate")?;
            if let Ok(dates_str) = delivery_dates.str() {
                for i in 0..dates_str.len() {
                    if let Some(date_str) = dates_str.get(i) {
                        if let Ok(date) = NaiveDate::parse_from_str(date_str, "%m/%d/%Y") {
                            // Calculate days since Unix epoch (1970-01-01)
                            let days_since_epoch = (date - epoch).num_days() as i32;
                            parsed_dates.push(Some(days_since_epoch));
                            
                            // Format as ISO date with timezone indicator
                            let formatted = format!("{}T00:00:00-06:00", date.format("%Y-%m-%d"));
                            date_strings.push(Some(formatted));
                        } else {
                            parsed_dates.push(None);
                            date_strings.push(None);
                        }
                    } else {
                        parsed_dates.push(None);
                        date_strings.push(None);
                    }
                }
            }
        }
        
        // Create new Date column and replace the string one
        let date_series = Series::new("DeliveryDate", parsed_dates);
        df.with_column(date_series.cast(&DataType::Date)?)?;
        
        // Add string representation with timezone
        let date_str_series = Series::new("DeliveryDateStr", date_strings);
        df.with_column(date_str_series)?;
        
        // Create datetime column by combining date and hour
        let df = df.lazy()
            .with_column(
                // Hour Ending 1 = 00:00-01:00, so we subtract 1 hour
                (col("DeliveryDate").cast(DataType::Datetime(TimeUnit::Milliseconds, None)) +
                 duration_hours(col("hour") - lit(1)))
                .alias("datetime")
            )
            .collect()?;
        
        Ok(df)
    }
    
    fn normalize_dam_load_dataframes(&self, dfs: Vec<DataFrame>) -> Result<Vec<DataFrame>> {
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
                    select_exprs.push(col(col_name));
                } else {
                    // Create with appropriate default
                    if col_name.contains("Awarded") || col_name.contains("MCPC") || 
                       col_name.contains("Power") || col_name.contains("Consumption") {
                        select_exprs.push(lit(0.0f64).alias(col_name));
                    } else if col_name == "hour" || col_name.contains("Hour") {
                        select_exprs.push(lit(0i32).alias(col_name));
                    } else if col_name == "datetime" || col_name == "DeliveryDate" {
                        // Use existing datetime/date column
                        if existing_cols.contains(&"datetime") {
                            select_exprs.push(col("datetime").alias(col_name));
                        } else if existing_cols.contains(&"DeliveryDate") {
                            select_exprs.push(col("DeliveryDate").alias(col_name));
                        } else {
                            select_exprs.push(lit(NULL).alias(col_name));
                        }
                    } else {
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
    
    fn normalize_cop_dataframe(&self, mut df: DataFrame) -> Result<DataFrame> {
        // Normalize COP dataframes to have consistent columns
        // Evolution of COP file formats:
        // - 13 columns: Before Dec 13, 2022 (with single RRS column)
        // - 15 columns: Dec 13-31, 2022 (RRS split into RRSPFR/RRSFFR/RRSUFR)
        // - 16 columns: 2023 (added ECRS)
        // - 19 columns: 2024+ (added Minimum/Maximum/Hour Beginning Planned SOC)
        
        let height = df.height();
        let columns = df.get_column_names();
        let has_rrspfr = columns.contains(&"RRSPFR");
        let has_rrs = columns.contains(&"RRS");
        let has_ecrs = columns.contains(&"ECRS");
        let has_min_soc = columns.contains(&"Minimum SOC");
        let has_max_soc = columns.contains(&"Maximum SOC");
        let has_planned_soc = columns.contains(&"Hour Beginning Planned SOC");
        
        // If it's the old format with single RRS column, we need to:
        // 1. Copy RRS values to RRSPFR (as a reasonable default)
        // 2. Create RRSFFR and RRSUFR with nulls
        // 3. Remove the original RRS column
        if !has_rrspfr && has_rrs {
            // Get RRS values to copy to RRSPFR
            let rrs_col = df.column("RRS")?;
            df.with_column(rrs_col.clone().with_name("RRSPFR"))?;
            df.with_column(Series::new("RRSFFR", vec![None::<f64>; height]))?;
            df.with_column(Series::new("RRSUFR", vec![None::<f64>; height]))?;
            
            // Remove the original RRS column
            let cols_to_keep: Vec<&str> = df.get_column_names()
                .into_iter()
                .filter(|&name| name != "RRS")
                .collect();
            df = df.select(cols_to_keep)?;
        }
        
        // Add ECRS column if missing (added in 2023)
        if !has_ecrs {
            df.with_column(Series::new("ECRS", vec![None::<f64>; height]))?;
        }
        
        // Add SOC columns if missing (added in 2024)
        if !has_min_soc {
            df.with_column(Series::new("Minimum SOC", vec![None::<f64>; height]))?;
        }
        if !has_max_soc {
            df.with_column(Series::new("Maximum SOC", vec![None::<f64>; height]))?;
        }
        if !has_planned_soc {
            df.with_column(Series::new("Hour Beginning Planned SOC", vec![None::<f64>; height]))?;
        }
        
        // Ensure consistent column ordering
        let standard_columns = vec![
            "Delivery Date", "QSE Name", "Resource Name", "Hour Ending", "Status",
            "High Sustained Limit", "Low Sustained Limit", "High Emergency Limit", "Low Emergency Limit",
            "Reg Up", "Reg Down", "RRSPFR", "RRSFFR", "RRSUFR", "NSPIN", "ECRS",
            "Minimum SOC", "Maximum SOC", "Hour Beginning Planned SOC"
        ];
        
        // Select columns in the standard order
        let available_cols: Vec<&str> = standard_columns.iter()
            .filter(|&&col| df.get_column_names().contains(&col))
            .copied()
            .collect();
            
        df = df.select(available_cols)?;
        
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
                    Ok(df) => {
                        // Normalize to handle schema evolution
                        match self.normalize_cop_dataframe(df) {
                            Ok(normalized) => all_dfs.push(normalized),
                            Err(e) => eprintln!("    Error normalizing {}: {}", file.display(), e),
                        }
                    },
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
        // Use the robust COP file reader that handles all format variations
        // This handles late 2014 files without headers, schema evolution, etc.
        crate::cop_file_reader::read_cop_file(file)
    }
    
    fn _read_csv_with_schema_detection(&self, file: &Path) -> Result<DataFrame> {
        // Step 1: Read a sample to detect the actual columns
        let sample_df = CsvReader::from_path(file)?
            .has_header(true)
            .infer_schema(Some(2000))  // Sample enough rows to understand the data
            .finish()?;
        
        // Step 2: Build intelligent schema based on column name patterns
        let mut schema_overrides = Schema::new();
        
        // Define patterns for different data types
        let text_patterns = vec![
            "Date", "Hour", "Name", "Type", "QSE", "Settlement", "Status", 
            "Resource Name", "Load Resource", "Gen Resource", "Point"
        ];
        
        let numeric_patterns = vec![
            "Awarded", "MCPC", "Power", "Consumption", "MW", "Price", "Curve",
            "LSL", "HSL", "LDL", "HDL", "Base Point", "Output", "Telemetered",
            "Limit", "Schedule", "Bid", "Offer", "Quantity", "Capacity",
            "AS ", "RegUp", "RegDown", "RRS", "ECRS", "NonSpin", "NSRS"
        ];
        
        for col_name in sample_df.get_column_names() {
            let col_str = col_name.to_string();
            let col_lower = col_str.to_lowercase();
            
            // Check if it's definitely a text column
            let is_text = text_patterns.iter().any(|pattern| {
                col_lower.contains(&pattern.to_lowercase())
            }) && !numeric_patterns.iter().any(|pattern| {
                col_lower.contains(&pattern.to_lowercase())
            });
            
            // Check if it's definitely a numeric column  
            let is_numeric = numeric_patterns.iter().any(|pattern| {
                col_lower.contains(&pattern.to_lowercase())
            });
            
            let dtype = if is_text {
                DataType::String
            } else if is_numeric {
                DataType::Float64
            } else {
                // For unknown columns, check the actual data in the sample
                match sample_df.column(col_name) {
                    Ok(col) => {
                        // Check if the column contains numeric-looking data
                        match col.dtype() {
                            DataType::Int64 | DataType::Float64 | 
                            DataType::Int32 | DataType::Float32 => DataType::Float64,
                            _ => DataType::String,
                        }
                    },
                    Err(_) => DataType::String,  // Default to string if we can't determine
                }
            };
            
            schema_overrides.with_column(col_str.into(), dtype);
        }
        
        // Step 3: Define comprehensive null values
        let null_values = NullValues::AllColumns(vec![
            "".to_string(),
            "NA".to_string(), 
            "N/A".to_string(),
            "null".to_string(),
            "NULL".to_string(),
            "#N/A".to_string(),
            "NaN".to_string(),
        ]);
        
        // Step 4: Read the file with the determined schema
        let df = CsvReader::from_path(file)?
            .has_header(true)
            .with_dtypes(Some(Arc::new(schema_overrides)))
            .infer_schema(None)  // Don't infer - use our schema
            .with_null_values(Some(null_values))
            .finish()?;
        
        Ok(df)
    }
    
    fn combine_dataframes(&self, dfs: Vec<DataFrame>) -> Result<DataFrame> {
        if dfs.is_empty() {
            return Err(anyhow::anyhow!("No dataframes to combine"));
        }
        
        // For small numbers of dataframes, use the simple approach
        if dfs.len() <= 10 {
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
            
            return Ok(combined);
        }
        
        // For large numbers of dataframes, use parallel tree reduction
        // Don't print for every batch combination - too verbose
        
        // Convert to Arc<Mutex<>> for thread-safe access
        let mut work_queue: Vec<DataFrame> = dfs;
        
        // Tree reduction - combine pairs in parallel until we have one
        while work_queue.len() > 1 {
            // Process pairs in parallel
            let paired_results: Vec<DataFrame> = work_queue
                .par_chunks(2)
                .map(|pair| {
                    if pair.len() == 2 {
                        // Combine two dataframes
                        pair[0].vstack(&pair[1])
                    } else {
                        // Odd one out, just return it
                        Ok(pair[0].clone())
                    }
                })
                .collect::<Result<Vec<_>, _>>()?;
            
            work_queue = paired_results;
        }
        
        let mut combined = work_queue.into_iter().next()
            .ok_or_else(|| anyhow::anyhow!("Failed to combine dataframes"))?;
        
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
    DAMLoadResource,
    DAMEnergyBidAwards,
    SCEDGenResource,
    SCEDLoadResource,
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