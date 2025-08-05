use anyhow::Result;
use chrono::{NaiveDate, NaiveDateTime, NaiveTime};
use glob::glob;
use indicatif::{ProgressBar, ProgressStyle};
use polars::prelude::*;
use rayon::prelude::*;
use serde_json::json;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

pub struct ErcotPriceProcessor {
    base_path: PathBuf,
    output_path: PathBuf,
}

impl ErcotPriceProcessor {
    pub fn new(base_path: PathBuf) -> Self {
        let output_path = base_path.join("rollup_files");
        Self {
            base_path,
            output_path,
        }
    }

    pub fn process_all(&self) -> Result<()> {
        println!("\n📊 ERCOT Price Data Processor");
        println!("{}", "=".repeat(60));
        println!("Base path: {}", self.base_path.display());
        println!("Output path: {}", self.output_path.display());

        // Create output directories
        self.create_output_directories()?;

        // Process each price type
        self.process_realtime_prices()?;
        self.process_dayahead_prices()?;
        self.process_ancillary_prices()?;

        println!("\n✅ All price data processed successfully!");
        Ok(())
    }

    fn create_output_directories(&self) -> Result<()> {
        let dirs = ["RT_prices", "DA_prices", "AS_prices"];
        for dir in &dirs {
            let path = self.output_path.join(dir);
            fs::create_dir_all(&path)?;
            println!("📁 Created directory: {}", path.display());
        }
        Ok(())
    }

    pub fn process_realtime_prices(&self) -> Result<()> {
        println!("\n🔄 Processing Real-Time Prices...");
        
        let source_dir = self.base_path.join("Settlement_Point_Prices_at_Resource_Nodes,_Hubs_and_Load_Zones/csv");
        let output_dir = self.output_path.join("RT_prices");
        
        // Find all RT price CSV files
        let pattern = format!("{}/*.csv", source_dir.display());
        let files: Vec<PathBuf> = glob(&pattern)?
            .filter_map(Result::ok)
            .collect();

        println!("Found {} RT price files", files.len());

        // Group files by year
        let files_by_year = self.group_files_by_year(&files)?;

        // Process each year in parallel
        let pb = ProgressBar::new(files_by_year.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
                .unwrap()
                .progress_chars("##-"),
        );

        files_by_year.par_iter().for_each(|(year, year_files)| {
            pb.set_message(format!("Processing RT prices for {}", year));
            if let Err(e) = self.process_rt_year(*year, year_files, &output_dir) {
                eprintln!("Error processing RT year {}: {}", year, e);
            }
            pb.inc(1);
        });

        pb.finish_with_message("RT prices processed");

        // Create schema JSON
        self.create_rt_schema(&output_dir)?;

        Ok(())
    }

    fn process_rt_year(&self, year: i32, files: &[PathBuf], output_dir: &Path) -> Result<()> {
        let mut all_data = Vec::new();

        for file in files {
            match self.read_rt_file(file) {
                Ok(df) => all_data.push(df),
                Err(e) => eprintln!("Error reading RT file {:?}: {}", file, e),
            }
        }

        if all_data.is_empty() {
            return Ok(());
        }

        // Concatenate all dataframes
        let lazy_frames: Vec<LazyFrame> = all_data.into_iter()
            .map(|df| df.lazy())
            .collect();
        let df = concat(&lazy_frames, UnionArgs::default())?.collect()?;

        // Save to parquet
        let output_file = output_dir.join(format!("{}.parquet", year));
        let mut file = std::fs::File::create(&output_file)?;
        ParquetWriter::new(&mut file).finish(&mut df.clone())?;
        
        println!("✅ Saved RT prices for {} to {}", year, output_file.display());

        Ok(())
    }

    fn read_rt_file(&self, file: &Path) -> Result<DataFrame> {
        let df = CsvReader::from_path(file)?
            .has_header(true)
            .finish()?;

        // Get columns as series
        let delivery_date_str = df.column("DeliveryDate")?;
        let hour = df.column("DeliveryHour")?.cast(&DataType::UInt32)?;
        let interval = df.column("DeliveryInterval")?.cast(&DataType::UInt32)?;

        // Parse dates and create datetime column
        let mut datetimes = Vec::new();
        
        let date_str_series = delivery_date_str.cast(&DataType::Utf8)?;
        let date_str_ca = date_str_series.utf8()?;
        let hour_ca = hour.u32()?;
        let interval_ca = interval.u32()?;
        
        for i in 0..df.height() {
            if let Some(date_str) = date_str_ca.get(i) {
                if let Ok(date) = NaiveDate::parse_from_str(date_str, "%m/%d/%Y") {
                    let h = hour_ca.get(i).unwrap_or(1);
                    let int = interval_ca.get(i).unwrap_or(1);
                    let minutes = (int - 1) * 15;
                    
                    if let Some(time) = NaiveTime::from_hms_opt((h - 1) as u32, minutes, 0) {
                        let dt = NaiveDateTime::new(date, time);
                        datetimes.push(Some(dt.and_utc().timestamp_millis()));
                    } else {
                        datetimes.push(None);
                    }
                } else {
                    datetimes.push(None);
                }
            } else {
                datetimes.push(None);
            }
        }

        // Force price column to Float64
        let price_col = df.column("SettlementPointPrice")?
            .cast(&DataType::Float64)?;

        // Build final dataframe
        let mut final_df = df.clone();
        final_df.with_column(Series::new("datetime", datetimes))?;
        final_df.replace("SettlementPointPrice", price_col)?;

        Ok(final_df)
    }

    pub fn process_dayahead_prices(&self) -> Result<()> {
        println!("\n🔄 Processing Day-Ahead Prices...");
        
        let source_dir = self.base_path.join("DAM_Settlement_Point_Prices/csv");
        let output_dir = self.output_path.join("DA_prices");
        
        // Find all DA price CSV files
        let pattern = format!("{}/*.csv", source_dir.display());
        let files: Vec<PathBuf> = glob(&pattern)?
            .filter_map(Result::ok)
            .collect();

        println!("Found {} DA price files", files.len());

        // Group files by year
        let files_by_year = self.group_files_by_year(&files)?;

        // Process each year
        let pb = ProgressBar::new(files_by_year.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
                .unwrap()
                .progress_chars("##-"),
        );

        files_by_year.par_iter().for_each(|(year, year_files)| {
            pb.set_message(format!("Processing DA prices for {}", year));
            if let Err(e) = self.process_da_year(*year, year_files, &output_dir) {
                eprintln!("Error processing DA year {}: {}", year, e);
            }
            pb.inc(1);
        });

        pb.finish_with_message("DA prices processed");

        // Create schema JSON
        self.create_da_schema(&output_dir)?;

        Ok(())
    }

    fn process_da_year(&self, year: i32, files: &[PathBuf], output_dir: &Path) -> Result<()> {
        let mut all_data = Vec::new();

        for file in files {
            match self.read_da_file(file) {
                Ok(df) => all_data.push(df),
                Err(e) => eprintln!("Error reading DA file {:?}: {}", file, e),
            }
        }

        if all_data.is_empty() {
            return Ok(());
        }

        // Concatenate all dataframes
        let lazy_frames: Vec<LazyFrame> = all_data.into_iter()
            .map(|df| df.lazy())
            .collect();
        let df = concat(&lazy_frames, UnionArgs::default())?.collect()?;

        // Save to parquet
        let output_file = output_dir.join(format!("{}.parquet", year));
        let mut file = std::fs::File::create(&output_file)?;
        ParquetWriter::new(&mut file).finish(&mut df.clone())?;
        
        println!("✅ Saved DA prices for {} to {}", year, output_file.display());

        Ok(())
    }

    fn read_da_file(&self, file: &Path) -> Result<DataFrame> {
        let df = CsvReader::from_path(file)?
            .has_header(true)
            .finish()?;

        // Get columns as series
        let delivery_date_str = df.column("DeliveryDate")?;
        let hour_ending_str = df.column("HourEnding")?;

        // Parse dates and create datetime column
        let mut datetimes = Vec::new();
        let date_str_series = delivery_date_str.cast(&DataType::Utf8)?;
        let date_str_ca = date_str_series.utf8()?;
        let hour_str_series = hour_ending_str.cast(&DataType::Utf8)?;
        let hour_str_ca = hour_str_series.utf8()?;
        
        for i in 0..df.height() {
            if let (Some(date_str), Some(hour_str)) = (
                date_str_ca.get(i),
                hour_str_ca.get(i)
            ) {
                if let Ok(date) = NaiveDate::parse_from_str(date_str, "%m/%d/%Y") {
                    // Parse hour from "HH:00" format
                    let hour = hour_str.split(':')
                        .next()
                        .and_then(|h| h.parse::<u32>().ok())
                        .unwrap_or(1);
                    
                    if let Some(time) = NaiveTime::from_hms_opt((hour - 1) as u32, 0, 0) {
                        let dt = NaiveDateTime::new(date, time);
                        datetimes.push(Some(dt.and_utc().timestamp_millis()));
                    } else {
                        datetimes.push(None);
                    }
                } else {
                    datetimes.push(None);
                }
            } else {
                datetimes.push(None);
            }
        }

        // Force price column to Float64
        let price_col = df.column("SettlementPointPrice")?
            .cast(&DataType::Float64)?;

        // Build final dataframe
        let mut final_df = df.clone();
        final_df.with_column(Series::new("datetime", datetimes))?;
        final_df.replace("SettlementPointPrice", price_col)?;

        Ok(final_df)
    }

    pub fn process_ancillary_prices(&self) -> Result<()> {
        println!("\n🔄 Processing Ancillary Services Prices...");
        
        let source_dir = self.base_path.join("DAM_Clearing_Prices_for_Capacity/csv");
        let output_dir = self.output_path.join("AS_prices");
        
        // Find all AS price CSV files
        let pattern = format!("{}/*.csv", source_dir.display());
        let files: Vec<PathBuf> = glob(&pattern)?
            .filter_map(Result::ok)
            .collect();

        println!("Found {} AS price files", files.len());

        // Group files by year
        let files_by_year = self.group_files_by_year(&files)?;

        // Process each year
        let pb = ProgressBar::new(files_by_year.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
                .unwrap()
                .progress_chars("##-"),
        );

        files_by_year.par_iter().for_each(|(year, year_files)| {
            pb.set_message(format!("Processing AS prices for {}", year));
            if let Err(e) = self.process_as_year(*year, year_files, &output_dir) {
                eprintln!("Error processing AS year {}: {}", year, e);
            }
            pb.inc(1);
        });

        pb.finish_with_message("AS prices processed");

        // Create schema JSON
        self.create_as_schema(&output_dir)?;

        Ok(())
    }

    fn process_as_year(&self, year: i32, files: &[PathBuf], output_dir: &Path) -> Result<()> {
        let mut all_data = Vec::new();

        for file in files {
            match self.read_as_file(file) {
                Ok(df) => all_data.push(df),
                Err(e) => eprintln!("Error reading AS file {:?}: {}", file, e),
            }
        }

        if all_data.is_empty() {
            return Ok(());
        }

        // Concatenate all dataframes
        let lazy_frames: Vec<LazyFrame> = all_data.into_iter()
            .map(|df| df.lazy())
            .collect();
        let df = concat(&lazy_frames, UnionArgs::default())?.collect()?;

        // Save to parquet
        let output_file = output_dir.join(format!("{}.parquet", year));
        let mut file = std::fs::File::create(&output_file)?;
        ParquetWriter::new(&mut file).finish(&mut df.clone())?;
        
        println!("✅ Saved AS prices for {} to {}", year, output_file.display());

        Ok(())
    }

    fn read_as_file(&self, file: &Path) -> Result<DataFrame> {
        let df = CsvReader::from_path(file)?
            .has_header(true)
            .finish()?;

        // Get columns as series
        let delivery_date_str = df.column("DeliveryDate")?;
        let hour_ending_str = df.column("HourEnding")?;

        // Parse dates and create datetime column
        let mut datetimes = Vec::new();
        let date_str_series = delivery_date_str.cast(&DataType::Utf8)?;
        let date_str_ca = date_str_series.utf8()?;
        let hour_str_series = hour_ending_str.cast(&DataType::Utf8)?;
        let hour_str_ca = hour_str_series.utf8()?;
        
        for i in 0..df.height() {
            if let (Some(date_str), Some(hour_str)) = (
                date_str_ca.get(i),
                hour_str_ca.get(i)
            ) {
                if let Ok(date) = NaiveDate::parse_from_str(date_str, "%m/%d/%Y") {
                    // Parse hour from "HH:00" format
                    let hour = hour_str.split(':')
                        .next()
                        .and_then(|h| h.parse::<u32>().ok())
                        .unwrap_or(1);
                    
                    if let Some(time) = NaiveTime::from_hms_opt((hour - 1) as u32, 0, 0) {
                        let dt = NaiveDateTime::new(date, time);
                        datetimes.push(Some(dt.and_utc().timestamp_millis()));
                    } else {
                        datetimes.push(None);
                    }
                } else {
                    datetimes.push(None);
                }
            } else {
                datetimes.push(None);
            }
        }

        // Force MCPC column to Float64
        let mcpc_col = df.column("MCPC")?
            .cast(&DataType::Float64)?;

        // Build final dataframe
        let mut final_df = df.clone();
        final_df.with_column(Series::new("datetime", datetimes))?;
        final_df.replace("MCPC", mcpc_col)?;

        Ok(final_df)
    }

    fn group_files_by_year(&self, files: &[PathBuf]) -> Result<HashMap<i32, Vec<PathBuf>>> {
        let mut files_by_year: HashMap<i32, Vec<PathBuf>> = HashMap::new();

        for file in files {
            // Extract year from filename (format: YYYYMMDD)
            if let Some(filename) = file.file_name().and_then(|f| f.to_str()) {
                // Find the date part in the filename
                let parts: Vec<&str> = filename.split('.').collect();
                for part in parts {
                    if part.len() == 8 && part.chars().all(|c| c.is_numeric()) {
                        if let Ok(year) = part[0..4].parse::<i32>() {
                            files_by_year.entry(year).or_insert_with(Vec::new).push(file.clone());
                            break;
                        }
                    }
                }
            }
        }

        Ok(files_by_year)
    }

    fn create_rt_schema(&self, output_dir: &Path) -> Result<()> {
        let schema = json!({
            "description": "Real-time settlement point prices at 15-minute intervals",
            "columns": {
                "datetime": {
                    "type": "Datetime",
                    "description": "Timestamp of the price interval"
                },
                "DeliveryDate": {
                    "type": "Utf8",
                    "description": "Operating day (MM/DD/YYYY)"
                },
                "DeliveryHour": {
                    "type": "UInt32",
                    "description": "Hour of the day (1-24)"
                },
                "DeliveryInterval": {
                    "type": "UInt32",
                    "description": "15-minute interval within hour (1-4)"
                },
                "SettlementPointName": {
                    "type": "Utf8",
                    "description": "Name of the settlement point"
                },
                "SettlementPointType": {
                    "type": "Utf8",
                    "description": "Type: RN (Resource Node), HB (Hub), LZ (Load Zone)"
                },
                "SettlementPointPrice": {
                    "type": "Float64",
                    "description": "Price in $/MWh (includes scarcity adders)"
                },
                "DSTFlag": {
                    "type": "Utf8",
                    "description": "Daylight Saving Time flag (Y/N)"
                }
            }
        });

        let schema_file = output_dir.join("schema.json");
        fs::write(&schema_file, serde_json::to_string_pretty(&schema)?)?;
        println!("📄 Created schema file: {}", schema_file.display());

        Ok(())
    }

    fn create_da_schema(&self, output_dir: &Path) -> Result<()> {
        let schema = json!({
            "description": "Day-ahead market settlement point prices (hourly)",
            "columns": {
                "datetime": {
                    "type": "Datetime",
                    "description": "Timestamp of the price hour"
                },
                "DeliveryDate": {
                    "type": "Utf8",
                    "description": "Operating day (MM/DD/YYYY)"
                },
                "HourEnding": {
                    "type": "Utf8",
                    "description": "Hour ending time (01:00 - 24:00)"
                },
                "SettlementPoint": {
                    "type": "Utf8",
                    "description": "Settlement point name"
                },
                "SettlementPointPrice": {
                    "type": "Float64",
                    "description": "Price in $/MWh (includes scarcity adders)"
                },
                "DSTFlag": {
                    "type": "Utf8",
                    "description": "Daylight Saving Time flag (Y/N)"
                }
            }
        });

        let schema_file = output_dir.join("schema.json");
        fs::write(&schema_file, serde_json::to_string_pretty(&schema)?)?;
        println!("📄 Created schema file: {}", schema_file.display());

        Ok(())
    }

    fn create_as_schema(&self, output_dir: &Path) -> Result<()> {
        let schema = json!({
            "description": "Day-ahead ancillary services clearing prices",
            "columns": {
                "datetime": {
                    "type": "Datetime",
                    "description": "Timestamp of the price hour"
                },
                "DeliveryDate": {
                    "type": "Utf8",
                    "description": "Operating day (MM/DD/YYYY)"
                },
                "HourEnding": {
                    "type": "Utf8",
                    "description": "Hour ending time (01:00 - 24:00)"
                },
                "AncillaryType": {
                    "type": "Utf8",
                    "description": "Type of ancillary service (REGUP, REGDN, RRS, ECRS, NSPIN)"
                },
                "MCPC": {
                    "type": "Float64",
                    "description": "Market Clearing Price for Capacity in $/MW"
                },
                "DSTFlag": {
                    "type": "Utf8",
                    "description": "Daylight Saving Time flag (Y/N)"
                }
            }
        });

        let schema_file = output_dir.join("schema.json");
        fs::write(&schema_file, serde_json::to_string_pretty(&schema)?)?;
        println!("📄 Created schema file: {}", schema_file.display());

        Ok(())
    }
}