use anyhow::Result;
use chrono::{DateTime, NaiveDateTime};
use polars::prelude::*;
use std::path::{Path, PathBuf};

pub struct GapAnalyzer {
    data_path: PathBuf,
}

#[derive(Debug, Clone)]
pub struct DataGap {
    pub dataset: String,
    pub year: i32,
    pub gap_start: NaiveDateTime,
    pub gap_end: NaiveDateTime,
    pub duration_hours: f64,
    pub missing_records: i64,
}

#[derive(Debug)]
pub struct DatasetSummary {
    pub dataset_name: String,
    pub years_analyzed: Vec<i32>,
    pub total_gaps: usize,
    pub total_missing_hours: f64,
    pub largest_gap_hours: f64,
    pub data_completeness_pct: f64,
}

impl GapAnalyzer {
    pub fn new(data_path: PathBuf) -> Self {
        Self { data_path }
    }

    pub fn analyze_all_datasets(&self) -> Result<()> {
        println!("\n🔍 ERCOT Data Gap Analysis");
        println!("{}", "=".repeat(60));
        
        let rt_summary = self.analyze_rt_gaps()?;
        let da_summary = self.analyze_da_gaps()?;
        let as_summary = self.analyze_as_gaps()?;
        
        self.print_summary(&[rt_summary, da_summary, as_summary]);
        
        Ok(())
    }

    pub fn analyze_rt_gaps(&self) -> Result<DatasetSummary> {
        println!("\n📊 Analyzing Real-Time Price Gaps (15-minute intervals)");
        
        let rt_dir = self.data_path.join("rollup_files/RT_prices");
        let files = self.get_parquet_files(&rt_dir)?;
        
        let mut all_gaps = Vec::new();
        let mut years_analyzed = Vec::new();
        let mut total_expected_records = 0i64;
        let mut total_actual_records = 0i64;
        
        for file in files {
            if let Some(year) = self.extract_year_from_filename(&file) {
                years_analyzed.push(year);
                println!("  Checking {} RT data...", year);
                
                let df = LazyFrame::scan_parquet(&file, ScanArgsParquet::default())?
                    .collect()?;
                
                let actual_records = df.height() as i64;
                
                // Get unique settlement points to calculate expected records correctly
                let settlement_points = df.column("SettlementPointName")?
                    .unique()?
                    .len() as i64;
                
                total_actual_records += actual_records;
                
                // Expected records: 4 intervals per hour * 24 hours * days in year * settlement points
                let days_in_year = if self.is_leap_year(year) { 366 } else { 365 };
                let expected_records = 4 * 24 * days_in_year * settlement_points;
                total_expected_records += expected_records;
                
                // Analyze temporal gaps (sample a few settlement points)
                let gaps = self.find_rt_temporal_gaps(&df, year)?;
                println!("    {} settlement points, {}/{} records ({:.2}% complete)", 
                    settlement_points, actual_records, expected_records,
                    (actual_records as f64 / expected_records as f64) * 100.0);
                
                all_gaps.extend(gaps);
            }
        }
        
        let total_gap_hours: f64 = all_gaps.iter().map(|g| g.duration_hours).sum();
        let largest_gap = all_gaps.iter()
            .max_by(|a, b| a.duration_hours.partial_cmp(&b.duration_hours).unwrap())
            .map(|g| g.duration_hours)
            .unwrap_or(0.0);
        
        let completeness = (total_actual_records as f64 / total_expected_records as f64) * 100.0;
        
        Ok(DatasetSummary {
            dataset_name: "Real-Time Prices".to_string(),
            years_analyzed,
            total_gaps: all_gaps.len(),
            total_missing_hours: total_gap_hours,
            largest_gap_hours: largest_gap,
            data_completeness_pct: completeness,
        })
    }

    pub fn analyze_da_gaps(&self) -> Result<DatasetSummary> {
        println!("\n📊 Analyzing Day-Ahead Price Gaps (hourly intervals)");
        
        let da_dir = self.data_path.join("rollup_files/DA_prices");
        let files = self.get_parquet_files(&da_dir)?;
        
        let mut all_gaps = Vec::new();
        let mut years_analyzed = Vec::new();
        let mut total_expected_records = 0i64;
        let mut total_actual_records = 0i64;
        
        for file in files {
            if let Some(year) = self.extract_year_from_filename(&file) {
                years_analyzed.push(year);
                println!("  Checking {} DA data...", year);
                
                let df = LazyFrame::scan_parquet(&file, ScanArgsParquet::default())?
                    .collect()?;
                
                let actual_records = df.height() as i64;
                
                // Get unique settlement points to calculate expected records correctly
                let settlement_points = df.column("SettlementPoint")?
                    .unique()?
                    .len() as i64;
                
                total_actual_records += actual_records;
                
                // Expected records: 24 hours * days in year * settlement points
                let days_in_year = if self.is_leap_year(year) { 366 } else { 365 };
                let expected_records = 24 * days_in_year * settlement_points;
                total_expected_records += expected_records;
                
                // Analyze temporal gaps
                let gaps = self.find_da_temporal_gaps(&df, year)?;
                println!("    {} settlement points, {}/{} records ({:.2}% complete)", 
                    settlement_points, actual_records, expected_records,
                    (actual_records as f64 / expected_records as f64) * 100.0);
                
                all_gaps.extend(gaps);
            }
        }
        
        let total_gap_hours: f64 = all_gaps.iter().map(|g| g.duration_hours).sum();
        let largest_gap = all_gaps.iter()
            .max_by(|a, b| a.duration_hours.partial_cmp(&b.duration_hours).unwrap())
            .map(|g| g.duration_hours)
            .unwrap_or(0.0);
        
        let completeness = (total_actual_records as f64 / total_expected_records as f64) * 100.0;
        
        Ok(DatasetSummary {
            dataset_name: "Day-Ahead Prices".to_string(),
            years_analyzed,
            total_gaps: all_gaps.len(),
            total_missing_hours: total_gap_hours,
            largest_gap_hours: largest_gap,
            data_completeness_pct: completeness,
        })
    }

    pub fn analyze_as_gaps(&self) -> Result<DatasetSummary> {
        println!("\n📊 Analyzing Ancillary Services Price Gaps (hourly intervals)");
        
        let as_dir = self.data_path.join("rollup_files/AS_prices");
        let files = self.get_parquet_files(&as_dir)?;
        
        let mut all_gaps = Vec::new();
        let mut years_analyzed = Vec::new();
        let mut total_expected_records = 0i64;
        let mut total_actual_records = 0i64;
        
        for file in files {
            if let Some(year) = self.extract_year_from_filename(&file) {
                years_analyzed.push(year);
                println!("  Checking {} AS data...", year);
                
                let df = LazyFrame::scan_parquet(&file, ScanArgsParquet::default())?
                    .collect()?;
                
                let actual_records = df.height() as i64;
                
                // Get unique service types to calculate expected records correctly
                let service_types = df.column("AncillaryType")?
                    .unique()?
                    .len() as i64;
                
                total_actual_records += actual_records;
                
                // Expected records: service_types * 24 hours * days in year
                let days_in_year = if self.is_leap_year(year) { 366 } else { 365 };
                let expected_records = service_types * 24 * days_in_year;
                total_expected_records += expected_records;
                
                // Analyze temporal gaps
                let gaps = self.find_as_temporal_gaps(&df, year)?;
                println!("    {} service types, {}/{} records ({:.2}% complete)", 
                    service_types, actual_records, expected_records,
                    (actual_records as f64 / expected_records as f64) * 100.0);
                
                all_gaps.extend(gaps);
            }
        }
        
        let total_gap_hours: f64 = all_gaps.iter().map(|g| g.duration_hours).sum();
        let largest_gap = all_gaps.iter()
            .max_by(|a, b| a.duration_hours.partial_cmp(&b.duration_hours).unwrap())
            .map(|g| g.duration_hours)
            .unwrap_or(0.0);
        
        let completeness = (total_actual_records as f64 / total_expected_records as f64) * 100.0;
        
        Ok(DatasetSummary {
            dataset_name: "Ancillary Services".to_string(),
            years_analyzed,
            total_gaps: all_gaps.len(),
            total_missing_hours: total_gap_hours,
            largest_gap_hours: largest_gap,
            data_completeness_pct: completeness,
        })
    }

    fn find_rt_temporal_gaps(&self, df: &DataFrame, year: i32) -> Result<Vec<DataGap>> {
        let mut gaps = Vec::new();
        
        // Get unique datetime values and sort them
        let datetime_col = df.column("datetime")?;
        let mut datetimes: Vec<i64> = datetime_col
            .cast(&DataType::Int64)?
            .i64()?
            .into_no_null_iter()
            .collect();
        datetimes.sort();
        
        // Check for gaps (expecting 15-minute intervals = 900,000 ms)
        let interval_ms = 15 * 60 * 1000; // 15 minutes in milliseconds
        
        for window in datetimes.windows(2) {
            let current = window[0];
            let next = window[1];
            let gap_ms = next - current;
            
            if gap_ms > interval_ms {
                let gap_start = DateTime::from_timestamp_millis(current).unwrap().naive_utc();
                let gap_end = DateTime::from_timestamp_millis(next).unwrap().naive_utc();
                let duration_hours = (gap_ms as f64) / (1000.0 * 3600.0);
                let missing_records = (gap_ms / interval_ms) - 1;
                
                gaps.push(DataGap {
                    dataset: "RT".to_string(),
                    year,
                    gap_start,
                    gap_end,
                    duration_hours,
                    missing_records,
                });
            }
        }
        
        Ok(gaps)
    }

    fn find_da_temporal_gaps(&self, df: &DataFrame, year: i32) -> Result<Vec<DataGap>> {
        let mut gaps = Vec::new();
        
        // Get unique datetime values and sort them
        let datetime_col = df.column("datetime")?;
        let mut datetimes: Vec<i64> = datetime_col
            .cast(&DataType::Int64)?
            .i64()?
            .into_no_null_iter()
            .collect();
        datetimes.sort();
        datetimes.dedup(); // Remove duplicates from multiple settlement points
        
        // Check for gaps (expecting hourly intervals = 3,600,000 ms)
        let interval_ms = 60 * 60 * 1000; // 1 hour in milliseconds
        
        for window in datetimes.windows(2) {
            let current = window[0];
            let next = window[1];
            let gap_ms = next - current;
            
            if gap_ms > interval_ms {
                let gap_start = DateTime::from_timestamp_millis(current).unwrap().naive_utc();
                let gap_end = DateTime::from_timestamp_millis(next).unwrap().naive_utc();
                let duration_hours = (gap_ms as f64) / (1000.0 * 3600.0);
                let missing_records = (gap_ms / interval_ms) - 1;
                
                gaps.push(DataGap {
                    dataset: "DA".to_string(),
                    year,
                    gap_start,
                    gap_end,
                    duration_hours,
                    missing_records,
                });
            }
        }
        
        Ok(gaps)
    }

    fn find_as_temporal_gaps(&self, df: &DataFrame, year: i32) -> Result<Vec<DataGap>> {
        let mut gaps = Vec::new();
        
        // Get unique datetime values and sort them
        let datetime_col = df.column("datetime")?;
        let mut datetimes: Vec<i64> = datetime_col
            .cast(&DataType::Int64)?
            .i64()?
            .into_no_null_iter()
            .collect();
        datetimes.sort();
        datetimes.dedup(); // Remove duplicates from multiple service types
        
        // Check for gaps (expecting hourly intervals = 3,600,000 ms)
        let interval_ms = 60 * 60 * 1000; // 1 hour in milliseconds
        
        for window in datetimes.windows(2) {
            let current = window[0];
            let next = window[1];
            let gap_ms = next - current;
            
            if gap_ms > interval_ms {
                let gap_start = DateTime::from_timestamp_millis(current).unwrap().naive_utc();
                let gap_end = DateTime::from_timestamp_millis(next).unwrap().naive_utc();
                let duration_hours = (gap_ms as f64) / (1000.0 * 3600.0);
                let missing_records = (gap_ms / interval_ms) - 1;
                
                gaps.push(DataGap {
                    dataset: "AS".to_string(),
                    year,
                    gap_start,
                    gap_end,
                    duration_hours,
                    missing_records,
                });
            }
        }
        
        Ok(gaps)
    }

    fn get_parquet_files(&self, dir: &Path) -> Result<Vec<PathBuf>> {
        let mut files = Vec::new();
        
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            
            if path.extension().and_then(|s| s.to_str()) == Some("parquet") {
                files.push(path);
            }
        }
        
        files.sort();
        Ok(files)
    }

    fn extract_year_from_filename(&self, path: &Path) -> Option<i32> {
        path.file_stem()
            .and_then(|s| s.to_str())
            .and_then(|s| s.parse::<i32>().ok())
    }

    fn is_leap_year(&self, year: i32) -> bool {
        (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
    }

    fn print_summary(&self, summaries: &[DatasetSummary]) {
        println!("\n📋 DATA COMPLETENESS SUMMARY");
        println!("{}", "=".repeat(80));
        
        for summary in summaries {
            println!("\n{}", summary.dataset_name);
            println!("{}", "-".repeat(summary.dataset_name.len()));
            println!("Years analyzed: {:?}", summary.years_analyzed);
            println!("Total gaps found: {}", summary.total_gaps);
            println!("Total missing hours: {:.1}", summary.total_missing_hours);
            println!("Largest gap: {:.1} hours", summary.largest_gap_hours);
            println!("Data completeness: {:.2}%", summary.data_completeness_pct);
            
            if summary.data_completeness_pct < 95.0 {
                println!("⚠️  WARNING: Data completeness below 95%");
            } else if summary.data_completeness_pct >= 99.0 {
                println!("✅ EXCELLENT: Data completeness above 99%");
            } else {
                println!("✓ GOOD: Data completeness above 95%");
            }
        }
        
        println!("\n{}", "=".repeat(80));
    }
}