use anyhow::Result;
use arrow::array::{StringArray, Float64Array, TimestampMicrosecondArray};
use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use arrow::record_batch::RecordBatch;
use chrono::{DateTime, Duration, NaiveDate, NaiveDateTime, TimeZone, Utc};
use csv::ReaderBuilder;
use dashmap::DashMap;
use glob::glob;
use indicatif::{ProgressBar, ProgressStyle};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use rayon::prelude::*;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use itertools::Itertools;

#[derive(Debug, Deserialize)]
struct RTRecord {
    #[serde(rename = "DeliveryDate")]
    delivery_date: String,
    #[serde(rename = "DeliveryHour")]
    delivery_hour: u8,
    #[serde(rename = "DeliveryInterval")]
    delivery_interval: u8,
    #[serde(rename = "SettlementPoint")]
    settlement_point: String,
    #[serde(rename = "SettlementPointPrice")]
    settlement_point_price: f64,
}

#[derive(Debug, Clone)]
struct ProcessedRecord {
    datetime: DateTime<Utc>,
    settlement_point: String,
    price: f64,
}

fn parse_datetime(date_str: &str, hour: u8, interval: u8) -> Result<DateTime<Utc>> {
    // Parse MM/DD/YYYY format
    let naive_date = NaiveDate::parse_from_str(date_str, "%m/%d/%Y")?;
    
    // Calculate actual hour and minute
    let actual_hour = if hour == 24 { 0 } else { hour - 1 };
    let minute = (interval - 1) * 15;
    
    let mut naive_datetime = naive_date.and_hms_opt(actual_hour as u32, minute as u32, 0)
        .ok_or_else(|| anyhow::anyhow!("Invalid time"))?;
    
    // Handle hour 24 by adding a day
    if hour == 24 {
        naive_datetime = naive_datetime + Duration::days(1);
    }
    
    Ok(Utc.from_utc_datetime(&naive_datetime))
}

fn extract_year_from_filename(filename: &str) -> Option<u16> {
    // Look for pattern like .20101201. (YYYYMMDD)
    if let Some(start) = filename.find(".20") {
        if let Some(year_str) = filename.get(start + 1..start + 5) {
            return year_str.parse().ok();
        }
    }
    None
}

fn process_csv_file(path: &Path) -> Result<Vec<ProcessedRecord>> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)?;
    
    let mut records = Vec::new();
    
    for result in reader.deserialize() {
        let record: RTRecord = match result {
            Ok(r) => r,
            Err(_) => continue, // Skip bad records
        };
        
        if let Ok(datetime) = parse_datetime(&record.delivery_date, record.delivery_hour, record.delivery_interval) {
            records.push(ProcessedRecord {
                datetime,
                settlement_point: record.settlement_point,
                price: record.settlement_point_price,
            });
        }
    }
    
    Ok(records)
}

fn save_year_data(year: u16, records: Vec<ProcessedRecord>, output_dir: &Path) -> Result<()> {
    println!("  Sorting {} records for year {}...", records.len(), year);
    
    // Sort by datetime and settlement point
    let mut sorted_records = records;
    sorted_records.sort_unstable_by(|a, b| {
        a.datetime.cmp(&b.datetime)
            .then_with(|| a.settlement_point.cmp(&b.settlement_point))
    });
    
    // Remove duplicates
    sorted_records.dedup_by(|a, b| {
        a.datetime == b.datetime && a.settlement_point == b.settlement_point
    });
    
    let record_count = sorted_records.len();
    println!("  Final record count: {}", record_count);
    
    let base_name = format!("RT_Settlement_Point_Prices_{}", year);
    
    // Save CSV
    println!("  Saving CSV...");
    let csv_path = output_dir.join(format!("{}.csv", base_name));
    let csv_file = File::create(&csv_path)?;
    let mut csv_writer = BufWriter::new(csv_file);
    
    writeln!(csv_writer, "datetime,SettlementPoint,SettlementPointPrice")?;
    for record in &sorted_records {
        writeln!(
            csv_writer,
            "{},{},{}",
            record.datetime.format("%Y-%m-%d %H:%M:%S"),
            record.settlement_point,
            record.price
        )?;
    }
    csv_writer.flush()?;
    
    // Convert to Arrow arrays
    println!("  Creating Arrow arrays...");
    let timestamps: Vec<i64> = sorted_records.iter()
        .map(|r| r.datetime.timestamp_micros())
        .collect();
    let settlement_points: Vec<&str> = sorted_records.iter()
        .map(|r| r.settlement_point.as_str())
        .collect();
    let prices: Vec<f64> = sorted_records.iter()
        .map(|r| r.price)
        .collect();
    
    let timestamp_array = TimestampMicrosecondArray::from(timestamps);
    let settlement_point_array = StringArray::from(settlement_points);
    let price_array = Float64Array::from(prices);
    
    // Create Arrow schema
    let schema = Schema::new(vec![
        Field::new("datetime", DataType::Timestamp(TimeUnit::Microsecond, None), false),
        Field::new("SettlementPoint", DataType::Utf8, false),
        Field::new("SettlementPointPrice", DataType::Float64, false),
    ]);
    
    // Create record batch
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(timestamp_array),
            Arc::new(settlement_point_array),
            Arc::new(price_array),
        ],
    )?;
    
    // Save Arrow file
    println!("  Saving Arrow file...");
    let arrow_path = output_dir.join(format!("{}.arrow", base_name));
    let arrow_file = File::create(&arrow_path)?;
    let mut arrow_writer = arrow::ipc::writer::FileWriter::try_new(arrow_file, &schema)?;
    arrow_writer.write(&batch)?;
    arrow_writer.finish()?;
    
    // Save Parquet file
    println!("  Saving Parquet file...");
    let parquet_path = output_dir.join(format!("{}.parquet", base_name));
    let parquet_file = File::create(&parquet_path)?;
    let props = WriterProperties::builder().build();
    let mut parquet_writer = ArrowWriter::try_new(parquet_file, Arc::new(schema), Some(props))?;
    parquet_writer.write(&batch)?;
    parquet_writer.close()?;
    
    println!("  ✓ Saved all formats for year {}", year);
    Ok(())
}

fn main() -> Result<()> {
    println!("🚀 ERCOT RT Settlement Point Prices - Rust Processor");
    println!("Using {} CPU cores", num_cpus::get());
    println!("=" .repeat(60));
    
    let data_dir = PathBuf::from("/Users/enrico/data/ERCOT_data/Settlement_Point_Prices_at_Resource_Nodes,_Hubs_and_Load_Zones/csv");
    let output_dir = PathBuf::from("annual_data");
    std::fs::create_dir_all(&output_dir)?;
    
    // Find all CSV files
    let pattern = data_dir.join("*.csv");
    let csv_files: Vec<PathBuf> = glob(pattern.to_str().unwrap())?
        .filter_map(Result::ok)
        .collect();
    
    println!("Found {} RT CSV files", csv_files.len());
    
    // Group files by year
    let mut files_by_year: HashMap<u16, Vec<PathBuf>> = HashMap::new();
    for file in csv_files {
        if let Some(year) = extract_year_from_filename(file.file_name().unwrap().to_str().unwrap()) {
            files_by_year.entry(year).or_insert_with(Vec::new).push(file);
        }
    }
    
    let years: Vec<u16> = files_by_year.keys().cloned().collect();
    println!("Years found: {:?}", years);
    
    // Process each year
    for year in years.into_iter().sorted() {
        let year_files = &files_by_year[&year];
        println!("\nProcessing year {}: {} files", year, year_files.len());
        
        // Create progress bar
        let pb = ProgressBar::new(year_files.len() as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
            .unwrap());
        
        // Process files in parallel
        let year_data: DashMap<String, Vec<ProcessedRecord>> = DashMap::new();
        
        year_files.par_iter().for_each(|file| {
            if let Ok(records) = process_csv_file(file) {
                // Group by a key to reduce contention
                let key = format!("{}", file.file_name().unwrap().to_str().unwrap());
                year_data.insert(key, records);
            }
            pb.inc(1);
        });
        
        pb.finish_with_message("Processing complete");
        
        // Combine all records for the year
        println!("  Combining data for year {}...", year);
        let mut all_records = Vec::new();
        for (_, records) in year_data.into_iter() {
            all_records.extend(records);
        }
        
        // Save year data
        if !all_records.is_empty() {
            save_year_data(year, all_records, &output_dir)?;
        }
    }
    
    println!("\n✅ Processing complete!");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_parse_datetime() {
        let dt = parse_datetime("01/15/2023", 14, 3).unwrap();
        assert_eq!(dt.format("%Y-%m-%d %H:%M:%S").to_string(), "2023-01-15 13:30:00");
        
        // Test hour 24
        let dt = parse_datetime("01/15/2023", 24, 1).unwrap();
        assert_eq!(dt.format("%Y-%m-%d %H:%M:%S").to_string(), "2023-01-16 00:00:00");
    }
    
    #[test]
    fn test_extract_year() {
        assert_eq!(extract_year_from_filename("cdr.00012328.0000000000000000.20230115.151202.csv"), Some(2023));
        assert_eq!(extract_year_from_filename("SPPHLZNP6345_20230115_2315_csv.zip"), Some(2023));
    }
}