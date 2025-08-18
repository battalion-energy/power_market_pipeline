use anyhow::{Context, Result};
use arrow::array::{Array, Float64Array, TimestampMillisecondArray, TimestampNanosecondArray, TimestampMicrosecondArray};
use arrow::record_batch::RecordBatch;
use chrono::{DateTime, Datelike, Utc};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::fs::File;
use std::path::Path;

use crate::models::PriceType;

pub async fn load_parquet_file(data_dir: &Path, key: &str) -> Result<RecordBatch> {
    let path = data_dir.join(format!("{}.parquet", key));
    
    let file = File::open(&path)
        .with_context(|| format!("Failed to open parquet file: {:?}", path))?;
    
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let mut reader = builder.build()?;
    
    let batch = reader
        .next()
        .ok_or_else(|| anyhow::anyhow!("No data in parquet file"))??;
    
    Ok(batch)
}

pub fn get_file_key(
    price_type: &PriceType,
    date: &DateTime<Utc>,
    combined: bool,
) -> String {
    let year = date.year();
    let month = date.month();
    
    if combined {
        // Use the 15-minute combined files for better granularity
        // Option 1: Use yearly file (simpler, single file per year)
        format!("combined/DA_AS_RT_15min_combined_{:04}", year)
        // Option 2: Use monthly files (would need to handle month)
        // format!("combined/monthly/DA_AS_RT_15min_combined/DA_AS_RT_15min_combined_{:04}_{:02}", year, month)
    } else {
        match price_type {
            PriceType::DayAhead => {
                format!("flattened/DA_prices_{:04}", year)
            }
            PriceType::RealTime => {
                format!("flattened/RT_prices_15min_{:04}", year)
            }
            PriceType::AncillaryServices => {
                format!("flattened/AS_prices_{:04}", year)
            }
            PriceType::Combined => {
                format!("combined/DA_AS_RT_combined_{:04}", year)
            }
        }
    }
}

pub fn filter_batch_by_date_range(
    batch: &RecordBatch,
    start: &DateTime<Utc>,
    end: &DateTime<Utc>,
) -> Result<RecordBatch> {
    let datetime_col = batch
        .column_by_name("datetime")
        .ok_or_else(|| anyhow::anyhow!("datetime column not found"))?;
    
    let mut indices = Vec::new();
    
    // Try different timestamp types
    if let Some(timestamps) = datetime_col
        .as_any()
        .downcast_ref::<TimestampNanosecondArray>()
    {
        // Handle nanosecond timestamps
        let start_nanos = start.timestamp_nanos_opt().unwrap_or(start.timestamp() * 1_000_000_000);
        let end_nanos = end.timestamp_nanos_opt().unwrap_or(end.timestamp() * 1_000_000_000);
        
        for i in 0..timestamps.len() {
            let ts = timestamps.value(i);
            if ts >= start_nanos && ts <= end_nanos {
                indices.push(i);
            }
        }
    } else if let Some(timestamps) = datetime_col
        .as_any()
        .downcast_ref::<TimestampMicrosecondArray>()
    {
        // Handle microsecond timestamps
        let start_micros = start.timestamp_micros();
        let end_micros = end.timestamp_micros();
        
        for i in 0..timestamps.len() {
            let ts = timestamps.value(i);
            if ts >= start_micros && ts <= end_micros {
                indices.push(i);
            }
        }
    } else if let Some(timestamps) = datetime_col
        .as_any()
        .downcast_ref::<TimestampMillisecondArray>()
    {
        // Handle millisecond timestamps
        let start_millis = start.timestamp_millis();
        let end_millis = end.timestamp_millis();
        
        for i in 0..timestamps.len() {
            let ts = timestamps.value(i);
            if ts >= start_millis && ts <= end_millis {
                indices.push(i);
            }
        }
    } else {
        return Err(anyhow::anyhow!("datetime column is not a supported timestamp type"));
    }
    
    if indices.is_empty() {
        return Ok(RecordBatch::new_empty(batch.schema()));
    }
    
    let columns: Result<Vec<_>> = batch
        .columns()
        .iter()
        .map(|col| {
            arrow::compute::take(col, &arrow::array::UInt32Array::from(
                indices.iter().map(|&i| i as u32).collect::<Vec<_>>()
            ), None)
            .map_err(|e| anyhow::anyhow!("Failed to filter column: {}", e))
        })
        .collect();
    
    Ok(RecordBatch::try_new(batch.schema(), columns?)?)
}

pub fn extract_hub_prices(
    batch: &RecordBatch,
    hubs: &[String],
    price_type: &PriceType,
) -> Result<Vec<(String, Vec<Option<f64>>)>> {
    let mut result = Vec::new();
    
    // Special handling for Ancillary Services - return all AS types
    if matches!(price_type, PriceType::AncillaryServices) {
        // AS columns in combined files are prefixed with AS_
        let as_columns = vec!["AS_ECRS", "AS_NSPIN", "AS_REGDN", "AS_REGUP", "AS_RRS"];
        for as_type in as_columns {
            if let Some(col) = batch.column_by_name(as_type) {
                let prices = col
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .ok_or_else(|| anyhow::anyhow!("Column {} is not float64", as_type))?;
                
                let values: Vec<Option<f64>> = (0..prices.len())
                    .map(|i| {
                        if prices.is_null(i) {
                            None
                        } else {
                            Some(prices.value(i))
                        }
                    })
                    .collect();
                
                // Return clean name without AS_ prefix for frontend
                let clean_name = as_type.strip_prefix("AS_").unwrap_or(as_type);
                result.push((clean_name.to_string(), values));
            }
        }
        return Ok(result);
    }
    
    // Normal hub-based processing for DA and RT
    for hub in hubs {
        // In combined files, columns are prefixed with DA_, RT_, etc.
        let column_name = match price_type {
            PriceType::DayAhead => format!("DA_{}", hub),
            PriceType::RealTime => format!("RT_{}", hub),
            PriceType::AncillaryServices => hub.clone(),
            PriceType::Combined => format!("DA_{}", hub),
        };
        
        if let Some(col) = batch.column_by_name(&column_name) {
            let prices = col
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| anyhow::anyhow!("Column {} is not float64", column_name))?;
            
            let values: Vec<Option<f64>> = (0..prices.len())
                .map(|i| Some(prices.value(i)))
                .collect();
            
            result.push((hub.clone(), values));
        } else {
            result.push((hub.clone(), vec![None; batch.num_rows()]));
        }
    }
    
    Ok(result)
}

pub fn extract_timestamps(batch: &RecordBatch) -> Result<Vec<DateTime<Utc>>> {
    let datetime_col = batch
        .column_by_name("datetime")
        .ok_or_else(|| anyhow::anyhow!("datetime column not found"))?;
    
    // Try to cast to different timestamp types
    let result: Vec<DateTime<Utc>> = if let Some(timestamps) = datetime_col
        .as_any()
        .downcast_ref::<TimestampNanosecondArray>()
    {
        // Handle nanosecond timestamps
        (0..timestamps.len())
            .filter_map(|i| {
                let ns = timestamps.value(i);
                let seconds = ns / 1_000_000_000;
                let nanos = (ns % 1_000_000_000) as u32;
                DateTime::from_timestamp(seconds, nanos)
            })
            .collect()
    } else if let Some(timestamps) = datetime_col
        .as_any()
        .downcast_ref::<TimestampMicrosecondArray>()
    {
        // Handle microsecond timestamps
        (0..timestamps.len())
            .filter_map(|i| {
                let us = timestamps.value(i);
                DateTime::from_timestamp_micros(us)
            })
            .collect()
    } else if let Some(timestamps) = datetime_col
        .as_any()
        .downcast_ref::<TimestampMillisecondArray>()
    {
        // Handle millisecond timestamps
        (0..timestamps.len())
            .filter_map(|i| {
                let ms = timestamps.value(i);
                DateTime::from_timestamp_millis(ms)
            })
            .collect()
    } else {
        return Err(anyhow::anyhow!("datetime column is not a supported timestamp type"));
    };
    
    Ok(result)
}