use anyhow::{Result, Context};
use chrono::NaiveDate;
use polars::prelude::*;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use indicatif::{ProgressBar, ProgressStyle};

pub fn process_simple_daily_revenues() -> Result<()> {
    println!("\n💰 ERCOT BESS Simple Daily Revenue Test");
    println!("{}", "=".repeat(80));
    
    // Load BESS resources
    let bess_resources = load_bess_resources()?;
    println!("📋 Loaded {} BESS resources", bess_resources.len());
    
    // Process one test file
    let dam_dir = PathBuf::from("/Users/enrico/data/ERCOT_data/60-Day_DAM_Disclosure_Reports/csv");
    let test_pattern = dam_dir.join("*DAM_Gen_Resource_Data*24.csv");
    
    let dam_files: Vec<PathBuf> = glob::glob(test_pattern.to_str().unwrap())?
        .filter_map(Result::ok)
        .take(1)  // Just process one file for testing
        .collect();
    
    if dam_files.is_empty() {
        println!("❌ No DAM files found");
        return Ok(());
    }
    
    let test_file = &dam_files[0];
    println!("\n📊 Processing test file: {}", test_file.file_name().unwrap().to_str().unwrap());
    
    // Read with all columns as strings first
    let mut df = CsvReader::new(std::fs::File::open(test_file)?)
        .has_header(true)
        .finish()?;
    
    println!("   Total rows: {}", df.height());
    
    // Filter for PWRSTR
    if let Ok(resource_types) = df.column("Resource Type") {
        let mask = resource_types.str()?.equal("PWRSTR");
        if let Ok(filtered) = df.filter(&mask) {
            println!("   PWRSTR rows: {}", filtered.height());
            
            // Process energy revenues
            if let (Ok(resources), Ok(awards), Ok(prices)) = (
                filtered.column("Resource Name"),
                filtered.column("Awarded Quantity"),
                filtered.column("Energy Settlement Point Price")
            ) {
                let resources_str = resources.str()?;
                let awards_f64 = parse_numeric_column(awards)?;
                let prices_f64 = parse_numeric_column(prices)?;
                
                // Calculate revenues for first BESS
                if let Some(first_resource) = resources_str.get(0) {
                    let mut energy_revenue = 0.0;
                    
                    for i in 0..filtered.height() {
                        if let (Some(resource), Some(award), Some(price)) = 
                            (resources_str.get(i), awards_f64.get(i), prices_f64.get(i)) {
                            if resource == first_resource {
                                energy_revenue += award * price;
                            }
                        }
                    }
                    
                    println!("\n💰 Daily energy revenue for {}: ${:.2}", first_resource, energy_revenue);
                }
            }
        }
    }
    
    println!("\n✅ Test completed successfully!");
    Ok(())
}

fn load_bess_resources() -> Result<HashMap<String, String>> {
    let mut resources = HashMap::new();
    
    let master_list_path = PathBuf::from("bess_analysis/bess_resources_master_list.csv");
    if master_list_path.exists() {
        let file = std::fs::File::open(&master_list_path)?;
        let df = CsvReader::new(file).has_header(true).finish()?;
        
        if let (Ok(names), Ok(settlement_points)) = (
            df.column("Resource_Name"),
            df.column("Settlement_Point")
        ) {
            let names_str = names.str()?;
            let sp_str = settlement_points.str()?;
            
            for i in 0..df.height() {
                if let (Some(name), Some(sp)) = (names_str.get(i), sp_str.get(i)) {
                    resources.insert(name.to_string(), sp.to_string());
                }
            }
        }
    }
    
    Ok(resources)
}

fn parse_numeric_column(series: &Series) -> Result<Float64Chunked> {
    match series.dtype() {
        DataType::Float64 => Ok(series.f64()?.clone()),
        DataType::Int64 => {
            let i64_values = series.i64()?;
            let f64_values: Float64Chunked = i64_values.cast(&DataType::Float64)?
                .f64()?
                .clone();
            Ok(f64_values)
        },
        DataType::String => {
            let str_values = series.str()?;
            let f64_values: Vec<Option<f64>> = str_values
                .into_iter()
                .map(|opt_str| {
                    opt_str.and_then(|s| {
                        if s.is_empty() || s == "NaN" {
                            Some(0.0)
                        } else {
                            s.replace(",", "").parse::<f64>().ok()
                        }
                    })
                })
                .collect();
            Ok(Float64Chunked::from_iter(f64_values))
        },
        _ => {
            // For any other type, try to convert to string first
            let str_series = series.cast(&DataType::String)?;
            parse_numeric_column(&str_series)
        }
    }
}