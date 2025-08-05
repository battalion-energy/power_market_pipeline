use anyhow::{Result, Context};
use chrono::{NaiveDate, NaiveDateTime, Datelike};
use polars::prelude::*;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use indicatif::{ProgressBar, ProgressStyle};

pub fn process_daily_revenues_fixed() -> Result<()> {
    println!("\n💰 ERCOT BESS Daily Revenue Processing (Fixed Schema)");
    println!("{}", "=".repeat(80));
    
    // Set up paths
    let dam_dir = PathBuf::from("/Users/enrico/data/ERCOT_data/60-Day_DAM_Disclosure_Reports/csv");
    let sced_dir = PathBuf::from("/Users/enrico/data/ERCOT_data/60-Day_SCED_Disclosure_Reports/csv");
    let output_dir = PathBuf::from("bess_daily_revenues_fixed");
    
    std::fs::create_dir_all(&output_dir)?;
    std::fs::create_dir_all(output_dir.join("daily"))?;
    std::fs::create_dir_all(output_dir.join("monthly"))?;
    
    // Load BESS resources
    let bess_resources = load_bess_resources()?;
    println!("📋 Loaded {} BESS resources", bess_resources.len());
    
    // Get available years - just process 2024 for testing
    let years = vec![2024];
    println!("\n📅 Processing years: {:?}", years);
    
    for year in years {
        println!("\n📊 Processing year {}", year);
        process_year_fixed(year, &dam_dir, &sced_dir, &output_dir, &bess_resources)?;
    }
    
    println!("\n✅ Processing complete!");
    Ok(())
}

fn process_year_fixed(
    year: i32,
    dam_dir: &Path,
    sced_dir: &Path,
    output_dir: &Path,
    bess_resources: &HashMap<String, BessResource>,
) -> Result<()> {
    let mut daily_revenues = HashMap::new();
    
    // Initialize revenues for all BESS resources
    for (name, resource) in bess_resources {
        // Create entries for each day of the year (simplified - just a few days for testing)
        for month in 1..=12 {
            for day in 1..=1 {  // Just first day of each month for testing
                if let Some(date) = NaiveDate::from_ymd_opt(year, month, day) {
                    let key = (name.clone(), date);
                    daily_revenues.insert(key, BessDailyRevenue {
                        resource_name: name.clone(),
                        date,
                        settlement_point: resource.settlement_point.clone(),
                        capacity_mw: resource.capacity_mw,
                        rt_energy_revenue: 0.0,
                        dam_energy_revenue: 0.0,
                        reg_up_revenue: 0.0,
                        reg_down_revenue: 0.0,
                        spin_revenue: 0.0,
                        non_spin_revenue: 0.0,
                        ecrs_revenue: 0.0,
                        total_revenue: 0.0,
                        energy_revenue: 0.0,
                        as_revenue: 0.0,
                        cycles: 0.0,
                    });
                }
            }
        }
    }
    
    // Process DAM data with fixed schema
    process_dam_data_fixed(year, dam_dir, &mut daily_revenues)?;
    
    // Save results
    save_daily_data(year, output_dir, &daily_revenues)?;
    
    Ok(())
}

fn process_dam_data_fixed(
    year: i32,
    dam_dir: &Path,
    daily_revenues: &mut HashMap<(String, NaiveDate), BessDailyRevenue>,
) -> Result<()> {
    let pattern = format!("*DAM_Gen_Resource_Data*{:02}.csv", year % 100);
    let dam_files: Vec<PathBuf> = glob::glob(dam_dir.join(&pattern).to_str().unwrap())?
        .filter_map(Result::ok)
        .take(2)  // Just process 2 files for testing
        .collect();
    
    println!("  Processing {} DAM files", dam_files.len());
    
    for file in dam_files {
        println!("  📁 Processing: {}", file.file_name().unwrap().to_str().unwrap());
        
        // Read CSV with schema overrides to handle mixed types
        let df = CsvReader::new(std::fs::File::open(&file)?)
            .has_header(true)
            .infer_schema(Some(50000))
            .with_dtypes(Some(Arc::new(get_dam_schema())))
            .finish()?;
        
        println!("     Total rows: {}", df.height());
        
        // Filter for PWRSTR
        if let Ok(resource_types) = df.column("Resource Type") {
            let mask = resource_types.utf8()?.equal("PWRSTR");
            if let Ok(filtered) = df.filter(&mask) {
                println!("     PWRSTR rows: {}", filtered.height());
                
                // Process the filtered data
                process_dam_awards(&filtered, daily_revenues)?;
            }
        }
    }
    
    Ok(())
}

fn get_dam_schema() -> Schema {
    // Define schema with LSL as Float64 to handle mixed int/float values
    let mut schema = Schema::new();
    
    // Add key columns
    schema.with_column("Delivery Date".to_string().into(), DataType::Utf8);
    schema.with_column("Hour Ending".to_string().into(), DataType::Int64);
    schema.with_column("Resource Name".to_string().into(), DataType::Utf8);
    schema.with_column("Resource Type".to_string().into(), DataType::Utf8);
    
    // Numeric columns that might have mixed types
    schema.with_column("LSL".to_string().into(), DataType::Float64);
    schema.with_column("HSL".to_string().into(), DataType::Float64);
    
    // Award and price columns
    schema.with_column("Awarded Quantity".to_string().into(), DataType::Float64);
    schema.with_column("Energy Settlement Point Price".to_string().into(), DataType::Float64);
    schema.with_column("RegUp Awarded".to_string().into(), DataType::Float64);
    schema.with_column("RegUp MCPC".to_string().into(), DataType::Float64);
    schema.with_column("RegDown Awarded".to_string().into(), DataType::Float64);
    schema.with_column("RegDown MCPC".to_string().into(), DataType::Float64);
    
    // Other columns can be inferred
    schema
}

fn process_dam_awards(
    df: &DataFrame,
    daily_revenues: &mut HashMap<(String, NaiveDate), BessDailyRevenue>,
) -> Result<()> {
    // Extract date column
    let dates = df.column("Delivery Date")?.utf8()?;
    let resources = df.column("Resource Name")?.utf8()?;
    let hours = df.column("Hour Ending")?.i64()?;
    
    // Energy awards
    let awards = df.column("Awarded Quantity")?.f64()?;
    let prices = df.column("Energy Settlement Point Price")?.f64()?;
    
    // Process each row
    for i in 0..df.height() {
        if let (Some(date_str), Some(resource), Some(hour), Some(award), Some(price)) = 
            (dates.get(i), resources.get(i), hours.get(i), awards.get(i), prices.get(i)) {
            
            // Parse date
            if let Ok(date) = NaiveDate::parse_from_str(date_str, "%m/%d/%Y") {
                let key = (resource.to_string(), date);
                
                if let Some(revenue) = daily_revenues.get_mut(&key) {
                    revenue.dam_energy_revenue += award * price;
                }
            }
        }
    }
    
    // RegUp awards
    if let (Ok(reg_up_awards), Ok(reg_up_prices)) = 
        (df.column("RegUp Awarded"), df.column("RegUp MCPC")) {
        
        let reg_up_awards_f64 = reg_up_awards.f64()?;
        let reg_up_prices_f64 = reg_up_prices.f64()?;
        
        for i in 0..df.height() {
            if let (Some(date_str), Some(resource), Some(award), Some(price)) = 
                (dates.get(i), resources.get(i), reg_up_awards_f64.get(i), reg_up_prices_f64.get(i)) {
                
                if let Ok(date) = NaiveDate::parse_from_str(date_str, "%m/%d/%Y") {
                    let key = (resource.to_string(), date);
                    
                    if let Some(revenue) = daily_revenues.get_mut(&key) {
                        revenue.reg_up_revenue += award * price;
                    }
                }
            }
        }
    }
    
    Ok(())
}

fn save_daily_data(
    year: i32,
    output_dir: &Path,
    daily_revenues: &HashMap<(String, NaiveDate), BessDailyRevenue>,
) -> Result<()> {
    // Convert to vectors for DataFrame
    let mut data: Vec<BessDailyRevenue> = daily_revenues.values()
        .filter(|r| r.date.year() == year && r.total_revenue > 0.0)
        .cloned()
        .collect();
    
    // Calculate totals
    for revenue in &mut data {
        revenue.energy_revenue = revenue.rt_energy_revenue + revenue.dam_energy_revenue;
        revenue.as_revenue = revenue.reg_up_revenue + revenue.reg_down_revenue + 
                            revenue.spin_revenue + revenue.non_spin_revenue + revenue.ecrs_revenue;
        revenue.total_revenue = revenue.energy_revenue + revenue.as_revenue;
    }
    
    if data.is_empty() {
        println!("  ⚠️  No revenue data for year {}", year);
        return Ok(());
    }
    
    // Create DataFrame
    let df = DataFrame::new(vec![
        Series::new("resource_name", data.iter().map(|r| &r.resource_name).collect::<Vec<_>>()),
        Series::new("date", data.iter().map(|r| r.date.to_string()).collect::<Vec<_>>()),
        Series::new("settlement_point", data.iter().map(|r| &r.settlement_point).collect::<Vec<_>>()),
        Series::new("capacity_mw", data.iter().map(|r| r.capacity_mw).collect::<Vec<_>>()),
        Series::new("rt_energy_revenue", data.iter().map(|r| r.rt_energy_revenue).collect::<Vec<_>>()),
        Series::new("dam_energy_revenue", data.iter().map(|r| r.dam_energy_revenue).collect::<Vec<_>>()),
        Series::new("reg_up_revenue", data.iter().map(|r| r.reg_up_revenue).collect::<Vec<_>>()),
        Series::new("reg_down_revenue", data.iter().map(|r| r.reg_down_revenue).collect::<Vec<_>>()),
        Series::new("total_revenue", data.iter().map(|r| r.total_revenue).collect::<Vec<_>>()),
    ])?;
    
    // Save as Parquet
    let output_file = output_dir.join("daily").join(format!("bess_daily_revenue_{}.parquet", year));
    let file = std::fs::File::create(&output_file)?;
    ParquetWriter::new(file).finish(&mut df.clone())?;
    
    println!("  ✅ Saved {} daily records to {:?}", data.len(), output_file);
    
    Ok(())
}

#[derive(Debug, Clone)]
struct BessResource {
    settlement_point: String,
    capacity_mw: f64,
}

#[derive(Debug, Clone)]
struct BessDailyRevenue {
    resource_name: String,
    date: NaiveDate,
    settlement_point: String,
    capacity_mw: f64,
    rt_energy_revenue: f64,
    dam_energy_revenue: f64,
    reg_up_revenue: f64,
    reg_down_revenue: f64,
    spin_revenue: f64,
    non_spin_revenue: f64,
    ecrs_revenue: f64,
    total_revenue: f64,
    energy_revenue: f64,
    as_revenue: f64,
    cycles: f64,
}

fn load_bess_resources() -> Result<HashMap<String, BessResource>> {
    let mut resources = HashMap::new();
    
    let master_list_path = PathBuf::from("bess_analysis/bess_resources_master_list.csv");
    if master_list_path.exists() {
        let file = std::fs::File::open(&master_list_path)?;
        let df = CsvReader::new(file).has_header(true).finish()?;
        
        if let (Ok(names), Ok(settlement_points), Ok(capacities)) = (
            df.column("Resource_Name"),
            df.column("Settlement_Point"),
            df.column("Max_Capacity_MW")
        ) {
            let names_str = names.utf8()?;
            let sp_str = settlement_points.utf8()?;
            let cap_f64 = capacities.f64()?;
            
            for i in 0..df.height() {
                if let (Some(name), Some(sp), Some(cap)) = 
                    (names_str.get(i), sp_str.get(i), cap_f64.get(i)) {
                    resources.insert(name.to_string(), BessResource {
                        settlement_point: sp.to_string(),
                        capacity_mw: cap,
                    });
                }
            }
        }
    }
    
    Ok(resources)
}