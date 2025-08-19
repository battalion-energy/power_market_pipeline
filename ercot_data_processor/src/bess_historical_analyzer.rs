use anyhow::Result;
use chrono::NaiveDate;
use polars::prelude::*;
use std::collections::HashMap;
use std::path::PathBuf;

/// BESS Historical Revenue Analyzer
/// 
/// This performs ACCOUNTING of actual BESS operations, NOT optimization.
/// We're calculating what batteries actually earned from their actual operations.
pub struct BessHistoricalAnalyzer {
    disclosure_dir: PathBuf,
    #[allow(dead_code)]
    price_dir: PathBuf,
    #[allow(dead_code)]
    output_dir: PathBuf,
}

#[derive(Debug, Clone)]
pub struct BessResource {
    pub name: String,
    #[allow(dead_code)]
    pub qse: String,
    #[allow(dead_code)]
    pub settlement_point: String,
    pub max_discharge_mw: f64,
    pub max_charge_mw: f64,
}

#[derive(Debug, Clone)]
pub struct DailyRevenue {
    #[allow(dead_code)]
    pub resource_name: String,
    #[allow(dead_code)]
    pub date: NaiveDate,
    pub dam_energy_revenue: f64,
    pub rt_deviation_revenue: f64,
    pub regup_revenue: f64,
    pub regdown_revenue: f64,
    pub rrs_revenue: f64,
    pub ecrs_revenue: f64,
    pub nonspin_revenue: f64,
    pub total_revenue: f64,
    pub mwh_charged: f64,
    pub mwh_discharged: f64,
}

impl BessHistoricalAnalyzer {
    pub fn new(disclosure_dir: PathBuf, price_dir: PathBuf) -> Self {
        let output_dir = PathBuf::from("bess_historical_analysis");
        std::fs::create_dir_all(&output_dir).ok();
        
        Self {
            disclosure_dir,
            price_dir,
            output_dir,
        }
    }
    
    /// Extract BESS resources from DAM disclosure data
    /// These are REAL resources, not mock data!
    pub fn extract_bess_resources(&self) -> Result<Vec<BessResource>> {
        println!("📋 Extracting REAL BESS resources from disclosure data...");
        
        let dam_dir = self.disclosure_dir.join("60-Day_DAM_Disclosure_Reports").join("csv");
        let pattern = dam_dir.join("60d_DAM_Gen_Resource_Data-*.csv");
        
        let files: Vec<PathBuf> = glob::glob(pattern.to_str().unwrap())?
            .filter_map(Result::ok)
            .collect();
        
        // Use recent files to get current fleet
        let sample_files: Vec<_> = files.iter().rev().take(10).collect();
        
        let mut resources: HashMap<String, BessResource> = HashMap::new();
        
        for file in sample_files {
            println!("  Reading {}", file.file_name().unwrap().to_str().unwrap());
            
            let df = CsvReader::from_path(file)?
                .has_header(true)
                .finish()?;
            
            // Filter for PWRSTR (Power Storage) resources
            if let Ok(resource_type) = df.column("Resource Type") {
                if let Ok(resource_type_str) = resource_type.cast(&DataType::String) {
                    let mask = resource_type_str.str()?.equal_missing("PWRSTR");
                    
                    if let Ok(bess_df) = df.filter(&mask) {
                        if let (Ok(names), Ok(hsls), Ok(lsls)) = (
                            bess_df.column("Resource Name"),
                            bess_df.column("HSL"),
                            bess_df.column("LSL"),
                        ) {
                            if let (Ok(names_str), Ok(hsls_f64), Ok(lsls_f64)) = (
                                names.cast(&DataType::String),
                                hsls.cast(&DataType::Float64),
                                lsls.cast(&DataType::Float64),
                            ) {
                                let names_arr = names_str.str()?;
                                let hsls_arr = hsls_f64.f64()?;
                                let lsls_arr = lsls_f64.f64()?;
                                
                                for i in 0..bess_df.height() {
                                    if let (Some(name), Some(hsl), Some(lsl)) = 
                                        (names_arr.get(i), hsls_arr.get(i), lsls_arr.get(i)) {
                                        
                                        resources.entry(name.to_string())
                                            .and_modify(|r| {
                                                r.max_discharge_mw = r.max_discharge_mw.max(hsl);
                                                r.max_charge_mw = r.max_charge_mw.max(lsl.abs());
                                            })
                                            .or_insert(BessResource {
                                                name: name.to_string(),
                                                qse: String::new(), // Would get from QSE column
                                                settlement_point: String::new(), // Would get from mapping
                                                max_discharge_mw: hsl,
                                                max_charge_mw: lsl.abs(),
                                            });
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        let mut resource_list: Vec<BessResource> = resources.into_values().collect();
        resource_list.sort_by(|a, b| b.max_discharge_mw.partial_cmp(&a.max_discharge_mw).unwrap());
        
        println!("✅ Found {} BESS resources", resource_list.len());
        println!("   Total capacity: {:.1} MW", 
                 resource_list.iter().map(|r| r.max_discharge_mw).sum::<f64>());
        
        Ok(resource_list)
    }
    
    /// Calculate revenue for a specific BESS on a specific date
    /// This is pure ACCOUNTING - what they did × price = revenue
    pub fn calculate_daily_revenue(
        &self,
        resource: &BessResource,
        date: NaiveDate,
    ) -> Result<DailyRevenue> {
        let mut revenue = DailyRevenue {
            resource_name: resource.name.clone(),
            date,
            dam_energy_revenue: 0.0,
            rt_deviation_revenue: 0.0,
            regup_revenue: 0.0,
            regdown_revenue: 0.0,
            rrs_revenue: 0.0,
            ecrs_revenue: 0.0,
            nonspin_revenue: 0.0,
            total_revenue: 0.0,
            mwh_charged: 0.0,
            mwh_discharged: 0.0,
        };
        
        // Get DAM awards (what they won)
        let dam_awards = self.get_dam_awards(resource, date)?;
        
        if let Some(awards) = dam_awards {
            // Calculate DAM energy revenue
            if let Ok(awarded_qty) = awards.column("Awarded Quantity") {
                if let Ok(prices) = awards.column("Energy Settlement Point Price") {
                    let qtys = awarded_qty.f64()?;
                    let prices = prices.f64()?;
                    
                    for i in 0..awards.height() {
                        if let (Some(mw), Some(price)) = (qtys.get(i), prices.get(i)) {
                            let hourly_revenue = mw * price;
                            revenue.dam_energy_revenue += hourly_revenue;
                            
                            if mw > 0.0 {
                                revenue.mwh_discharged += mw;
                            } else {
                                revenue.mwh_charged += mw.abs();
                            }
                        }
                    }
                }
            }
            
            // Calculate AS revenues using MCPCs from the file
            // RegUp
            if let (Ok(regup_qty), Ok(regup_price)) = 
                (awards.column("RegUp Awarded"), awards.column("RegUp MCPC")) {
                let qtys = regup_qty.f64()?;
                let prices = regup_price.f64()?;
                for i in 0..awards.height() {
                    if let (Some(mw), Some(price)) = (qtys.get(i), prices.get(i)) {
                        revenue.regup_revenue += mw * price;
                    }
                }
            }
            
            // RegDown
            if let (Ok(regdown_qty), Ok(regdown_price)) = 
                (awards.column("RegDown Awarded"), awards.column("RegDown MCPC")) {
                let qtys = regdown_qty.f64()?;
                let prices = regdown_price.f64()?;
                for i in 0..awards.height() {
                    if let (Some(mw), Some(price)) = (qtys.get(i), prices.get(i)) {
                        revenue.regdown_revenue += mw * price;
                    }
                }
            }
            
            // RRS (combination of RRSPFR, RRSFFR, RRSUFR)
            if let Ok(rrs_price) = awards.column("RRS MCPC") {
                let prices = rrs_price.f64()?;
                for col in ["RRSPFR Awarded", "RRSFFR Awarded", "RRSUFR Awarded"] {
                    if let Ok(rrs_qty) = awards.column(col) {
                        let qtys = rrs_qty.f64()?;
                        for i in 0..awards.height() {
                            if let (Some(mw), Some(price)) = (qtys.get(i), prices.get(i)) {
                                revenue.rrs_revenue += mw * price;
                            }
                        }
                    }
                }
            }
            
            // ECRS
            if let (Ok(ecrs_qty), Ok(ecrs_price)) = 
                (awards.column("ECRSSD Awarded"), awards.column("ECRS MCPC")) {
                let qtys = ecrs_qty.f64()?;
                let prices = ecrs_price.f64()?;
                for i in 0..awards.height() {
                    if let (Some(mw), Some(price)) = (qtys.get(i), prices.get(i)) {
                        revenue.ecrs_revenue += mw * price;
                    }
                }
            }
            
            // NonSpin
            if let (Ok(nonspin_qty), Ok(nonspin_price)) = 
                (awards.column("NonSpin Awarded"), awards.column("NonSpin MCPC")) {
                let qtys = nonspin_qty.f64()?;
                let prices = nonspin_price.f64()?;
                for i in 0..awards.height() {
                    if let (Some(mw), Some(price)) = (qtys.get(i), prices.get(i)) {
                        revenue.nonspin_revenue += mw * price;
                    }
                }
            }
        }
        
        // RT deviations would be calculated from SCED files
        // For now, set to 0
        revenue.rt_deviation_revenue = 0.0;
        
        // Total it up
        revenue.total_revenue = revenue.dam_energy_revenue 
            + revenue.rt_deviation_revenue
            + revenue.regup_revenue
            + revenue.regdown_revenue
            + revenue.rrs_revenue
            + revenue.ecrs_revenue
            + revenue.nonspin_revenue;
        
        Ok(revenue)
    }
    
    fn get_dam_awards(&self, resource: &BessResource, date: NaiveDate) -> Result<Option<DataFrame>> {
        // Format: 60d_DAM_Gen_Resource_Data-DD-MMM-YY.csv
        let date_str = date.format("%d-%b-%y").to_string().to_uppercase();
        let filename = format!("60d_DAM_Gen_Resource_Data-{}.csv", date_str);
        let file_path = self.disclosure_dir
            .join("60-Day_DAM_Disclosure_Reports")
            .join("csv")
            .join(&filename);
        
        if !file_path.exists() {
            println!("  DAM file not found: {}", filename);
            return Ok(None);
        }
        
        // Set up schema overrides for numeric columns
        let mut schema_overrides = Schema::new();
        schema_overrides.with_column("HSL".into(), DataType::Float64);
        schema_overrides.with_column("LSL".into(), DataType::Float64);
        schema_overrides.with_column("Awarded Quantity".into(), DataType::Float64);
        schema_overrides.with_column("Energy Settlement Point Price".into(), DataType::Float64);
        schema_overrides.with_column("RegUp Awarded".into(), DataType::Float64);
        schema_overrides.with_column("RegUp MCPC".into(), DataType::Float64);
        schema_overrides.with_column("RegDown Awarded".into(), DataType::Float64);
        schema_overrides.with_column("RegDown MCPC".into(), DataType::Float64);
        schema_overrides.with_column("RRSPFR Awarded".into(), DataType::Float64);
        schema_overrides.with_column("RRSFFR Awarded".into(), DataType::Float64);
        schema_overrides.with_column("RRSUFR Awarded".into(), DataType::Float64);
        schema_overrides.with_column("RRS MCPC".into(), DataType::Float64);
        schema_overrides.with_column("ECRSSD Awarded".into(), DataType::Float64);
        schema_overrides.with_column("ECRS MCPC".into(), DataType::Float64);
        schema_overrides.with_column("NonSpin Awarded".into(), DataType::Float64);
        schema_overrides.with_column("NonSpin MCPC".into(), DataType::Float64);
        
        let df = CsvReader::from_path(&file_path)?
            .has_header(true)
            .with_dtypes(Some(schema_overrides.into()))
            .finish()?;
        
        // Filter for this resource and PWRSTR type
        if let (Ok(names), Ok(types)) = (df.column("Resource Name"), df.column("Resource Type")) {
            if let (Ok(names_str), Ok(types_str)) = (names.cast(&DataType::String), types.cast(&DataType::String)) {
                let name_mask = names_str.str()?.equal_missing(resource.name.as_str());
                let type_mask = types_str.str()?.equal_missing("PWRSTR");
                let mask = name_mask & type_mask;
                let resource_df = df.filter(&mask)?;
                
                if resource_df.height() > 0 {
                    return Ok(Some(resource_df));
                } else {
                    println!("  No awards found for {} on {}", resource.name, date);
                }
            }
        }
        
        Ok(None)
    }
    
    #[allow(dead_code)]
    fn get_dam_prices(&self, date: NaiveDate) -> Result<Option<DataFrame>> {
        let year = date.format("%Y").to_string();
        let price_file = self.price_dir
            .join("DA_prices")
            .join(format!("{}.parquet", year));
        
        if !price_file.exists() {
            return Ok(None);
        }
        
        let df = ParquetReader::new(std::fs::File::open(&price_file)?)
            .finish()?;
        
        // Filter for this date
        // Would need proper date filtering here
        
        Ok(Some(df))
    }
    
    
    /// Analyze a BESS resource over a date range
    pub fn analyze_resource(
        &self,
        resource: &BessResource,
        start_date: NaiveDate,
        end_date: NaiveDate,
    ) -> Result<Vec<DailyRevenue>> {
        println!("\n📊 Analyzing {} from {} to {}", 
                 resource.name, start_date, end_date);
        
        let mut results = Vec::new();
        let mut current_date = start_date;
        
        while current_date <= end_date {
            match self.calculate_daily_revenue(resource, current_date) {
                Ok(revenue) => results.push(revenue),
                Err(e) => eprintln!("  Error for {}: {}", current_date, e),
            }
            current_date = current_date.succ_opt().unwrap();
        }
        
        // Calculate summary statistics
        let total_revenue: f64 = results.iter().map(|r| r.total_revenue).sum();
        let total_charged: f64 = results.iter().map(|r| r.mwh_charged).sum();
        let total_discharged: f64 = results.iter().map(|r| r.mwh_discharged).sum();
        
        println!("  Total Revenue: ${:.2}", total_revenue);
        println!("  MWh Charged: {:.1}", total_charged);
        println!("  MWh Discharged: {:.1}", total_discharged);
        
        if total_charged > 0.0 {
            let efficiency = total_discharged / total_charged;
            println!("  Implied Efficiency: {:.1}%", efficiency * 100.0);
        }
        
        Ok(results)
    }
}

/// Main entry point for BESS historical analysis
pub fn analyze_bess_historical() -> Result<()> {
    println!("🔋 BESS Historical Revenue Analysis");
    println!("{}", "=".repeat(60));
    println!("This is ACCOUNTING, not optimization!");
    println!("We're calculating what actually happened.\n");
    
    let analyzer = BessHistoricalAnalyzer::new(
        PathBuf::from("/Users/enrico/data/ERCOT_data"),
        PathBuf::from("/Users/enrico/data/ERCOT_data/rollup_files"),
    );
    
    // Get real BESS resources
    let resources = analyzer.extract_bess_resources()?;
    
    // Analyze top resources
    println!("\nTop 5 BESS Resources by Capacity:");
    for resource in resources.iter().take(5) {
        println!("  {} - {:.1} MW", resource.name, resource.max_discharge_mw);
    }
    
    // Analyze a sample resource for a week
    if let Some(resource) = resources.first() {
        let start_date = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
        let end_date = NaiveDate::from_ymd_opt(2024, 1, 7).unwrap();
        
        let _results = analyzer.analyze_resource(resource, start_date, end_date)?;
    }
    
    println!("\n✅ Analysis complete!");
    Ok(())
}