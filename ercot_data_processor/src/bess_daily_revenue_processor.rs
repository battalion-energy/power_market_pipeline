use anyhow::{Result, Context};
use chrono::{DateTime, NaiveDate, NaiveDateTime, Datelike};
use polars::prelude::*;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use indicatif::{ProgressBar, ProgressStyle};

#[derive(Debug, Clone)]
pub struct BessResource {
    #[allow(dead_code)]
    pub name: String,
    pub settlement_point: String,
    pub capacity_mw: f64,
    #[allow(dead_code)]
    pub qse: String,
}

#[derive(Debug, Clone, Default)]
pub struct BessDailyRevenue {
    pub resource_name: String,
    pub date: NaiveDate,
    pub settlement_point: String,
    pub capacity_mw: f64,
    
    // Energy revenues
    pub rt_energy_revenue: f64,
    pub dam_energy_revenue: f64,
    
    // Ancillary service revenues
    pub reg_up_revenue: f64,
    pub reg_down_revenue: f64,
    pub spin_revenue: f64,      // RRS (all types)
    pub non_spin_revenue: f64,
    pub ecrs_revenue: f64,
    
    // Calculated fields
    pub total_revenue: f64,
    pub energy_revenue: f64,     // rt + dam energy
    pub as_revenue: f64,         // all ancillary services
    
    // Operational metrics
    pub dam_awarded_mw: f64,
    pub rt_dispatch_mwh: f64,
    #[allow(dead_code)]
    pub cycles: f64,
}

#[derive(Debug, Clone)]
pub struct BessMonthlyRevenue {
    pub resource_name: String,
    pub year: i32,
    pub month: u32,
    pub settlement_point: String,
    pub capacity_mw: f64,
    
    // Monthly totals
    pub rt_energy_revenue: f64,
    pub dam_energy_revenue: f64,
    pub reg_up_revenue: f64,
    pub reg_down_revenue: f64,
    pub spin_revenue: f64,
    pub non_spin_revenue: f64,
    pub ecrs_revenue: f64,
    pub total_revenue: f64,
    pub energy_revenue: f64,
    pub as_revenue: f64,
    
    // Monthly averages
    pub avg_daily_revenue: f64,
    pub avg_daily_cycles: f64,
    
    // Monthly metrics
    pub days_active: i32,
    pub total_cycles: f64,
}

pub struct BessDailyRevenueProcessor {
    dam_disclosure_dir: PathBuf,
    sced_disclosure_dir: PathBuf,
    price_data_dir: PathBuf,
    output_dir: PathBuf,
    bess_resources: HashMap<String, BessResource>,
}

impl BessDailyRevenueProcessor {
    pub fn new() -> Result<Self> {
        // Set up paths
        let dam_disclosure_dir = PathBuf::from("/Users/enrico/data/ERCOT_data/60-Day_DAM_Disclosure_Reports/csv");
        let sced_disclosure_dir = PathBuf::from("/Users/enrico/data/ERCOT_data/60-Day_SCED_Disclosure_Reports/csv");
        let price_data_dir = PathBuf::from("/Users/enrico/data/ERCOT_data/processed");
        let output_dir = PathBuf::from("/Users/enrico/data/ERCOT_data/processed/bess_revenues");
        
        std::fs::create_dir_all(&output_dir)?;
        std::fs::create_dir_all(output_dir.join("daily"))?;
        std::fs::create_dir_all(output_dir.join("monthly"))?;
        
        // Load BESS resources
        let bess_resources = Self::load_bess_resources()?;
        println!("📋 Loaded {} BESS resources", bess_resources.len());
        
        Ok(Self {
            dam_disclosure_dir,
            sced_disclosure_dir,
            price_data_dir,
            output_dir,
            bess_resources,
        })
    }
    
    fn load_bess_resources() -> Result<HashMap<String, BessResource>> {
        let mut resources = HashMap::new();
        
        // Try multiple possible locations for the master list
        let possible_paths = vec![
            PathBuf::from("bess_complete_analysis/bess_resources_master_list.csv"),
            PathBuf::from("bess_analysis/bess_resources_master_list.csv"),
            PathBuf::from("/Users/enrico/proj/power_market_pipeline/ercot_data_processor/bess_complete_analysis/bess_resources_master_list.csv"),
        ];
        
        let mut master_list_path = None;
        for path in possible_paths {
            if path.exists() {
                master_list_path = Some(path);
                break;
            }
        }
        
        if let Some(path) = master_list_path {
            println!("Loading BESS resources from: {}", path.display());
            let file = std::fs::File::open(&path)?;
            let df = CsvReader::new(file).has_header(true).finish()?;
            
            let names = df.column("Resource_Name")?.utf8()?;
            let settlement_points = df.column("Settlement_Point")?.utf8()?;
            let capacities = df.column("Max_Capacity_MW")?.f64()?;
            let qses = df.column("QSE").ok().and_then(|c| c.utf8().ok());
            
            for i in 0..df.height() {
                if let (Some(name), Some(sp), Some(capacity)) = 
                    (names.get(i), settlement_points.get(i), capacities.get(i)) {
                    
                    let qse = qses.as_ref().and_then(|q| q.get(i)).unwrap_or("UNKNOWN");
                    
                    resources.insert(name.to_string(), BessResource {
                        name: name.to_string(),
                        settlement_point: sp.to_string(),
                        capacity_mw: capacity,
                        qse: qse.to_string(),
                    });
                }
            }
        } else {
            println!("⚠️  No BESS master list found. Will identify from disclosure data.");
        }
        
        Ok(resources)
    }
    
    pub fn process_all_data(&self) -> Result<()> {
        println!("\n💰 ERCOT BESS Daily Revenue Processing");
        println!("{}", "=".repeat(80));
        
        // Get available years from disclosure files
        let years = self.get_available_years()?;
        println!("\n📅 Processing years: {:?}", years);
        
        for year in years {
            println!("\n📊 Processing year {}", year);
            self.process_year(year)?;
        }
        
        println!("\n✅ Processing complete!");
        Ok(())
    }
    
    fn get_available_years(&self) -> Result<Vec<i32>> {
        let mut years = std::collections::HashSet::new();
        
        // Check DAM files
        let dam_pattern = self.dam_disclosure_dir.join("*Gen_Resource_Data*.csv");
        let dam_files: Vec<PathBuf> = glob::glob(dam_pattern.to_str().unwrap())?
            .filter_map(Result::ok)
            .collect();
        
        for file in dam_files {
            if let Some(year) = Self::extract_year_from_filename(&file) {
                years.insert(year);
            }
        }
        
        let mut years_vec: Vec<i32> = years.into_iter().collect();
        years_vec.sort();
        Ok(years_vec)
    }
    
    fn extract_year_from_filename(path: &Path) -> Option<i32> {
        let filename = path.file_name()?.to_str()?;
        
        // Try to find year in format DD-MMM-YY
        let parts: Vec<&str> = filename.split('-').collect();
        if parts.len() >= 3 {
            if let Some(year_part) = parts.last() {
                let year_str = year_part.trim_end_matches(".csv");
                if let Ok(year) = year_str.parse::<i32>() {
                    // Convert 2-digit year to 4-digit
                    if year < 100 {
                        return Some(if year < 50 { 2000 + year } else { 1900 + year });
                    }
                    return Some(year);
                }
            }
        }
        None
    }
    
    fn process_year(&self, year: i32) -> Result<()> {
        // Initialize daily revenue map
        let mut daily_revenues: HashMap<(String, NaiveDate), BessDailyRevenue> = HashMap::new();
        
        // Process DAM data
        self.process_dam_data_daily(year, &mut daily_revenues)?;
        
        // Process RT data
        self.process_rt_data_daily(year, &mut daily_revenues)?;
        
        // Calculate totals for each day
        for (_, revenue) in daily_revenues.iter_mut() {
            revenue.energy_revenue = revenue.dam_energy_revenue + revenue.rt_energy_revenue;
            revenue.as_revenue = revenue.reg_up_revenue + revenue.reg_down_revenue + 
                                revenue.spin_revenue + revenue.non_spin_revenue + revenue.ecrs_revenue;
            revenue.total_revenue = revenue.energy_revenue + revenue.as_revenue;
        }
        
        // Save daily data
        self.save_daily_data(year, &daily_revenues)?;
        
        // Generate and save monthly rollups
        let monthly_revenues = self.generate_monthly_rollups(&daily_revenues);
        self.save_monthly_data(year, &monthly_revenues)?;
        
        // Print summary
        self.print_year_summary(year, &daily_revenues, &monthly_revenues);
        
        Ok(())
    }
    
    fn process_dam_data_daily(&self, year: i32, daily_revenues: &mut HashMap<(String, NaiveDate), BessDailyRevenue>) -> Result<()> {
        let pattern = format!("*DAM_Gen_Resource_Data*{:02}.csv", year % 100);
        let dam_files: Vec<PathBuf> = glob::glob(self.dam_disclosure_dir.join(&pattern).to_str().unwrap())?
            .filter_map(Result::ok)
            .collect();
        
        println!("  Processing {} DAM files", dam_files.len());
        
        let pb = ProgressBar::new(dam_files.len() as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len}")
            .unwrap());
        
        for file in dam_files {
            pb.inc(1);
            self.process_dam_file_daily(&file, daily_revenues)?;
        }
        
        pb.finish();
        Ok(())
    }
    
    fn process_dam_file_daily(&self, file: &Path, daily_revenues: &mut HashMap<(String, NaiveDate), BessDailyRevenue>) -> Result<()> {
        let df = CsvReader::new(std::fs::File::open(file)?)
            .has_header(true)
            .finish()?;
        
        // Extract date from filename
        let date = Self::extract_date_from_filename(file).context("Failed to extract date from DAM file")?;
        
        // Filter for BESS resources
        if let Ok(resource_types) = df.column("Resource Type") {
            let mask = resource_types.utf8()?.equal("PWRSTR");
            
            if let Ok(filtered) = df.filter(&mask) {
                // Process each resource
                if let Ok(resources) = filtered.column("Resource Name") {
                    let resources_str = resources.utf8()?;
                    
                    for i in 0..filtered.height() {
                        if let Some(resource_name) = resources_str.get(i) {
                            // Get or create daily revenue entry
                            let key = (resource_name.to_string(), date);
                            let revenue = daily_revenues.entry(key.clone()).or_insert_with(|| {
                                let mut rev = BessDailyRevenue::default();
                                rev.resource_name = resource_name.to_string();
                                rev.date = date;
                                
                                // Set resource info if available
                                if let Some(bess_info) = self.bess_resources.get(resource_name) {
                                    rev.settlement_point = bess_info.settlement_point.clone();
                                    rev.capacity_mw = bess_info.capacity_mw;
                                }
                                
                                rev
                            });
                            
                            // Process energy awards for this hour
                            if let (Ok(hour_col), Ok(awards), Ok(prices)) = (
                                filtered.column("Hour Ending"),
                                filtered.column("Awarded Quantity"),
                                filtered.column("Energy Settlement Point Price")
                            ) {
                                let hours = Self::parse_numeric_column(hour_col)?;
                                let awards_f64 = Self::parse_numeric_column(awards)?;
                                let prices_f64 = Self::parse_numeric_column(prices)?;
                                
                                if let (Some(_hour), Some(award), Some(price)) = 
                                    (hours.get(i), awards_f64.get(i), prices_f64.get(i)) {
                                    
                                    revenue.dam_energy_revenue += award * price;
                                    revenue.dam_awarded_mw += award;
                                }
                            }
                            
                            // Process AS awards
                            self.process_dam_as_hourly(&filtered, i, revenue)?;
                        }
                    }
                }
            }
        }
        
        Ok(())
    }
    
    fn process_dam_as_hourly(&self, df: &DataFrame, row_idx: usize, revenue: &mut BessDailyRevenue) -> Result<()> {
        // RegUp
        if let (Ok(awards), Ok(prices)) = (
            df.column("RegUp Awarded"),
            df.column("RegUp MCPC")
        ) {
            let awards_f64 = Self::parse_numeric_column(awards)?;
            let prices_f64 = Self::parse_numeric_column(prices)?;
            
            if let (Some(award), Some(price)) = (awards_f64.get(row_idx), prices_f64.get(row_idx)) {
                revenue.reg_up_revenue += award * price;
            }
        }
        
        // RegDown
        if let (Ok(awards), Ok(prices)) = (
            df.column("RegDown Awarded"),
            df.column("RegDown MCPC")
        ) {
            let awards_f64 = Self::parse_numeric_column(awards)?;
            let prices_f64 = Self::parse_numeric_column(prices)?;
            
            if let (Some(award), Some(price)) = (awards_f64.get(row_idx), prices_f64.get(row_idx)) {
                revenue.reg_down_revenue += award * price;
            }
        }
        
        // RRS (combines RRSPFR, RRSFFR, RRSUFR)
        let mut rrs_total = 0.0;
        for rrs_type in ["RRSPFR Awarded", "RRSFFR Awarded", "RRSUFR Awarded"] {
            if let Ok(awards) = df.column(rrs_type) {
                let awards_f64 = Self::parse_numeric_column(awards)?;
                if let Some(award) = awards_f64.get(row_idx) {
                    rrs_total += award;
                }
            }
        }
        
        if let Ok(prices) = df.column("RRS MCPC") {
            let prices_f64 = Self::parse_numeric_column(prices)?;
            if let Some(price) = prices_f64.get(row_idx) {
                revenue.spin_revenue += rrs_total * price;
            }
        }
        
        // ECRS
        if let (Ok(awards), Ok(prices)) = (
            df.column("ECRSSD Awarded"),
            df.column("ECRS MCPC")
        ) {
            let awards_f64 = Self::parse_numeric_column(awards)?;
            let prices_f64 = Self::parse_numeric_column(prices)?;
            
            if let (Some(award), Some(price)) = (awards_f64.get(row_idx), prices_f64.get(row_idx)) {
                revenue.ecrs_revenue += award * price;
            }
        }
        
        // NonSpin
        if let (Ok(awards), Ok(prices)) = (
            df.column("NonSpin Awarded"),
            df.column("NonSpin MCPC")
        ) {
            let awards_f64 = Self::parse_numeric_column(awards)?;
            let prices_f64 = Self::parse_numeric_column(prices)?;
            
            if let (Some(award), Some(price)) = (awards_f64.get(row_idx), prices_f64.get(row_idx)) {
                revenue.non_spin_revenue += award * price;
            }
        }
        
        Ok(())
    }
    
    fn process_rt_data_daily(&self, year: i32, daily_revenues: &mut HashMap<(String, NaiveDate), BessDailyRevenue>) -> Result<()> {
        let pattern = format!("*SCED_Gen_Resource_Data*{:02}.csv", year % 100);
        let sced_files: Vec<PathBuf> = glob::glob(self.sced_disclosure_dir.join(&pattern).to_str().unwrap())?
            .filter_map(Result::ok)
            .collect();
        
        println!("  Processing {} SCED files", sced_files.len());
        
        // Load RT prices (organized by date for efficiency)
        let rt_prices = self.load_rt_prices_by_date(year)?;
        println!("    Loaded RT prices for {} days", rt_prices.len());
        
        let pb = ProgressBar::new(sced_files.len() as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len}")
            .unwrap());
        
        for file in sced_files {
            pb.inc(1);
            self.process_sced_file_daily(&file, &rt_prices, daily_revenues)?;
        }
        
        pb.finish();
        Ok(())
    }
    
    fn load_rt_prices_by_date(&self, year: i32) -> Result<HashMap<NaiveDate, HashMap<(String, NaiveDateTime), f64>>> {
        let mut prices_by_date = HashMap::new();
        
        // Load prices from parquet files organized by date
        let price_dir = self.price_data_dir.join("rtm");
        
        for entry in std::fs::read_dir(&price_dir)? {
            let entry = entry?;
            let path = entry.path();
            
            if path.extension().and_then(|s| s.to_str()) == Some("parquet") {
                if let Some(filename) = path.file_stem().and_then(|s| s.to_str()) {
                    // Extract date from filename (format: YYYYMMDD_rtm_spp)
                    let date_str = filename[0..8].to_string();
                    if let Ok(file_year) = date_str[0..4].parse::<i32>() {
                        if file_year == year {
                            let date = NaiveDate::parse_from_str(&date_str, "%Y%m%d")?;
                            let day_prices = self.load_rt_prices_for_date(&path)?;
                            prices_by_date.insert(date, day_prices);
                        }
                    }
                }
            }
        }
        
        Ok(prices_by_date)
    }
    
    fn load_rt_prices_for_date(&self, file_path: &Path) -> Result<HashMap<(String, NaiveDateTime), f64>> {
        let mut prices = HashMap::new();
        
        let file = std::fs::File::open(file_path)?;
        let df = ParquetReader::new(file).finish()?;
        
        if let (Ok(timestamps), Ok(names), Ok(lmps)) = (
            df.column("interval_start"),
            df.column("location"),
            df.column("lmp")
        ) {
            let timestamps_i64 = timestamps.i64()?;
            let names_str = names.utf8()?;
            let lmps_f64 = lmps.f64()?;
            
            for i in 0..df.height() {
                if let (Some(ts), Some(name), Some(lmp)) = 
                    (timestamps_i64.get(i), names_str.get(i), lmps_f64.get(i)) {
                    
                    let datetime = DateTime::from_timestamp(ts / 1000, 0)
                        .context("Invalid timestamp")?
                        .naive_utc();
                    prices.insert((name.to_string(), datetime), lmp);
                }
            }
        }
        
        Ok(prices)
    }
    
    fn process_sced_file_daily(&self, file: &Path, rt_prices: &HashMap<NaiveDate, HashMap<(String, NaiveDateTime), f64>>, 
                                daily_revenues: &mut HashMap<(String, NaiveDate), BessDailyRevenue>) -> Result<()> {
        let df = CsvReader::new(std::fs::File::open(file)?)
            .has_header(true)
            .finish()?;
        
        // Extract date from filename
        let date = Self::extract_date_from_filename(file).context("Failed to extract date from SCED file")?;
        
        // Get prices for this date
        let day_prices = rt_prices.get(&date);
        if day_prices.is_none() {
            return Ok(()); // Skip if no prices available
        }
        let day_prices = day_prices.unwrap();
        
        // Filter for BESS resources
        if let Ok(resource_types) = df.column("Resource Type") {
            let mask = resource_types.utf8()?.equal("ERCOT_BATT");
            
            if let Ok(filtered) = df.filter(&mask) {
                // Process base points
                if let (Ok(resources), Ok(timestamps), Ok(base_points)) = (
                    filtered.column("Resource Name"),
                    filtered.column("SCED Timestamp"),
                    filtered.column("Base Point")
                ) {
                    let resources_str = resources.utf8()?;
                    let timestamps_str = timestamps.utf8()?;
                    let base_points_f64 = Self::parse_numeric_column(base_points)?;
                    
                    for i in 0..filtered.height() {
                        if let (Some(resource), Some(ts_str), Some(base_point)) = 
                            (resources_str.get(i), timestamps_str.get(i), base_points_f64.get(i)) {
                            
                            // Parse timestamp
                            if let Ok(timestamp) = NaiveDateTime::parse_from_str(ts_str, "%m/%d/%Y %H:%M:%S") {
                                // Get daily revenue entry
                                let key = (resource.to_string(), date);
                                if let Some(revenue) = daily_revenues.get_mut(&key) {
                                    // Look up RT price
                                    let price_key = (revenue.settlement_point.clone(), timestamp);
                                    if let Some(&price) = day_prices.get(&price_key) {
                                        // Calculate 5-minute revenue
                                        let revenue_5min = base_point * price * (5.0 / 60.0);
                                        revenue.rt_energy_revenue += revenue_5min;
                                        
                                        // Track dispatch for cycles calculation
                                        if base_point.abs() > 0.1 {
                                            revenue.rt_dispatch_mwh += base_point.abs() * (5.0 / 60.0);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        Ok(())
    }
    
    fn extract_date_from_filename(path: &Path) -> Option<NaiveDate> {
        let filename = path.file_name()?.to_str()?;
        
        // Try to parse date from filename (format: DD-MMM-YY)
        let parts: Vec<&str> = filename.split('_').collect();
        for part in parts {
            if let Ok(date) = NaiveDate::parse_from_str(part, "%d-%b-%y") {
                return Some(date);
            }
            // Also try without extension
            let part_no_ext = part.trim_end_matches(".csv");
            if let Ok(date) = NaiveDate::parse_from_str(part_no_ext, "%d-%b-%y") {
                return Some(date);
            }
        }
        None
    }
    
    fn generate_monthly_rollups(&self, daily_revenues: &HashMap<(String, NaiveDate), BessDailyRevenue>) -> Vec<BessMonthlyRevenue> {
        let mut monthly_map: HashMap<(String, i32, u32), BessMonthlyRevenue> = HashMap::new();
        
        for ((resource_name, date), daily) in daily_revenues {
            let key = (resource_name.clone(), date.year(), date.month());
            
            let monthly = monthly_map.entry(key).or_insert_with(|| {
                BessMonthlyRevenue {
                    resource_name: resource_name.clone(),
                    year: date.year(),
                    month: date.month(),
                    settlement_point: daily.settlement_point.clone(),
                    capacity_mw: daily.capacity_mw,
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
                    avg_daily_revenue: 0.0,
                    avg_daily_cycles: 0.0,
                    days_active: 0,
                    total_cycles: 0.0,
                }
            });
            
            // Aggregate revenues
            monthly.rt_energy_revenue += daily.rt_energy_revenue;
            monthly.dam_energy_revenue += daily.dam_energy_revenue;
            monthly.reg_up_revenue += daily.reg_up_revenue;
            monthly.reg_down_revenue += daily.reg_down_revenue;
            monthly.spin_revenue += daily.spin_revenue;
            monthly.non_spin_revenue += daily.non_spin_revenue;
            monthly.ecrs_revenue += daily.ecrs_revenue;
            monthly.total_revenue += daily.total_revenue;
            monthly.energy_revenue += daily.energy_revenue;
            monthly.as_revenue += daily.as_revenue;
            
            // Count active days
            if daily.total_revenue > 0.0 {
                monthly.days_active += 1;
            }
            
            // Estimate cycles
            if daily.capacity_mw > 0.0 {
                let daily_cycles = daily.rt_dispatch_mwh / (daily.capacity_mw * 2.0); // Assume 2-hour battery
                monthly.total_cycles += daily_cycles;
            }
        }
        
        // Calculate averages
        for monthly in monthly_map.values_mut() {
            if monthly.days_active > 0 {
                monthly.avg_daily_revenue = monthly.total_revenue / monthly.days_active as f64;
                monthly.avg_daily_cycles = monthly.total_cycles / monthly.days_active as f64;
            }
        }
        
        monthly_map.into_values().collect()
    }
    
    fn save_daily_data(&self, year: i32, daily_revenues: &HashMap<(String, NaiveDate), BessDailyRevenue>) -> Result<()> {
        // Convert to DataFrame
        let mut resources: Vec<String> = Vec::new();
        let mut dates: Vec<String> = Vec::new();
        let mut settlement_points: Vec<String> = Vec::new();
        let mut capacities: Vec<f64> = Vec::new();
        let mut rt_energy: Vec<f64> = Vec::new();
        let mut dam_energy: Vec<f64> = Vec::new();
        let mut reg_up: Vec<f64> = Vec::new();
        let mut reg_down: Vec<f64> = Vec::new();
        let mut spin: Vec<f64> = Vec::new();
        let mut non_spin: Vec<f64> = Vec::new();
        let mut ecrs: Vec<f64> = Vec::new();
        let mut total: Vec<f64> = Vec::new();
        let mut energy: Vec<f64> = Vec::new();
        let mut ancillary: Vec<f64> = Vec::new();
        
        // Sort by resource and date
        let mut sorted_revenues: Vec<(&(String, NaiveDate), &BessDailyRevenue)> = 
            daily_revenues.iter().collect();
        sorted_revenues.sort_by_key(|(k, _)| (k.0.clone(), k.1));
        
        for ((resource_name, date), revenue) in sorted_revenues {
            resources.push(resource_name.clone());
            dates.push(date.format("%Y-%m-%d").to_string());
            settlement_points.push(revenue.settlement_point.clone());
            capacities.push(revenue.capacity_mw);
            rt_energy.push(revenue.rt_energy_revenue);
            dam_energy.push(revenue.dam_energy_revenue);
            reg_up.push(revenue.reg_up_revenue);
            reg_down.push(revenue.reg_down_revenue);
            spin.push(revenue.spin_revenue);
            non_spin.push(revenue.non_spin_revenue);
            ecrs.push(revenue.ecrs_revenue);
            total.push(revenue.total_revenue);
            energy.push(revenue.energy_revenue);
            ancillary.push(revenue.as_revenue);
        }
        
        let df = DataFrame::new(vec![
            Series::new("resource_name", resources),
            Series::new("date", dates),
            Series::new("settlement_point", settlement_points),
            Series::new("capacity_mw", capacities),
            Series::new("rt_energy_revenue", rt_energy),
            Series::new("dam_energy_revenue", dam_energy),
            Series::new("reg_up_revenue", reg_up),
            Series::new("reg_down_revenue", reg_down),
            Series::new("spin_revenue", spin),
            Series::new("non_spin_revenue", non_spin),
            Series::new("ecrs_revenue", ecrs),
            Series::new("total_revenue", total),
            Series::new("energy_revenue", energy),
            Series::new("as_revenue", ancillary),
        ])?;
        
        // Save as Parquet
        let output_path = self.output_dir.join("daily").join(format!("bess_daily_revenues_{}.parquet", year));
        let file = std::fs::File::create(&output_path)?;
        ParquetWriter::new(file).finish(&mut df.clone())?;
        
        println!("    ✓ Saved daily data to: {}", output_path.display());
        
        Ok(())
    }
    
    fn save_monthly_data(&self, year: i32, monthly_revenues: &Vec<BessMonthlyRevenue>) -> Result<()> {
        // Convert to DataFrame
        let mut resources: Vec<String> = Vec::new();
        let mut years: Vec<i32> = Vec::new();
        let mut months: Vec<i32> = Vec::new();
        let mut settlement_points: Vec<String> = Vec::new();
        let mut capacities: Vec<f64> = Vec::new();
        let mut rt_energy: Vec<f64> = Vec::new();
        let mut dam_energy: Vec<f64> = Vec::new();
        let mut reg_up: Vec<f64> = Vec::new();
        let mut reg_down: Vec<f64> = Vec::new();
        let mut spin: Vec<f64> = Vec::new();
        let mut non_spin: Vec<f64> = Vec::new();
        let mut ecrs: Vec<f64> = Vec::new();
        let mut total: Vec<f64> = Vec::new();
        let mut energy: Vec<f64> = Vec::new();
        let mut ancillary: Vec<f64> = Vec::new();
        let mut avg_daily: Vec<f64> = Vec::new();
        let mut days_active: Vec<i32> = Vec::new();
        
        // Sort by resource and month
        let mut sorted_revenues = monthly_revenues.clone();
        sorted_revenues.sort_by_key(|r| (r.resource_name.clone(), r.year, r.month));
        
        for revenue in sorted_revenues {
            resources.push(revenue.resource_name);
            years.push(revenue.year);
            months.push(revenue.month as i32);
            settlement_points.push(revenue.settlement_point);
            capacities.push(revenue.capacity_mw);
            rt_energy.push(revenue.rt_energy_revenue);
            dam_energy.push(revenue.dam_energy_revenue);
            reg_up.push(revenue.reg_up_revenue);
            reg_down.push(revenue.reg_down_revenue);
            spin.push(revenue.spin_revenue);
            non_spin.push(revenue.non_spin_revenue);
            ecrs.push(revenue.ecrs_revenue);
            total.push(revenue.total_revenue);
            energy.push(revenue.energy_revenue);
            ancillary.push(revenue.as_revenue);
            avg_daily.push(revenue.avg_daily_revenue);
            days_active.push(revenue.days_active);
        }
        
        let df = DataFrame::new(vec![
            Series::new("resource_name", resources),
            Series::new("year", years),
            Series::new("month", months),
            Series::new("settlement_point", settlement_points),
            Series::new("capacity_mw", capacities),
            Series::new("rt_energy_revenue", rt_energy),
            Series::new("dam_energy_revenue", dam_energy),
            Series::new("reg_up_revenue", reg_up),
            Series::new("reg_down_revenue", reg_down),
            Series::new("spin_revenue", spin),
            Series::new("non_spin_revenue", non_spin),
            Series::new("ecrs_revenue", ecrs),
            Series::new("total_revenue", total),
            Series::new("energy_revenue", energy),
            Series::new("as_revenue", ancillary),
            Series::new("avg_daily_revenue", avg_daily),
            Series::new("days_active", days_active),
        ])?;
        
        // Save as Parquet
        let output_path = self.output_dir.join("monthly").join(format!("bess_monthly_revenues_{}.parquet", year));
        let file = std::fs::File::create(&output_path)?;
        ParquetWriter::new(file).finish(&mut df.clone())?;
        
        println!("    ✓ Saved monthly data to: {}", output_path.display());
        
        Ok(())
    }
    
    fn print_year_summary(&self, year: i32, daily_revenues: &HashMap<(String, NaiveDate), BessDailyRevenue>, 
                          monthly_revenues: &Vec<BessMonthlyRevenue>) {
        println!("\n📊 Year {} Summary:", year);
        println!("    Total BESS tracked: {}", self.bess_resources.len());
        println!("    Daily records: {}", daily_revenues.len());
        println!("    Monthly records: {}", monthly_revenues.len());
        
        // Calculate totals
        let total_revenue: f64 = daily_revenues.values().map(|r| r.total_revenue).sum();
        let total_energy: f64 = daily_revenues.values().map(|r| r.energy_revenue).sum();
        let total_as: f64 = daily_revenues.values().map(|r| r.as_revenue).sum();
        
        println!("    Total Revenue: ${:.2}M", total_revenue / 1_000_000.0);
        println!("    Energy Revenue: ${:.2}M ({:.1}%)", 
                 total_energy / 1_000_000.0, 
                 (total_energy / total_revenue * 100.0));
        println!("    AS Revenue: ${:.2}M ({:.1}%)", 
                 total_as / 1_000_000.0, 
                 (total_as / total_revenue * 100.0));
    }
    
    fn parse_numeric_column(series: &Series) -> Result<Float64Chunked> {
        match series.dtype() {
            DataType::Float64 => Ok(series.f64()?.clone()),
            DataType::Float32 => {
                let f32_values = series.f32()?;
                let f64_values: Float64Chunked = f32_values.cast(&DataType::Float64)?
                    .f64()?
                    .clone();
                Ok(f64_values)
            },
            DataType::Int64 => {
                let i64_values = series.i64()?;
                let f64_values: Float64Chunked = i64_values.cast(&DataType::Float64)?
                    .f64()?
                    .clone();
                Ok(f64_values)
            },
            DataType::Int32 => {
                let i32_values = series.i32()?;
                let f64_values: Float64Chunked = i32_values.cast(&DataType::Float64)?
                    .f64()?
                    .clone();
                Ok(f64_values)
            },
            DataType::Utf8 => {
                let str_values = series.utf8()?;
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
                // For any other type, try to convert to string first then parse
                match series.cast(&DataType::Utf8) {
                    Ok(str_series) => Self::parse_numeric_column(&str_series),
                    Err(_) => Err(anyhow::anyhow!("Unsupported column type for numeric conversion: {:?}", series.dtype()))
                }
            }
        }
    }
}

pub fn process_bess_daily_revenues() -> Result<()> {
    let processor = BessDailyRevenueProcessor::new()?;
    processor.process_all_data()?;
    Ok(())
}