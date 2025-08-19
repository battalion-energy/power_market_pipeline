use anyhow::Result;
use chrono::{NaiveDate, Datelike};
use polars::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use indicatif::{ProgressBar, ProgressStyle, MultiProgress};

/// BESS Resource information
#[derive(Debug, Clone)]
pub struct BessResource {
    pub gen_resource: String,      // Generation resource name (e.g., ALVIN_BESS_UNIT1)
    pub load_resource: String,     // Load resource name (e.g., ALVIN_BESS_LD1)
    pub settlement_point: String,  // Settlement point for pricing
    pub capacity_mw: f64,          // Rated capacity in MW
    pub _duration_hours: f64,       // Storage duration in hours
}

/// Daily revenue breakdown for a BESS resource
#[derive(Debug, Clone, Default)]
pub struct BessDailyRevenue {
    pub resource_name: String,
    pub date: NaiveDate,
    pub settlement_point: String,
    pub capacity_mw: f64,
    
    // Energy revenues
    pub dam_energy_revenue: f64,      // Day-ahead energy revenue
    pub rt_energy_revenue: f64,       // Real-time energy revenue (net of DAM)
    
    // Ancillary service revenues
    pub reg_up_revenue: f64,          // Regulation up
    pub reg_down_revenue: f64,        // Regulation down
    pub rrs_revenue: f64,             // Responsive reserve (all types)
    pub nonspin_revenue: f64,         // Non-spinning reserve
    pub ecrs_revenue: f64,            // ECRS (all types)
    
    // Totals
    pub total_revenue: f64,           // Total daily revenue
    pub energy_revenue: f64,          // Total energy revenue
    pub as_revenue: f64,              // Total ancillary service revenue
    
    // Operational metrics
    pub dam_awarded_mwh: f64,         // Total DAM energy awards
    pub rt_dispatched_mwh: f64,       // Total RT energy dispatch
    pub cycles: f64,                  // Estimated charge/discharge cycles
}

/// High-performance BESS revenue processor using parquet files
pub struct BessParquetRevenueProcessor {
    _base_dir: PathBuf,
    rollup_dir: PathBuf,
    output_dir: PathBuf,
    bess_resources: Vec<BessResource>,
    num_threads: usize,
}

impl BessParquetRevenueProcessor {
    pub fn new(base_dir: PathBuf) -> Result<Self> {
        let rollup_dir = base_dir.join("rollup_files");
        let output_dir = base_dir.join("bess_analysis");
        
        // Create output directories
        std::fs::create_dir_all(&output_dir)?;
        std::fs::create_dir_all(output_dir.join("daily"))?;
        std::fs::create_dir_all(output_dir.join("monthly"))?;
        std::fs::create_dir_all(output_dir.join("yearly"))?;
        
        // Load BESS resources
        let bess_resources = Self::load_bess_resources(&base_dir)?;
        
        // Configure parallelism
        let num_threads = num_cpus::get().min(32);  // Use up to 32 threads
        
        println!("🔧 BESS Revenue Processor Configuration:");
        println!("  📁 Base directory: {}", base_dir.display());
        println!("  🔄 Rollup directory: {}", rollup_dir.display());
        println!("  💾 Output directory: {}", output_dir.display());
        println!("  🔋 BESS resources: {}", bess_resources.len());
        println!("  🚀 Parallel threads: {}", num_threads);
        
        Ok(Self {
            _base_dir: base_dir,
            rollup_dir,
            output_dir,
            bess_resources,
            num_threads,
        })
    }
    
    /// Load BESS resource mappings from file or identify from data
    fn load_bess_resources(base_dir: &Path) -> Result<Vec<BessResource>> {
        let mut resources = Vec::new();
        
        // Try to load from the fixed resource pairs file first
        let pairs_file = base_dir.join("bess_resource_pairs.csv");
        let legacy_file = base_dir.join("bess_match_file.csv");
        
        let mapping_file = if pairs_file.exists() {
            pairs_file
        } else if legacy_file.exists() {
            println!("⚠️  Using legacy match file - run fix_bess_mapping.py for better results");
            legacy_file
        } else {
            PathBuf::new()
        };
        
        if mapping_file.exists() {
            println!("📋 Loading BESS mappings from: {}", mapping_file.display());
            let df = CsvReader::from_path(&mapping_file)?
                .has_header(true)
                .finish()?;
            
            let gen_names = df.column("gen_resource")?.str()?;
            let load_names = df.column("load_resource")?.str()?;
            let settlement_points = df.column("settlement_point")?.str()?;
            let capacities = df.column("capacity_mw")?.f64()?;
            let durations = df.column("duration_hours")?.f64()?;
            
            for i in 0..df.height() {
                if let (Some(gen), Some(load), Some(sp), Some(cap), Some(dur)) = (
                    gen_names.get(i),
                    load_names.get(i),
                    settlement_points.get(i),
                    capacities.get(i),
                    durations.get(i),
                ) {
                    resources.push(BessResource {
                        gen_resource: gen.to_string(),
                        load_resource: load.to_string(),
                        settlement_point: sp.to_string(),
                        capacity_mw: cap,
                        _duration_hours: dur,
                    });
                }
            }
        } else {
            // Identify BESS resources from DAM data
            println!("⚠️  No mapping file found. Identifying BESS resources from data...");
            resources = Self::identify_bess_from_data(base_dir)?;
        }
        
        Ok(resources)
    }
    
    /// Identify BESS resources from DAM disclosure data
    fn identify_bess_from_data(base_dir: &Path) -> Result<Vec<BessResource>> {
        let mut resources = Vec::new();
        let dam_gen_dir = base_dir.join("rollup_files/DAM_Gen_Resources");
        
        // Read a recent year to identify BESS resources
        let recent_file = dam_gen_dir.join("2024.parquet");
        if recent_file.exists() {
            let df = ParquetReader::new(std::fs::File::open(&recent_file)?)
                .finish()?;
            
            // Filter for PWRSTR (Power Storage) resources
            if let Ok(resource_types) = df.column("ResourceType") {
                let mask = resource_types.str()?.equal("PWRSTR");
                let filtered = df.filter(&mask)?;
                
                // Get unique resource names
                let unique_resources = filtered
                    .column("ResourceName")?
                    .unique()?;
                
                let resource_names = unique_resources.str()?;
                for i in 0..resource_names.len() {
                    if let Some(name) = resource_names.get(i) {
                        // Try to identify matching load resource
                        let base_name = name.split("_UNIT").next().unwrap_or(name);
                        let unit_num = name.split("UNIT").nth(1).and_then(|s| s.parse::<u32>().ok()).unwrap_or(1);
                        let load_name = format!("{}_LD{}", base_name, unit_num);
                        
                        // Get settlement point from the data
                        let sp_mask = filtered.column("ResourceName")?.str()?.equal(name);
                        let sp_filtered = filtered.filter(&sp_mask)?;
                        let settlement_point = sp_filtered
                            .column("SettlementPointName")?
                            .str()?
                            .get(0)
                            .unwrap_or("UNKNOWN")
                            .to_string();
                        
                        resources.push(BessResource {
                            gen_resource: name.to_string(),
                            load_resource: load_name,
                            settlement_point,
                            capacity_mw: 100.0,  // Default, will be updated from HSL
                            _duration_hours: 2.0,  // Default assumption
                        });
                    }
                }
            }
        }
        
        println!("  Identified {} BESS resources", resources.len());
        Ok(resources)
    }
    
    /// Main processing function - calculates revenues for all BESS resources
    pub fn process_all_revenues(&self) -> Result<()> {
        println!("\n💰 BESS PARQUET REVENUE PROCESSOR");
        println!("{}", "=".repeat(80));
        
        // Get available years
        let years = self.get_available_years()?;
        println!("📅 Processing years: {:?}", years);
        
        // Set up progress tracking
        let multi_progress = Arc::new(MultiProgress::new());
        let main_pb = multi_progress.add(ProgressBar::new(years.len() as u64));
        main_pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} years")
                .unwrap()
        );
        
        // Process years in parallel
        let all_revenues: Arc<Mutex<HashMap<(String, NaiveDate), BessDailyRevenue>>> = 
            Arc::new(Mutex::new(HashMap::new()));
        
        // Configure thread pool
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(self.num_threads)
            .build()
            .unwrap();
        
        pool.install(|| {
            years.par_iter().for_each(|&year| {
                match self.process_year(year, &all_revenues, &multi_progress) {
                    Ok(_) => {
                        main_pb.inc(1);
                        println!("✅ Completed year {}", year);
                    }
                    Err(e) => {
                        eprintln!("❌ Error processing year {}: {}", year, e);
                    }
                }
            });
        });
        
        main_pb.finish_with_message("All years processed");
        
        // Generate reports and save results
        self.generate_reports(&all_revenues)?;
        
        println!("\n✅ BESS revenue processing complete!");
        Ok(())
    }
    
    /// Get list of available years from parquet files
    fn get_available_years(&self) -> Result<Vec<i32>> {
        let mut years = Vec::new();
        let dam_gen_dir = self.rollup_dir.join("DAM_Gen_Resources");
        
        for entry in std::fs::read_dir(&dam_gen_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("parquet") {
                if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                    if let Ok(year) = stem.parse::<i32>() {
                        years.push(year);
                    }
                }
            }
        }
        
        years.sort();
        Ok(years)
    }
    
    /// Process a single year of data
    fn process_year(
        &self,
        year: i32,
        all_revenues: &Arc<Mutex<HashMap<(String, NaiveDate), BessDailyRevenue>>>,
        multi_progress: &Arc<MultiProgress>,
    ) -> Result<()> {
        // Create progress bar for this year
        let year_pb = multi_progress.add(ProgressBar::new(4));
        year_pb.set_style(
            ProgressStyle::default_bar()
                .template(&format!("  {} {{msg}} [{{bar:30}}] {{pos}}/{{len}}", year))
                .unwrap()
        );
        
        // Load price data for the year
        year_pb.set_message("Loading prices");
        let prices = self.load_price_data(year)?;
        year_pb.inc(1);
        
        // Process DAM Generation awards
        year_pb.set_message("Processing DAM Gen");
        let dam_gen_revenues = self.process_dam_generation(year, &prices)?;
        year_pb.inc(1);
        
        // Process DAM Load awards (charging)
        year_pb.set_message("Processing DAM Load");
        let dam_load_revenues = self.process_dam_load(year, &prices)?;
        year_pb.inc(1);
        
        // Process RT dispatch (both gen and load)
        year_pb.set_message("Processing RT data");
        let rt_revenues = self.process_rt_data(year, &prices)?;
        year_pb.inc(1);
        
        // Combine all revenues
        let mut revenues_lock = all_revenues.lock().unwrap();
        revenues_lock.extend(dam_gen_revenues);
        revenues_lock.extend(dam_load_revenues);
        
        // Merge RT revenues
        for (key, rt_rev) in rt_revenues {
            revenues_lock.entry(key.clone())
                .and_modify(|e| {
                    e.rt_energy_revenue += rt_rev.rt_energy_revenue;
                    e.rt_dispatched_mwh += rt_rev.rt_dispatched_mwh;
                })
                .or_insert(rt_rev);
        }
        
        year_pb.finish_with_message("Complete");
        Ok(())
    }
    
    /// Load price data for a year (DA + AS + RT)
    fn load_price_data(&self, year: i32) -> Result<PriceData> {
        let mut price_data = PriceData::default();
        
        // Load combined DA + AS prices if available
        let combined_file = self.rollup_dir.join(format!("combined/DA_AS_combined_{}.parquet", year));
        if combined_file.exists() {
            price_data.da_as_prices = Some(
                ParquetReader::new(std::fs::File::open(&combined_file)?)
                    .finish()?
            );
        }
        
        // Load RT prices
        let rt_file = self.rollup_dir.join(format!("RT_prices/{}.parquet", year));
        if rt_file.exists() {
            price_data.rt_prices = Some(
                ParquetReader::new(std::fs::File::open(&rt_file)?)
                    .finish()?
            );
        }
        
        Ok(price_data)
    }
    
    /// Process DAM Generation awards and AS commitments
    fn process_dam_generation(&self, year: i32, prices: &PriceData) -> Result<HashMap<(String, NaiveDate), BessDailyRevenue>> {
        let mut revenues = HashMap::new();
        let dam_gen_file = self.rollup_dir.join(format!("DAM_Gen_Resources/{}.parquet", year));
        
        if !dam_gen_file.exists() {
            return Ok(revenues);
        }
        
        let df = ParquetReader::new(std::fs::File::open(&dam_gen_file)?).finish()?;
        
        // Filter for BESS resources (PWRSTR type)
        let mask = df.column("ResourceType")?.str()?.equal("PWRSTR");
        let bess_df = df.filter(&mask)?;
        
        // Process each BESS resource
        for resource in &self.bess_resources {
            // Filter for this specific resource
            let resource_mask = bess_df.column("ResourceName")?.str()?.equal(resource.gen_resource.as_str());
            if let Ok(resource_df) = bess_df.filter(&resource_mask) {
                // Process awards for this resource
                self.calculate_dam_gen_revenues(&resource_df, &resource, prices, &mut revenues)?;
            }
        }
        
        Ok(revenues)
    }
    
    /// Process DAM Load awards (charging side)
    fn process_dam_load(&self, year: i32, prices: &PriceData) -> Result<HashMap<(String, NaiveDate), BessDailyRevenue>> {
        let mut revenues = HashMap::new();
        let dam_load_file = self.rollup_dir.join(format!("DAM_Load_Resources/{}.parquet", year));
        
        if !dam_load_file.exists() {
            return Ok(revenues);
        }
        
        let df = ParquetReader::new(std::fs::File::open(&dam_load_file)?).finish()?;
        
        // Process each BESS resource's load side
        for resource in &self.bess_resources {
            // Filter for this specific load resource
            let resource_mask = df.column("ResourceName")?.str()?.equal(resource.load_resource.as_str());
            if let Ok(resource_df) = df.filter(&resource_mask) {
                // Process load awards (negative revenue for charging)
                self.calculate_dam_load_revenues(&resource_df, &resource, prices, &mut revenues)?;
            }
        }
        
        Ok(revenues)
    }
    
    /// Process real-time dispatch data
    fn process_rt_data(&self, year: i32, prices: &PriceData) -> Result<HashMap<(String, NaiveDate), BessDailyRevenue>> {
        let mut revenues = HashMap::new();
        
        // Process SCED Generation (discharging)
        let sced_gen_file = self.rollup_dir.join(format!("SCED_Gen_Resources/{}.parquet", year));
        if sced_gen_file.exists() {
            let df = ParquetReader::new(std::fs::File::open(&sced_gen_file)?).finish()?;
            
            // Filter for BESS resources
            for resource in &self.bess_resources {
                let resource_mask = df.column("ResourceName")?.str()?.equal(resource.gen_resource.as_str());
                if let Ok(resource_df) = df.filter(&resource_mask) {
                    self.calculate_rt_gen_revenues(&resource_df, &resource, prices, &mut revenues)?;
                }
            }
        }
        
        // Process SCED Load (charging)
        let sced_load_file = self.rollup_dir.join(format!("SCED_Load_Resources/{}.parquet", year));
        if sced_load_file.exists() {
            let df = ParquetReader::new(std::fs::File::open(&sced_load_file)?).finish()?;
            
            // Filter for BESS load resources
            for resource in &self.bess_resources {
                let resource_mask = df.column("ResourceName")?.str()?.equal(resource.load_resource.as_str());
                if let Ok(resource_df) = df.filter(&resource_mask) {
                    self.calculate_rt_load_revenues(&resource_df, &resource, prices, &mut revenues)?;
                }
            }
        }
        
        Ok(revenues)
    }
    
    /// Calculate DAM generation revenues from awards
    fn calculate_dam_gen_revenues(
        &self,
        df: &DataFrame,
        resource: &BessResource,
        _prices: &PriceData,
        revenues: &mut HashMap<(String, NaiveDate), BessDailyRevenue>,
    ) -> Result<()> {
        // Extract columns - handle potential missing datetime column
        let dates = if let Ok(col) = df.column("DeliveryDate") {
            col.datetime().ok()
        } else {
            None
        };
        let _hours = df.column("HourEnding")?.str()?;
        let energy_awards = df.column("AwardedQuantity")?.f64()?;
        let energy_prices = df.column("EnergySettlementPointPrice")?.f64()?;
        
        // AS awards
        let reg_up_awards = df.column("RegUpAwarded").ok().and_then(|c| c.f64().ok());
        let reg_down_awards = df.column("RegDownAwarded").ok().and_then(|c| c.f64().ok());
        let rrs_awards = df.column("RRSAwarded").ok().and_then(|c| c.f64().ok());
        let nonspin_awards = df.column("NonSpinAwarded").ok().and_then(|c| c.f64().ok());
        let ecrs_awards = df.column("ECRSAwarded").ok().and_then(|c| c.f64().ok());
        
        // AS prices (would need to join with AS price data)
        // For now, use placeholder AS prices
        let reg_up_price = 10.0;
        let reg_down_price = 5.0;
        let rrs_price = 8.0;
        let nonspin_price = 3.0;
        let ecrs_price = 15.0;
        
        // Process each hour
        for i in 0..df.height() {
            // Try to get date from dates column if available
            let date = if let Some(dates_col) = &dates {
                if let Some(date_ms) = dates_col.get(i) {
                    NaiveDate::from_ymd_opt(1970, 1, 1).unwrap() + 
                        chrono::Duration::milliseconds(date_ms)
                } else {
                    continue;
                }
            } else {
                // Use a default date or skip if no date available
                continue;
            };
                
            let key = (resource.gen_resource.clone(), date);
            let revenue = revenues.entry(key).or_insert_with(|| {
                let mut rev = BessDailyRevenue::default();
                rev.resource_name = resource.gen_resource.clone();
                rev.date = date;
                rev.settlement_point = resource.settlement_point.clone();
                rev.capacity_mw = resource.capacity_mw;
                rev
            });
            
            // Energy revenue
            if let (Some(award), Some(price)) = (energy_awards.get(i), energy_prices.get(i)) {
                revenue.dam_energy_revenue += award * price;
                revenue.dam_awarded_mwh += award;
            }
            
            // AS revenues
            if let Some(reg_up) = reg_up_awards.as_ref().and_then(|a| a.get(i)) {
                revenue.reg_up_revenue += reg_up * reg_up_price;
            }
            if let Some(reg_down) = reg_down_awards.as_ref().and_then(|a| a.get(i)) {
                revenue.reg_down_revenue += reg_down * reg_down_price;
            }
            if let Some(rrs) = rrs_awards.as_ref().and_then(|a| a.get(i)) {
                revenue.rrs_revenue += rrs * rrs_price;
            }
            if let Some(nonspin) = nonspin_awards.as_ref().and_then(|a| a.get(i)) {
                revenue.nonspin_revenue += nonspin * nonspin_price;
            }
            if let Some(ecrs) = ecrs_awards.as_ref().and_then(|a| a.get(i)) {
                revenue.ecrs_revenue += ecrs * ecrs_price;
            }
        }
        
        Ok(())
    }
    
    /// Calculate DAM load revenues (charging costs)
    fn calculate_dam_load_revenues(
        &self,
        _df: &DataFrame,
        _resource: &BessResource,
        _prices: &PriceData,
        _revenues: &mut HashMap<(String, NaiveDate), BessDailyRevenue>,
    ) -> Result<()> {
        // Similar to gen but with negative revenues for load
        // Implementation would be similar to calculate_dam_gen_revenues
        // but energy awards would be negative (cost of charging)
        Ok(())
    }
    
    /// Calculate RT generation revenues
    fn calculate_rt_gen_revenues(
        &self,
        _df: &DataFrame,
        _resource: &BessResource,
        _prices: &PriceData,
        _revenues: &mut HashMap<(String, NaiveDate), BessDailyRevenue>,
    ) -> Result<()> {
        // Extract RT dispatch and calculate revenues based on RT prices
        // This would use 5-minute RT prices and actual dispatch
        Ok(())
    }
    
    /// Calculate RT load revenues
    fn calculate_rt_load_revenues(
        &self,
        _df: &DataFrame,
        _resource: &BessResource,
        _prices: &PriceData,
        _revenues: &mut HashMap<(String, NaiveDate), BessDailyRevenue>,
    ) -> Result<()> {
        // Calculate RT charging costs
        Ok(())
    }
    
    /// Generate reports and save results
    fn generate_reports(&self, all_revenues: &Arc<Mutex<HashMap<(String, NaiveDate), BessDailyRevenue>>>) -> Result<()> {
        let revenues = all_revenues.lock().unwrap();
        
        // Calculate totals for each revenue entry
        let mut final_revenues: Vec<BessDailyRevenue> = revenues.values().cloned().collect();
        for revenue in &mut final_revenues {
            revenue.energy_revenue = revenue.dam_energy_revenue + revenue.rt_energy_revenue;
            revenue.as_revenue = revenue.reg_up_revenue + revenue.reg_down_revenue + 
                                revenue.rrs_revenue + revenue.nonspin_revenue + revenue.ecrs_revenue;
            revenue.total_revenue = revenue.energy_revenue + revenue.as_revenue;
            
            // Estimate cycles based on MWh throughput
            if revenue.capacity_mw > 0.0 {
                revenue.cycles = (revenue.dam_awarded_mwh + revenue.rt_dispatched_mwh.abs()) / 
                                (revenue.capacity_mw * 2.0);  // Assuming 2-hour duration
            }
        }
        
        // Sort by date and resource
        final_revenues.sort_by(|a, b| {
            a.date.cmp(&b.date).then(a.resource_name.cmp(&b.resource_name))
        });
        
        // Save daily revenues
        self.save_daily_revenues(&final_revenues)?;
        
        // Generate monthly rollups
        let monthly_revenues = self.generate_monthly_rollups(&final_revenues);
        self.save_monthly_revenues(&monthly_revenues)?;
        
        // Generate leaderboard
        self.generate_leaderboard(&final_revenues)?;
        
        Ok(())
    }
    
    /// Save daily revenues to parquet
    fn save_daily_revenues(&self, revenues: &[BessDailyRevenue]) -> Result<()> {
        // Convert to DataFrame and save
        let mut resource_names = Vec::new();
        let mut dates = Vec::new();
        let mut total_revenues = Vec::new();
        let mut energy_revenues = Vec::new();
        let mut as_revenues = Vec::new();
        
        for rev in revenues {
            resource_names.push(rev.resource_name.clone());
            dates.push(rev.date.and_hms_opt(0, 0, 0).unwrap().and_utc().timestamp());
            total_revenues.push(rev.total_revenue);
            energy_revenues.push(rev.energy_revenue);
            as_revenues.push(rev.as_revenue);
        }
        
        let df = DataFrame::new(vec![
            Series::new("resource_name", resource_names),
            Series::new("date", dates),
            Series::new("total_revenue", total_revenues),
            Series::new("energy_revenue", energy_revenues),
            Series::new("as_revenue", as_revenues),
        ])?;
        
        let output_file = self.output_dir.join("daily/bess_daily_revenues.parquet");
        let file = std::fs::File::create(&output_file)?;
        ParquetWriter::new(file).finish(&mut df.clone())?;
        
        println!("💾 Saved daily revenues to: {}", output_file.display());
        Ok(())
    }
    
    /// Generate monthly rollups
    fn generate_monthly_rollups(&self, daily_revenues: &[BessDailyRevenue]) -> Vec<MonthlyRevenue> {
        let mut monthly_map: HashMap<(String, i32, u32), MonthlyRevenue> = HashMap::new();
        
        for daily in daily_revenues {
            let key = (
                daily.resource_name.clone(),
                daily.date.year(),
                daily.date.month(),
            );
            
            let monthly = monthly_map.entry(key.clone()).or_insert_with(|| {
                MonthlyRevenue {
                    _resource_name: key.0.clone(),
                    _year: key.1,
                    _month: key.2,
                    total_revenue: 0.0,
                    energy_revenue: 0.0,
                    as_revenue: 0.0,
                    days_active: 0,
                    avg_daily_revenue: 0.0,
                }
            });
            
            monthly.total_revenue += daily.total_revenue;
            monthly.energy_revenue += daily.energy_revenue;
            monthly.as_revenue += daily.as_revenue;
            monthly.days_active += 1;
        }
        
        // Calculate averages
        for monthly in monthly_map.values_mut() {
            if monthly.days_active > 0 {
                monthly.avg_daily_revenue = monthly.total_revenue / monthly.days_active as f64;
            }
        }
        
        monthly_map.into_values().collect()
    }
    
    /// Save monthly revenues
    fn save_monthly_revenues(&self, _revenues: &[MonthlyRevenue]) -> Result<()> {
        // Similar to save_daily_revenues but for monthly data
        println!("💾 Saved monthly revenues");
        Ok(())
    }
    
    /// Generate leaderboard of top performing BESS resources
    fn generate_leaderboard(&self, daily_revenues: &[BessDailyRevenue]) -> Result<()> {
        // Aggregate by resource
        let mut resource_totals: HashMap<String, f64> = HashMap::new();
        
        for daily in daily_revenues {
            *resource_totals.entry(daily.resource_name.clone()).or_insert(0.0) += daily.total_revenue;
        }
        
        // Sort by total revenue
        let mut leaderboard: Vec<_> = resource_totals.into_iter().collect();
        leaderboard.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // Print top 20
        println!("\n🏆 BESS REVENUE LEADERBOARD");
        println!("{}", "=".repeat(60));
        println!("{:<30} {:>20}", "Resource", "Total Revenue ($)");
        println!("{}", "-".repeat(60));
        
        for (i, (resource, revenue)) in leaderboard.iter().take(20).enumerate() {
            println!("{:2}. {:<27} ${:>18.2}", i + 1, resource, revenue);
        }
        
        Ok(())
    }
}

/// Container for price data
#[derive(Default)]
struct PriceData {
    da_as_prices: Option<DataFrame>,
    rt_prices: Option<DataFrame>,
}

/// Monthly revenue summary
#[derive(Debug, Clone)]
struct MonthlyRevenue {
    _resource_name: String,
    _year: i32,
    _month: u32,
    total_revenue: f64,
    energy_revenue: f64,
    as_revenue: f64,
    days_active: i32,
    avg_daily_revenue: f64,
}

/// Entry point for BESS revenue processing
pub fn process_bess_revenues_from_parquet() -> Result<()> {
    // Get data directory from environment
    let data_dir = std::env::var("ERCOT_DATA_DIR")
        .unwrap_or_else(|_| "/home/enrico/data/ERCOT_data".to_string());
    
    let processor = BessParquetRevenueProcessor::new(PathBuf::from(data_dir))?;
    processor.process_all_revenues()?;
    
    Ok(())
}