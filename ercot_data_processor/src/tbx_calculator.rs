use anyhow::Result;
use chrono::{Datelike, NaiveDate};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use polars::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

/// TBX arbitrage calculation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TbxResult {
    pub date: String,
    pub year: i32,
    pub month: u32,
    pub day: u32,
    pub node: String,
    pub tb2_revenue: f64,      // 2-hour battery revenue
    pub tb4_revenue: f64,      // 4-hour battery revenue
    pub tb2_charge_cost: f64,
    pub tb2_discharge_revenue: f64,
    pub tb4_charge_cost: f64,
    pub tb4_discharge_revenue: f64,
    pub da_price_mean: f64,
    pub da_price_std: f64,
    pub rt_price_mean: f64,
    pub rt_price_std: f64,
}

/// Monthly aggregated results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonthlyTbxResult {
    pub year: i32,
    pub month: u32,
    pub node: String,
    pub tb2_total_revenue: f64,
    pub tb4_total_revenue: f64,
    pub tb2_revenue_per_mw_month: f64,
    pub tb4_revenue_per_mw_month: f64,
    pub days_calculated: i32,
    pub volatility_score: f64,
}

/// TBX Calculator configuration
pub struct TbxCalculator {
    data_dir: PathBuf,
    output_dir: PathBuf,
    efficiency: f64,
    num_threads: usize,
}

impl TbxCalculator {
    pub fn new(data_dir: PathBuf, output_dir: PathBuf, efficiency: f64) -> Result<Self> {
        std::fs::create_dir_all(&output_dir)?;
        let num_threads = num_cpus::get();
        
        println!("🔋 TBX Calculator initialized");
        println!("  📁 Data directory: {}", data_dir.display());
        println!("  📁 Output directory: {}", output_dir.display());
        println!("  ⚡ Efficiency: {:.0}%", efficiency * 100.0);
        println!("  🚀 Parallel threads: {}", num_threads);
        
        Ok(Self {
            data_dir,
            output_dir,
            efficiency,
            num_threads,
        })
    }
    
    /// Process all years and nodes
    pub fn process_all_years(&self, start_year: i32, end_year: i32) -> Result<()> {
        println!("\n💰 TBX BATTERY ARBITRAGE CALCULATOR");
        println!("{}", "=".repeat(80));
        println!("📅 Processing years: {} to {}", start_year, end_year);
        
        let years: Vec<i32> = (start_year..=end_year).collect();
        let multi_progress = Arc::new(MultiProgress::new());
        
        // Configure thread pool
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(self.num_threads)
            .build()?;
        
        // Process each year in parallel
        pool.install(|| {
            years.par_iter().for_each(|&year| {
                let year_pb = multi_progress.add(ProgressBar::new(100));
                year_pb.set_style(
                    ProgressStyle::default_bar()
                        .template(&format!("{} {{bar:40}} {{pos}}/{{len}} {{msg}}", year))
                        .unwrap()
                );
                
                match self.process_year(year, &year_pb) {
                    Ok(_) => year_pb.finish_with_message("✅ Complete"),
                    Err(e) => {
                        let msg = format!("❌ Error: {}", e);
                        year_pb.finish_with_message(msg)
                    }
                }
            });
        });
        
        println!("\n✅ All years processed successfully!");
        Ok(())
    }
    
    /// Process a single year
    fn process_year(&self, year: i32, pb: &ProgressBar) -> Result<()> {
        pb.set_message("Loading DA prices...");
        
        // Load DA prices
        let da_file = self.data_dir.join("rollup_files/flattened")
            .join(format!("DA_prices_{}.parquet", year));
        
        if !da_file.exists() {
            return Err(anyhow::anyhow!("DA prices not found for {}", year));
        }
        
        let da_prices = LazyFrame::scan_parquet(&da_file, Default::default())?.collect()?;
        
        pb.set_message("Loading RT prices...");
        
        // Load RT prices (hourly aggregated)
        let rt_file = self.data_dir.join("rollup_files/flattened")
            .join(format!("RT_prices_hourly_{}.parquet", year));
        
        let rt_prices = if rt_file.exists() {
            Some(LazyFrame::scan_parquet(&rt_file, Default::default())?.collect()?)
        } else {
            None
        };
        
        pb.set_message("Calculating arbitrage...");
        
        // Get list of nodes (settlement points)
        let nodes = self.get_nodes(&da_prices)?;
        pb.set_length(nodes.len() as u64);
        
        // Calculate TBX for each node
        let all_results: Vec<Vec<TbxResult>> = nodes
            .par_iter()
            .map(|node| {
                pb.inc(1);
                let msg = format!("Processing {}", node);
                pb.set_message(msg);
                self.calculate_node_tbx(&da_prices, rt_prices.as_ref(), node, year)
                    .unwrap_or_default()
            })
            .collect();
        
        // Flatten and save results
        let mut all_tbx_results = Vec::new();
        for node_results in all_results {
            all_tbx_results.extend(node_results);
        }
        
        if !all_tbx_results.is_empty() {
            self.save_results(&all_tbx_results, year)?;
            let msg = format!("Saved {} results", all_tbx_results.len());
            pb.set_message(msg);
        }
        
        Ok(())
    }
    
    /// Get list of nodes from price data
    fn get_nodes(&self, df: &DataFrame) -> Result<Vec<String>> {
        let mut nodes = Vec::new();
        
        // Skip datetime columns
        for col_name in df.get_column_names() {
            if !col_name.contains("datetime") && !col_name.contains("date") {
                nodes.push(col_name.to_string());
            }
        }
        
        // Limit to top nodes for performance
        if nodes.len() > 100 {
            nodes.truncate(100);  // Process top 100 nodes
        }
        
        Ok(nodes)
    }
    
    /// Calculate TBX for a single node
    fn calculate_node_tbx(
        &self,
        da_prices: &DataFrame,
        rt_prices: Option<&DataFrame>,
        node: &str,
        year: i32,
    ) -> Result<Vec<TbxResult>> {
        let mut results = Vec::new();
        
        // Get node prices
        let node_da = da_prices.column(node).ok();
        if node_da.is_none() {
            return Ok(results);
        }
        
        let node_da = node_da.unwrap().f64()?;
        
        // Get datetime column
        let datetime_col = da_prices.column("datetime")
            .or_else(|_| da_prices.column("datetime_ts"))?;
        
        // Convert to dates
        let dates = if let Ok(dt_series) = datetime_col.datetime() {
            dt_series.as_datetime_iter()
                .map(|d| d.map(|timestamp| timestamp.date()))
                .collect::<Vec<_>>()
        } else {
            return Ok(results);
        };
        
        // Get RT prices for node if available
        let node_rt = rt_prices.and_then(|rt| rt.column(node).ok())
            .and_then(|col| col.f64().ok());
        
        // Group by day and calculate arbitrage
        let mut day_data: HashMap<NaiveDate, Vec<(f64, Option<f64>)>> = HashMap::new();
        
        for (i, date_opt) in dates.iter().enumerate() {
            if let Some(date) = date_opt {
                if let Some(da_price) = node_da.get(i) {
                    let rt_price = node_rt.and_then(|rt| rt.get(i));
                    day_data.entry(*date)
                        .or_insert_with(Vec::new)
                        .push((da_price, rt_price));
                }
            }
        }
        
        // Calculate daily TBX
        for (date, hourly_prices) in day_data {
            if hourly_prices.len() < 24 {
                continue;  // Skip incomplete days
            }
            
            // Use DA prices for arbitrage calculation
            let da_hourly: Vec<f64> = hourly_prices.iter()
                .map(|(da, _)| *da)
                .collect();
            
            // Calculate TB2 (2-hour battery)
            let tb2 = self.calculate_battery_arbitrage(&da_hourly, 2);
            
            // Calculate TB4 (4-hour battery)
            let tb4 = self.calculate_battery_arbitrage(&da_hourly, 4);
            
            // Calculate statistics
            let da_mean = da_hourly.iter().sum::<f64>() / da_hourly.len() as f64;
            let da_std = self.calculate_std(&da_hourly, da_mean);
            
            let rt_hourly: Vec<f64> = hourly_prices.iter()
                .filter_map(|(_, rt)| *rt)
                .collect();
            
            let (rt_mean, rt_std) = if !rt_hourly.is_empty() {
                let mean = rt_hourly.iter().sum::<f64>() / rt_hourly.len() as f64;
                let std = self.calculate_std(&rt_hourly, mean);
                (mean, std)
            } else {
                (da_mean, da_std)  // Use DA as fallback
            };
            
            results.push(TbxResult {
                date: date.to_string(),
                year: date.year(),
                month: date.month(),
                day: date.day(),
                node: node.to_string(),
                tb2_revenue: tb2.2,
                tb4_revenue: tb4.2,
                tb2_charge_cost: tb2.0,
                tb2_discharge_revenue: tb2.1,
                tb4_charge_cost: tb4.0,
                tb4_discharge_revenue: tb4.1,
                da_price_mean: da_mean,
                da_price_std: da_std,
                rt_price_mean: rt_mean,
                rt_price_std: rt_std,
            });
        }
        
        Ok(results)
    }
    
    /// Calculate battery arbitrage for given hours
    fn calculate_battery_arbitrage(&self, prices: &[f64], hours: usize) -> (f64, f64, f64) {
        if prices.len() < hours * 2 {
            return (0.0, 0.0, 0.0);
        }
        
        // Find lowest hours for charging
        let mut indexed_prices: Vec<(usize, f64)> = prices.iter()
            .enumerate()
            .map(|(i, &p)| (i, p))
            .collect();
        
        indexed_prices.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        
        let charge_hours = &indexed_prices[..hours];
        let discharge_hours = &indexed_prices[prices.len() - hours..];
        
        // Calculate costs and revenues
        let charge_cost: f64 = charge_hours.iter().map(|(_, p)| p).sum();
        let discharge_revenue: f64 = discharge_hours.iter().map(|(_, p)| p).sum();
        
        // Apply efficiency on discharge
        let net_revenue = discharge_revenue * self.efficiency - charge_cost;
        
        (charge_cost, discharge_revenue, net_revenue)
    }
    
    /// Calculate standard deviation
    fn calculate_std(&self, values: &[f64], mean: f64) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        
        let variance = values.iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>() / values.len() as f64;
        
        variance.sqrt()
    }
    
    /// Save results to Parquet files
    fn save_results(&self, results: &[TbxResult], year: i32) -> Result<()> {
        // Convert to DataFrame
        let df = self.results_to_dataframe(results)?;
        
        // Save daily results
        let daily_file = self.output_dir.join(format!("tbx_daily_{}.parquet", year));
        let mut file = std::fs::File::create(&daily_file)?;
        ParquetWriter::new(&mut file).finish(&mut df.clone())?;
        
        // Create monthly aggregation
        let monthly = self.aggregate_monthly(&df)?;
        let monthly_file = self.output_dir.join(format!("tbx_monthly_{}.parquet", year));
        let mut file = std::fs::File::create(&monthly_file)?;
        ParquetWriter::new(&mut file).finish(&mut monthly.clone())?;
        
        // Create annual aggregation
        let annual = self.aggregate_annual(&df)?;
        let annual_file = self.output_dir.join(format!("tbx_annual_{}.parquet", year));
        let mut file = std::fs::File::create(&annual_file)?;
        ParquetWriter::new(&mut file).finish(&mut annual.clone())?;
        
        Ok(())
    }
    
    /// Convert results to DataFrame
    fn results_to_dataframe(&self, results: &[TbxResult]) -> Result<DataFrame> {
        Ok(DataFrame::new(vec![
            Series::new("date", results.iter().map(|r| r.date.clone()).collect::<Vec<_>>()),
            Series::new("year", results.iter().map(|r| r.year).collect::<Vec<_>>()),
            Series::new("month", results.iter().map(|r| r.month).collect::<Vec<_>>()),
            Series::new("day", results.iter().map(|r| r.day).collect::<Vec<_>>()),
            Series::new("node", results.iter().map(|r| r.node.clone()).collect::<Vec<_>>()),
            Series::new("tb2_revenue", results.iter().map(|r| r.tb2_revenue).collect::<Vec<_>>()),
            Series::new("tb4_revenue", results.iter().map(|r| r.tb4_revenue).collect::<Vec<_>>()),
            Series::new("tb2_charge_cost", results.iter().map(|r| r.tb2_charge_cost).collect::<Vec<_>>()),
            Series::new("tb2_discharge_revenue", results.iter().map(|r| r.tb2_discharge_revenue).collect::<Vec<_>>()),
            Series::new("tb4_charge_cost", results.iter().map(|r| r.tb4_charge_cost).collect::<Vec<_>>()),
            Series::new("tb4_discharge_revenue", results.iter().map(|r| r.tb4_discharge_revenue).collect::<Vec<_>>()),
            Series::new("da_price_mean", results.iter().map(|r| r.da_price_mean).collect::<Vec<_>>()),
            Series::new("da_price_std", results.iter().map(|r| r.da_price_std).collect::<Vec<_>>()),
            Series::new("rt_price_mean", results.iter().map(|r| r.rt_price_mean).collect::<Vec<_>>()),
            Series::new("rt_price_std", results.iter().map(|r| r.rt_price_std).collect::<Vec<_>>()),
        ])?)
    }
    
    /// Aggregate to monthly results
    fn aggregate_monthly(&self, df: &DataFrame) -> Result<DataFrame> {
        df.clone().lazy()
            .group_by([col("year"), col("month"), col("node")])
            .agg([
                col("tb2_revenue").sum().alias("tb2_total_revenue"),
                col("tb4_revenue").sum().alias("tb4_total_revenue"),
                col("tb2_revenue").mean().alias("tb2_daily_avg"),
                col("tb4_revenue").mean().alias("tb4_daily_avg"),
                col("da_price_std").mean().alias("volatility_score"),
                col("date").count().alias("days_calculated"),
            ])
            .with_column((col("tb2_total_revenue") / lit(1.0)).alias("tb2_revenue_per_mw_month"))
            .with_column((col("tb4_total_revenue") / lit(1.0)).alias("tb4_revenue_per_mw_month"))
            .sort_by_exprs(
                vec![col("year"), col("month"), col("tb4_total_revenue")],
                vec![false, false, false],
                false,
                false,
            )
            .collect()
            .map_err(Into::into)
    }
    
    /// Aggregate to annual results
    fn aggregate_annual(&self, df: &DataFrame) -> Result<DataFrame> {
        df.clone().lazy()
            .group_by([col("year"), col("node")])
            .agg([
                col("tb2_revenue").sum().alias("tb2_annual_revenue"),
                col("tb4_revenue").sum().alias("tb4_annual_revenue"),
                col("tb2_revenue").mean().alias("tb2_daily_avg"),
                col("tb4_revenue").mean().alias("tb4_daily_avg"),
                col("da_price_std").mean().alias("avg_volatility"),
                col("date").count().alias("days_calculated"),
            ])
            .with_column((col("tb2_annual_revenue") / lit(1.0)).alias("tb2_revenue_per_mw_year"))
            .with_column((col("tb4_annual_revenue") / lit(1.0)).alias("tb4_revenue_per_mw_year"))
            .sort_by_exprs(
                vec![col("year"), col("tb4_annual_revenue")],
                vec![false, false],
                false,
                false,
            )
            .collect()
            .map_err(Into::into)
    }
    
    /// Generate leaderboard report
    pub fn generate_leaderboard(&self) -> Result<()> {
        println!("\n🏆 Generating TBX Leaderboard...");
        
        let mut all_annual = Vec::new();
        
        // Load all annual files
        for entry in std::fs::read_dir(&self.output_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.starts_with("tbx_annual_"))
                .unwrap_or(false)
            {
                let df = LazyFrame::scan_parquet(&path, Default::default())?.collect()?;
                all_annual.push(df);
            }
        }
        
        if all_annual.is_empty() {
            println!("No annual data found");
            return Ok(());
        }
        
        // Combine all years
        let combined = if all_annual.len() == 1 {
            all_annual.into_iter().next().unwrap()
        } else {
            // Stack dataframes vertically
            let mut combined_df = all_annual[0].clone();
            for df in &all_annual[1..] {
                combined_df = combined_df.vstack(df)?;
            }
            combined_df
        };
        
        // Create overall leaderboard
        let leaderboard = combined.lazy()
            .group_by([col("node")])
            .agg([
                col("tb2_annual_revenue").sum().alias("tb2_total_all_years"),
                col("tb4_annual_revenue").sum().alias("tb4_total_all_years"),
                col("tb2_annual_revenue").mean().alias("tb2_avg_annual"),
                col("tb4_annual_revenue").mean().alias("tb4_avg_annual"),
                col("avg_volatility").mean().alias("avg_volatility"),
            ])
            .sort_by_exprs(
                vec![col("tb4_total_all_years")],
                vec![false],
                false,
                false,
            )
            .limit(20)  // Top 20 nodes
            .collect()?;
        
        // Save leaderboard
        let leaderboard_file = self.output_dir.join("tbx_leaderboard.parquet");
        let mut file = std::fs::File::create(&leaderboard_file)?;
        ParquetWriter::new(&mut file).finish(&mut leaderboard.clone())?;
        
        // Print top performers
        println!("\n🏆 TOP 10 NODES BY TB4 REVENUE (All Years):");
        println!("{}", "-".repeat(80));
        
        if let (Ok(nodes), Ok(tb4_revenues)) = (
            leaderboard.column("node")?.str(),
            leaderboard.column("tb4_total_all_years")?.f64(),
        ) {
            for i in 0..10.min(nodes.len()) {
                if let (Some(node), Some(revenue)) = (nodes.get(i), tb4_revenues.get(i)) {
                    println!("{:3}. {:20} ${:15.2}", i + 1, node, revenue);
                }
            }
        }
        
        println!("\n✅ Leaderboard saved to: {}", leaderboard_file.display());
        
        Ok(())
    }
}

/// Main entry point for TBX calculation
pub fn run_tbx_calculation() -> Result<()> {
    let data_dir = PathBuf::from("/home/enrico/data/ERCOT_data");
    let output_dir = data_dir.join("tbx_results");
    
    let calculator = TbxCalculator::new(data_dir, output_dir, 0.9)?;
    
    // Process 2021-2025
    calculator.process_all_years(2021, 2025)?;
    
    // Generate leaderboard
    calculator.generate_leaderboard()?;
    
    Ok(())
}