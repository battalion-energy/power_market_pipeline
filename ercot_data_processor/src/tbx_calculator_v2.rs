use anyhow::Result;
use chrono::{Datelike, NaiveDate, NaiveDateTime};
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
    pub node: String,
    pub tb2_revenue: f64,
    pub tb4_revenue: f64,
    pub price_mean: f64,
    pub price_std: f64,
}

/// TBX Calculator for processing ALL nodes from raw DA price data
pub struct TbxCalculatorV2 {
    data_dir: PathBuf,
    output_dir: PathBuf,
    efficiency: f64,
}

impl TbxCalculatorV2 {
    pub fn new(data_dir: PathBuf, output_dir: PathBuf) -> Result<Self> {
        std::fs::create_dir_all(&output_dir)?;
        
        println!("🔋 TBX Calculator V2 initialized (ALL nodes)");
        println!("  📁 Data directory: {}", data_dir.display());
        println!("  📁 Output directory: {}", output_dir.display());
        println!("  ⚡ Efficiency: 90%");
        
        Ok(Self {
            data_dir,
            output_dir,
            efficiency: 0.9,
        })
    }
    
    /// Process all years
    pub fn process_all_years(&self, start_year: i32, end_year: i32) -> Result<()> {
        println!("\n💰 TBX BATTERY ARBITRAGE CALCULATOR - ALL NODES");
        println!("{}", "=".repeat(80));
        
        for year in start_year..=end_year {
            println!("\n📅 Processing year {}", year);
            if let Err(e) = self.process_year(year) {
                eprintln!("  ❌ Error processing {}: {}", year, e);
            }
        }
        
        println!("\n✅ All years processed!");
        self.create_leaderboard(start_year, end_year)?;
        Ok(())
    }
    
    /// Process a single year from raw DA price data
    fn process_year(&self, year: i32) -> Result<()> {
        // Load raw DA prices (long format)
        let da_file = self.data_dir.join("rollup_files/DA_prices")
            .join(format!("{}.parquet", year));
        
        if !da_file.exists() {
            return Err(anyhow::anyhow!("DA prices not found for {} at {:?}", year, da_file));
        }
        
        println!("  📂 Loading {}", da_file.display());
        let df = LazyFrame::scan_parquet(&da_file, Default::default())?.collect()?;
        println!("  📊 Loaded {} rows", df.height());
        
        // Process in long format
        let results = self.calculate_tbx_long_format(&df, year)?;
        
        // Save results
        if !results.is_empty() {
            self.save_results(&results, year)?;
        }
        
        Ok(())
    }
    
    /// Calculate TBX from long format data
    fn calculate_tbx_long_format(&self, df: &DataFrame, year: i32) -> Result<Vec<TbxResult>> {
        println!("  🧮 Calculating TBX arbitrage...");
        
        // Group by DeliveryDate and SettlementPoint
        let mut results = Vec::new();
        let mut daily_prices: HashMap<(String, String), Vec<f64>> = HashMap::new();
        
        // Extract columns with correct types
        let dates_col = df.column("DeliveryDate")?;
        let dates = if let Ok(date_series) = dates_col.date() {
            // Convert dates to strings
            let date_strs: Vec<String> = date_series.as_date_iter()
                .map(|opt_date| opt_date.map(|d| d.to_string()).unwrap_or_default())
                .collect();
            date_strs
        } else {
            // Try as string
            dates_col.str()?
                .into_iter()
                .map(|opt| opt.map(|s| s.to_string()).unwrap_or_default())
                .collect()
        };
        
        let hours = df.column("HourEnding")?.str()?;
        let nodes = df.column("SettlementPoint")?.str()?;
        
        // Try both possible column names for price
        let prices_series = df.column("SettlementPointPrice")
            .or_else(|_| df.column("SPP"))?
            .cast(&DataType::Float64)?;
        let prices = prices_series.f64()?;
        
        // Collect prices by date and node
        for i in 0..df.height() {
            let date = &dates[i];
            if let (Some(hour), Some(node), Some(price)) = 
                (hours.get(i), nodes.get(i), prices.get(i)) {
                if date.is_empty() {
                    continue;
                }
                
                // Parse hour from "HH:MM" format
                let hour_int = if hour.starts_with("24:") {
                    24
                } else {
                    hour.split(':').next()
                        .and_then(|h| h.parse::<usize>().ok())
                        .unwrap_or(0)
                };
                
                if hour_int > 0 && hour_int <= 24 {
                    let key = (date.to_string(), node.to_string());
                    daily_prices.entry(key)
                        .or_insert_with(|| vec![f64::NAN; 24])
                        [hour_int - 1] = price;
                }
            }
        }
        
        println!("  📍 Processing {} node-days", daily_prices.len());
        
        // Calculate arbitrage for each node-day
        for ((date, node), prices) in daily_prices.iter() {
            // Skip incomplete days
            if prices.iter().any(|p| p.is_nan()) {
                continue;
            }
            
            let tb2 = self.calculate_battery_arbitrage(prices, 2);
            let tb4 = self.calculate_battery_arbitrage(prices, 4);
            
            let price_mean = prices.iter().sum::<f64>() / prices.len() as f64;
            let price_std = {
                let variance = prices.iter()
                    .map(|p| (p - price_mean).powi(2))
                    .sum::<f64>() / prices.len() as f64;
                variance.sqrt()
            };
            
            results.push(TbxResult {
                date: date.clone(),
                node: node.clone(),
                tb2_revenue: tb2,
                tb4_revenue: tb4,
                price_mean,
                price_std,
            });
        }
        
        println!("  ✅ Calculated {} daily results", results.len());
        Ok(results)
    }
    
    /// Calculate battery arbitrage for given hourly prices
    fn calculate_battery_arbitrage(&self, hourly_prices: &[f64], hours: usize) -> f64 {
        if hourly_prices.len() != 24 {
            return 0.0;
        }
        
        // Find best charge and discharge hours
        let mut indexed_prices: Vec<(usize, f64)> = hourly_prices.iter()
            .enumerate()
            .map(|(i, &p)| (i, p))
            .collect();
        
        // Sort by price
        indexed_prices.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        
        // Charge during lowest price hours
        let charge_hours = &indexed_prices[..hours];
        let charge_cost: f64 = charge_hours.iter()
            .map(|(_, price)| price / self.efficiency)
            .sum();
        
        // Discharge during highest price hours
        let discharge_hours = &indexed_prices[24 - hours..];
        let discharge_revenue: f64 = discharge_hours.iter()
            .map(|(_, price)| price * self.efficiency)
            .sum();
        
        discharge_revenue - charge_cost
    }
    
    /// Save results to parquet files
    fn save_results(&self, results: &[TbxResult], year: i32) -> Result<()> {
        // Convert to DataFrame
        let dates: Vec<&str> = results.iter().map(|r| r.date.as_str()).collect();
        let nodes: Vec<&str> = results.iter().map(|r| r.node.as_str()).collect();
        let tb2: Vec<f64> = results.iter().map(|r| r.tb2_revenue).collect();
        let tb4: Vec<f64> = results.iter().map(|r| r.tb4_revenue).collect();
        let means: Vec<f64> = results.iter().map(|r| r.price_mean).collect();
        let stds: Vec<f64> = results.iter().map(|r| r.price_std).collect();
        
        let df = DataFrame::new(vec![
            Series::new("date", dates),
            Series::new("node", nodes),
            Series::new("tb2_revenue", tb2),
            Series::new("tb4_revenue", tb4),
            Series::new("price_mean", means),
            Series::new("price_std", stds),
        ])?;
        
        // Save daily results
        let daily_file = self.output_dir.join(format!("tbx_daily_{}_all_nodes.parquet", year));
        let mut file = std::fs::File::create(&daily_file)?;
        ParquetWriter::new(&mut file).finish(&mut df.clone())?;
        println!("  💾 Saved {} daily records to {}", df.height(), daily_file.display());
        
        // Create monthly aggregation
        let monthly = df.lazy()
            .group_by([col("node")])
            .agg([
                col("tb2_revenue").sum().alias("tb2_total"),
                col("tb4_revenue").sum().alias("tb4_total"),
                col("price_mean").mean().alias("price_mean"),
                col("price_std").mean().alias("price_std"),
                col("date").count().alias("days"),
            ])
            .with_column(lit(year).alias("year"))
            .collect()?;
        
        let monthly_file = self.output_dir.join(format!("tbx_annual_{}_all_nodes.parquet", year));
        let mut file = std::fs::File::create(&monthly_file)?;
        ParquetWriter::new(&mut file).finish(&mut monthly.clone())?;
        println!("  💾 Saved {} annual summaries", monthly.height());
        
        Ok(())
    }
    
    /// Create leaderboard from all years
    fn create_leaderboard(&self, start_year: i32, end_year: i32) -> Result<()> {
        println!("\n🏆 Creating All-Nodes Leaderboard...");
        
        let mut all_annual = Vec::new();
        
        for year in start_year..=end_year {
            let annual_file = self.output_dir.join(format!("tbx_annual_{}_all_nodes.parquet", year));
            if annual_file.exists() {
                let df = LazyFrame::scan_parquet(&annual_file, Default::default())?.collect()?;
                all_annual.push(df);
            }
        }
        
        if all_annual.is_empty() {
            return Ok(());
        }
        
        // Concatenate all years
        let combined = if all_annual.len() == 1 {
            all_annual.into_iter().next().unwrap()
        } else {
            // Convert to LazyFrames for concatenation
            let lazy_frames: Vec<LazyFrame> = all_annual.into_iter()
                .map(|df| df.lazy())
                .collect();
            concat(&lazy_frames, UnionArgs::default())?.collect()?
        };
        
        // Aggregate across all years
        let leaderboard = combined.lazy()
            .group_by([col("node")])
            .agg([
                col("tb2_total").sum().alias("tb2_revenue"),
                col("tb4_total").sum().alias("tb4_revenue"),
                col("price_mean").mean().alias("price_mean"),
                col("price_std").mean().alias("price_std"),
                col("days").sum().alias("days"),
            ])
            .with_columns([
                (col("tb2_revenue") / col("days") * lit(365.0)).alias("tb2_per_mw_year"),
                (col("tb4_revenue") / col("days") * lit(365.0)).alias("tb4_per_mw_year"),
            ])
            .sort_by_exprs(
                vec![col("tb4_revenue")],
                vec![true],  // descending
                false,
                false,
            )
            .collect()?;
        
        // Save leaderboard
        let leader_file = self.output_dir.join("tbx_leaderboard_all_nodes.parquet");
        let mut file = std::fs::File::create(&leader_file)?;
        ParquetWriter::new(&mut file).finish(&mut leaderboard.clone())?;
        
        // Also save as CSV for easy viewing
        let csv_file = self.output_dir.join("tbx_leaderboard_all_nodes.csv");
        let mut csv_writer = CsvWriter::new(std::fs::File::create(&csv_file)?);
        csv_writer.include_header(true).finish(&mut leaderboard.clone())?;
        
        println!("  💾 Saved leaderboard with {} nodes", leaderboard.height());
        
        // Print top 10
        println!("\n  🏆 Top 10 Settlement Points (All-Time TB4 Revenue):");
        println!("  {:<20} {:>15} {:>15} {:>15}", "Node", "TB2 Revenue", "TB4 Revenue", "$/MW-year");
        println!("  {}", "-".repeat(70));
        
        let nodes = leaderboard.column("node")?.str()?;
        let tb2 = leaderboard.column("tb2_revenue")?.f64()?;
        let tb4 = leaderboard.column("tb4_revenue")?.f64()?;
        let per_year = leaderboard.column("tb4_per_mw_year")?.f64()?;
        
        for i in 0..10.min(leaderboard.height()) {
            if let (Some(node), Some(tb2_val), Some(tb4_val), Some(yearly)) = 
                (nodes.get(i), tb2.get(i), tb4.get(i), per_year.get(i)) {
                println!("  {:<20} ${:>14.2} ${:>14.2} ${:>14.2}", 
                         node, tb2_val, tb4_val, yearly);
            }
        }
        
        Ok(())
    }
}