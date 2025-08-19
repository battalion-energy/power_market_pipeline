use chrono::{DateTime, Datelike, NaiveDate};
use polars::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TbxConfig {
    pub efficiency: f64,  // 0.9 for 90% efficiency (10% losses)
    pub tb2_hours: usize, // 2 for TB2
    pub tb4_hours: usize, // 4 for TB4
    pub data_dir: PathBuf,
    pub output_dir: PathBuf,
    pub years: Vec<i32>,
}

impl Default for TbxConfig {
    fn default() -> Self {
        Self {
            efficiency: 0.9,
            tb2_hours: 2,
            tb4_hours: 4,
            data_dir: PathBuf::from("/home/enrico/data/ERCOT_data/rollup_files/flattened"),
            output_dir: PathBuf::from("/home/enrico/data/ERCOT_data/tbx_results"),
            years: vec![2021, 2022, 2023, 2024],
        }
    }
}

#[derive(Debug, Clone)]
pub struct DailyTbxResult {
    pub date: NaiveDate,
    pub node: String,
    pub tb2_da_revenue: f64,
    pub tb2_rt_revenue: f64,
    pub tb4_da_revenue: f64,
    pub tb4_rt_revenue: f64,
    pub tb2_charge_hours: Vec<u32>,
    pub tb2_discharge_hours: Vec<u32>,
    pub tb4_charge_hours: Vec<u32>,
    pub tb4_discharge_hours: Vec<u32>,
}

pub struct TbxCalculator {
    config: TbxConfig,
}

impl TbxCalculator {
    pub fn new(config: TbxConfig) -> Self {
        Self { config }
    }

    pub fn calculate_all(&self) -> Result<(), Box<dyn std::error::Error>> {
        std::fs::create_dir_all(&self.config.output_dir)?;

        for year in &self.config.years {
            println!("\n📅 Processing year {}", year);
            self.process_year(*year)?;
        }

        Ok(())
    }

    fn process_year(&self, year: i32) -> Result<(), Box<dyn std::error::Error>> {
        println!("  📂 Loading price data...");
        
        // Load DA prices
        let da_path = self.config.data_dir.join(format!("DA_prices_{}.parquet", year));
        let da_df = if da_path.exists() {
            Some(LazyFrame::scan_parquet(&da_path, Default::default())?.collect()?)
        } else {
            println!("    ⚠️  DA prices file not found for {}", year);
            None
        };

        // Load RT prices (15-minute)
        let rt_path = self.config.data_dir.join(format!("RT_prices_15min_{}.parquet", year));
        let rt_df = if rt_path.exists() {
            Some(LazyFrame::scan_parquet(&rt_path, Default::default())?.collect()?)
        } else {
            println!("    ⚠️  RT prices file not found for {}", year);
            None
        };

        if da_df.is_none() && rt_df.is_none() {
            println!("    ❌ No price data available for year {}", year);
            return Ok(());
        }

        // Get unique nodes from both dataframes
        let mut nodes = HashSet::new();
        
        if let Some(ref df) = da_df {
            for col in df.get_column_names() {
                if col != "datetime" {
                    nodes.insert(col.to_string());
                }
            }
        }
        
        if let Some(ref df) = rt_df {
            for col in df.get_column_names() {
                if col != "datetime" {
                    nodes.insert(col.to_string());
                }
            }
        }

        let nodes: Vec<String> = nodes.into_iter().collect();
        println!("  📊 Found {} unique nodes", nodes.len());

        // Process each node in parallel
        let chunk_size = (nodes.len() / rayon::current_num_threads()).max(1);
        let daily_results: Vec<DailyTbxResult> = nodes
            .par_chunks(chunk_size)
            .flat_map(|node_chunk| {
                node_chunk.iter().flat_map(|node| {
                    self.calculate_node_tbx_year(node, &da_df, &rt_df, year)
                        .unwrap_or_else(|e| {
                            eprintln!("    ⚠️  Error processing node {}: {}", node, e);
                            Vec::new()
                        })
                }).collect::<Vec<_>>()
            })
            .collect();

        println!("  💾 Calculated {} daily results", daily_results.len());

        // Save results
        self.save_results(&daily_results, year)?;
        
        println!("  ✅ Year {} complete!", year);
        Ok(())
    }

    fn calculate_node_tbx_year(
        &self,
        node: &str,
        da_df: &Option<DataFrame>,
        rt_df: &Option<DataFrame>,
        year: i32,
    ) -> Result<Vec<DailyTbxResult>, Box<dyn std::error::Error>> {
        let mut results = Vec::new();

        // Process DA prices
        let da_prices_by_day = if let Some(df) = da_df {
            self.extract_node_prices_by_day(df, node)?
        } else {
            HashMap::new()
        };

        // Process RT prices (convert 15-min to hourly)
        let rt_prices_by_day = if let Some(df) = rt_df {
            self.extract_rt_prices_by_day(df, node)?
        } else {
            HashMap::new()
        };

        // Get all unique days
        let mut all_days: HashSet<u32> = HashSet::new();
        all_days.extend(da_prices_by_day.keys());
        all_days.extend(rt_prices_by_day.keys());

        // Calculate TBX for each day
        for day_of_year in all_days {
            let date = NaiveDate::from_yo_opt(year, day_of_year)
                .ok_or_else(|| format!("Invalid day {} for year {}", day_of_year, year))?;
            
            let da_prices = da_prices_by_day.get(&day_of_year).cloned().unwrap_or_default();
            let rt_prices = rt_prices_by_day.get(&day_of_year).cloned().unwrap_or_default();

            if da_prices.is_empty() && rt_prices.is_empty() {
                continue;
            }

            // Calculate TB2 and TB4
            let (tb2_da_revenue, tb2_charge_hours, tb2_discharge_hours) = 
                self.calculate_tbx_for_day(&da_prices, 2);
            let (tb4_da_revenue, tb4_charge_hours, tb4_discharge_hours) = 
                self.calculate_tbx_for_day(&da_prices, 4);
            
            let (tb2_rt_revenue, _, _) = self.calculate_tbx_for_day(&rt_prices, 2);
            let (tb4_rt_revenue, _, _) = self.calculate_tbx_for_day(&rt_prices, 4);

            results.push(DailyTbxResult {
                date,
                node: node.to_string(),
                tb2_da_revenue,
                tb2_rt_revenue,
                tb4_da_revenue,
                tb4_rt_revenue,
                tb2_charge_hours,
                tb2_discharge_hours,
                tb4_charge_hours,
                tb4_discharge_hours,
            });
        }

        Ok(results)
    }

    fn extract_node_prices_by_day(
        &self,
        df: &DataFrame,
        node: &str,
    ) -> Result<HashMap<u32, Vec<f64>>, Box<dyn std::error::Error>> {
        let mut prices_by_day = HashMap::new();

        // Check if node column exists
        if !df.get_column_names().contains(&node) {
            return Ok(prices_by_day);
        }

        let datetime_col = df.column("datetime")?;
        let price_col = df.column(node)?;

        // Extract day of year from datetime
        let datetime_values = datetime_col.datetime()?.as_datetime_iter();
        let price_values = price_col.f64()?;

        for (idx, dt_opt) in datetime_values.enumerate() {
            if let Some(dt_naive) = dt_opt {
                // Convert NaiveDateTime to DateTime<Utc>
                let dt = DateTime::from_timestamp_millis(dt_naive.and_utc().timestamp_millis()).unwrap();
                let day_of_year = dt.ordinal();
                
                if let Some(price) = price_values.get(idx) {
                    prices_by_day
                        .entry(day_of_year)
                        .or_insert_with(Vec::new)
                        .push(price);
                }
            }
        }

        Ok(prices_by_day)
    }

    fn extract_rt_prices_by_day(
        &self,
        df: &DataFrame,
        node: &str,
    ) -> Result<HashMap<u32, Vec<f64>>, Box<dyn std::error::Error>> {
        let prices_by_day_15min = self.extract_node_prices_by_day(df, node)?;
        
        // Convert 15-minute prices to hourly averages
        let mut hourly_prices_by_day = HashMap::new();
        
        for (day, prices_15min) in prices_by_day_15min {
            let mut hourly_prices = Vec::new();
            
            // Group 15-min intervals into hours (4 intervals per hour)
            for hour_chunk in prices_15min.chunks(4) {
                if !hour_chunk.is_empty() {
                    let avg = hour_chunk.iter().sum::<f64>() / hour_chunk.len() as f64;
                    hourly_prices.push(avg);
                }
            }
            
            // Ensure we have 24 hours
            while hourly_prices.len() < 24 {
                hourly_prices.push(0.0);
            }
            
            hourly_prices_by_day.insert(day, hourly_prices);
        }
        
        Ok(hourly_prices_by_day)
    }

    fn calculate_tbx_for_day(
        &self,
        prices: &[f64],
        hours: usize,
    ) -> (f64, Vec<u32>, Vec<u32>) {
        if prices.len() < 24 {
            return (0.0, vec![], vec![]);
        }

        // Create indexed prices for sorting
        let mut indexed_prices: Vec<(usize, f64)> = prices[..24.min(prices.len())]
            .iter()
            .enumerate()
            .map(|(i, &p)| (i, p))
            .collect();

        // Sort by price
        indexed_prices.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Get lowest hours for charging and highest for discharging
        let charge_hours: Vec<u32> = indexed_prices[..hours]
            .iter()
            .map(|(i, _)| *i as u32)
            .collect();
        
        let discharge_hours: Vec<u32> = indexed_prices[24 - hours..]
            .iter()
            .map(|(i, _)| *i as u32)
            .collect();

        // Calculate revenue with efficiency losses
        let charge_cost: f64 = charge_hours.iter()
            .map(|&h| prices[h as usize])
            .sum();
        
        let discharge_revenue: f64 = discharge_hours.iter()
            .map(|&h| prices[h as usize])
            .sum();

        // Apply efficiency: we can only discharge efficiency * charged amount
        let net_revenue = discharge_revenue * self.config.efficiency - charge_cost;

        (net_revenue, charge_hours, discharge_hours)
    }

    fn save_results(
        &self,
        results: &[DailyTbxResult],
        year: i32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Prepare data for DataFrame
        let mut dates = Vec::new();
        let mut nodes = Vec::new();
        let mut tb2_da_revenues = Vec::new();
        let mut tb2_rt_revenues = Vec::new();
        let mut tb4_da_revenues = Vec::new();
        let mut tb4_rt_revenues = Vec::new();

        for result in results {
            dates.push(result.date.and_hms_opt(0, 0, 0).unwrap().and_utc().timestamp());
            nodes.push(result.node.clone());
            tb2_da_revenues.push(result.tb2_da_revenue);
            tb2_rt_revenues.push(result.tb2_rt_revenue);
            tb4_da_revenues.push(result.tb4_da_revenue);
            tb4_rt_revenues.push(result.tb4_rt_revenue);
        }

        // Create DataFrame
        let df = DataFrame::new(vec![
            Series::new("date", dates),
            Series::new("node", nodes),
            Series::new("tb2_da_revenue", tb2_da_revenues),
            Series::new("tb2_rt_revenue", tb2_rt_revenues),
            Series::new("tb4_da_revenue", tb4_da_revenues),
            Series::new("tb4_rt_revenue", tb4_rt_revenues),
        ])?;

        // Save daily results
        let daily_path = self.config.output_dir.join(format!("tbx_daily_{}.parquet", year));
        let mut file = std::fs::File::create(&daily_path)?;
        ParquetWriter::new(&mut file).finish(&mut df.clone())?;
        println!("    💾 Saved daily results to {:?}", daily_path);

        // Create monthly aggregation
        let monthly_df = self.aggregate_monthly(&df)?;
        let monthly_path = self.config.output_dir.join(format!("tbx_monthly_{}.parquet", year));
        let mut file = std::fs::File::create(&monthly_path)?;
        ParquetWriter::new(&mut file).finish(&mut monthly_df.clone())?;
        println!("    💾 Saved monthly results to {:?}", monthly_path);

        // Create annual aggregation
        let annual_df = self.aggregate_annual(&df)?;
        let annual_path = self.config.output_dir.join(format!("tbx_annual_{}.parquet", year));
        let mut file = std::fs::File::create(&annual_path)?;
        ParquetWriter::new(&mut file).finish(&mut annual_df.clone())?;
        println!("    💾 Saved annual results to {:?}", annual_path);

        // Print top 10 nodes by TB2 revenue
        self.print_top_nodes(&annual_df, year)?;

        Ok(())
    }

    fn aggregate_monthly(&self, df: &DataFrame) -> Result<DataFrame, Box<dyn std::error::Error>> {
        // Convert timestamp to month
        let date_series = df.column("date")?;
        let timestamps = date_series.i64()?;
        
        let mut months = Vec::new();
        for ts_opt in timestamps.into_iter() {
            if let Some(ts) = ts_opt {
                let dt = DateTime::from_timestamp(ts, 0).unwrap();
                months.push(dt.month() as i32);
            } else {
                months.push(0);
            }
        }
        
        let df_with_month = df.clone()
            .lazy()
            .with_column(Series::new("month", months).lit())
            .group_by([col("node"), col("month")])
            .agg([
                col("tb2_da_revenue").sum().alias("tb2_da_revenue"),
                col("tb2_rt_revenue").sum().alias("tb2_rt_revenue"),
                col("tb4_da_revenue").sum().alias("tb4_da_revenue"),
                col("tb4_rt_revenue").sum().alias("tb4_rt_revenue"),
                col("date").count().alias("days_count"),
            ])
            .collect()?;
        
        Ok(df_with_month)
    }

    fn aggregate_annual(&self, df: &DataFrame) -> Result<DataFrame, Box<dyn std::error::Error>> {
        let annual_df = df.clone()
            .lazy()
            .group_by([col("node")])
            .agg([
                col("tb2_da_revenue").sum().alias("tb2_da_revenue"),
                col("tb2_rt_revenue").sum().alias("tb2_rt_revenue"),
                col("tb4_da_revenue").sum().alias("tb4_da_revenue"),
                col("tb4_rt_revenue").sum().alias("tb4_rt_revenue"),
                col("date").count().alias("days_count"),
            ])
            .collect()?;
        
        Ok(annual_df)
    }

    fn print_top_nodes(&self, annual_df: &DataFrame, year: i32) -> Result<(), Box<dyn std::error::Error>> {
        // Sort by TB2 DA revenue
        let sorted = annual_df.clone()
            .lazy()
            .sort("tb2_da_revenue", SortOptions {
                descending: true,
                nulls_last: true,
                multithreaded: true,
                maintain_order: false,
            })
            .limit(10)
            .collect()?;

        println!("\n  🏆 Top 10 Nodes by TB2 Day-Ahead Revenue for {}:", year);
        println!("  {:30} {:>15} {:>15} {:>15}", "Node", "TB2 DA ($)", "TB4 DA ($)", "Days");
        println!("  {}", "-".repeat(80));

        let nodes = sorted.column("node")?;
        let tb2_da = sorted.column("tb2_da_revenue")?.f64()?;
        let tb4_da = sorted.column("tb4_da_revenue")?.f64()?;
        let days = sorted.column("days_count")?.u32()?;

        for i in 0..sorted.height().min(10) {
            let node = nodes.get(i)?.to_string();
            if let (Some(tb2), Some(tb4), Some(d)) = 
                (tb2_da.get(i), tb4_da.get(i), days.get(i)) {
                println!("  {:30} {:>15.2} {:>15.2} {:>15}", 
                    node, tb2, tb4, d);
            }
        }

        Ok(())
    }
}