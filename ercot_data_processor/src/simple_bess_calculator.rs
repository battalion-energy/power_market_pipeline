use anyhow::Result;
use polars::prelude::*;
use std::path::PathBuf;
use std::time::Instant;

/// Simple BESS revenue calculator for fast comparison
pub fn run_simple_bess_test() -> Result<()> {
    let separator = "=".repeat(60);
    println!("{}", separator);
    println!("SIMPLIFIED BESS REVENUE TEST - 2024");
    println!("{}", separator);
    
    let start = Instant::now();
    let data_dir = PathBuf::from("/home/enrico/data/ERCOT_data/rollup_files");
    
    // Load data
    println!("\nLoading data...");
    let da_prices = LazyFrame::scan_parquet(
        data_dir.join("flattened/DA_prices_2024.parquet"),
        Default::default()
    )?.collect()?;
    
    let as_prices = LazyFrame::scan_parquet(
        data_dir.join("flattened/AS_prices_2024.parquet"), 
        Default::default()
    )?.collect()?;
    
    let dam_gen = LazyFrame::scan_parquet(
        data_dir.join("DAM_Gen_Resources/2024.parquet"),
        Default::default()
    )?
    .filter(col("ResourceType").eq(lit("PWRSTR")))
    .collect()?;
    
    // Get unique BESS resources (first 10)
    let resource_col = dam_gen.column("ResourceName")?;
    let unique_resources = resource_col.unique()?.head(Some(10));
    let resource_names = unique_resources.str()?;
    
    println!("Processing {} BESS resources", unique_resources.len());
    
    // Process each BESS
    let mut results = Vec::new();
    
    for i in 0..unique_resources.len() {
        if let Some(bess_name) = resource_names.get(i) {
            // Filter for this BESS
            let bess_data = dam_gen.clone().lazy()
                .filter(col("ResourceName").eq(lit(bess_name)))
                .collect()?;
            
            if bess_data.height() == 0 {
                continue;
            }
            
            // Get awarded quantities
            let awards = bess_data.column("AwardedQuantity")?.f64()?;
            
            // Use HB_BUSAVG price (simplified - should match on datetime)
            let hub_prices = da_prices.column("HB_BUSAVG")?;
            
            // For simplicity, just use average price
            let avg_price = hub_prices.mean().unwrap_or(0.0);
            
            // Calculate DA revenue
            let total_mwh: f64 = awards.into_iter()
                .map(|v| v.unwrap_or(0.0))
                .sum();
            let da_revenue = total_mwh * avg_price;
            
            // Calculate AS revenues (simplified)
            let mut as_revenue = 0.0;
            if let Ok(regup_col) = bess_data.column("RegUpAwarded") {
                if let Ok(regup) = regup_col.f64() {
                    let regup_mw: f64 = regup.into_iter()
                        .map(|v| v.unwrap_or(0.0))
                        .sum();
                    
                    // Use average REGUP price
                    if let Ok(regup_prices) = as_prices.column("REGUP") {
                        let avg_regup_price = regup_prices.mean().unwrap_or(0.0);
                        as_revenue += regup_mw * avg_regup_price;
                    }
                }
            }
            
            if let Ok(regdn_col) = bess_data.column("RegDownAwarded") {
                if let Ok(regdn) = regdn_col.f64() {
                    let regdn_mw: f64 = regdn.into_iter()
                        .map(|v| v.unwrap_or(0.0))
                        .sum();
                    
                    // Use average REGDN price
                    if let Ok(regdn_prices) = as_prices.column("REGDN") {
                        let avg_regdn_price = regdn_prices.mean().unwrap_or(0.0);
                        as_revenue += regdn_mw * avg_regdn_price;
                    }
                }
            }
            
            let total_revenue = da_revenue + as_revenue;
            
            results.push((bess_name.to_string(), da_revenue, as_revenue, total_revenue));
            
            println!("  {}: DA=${:.0}, AS=${:.0}, Total=${:.0}", 
                bess_name, da_revenue, as_revenue, total_revenue);
        }
    }
    
    let elapsed = start.elapsed();
    
    let total_revenue: f64 = results.iter().map(|r| r.3).sum();
    
    println!("\n⏱️  Processing time: {:.2?}", elapsed);
    println!("📊 Total resources processed: {}", results.len());
    println!("💵 Total revenue: ${:.0}", total_revenue);
    
    // Save results
    let resources: Vec<String> = results.iter().map(|r| r.0.clone()).collect();
    let da_revs: Vec<f64> = results.iter().map(|r| r.1).collect();
    let as_revs: Vec<f64> = results.iter().map(|r| r.2).collect();
    let total_revs: Vec<f64> = results.iter().map(|r| r.3).collect();
    
    let df = DataFrame::new(vec![
        Series::new("resource", resources),
        Series::new("da_revenue", da_revs),
        Series::new("as_revenue", as_revs),
        Series::new("total_revenue", total_revs),
    ])?;
    
    let output_file = PathBuf::from("/tmp/rust_bess_results.parquet");
    let mut file = std::fs::File::create(&output_file)?;
    ParquetWriter::new(&mut file).finish(&mut df.clone())?;
    
    println!("\n✅ Results saved to /tmp/rust_bess_results.parquet");
    
    Ok(())
}