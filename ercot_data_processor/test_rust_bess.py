#!/usr/bin/env python3
"""Run simplified Rust BESS test directly"""

import subprocess
import sys

# Create a minimal Rust program that will compile and run
rust_code = '''
use polars::prelude::*;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", "=".repeat(60));
    println!("RUST BESS REVENUE TEST - SIMPLIFIED");
    println!("{}", "=".repeat(60));
    
    let start = Instant::now();
    
    // Load DA prices
    let da_prices = LazyFrame::scan_parquet(
        "/home/enrico/data/ERCOT_data/rollup_files/flattened/DA_prices_2024.parquet",
        Default::default()
    )?.collect()?;
    
    // Load DAM Gen data
    let dam_gen = LazyFrame::scan_parquet(
        "/home/enrico/data/ERCOT_data/rollup_files/DAM_Gen_Resources/2024.parquet",
        Default::default()
    )?
    .filter(col("ResourceType").eq(lit("PWRSTR")))
    .select([col("ResourceName"), col("AwardedQuantity")])
    .collect()?;
    
    // Get unique BESS resources
    let resources = dam_gen.column("ResourceName")?.unique()?;
    let num_resources = resources.len().min(10);
    
    println!("\\nProcessing {} BESS resources", num_resources);
    
    // Calculate simple revenue (just multiply awards by average price)
    let hub_prices = da_prices.column("HB_BUSAVG")?;
    let avg_price = hub_prices.mean().unwrap_or(0.0);
    
    let awards = dam_gen.column("AwardedQuantity")?;
    let total_mwh: f64 = awards.sum().unwrap_or(0.0);
    let total_revenue = total_mwh * avg_price;
    
    let elapsed = start.elapsed();
    
    println!("\\n⏱️  Processing time: {:.2?}", elapsed);
    println!("📊 Resources: {}", num_resources);  
    println!("💰 Avg price: ${:.2}/MWh", avg_price);
    println!("⚡ Total MWh: {:.0}", total_mwh);
    println!("💵 Total revenue: ${:.0}", total_revenue);
    
    Ok(())
}
'''

# Write temporary Rust file
with open('/tmp/test_bess.rs', 'w') as f:
    f.write(rust_code)

# Compile and run
print("Compiling Rust test...")
result = subprocess.run([
    'rustc', 
    '--edition', '2021',
    '-L', '/home/enrico/projects/power_market_pipeline/ercot_data_processor/target/release/deps',
    '--extern', 'polars=/home/enrico/projects/power_market_pipeline/ercot_data_processor/target/release/deps/libpolars-*.rlib',
    '-o', '/tmp/test_bess',
    '/tmp/test_bess.rs'
], capture_output=True, text=True)

if result.returncode != 0:
    print("Compilation failed. Trying cargo script instead...")
    # Use cargo to run it
    result = subprocess.run([
        'cargo', 'script', '/tmp/test_bess.rs'
    ], capture_output=True, text=True, cwd='/home/enrico/projects/power_market_pipeline/ercot_data_processor')
    
    if result.returncode != 0:
        print("Failed to run Rust test")
        print(result.stderr)
        sys.exit(1)
    else:
        print(result.stdout)
else:
    print("Running test...")
    result = subprocess.run(['/tmp/test_bess'], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)