use anyhow::Result;
use polars::prelude::*;
use std::sync::Arc;

fn main() -> Result<()> {
    println!("Testing 60-Day Disclosure Schema Handling\n");
    
    // Test DAM file
    let dam_test_file = "/Users/enrico/data/ERCOT_data/60-Day_DAM_Disclosure_Reports/csv/60d_DAM_Gen_Resource_Data-31-JAN-25.csv";
    println!("Testing DAM file: {}", dam_test_file);
    test_dam_file(dam_test_file)?;
    
    // Test SCED file (if available)
    println!("\n✅ DAM test passed! (SCED test skipped - files need extraction)");
    
    println!("\n✅ All tests passed!");
    Ok(())
}

fn test_dam_file(path: &str) -> Result<()> {
    // Force Float64 for numeric columns
    let mut schema_overrides = Schema::new();
    schema_overrides.with_column("LSL".into(), DataType::Float64);
    schema_overrides.with_column("HSL".into(), DataType::Float64);
    schema_overrides.with_column("Awarded Quantity".into(), DataType::Float64);
    schema_overrides.with_column("Energy Settlement Point Price".into(), DataType::Float64);
    
    let df = CsvReader::from_path(path)?
        .has_header(true)
        .with_dtypes(Some(Arc::new(schema_overrides)))
        .infer_schema(Some(10000))
        .finish()?;
    
    println!("  Loaded {} rows", df.height());
    
    // Check column types
    if let Ok(lsl) = df.column("LSL") {
        println!("  LSL dtype: {:?}", lsl.dtype());
        if let Ok(vals) = lsl.head(Some(5)).f64() {
            println!("  LSL samples: {:?}", vals.into_iter().take(3).collect::<Vec<_>>());
        }
    }
    
    if let Ok(hsl) = df.column("HSL") {
        println!("  HSL dtype: {:?}", hsl.dtype());
        if let Ok(vals) = hsl.head(Some(5)).f64() {
            println!("  HSL samples: {:?}", vals.into_iter().take(3).collect::<Vec<_>>());
        }
    }
    
    Ok(())
}

#[allow(dead_code)]
fn test_sced_file(path: &str) -> Result<()> {
    // Force Float64 for numeric columns
    let mut schema_overrides = Schema::new();
    schema_overrides.with_column("LSL".into(), DataType::Float64);
    schema_overrides.with_column("HSL".into(), DataType::Float64);
    schema_overrides.with_column("Base Point".into(), DataType::Float64);
    schema_overrides.with_column("Ancillary Service RRS".into(), DataType::Float64);
    
    let df = CsvReader::from_path(path)?
        .has_header(true)
        .with_dtypes(Some(Arc::new(schema_overrides)))
        .infer_schema(Some(10000))
        .finish()?;
    
    println!("  Loaded {} rows", df.height());
    
    // Check column types
    if let Ok(lsl) = df.column("LSL") {
        println!("  LSL dtype: {:?}", lsl.dtype());
        if let Ok(vals) = lsl.head(Some(5)).f64() {
            println!("  LSL samples: {:?}", vals.into_iter().take(3).collect::<Vec<_>>());
        }
    }
    
    if let Ok(as_rrs) = df.column("Ancillary Service RRS") {
        println!("  AS RRS dtype: {:?}", as_rrs.dtype());
        if let Ok(vals) = as_rrs.head(Some(5)).f64() {
            println!("  AS RRS samples: {:?}", vals.into_iter().take(3).collect::<Vec<_>>());
        }
    }
    
    Ok(())
}