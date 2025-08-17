use anyhow::Result;
use std::path::PathBuf;

// Import from the main crate
extern crate ercot_data_processor;
use ercot_data_processor::ercot_price_processor::ErcotPriceProcessor;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    
    if args.len() < 2 {
        println!("Usage: {} <ercot_data_directory>", args[0]);
        println!("Example: {} /Users/enrico/data/ERCOT_data", args[0]);
        return Ok(());
    }
    
    let base_path = PathBuf::from(&args[1]);
    let processor = ErcotPriceProcessor::new(base_path);
    processor.process_all()?;
    
    Ok(())
}