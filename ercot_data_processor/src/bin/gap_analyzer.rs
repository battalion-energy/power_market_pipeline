use anyhow::Result;
use rt_rust_processor::gap_analyzer::GapAnalyzer;
use std::env;
use std::path::PathBuf;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    
    if args.len() != 2 {
        eprintln!("Usage: {} <data_path>", args[0]);
        eprintln!("Example: {} /Users/enrico/data/ERCOT_data", args[0]);
        std::process::exit(1);
    }
    
    let data_path = PathBuf::from(&args[1]);
    
    if !data_path.exists() {
        eprintln!("Error: Data path does not exist: {}", data_path.display());
        std::process::exit(1);
    }
    
    let analyzer = GapAnalyzer::new(data_path);
    analyzer.analyze_all_datasets()?;
    
    Ok(())
}