use std::path::PathBuf;
use std::env;

pub mod ercot_price_processor;
pub mod gap_analyzer;

/// Helper function to get ERCOT data directory from environment or default
pub fn get_ercot_data_dir() -> PathBuf {
    // Load .env file if it exists
    dotenv::dotenv().ok();
    
    // Try to get from environment variable, otherwise use platform-specific default
    env::var("ERCOT_DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            // Default based on platform
            if cfg!(target_os = "linux") {
                PathBuf::from("/home/enrico/data/ERCOT_data")
            } else {
                PathBuf::from("/Users/enrico/data/ERCOT_data")
            }
        })
}