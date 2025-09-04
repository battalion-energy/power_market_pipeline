use anyhow::Result;
use glob::glob;
use polars::prelude::*;
use std::path::{Path, PathBuf};

mod ercot_processor;
mod comprehensive_processor;
mod process_historical;
mod dam_processor;
mod ancillary_processor;
mod lmp_processor;
mod lmp_fast_processor;
mod lmp_full_processor;
mod disclosure_processor;
mod disclosure_fast_processor;
mod bess_analyzer;
mod bess_revenue_calculator;
mod bess_visualization;
mod bess_market_report;
mod bess_yearly_analysis;
mod bess_comprehensive_calculator;
mod bess_parquet_calculator;
mod bess_disclosure_analyzer;
mod bess_full_disclosure_analyzer;
mod bess_complete_analyzer;
mod bess_daily_revenue_processor;
mod ercot_unified_processor;
mod unified_processor;
mod csv_extractor;
mod annual_processor;
// mod bess_daily_revenue_simple;
// mod bess_daily_revenue_fixed;
mod ercot_price_processor;
mod enhanced_annual_processor;
mod bess_parquet_analyzer;
mod schema_normalizer;
mod bess_historical_analyzer;
mod schema_detector;
mod enhanced_annual_processor_validated;
mod parquet_verifier;
mod bess_parquet_revenue_processor;
mod cop_file_reader;
mod tbx_calculator_polars;
mod tbx_calculator_v2;
mod unified_bess_calculator;
mod simple_bess_calculator;

fn verify_data_quality(_dir: &Path) -> Result<()> {
    println!("\n🔍 Data Quality Verification");
    println!("{}", "=".repeat(60));
    
    // Find all processed files
    let patterns = vec![
        "processed_ercot_data/**/*.parquet",
        "annual_data/*.parquet",
        "dam_annual_data/*.parquet",
        "lmp_annual_data/*.parquet",
        "ancillary_annual_data/*.parquet"
    ];
    
    let mut total_issues = 0;
    
    for pattern in patterns {
        let files: Vec<PathBuf> = glob(pattern)?
            .filter_map(Result::ok)
            .collect();
            
        if files.is_empty() {
            continue;
        }
        
        println!("\n📁 Checking {} files in {}", files.len(), pattern);
        
        for file in files {
            println!("\n  Verifying: {}", file.file_name().unwrap().to_str().unwrap());
            
            // Read the parquet file
            let df = LazyFrame::scan_parquet(&file, Default::default())?
                .collect()?;
                
            // Get datetime column name (could be datetime, DeliveryDate, etc)
            let datetime_col = if df.get_column_names().contains(&"datetime") {
                "datetime"
            } else if df.get_column_names().contains(&"DeliveryDate") {
                "DeliveryDate"
            } else if df.get_column_names().contains(&"timestamp") {
                "timestamp"
            } else {
                println!("    ⚠️  No datetime column found");
                continue;
            };
            
            // Get location column name (could be SettlementPoint, BusName, etc)
            let location_col = if df.get_column_names().contains(&"SettlementPoint") {
                "SettlementPoint"
            } else if df.get_column_names().contains(&"BusName") {
                "BusName"
            } else if df.get_column_names().contains(&"location") {
                "location"
            } else {
                println!("    ⚠️  No location column found");
                continue;
            };
            
            // Check for duplicates
            let duplicate_check = df.clone().lazy()
                .group_by([col(datetime_col), col(location_col)])
                .agg([col(datetime_col).count().alias("count")])
                .filter(col("count").gt(1))
                .collect()?;
                
            if duplicate_check.height() > 0 {
                println!("    ❌ Found {} duplicate entries", duplicate_check.height());
                total_issues += duplicate_check.height();
            } else {
                println!("    ✅ No duplicates found");
            }
            
            // Check for gaps (only for 5-minute interval data)
            if file.to_str().unwrap().contains("RT_") {
                // Sort by datetime and check intervals
                let sorted_df = df.clone().lazy()
                    .sort(datetime_col, Default::default())
                    .collect()?;
                    
                // Get unique timestamps
                let timestamps = sorted_df.column(datetime_col)?
                    .unique()?;
                    
                let mut gaps_found = 0;
                if let Ok(datetime_series) = timestamps.datetime() {
                    let values: Vec<Option<i64>> = datetime_series.into_iter().collect();
                    
                    for i in 1..values.len() {
                        if let (Some(prev), Some(curr)) = (values[i-1], values[i]) {
                            let diff_minutes = (curr - prev) / (60 * 1000); // milliseconds to minutes
                            
                            // For RT data, expect 5-minute intervals
                            if diff_minutes > 5 && diff_minutes < 60 {
                                gaps_found += 1;
                            }
                        }
                    }
                }
                
                if gaps_found > 0 {
                    println!("    ⚠️  Found {} gaps in time series", gaps_found);
                    total_issues += gaps_found;
                } else {
                    println!("    ✅ No gaps in time series");
                }
            }
            
            // Check if data is sorted
            let sorted_check = df.clone().lazy()
                .with_column(col(datetime_col).alias("datetime_sorted"))
                .sort("datetime_sorted", Default::default())
                .collect()?;
                
            let original_datetimes = df.column(datetime_col)?;
            let sorted_datetimes = sorted_check.column("datetime_sorted")?;
            
            let is_sorted = original_datetimes.equal(sorted_datetimes)?;
            if !is_sorted.all() {
                println!("    ⚠️  Data is not sorted by datetime");
                total_issues += 1;
            } else {
                println!("    ✅ Data is properly sorted");
            }
            
            // Basic statistics
            println!("    📊 Total records: {}", df.height());
            if let Ok(unique_points) = df.column(location_col) {
                println!("    📊 Unique locations: {}", unique_points.n_unique()?);
            }
        }
    }
    
    println!("\n{}", "=".repeat(60));
    if total_issues == 0 {
        println!("✅ Data quality verification passed! No issues found.");
    } else {
        println!("⚠️  Data quality verification found {} issues", total_issues);
    }
    
    Ok(())
}

// Use the shared helper from lib
use ercot_data_processor::get_ercot_data_dir;

// Removed old broken functions - now using enhanced_annual_processor exclusively

/// Check if Rust version meets minimum requirements
fn check_rust_version() -> Result<()> {
    const MIN_VERSION: &str = "1.75";
    const RECOMMENDED_VERSION: &str = "1.80";
    
    // Get version at runtime
    let output = std::process::Command::new("rustc")
        .arg("--version")
        .output()?;
    let version = std::str::from_utf8(&output.stdout)?;
    
    // Extract version number (e.g., "1.89.0" from "rustc 1.89.0 ...")
    let version_parts: Vec<&str> = version.split_whitespace().collect();
    if version_parts.len() < 2 {
        eprintln!("⚠️  Warning: Could not determine Rust version");
        return Ok(());
    }
    
    let version_str = version_parts[1];
    let version_nums: Vec<u32> = version_str
        .split('.')
        .take(2)
        .filter_map(|s| s.parse().ok())
        .collect();
    
    if version_nums.len() < 2 {
        eprintln!("⚠️  Warning: Could not parse Rust version: {}", version_str);
        return Ok(());
    }
    
    let major = version_nums[0];
    let minor = version_nums[1];
    
    // Check minimum version
    if major < 1 || (major == 1 && minor < 75) {
        return Err(anyhow::anyhow!(
            "❌ Rust version {} is too old!\n\
             Minimum required: {}\n\
             Recommended: {} or newer\n\n\
             To update Rust, run:\n\
             rustup update stable",
            version_str, MIN_VERSION, RECOMMENDED_VERSION
        ));
    }
    
    // Warn if below recommended version
    if major == 1 && minor < 80 {
        eprintln!(
            "⚠️  Warning: Rust version {} is older than recommended.\n\
             Recommended version: {} or newer\n\
             Some dependencies may require a newer version.\n\
             To update: rustup update stable\n",
            version_str, RECOMMENDED_VERSION
        );
    } else {
        println!("✅ Rust version {} meets requirements", version_str);
    }
    
    Ok(())
}

fn main() -> Result<()> {
    // Load environment variables from .env file
    dotenv::dotenv().ok();
    
    // Check Rust version requirement
    check_rust_version()?;
    
    // Set Rayon to use all available cores
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_cpus::get())
        .build_global()
        .unwrap();
    
    // Check command line arguments
    let args: Vec<String> = std::env::args().collect();
    
    if args.len() > 1 && args[1] == "--all" {
        // Process all ERCOT data types
        comprehensive_processor::process_all_ercot_data()?;
    } else if args.len() > 1 && args[1] == "--extract" {
        // Extract historical data
        process_historical::extract_and_process_historical()?;
    } else if args.len() > 1 && args[1] == "--dam" {
        // Process DAM settlement data
        dam_processor::process_all_dam_data()?;
    } else if args.len() > 1 && args[1] == "--ancillary" {
        // Process ancillary services data
        ancillary_processor::process_all_ancillary_data()?;
    } else if args.len() > 1 && args[1] == "--lmp" {
        // Process LMP data with nested extraction
        lmp_processor::process_all_lmp_data()?;
    } else if args.len() > 1 && args[1] == "--lmp-fast" {
        // Process existing LMP CSV files only
        lmp_fast_processor::process_existing_lmp_csv()?;
    } else if args.len() > 1 && args[1] == "--lmp-sample" {
        // Process sample of LMP data
        let sample_size = if args.len() > 2 {
            args[2].parse().unwrap_or(1000)
        } else {
            1000
        };
        lmp_fast_processor::process_lmp_sample(sample_size)?;
    } else if args.len() > 1 && args[1] == "--lmp-all" {
        // Process ALL LMP historical data
        lmp_full_processor::process_all_lmp_historical()?;
    } else if args.len() > 1 && args[1] == "--disclosure" {
        // Process 60-Day disclosure reports
        disclosure_processor::process_all_disclosures()?;
    } else if args.len() > 1 && args[1] == "--disclosure-fast" {
        // Process already extracted disclosure CSV files
        disclosure_fast_processor::process_disclosure_fast()?;
    } else if args.len() > 1 && args[1] == "--bess" {
        // Analyze BESS resources
        bess_analyzer::analyze_bess_resources()?;
    } else if args.len() > 1 && args[1] == "--bess-revenue" {
        // Calculate BESS revenues using Parquet files
        bess_parquet_calculator::calculate_bess_revenues_from_parquet()?;
    } else if args.len() > 1 && args[1] == "--bess-report" {
        // Generate comprehensive BESS market report
        bess_market_report::generate_market_report()?;
    } else if args.len() > 1 && args[1] == "--bess-yearly" {
        // Generate yearly BESS analysis
        bess_yearly_analysis::generate_yearly_analysis()?;
    } else if args.len() > 1 && args[1] == "--bess-viz" {
        // Generate BESS visualizations
        bess_visualization::generate_bess_visualizations()?;
    } else if args.len() > 1 && args[1] == "--bess-comprehensive" {
        // Run comprehensive BESS analysis using Parquet data
        bess_comprehensive_calculator::run_comprehensive_bess_analysis()?;
    } else if args.len() > 1 && args[1] == "--bess-disclosure" {
        // Analyze BESS revenues from 60-day disclosure data
        bess_disclosure_analyzer::analyze_bess_disclosure_revenues()?;
    } else if args.len() > 1 && args[1] == "--bess-full-disclosure" {
        // Run complete BESS analysis with full 60-day disclosure dataset
        bess_full_disclosure_analyzer::analyze_bess_with_full_disclosure()?;
    } else if args.len() > 1 && args[1] == "--bess-complete" {
        // Run complete BESS revenue analysis with all data sources
        bess_complete_analyzer::run_complete_bess_analysis()?;
    } else if args.len() > 1 && args[1] == "--bess-daily-revenue" {
        // Process BESS daily revenues from 60-day disclosure data
        bess_daily_revenue_processor::process_bess_daily_revenues()?;
    // } else if args.len() > 1 && args[1] == "--bess-daily-test" {
    //     // Test simple BESS daily revenue processing
    //     bess_daily_revenue_simple::process_simple_daily_revenues()?;
    // } else if args.len() > 1 && args[1] == "--bess-daily-fixed" {
    //     // Process BESS daily revenues with fixed schema
    //     bess_daily_revenue_fixed::process_daily_revenues_fixed()?;
    } else if args.len() > 1 && args[1] == "--process-ercot" {
        // Process all ERCOT data from source directories
        ercot_unified_processor::process_all_ercot_data()?;
    } else if args.len() > 1 && args[1] == "--unified" {
        // Process data with unified processor (recursive unzip, dedup, etc.)
        unified_processor::process_unified_data()?;
    } else if args.len() > 1 && args[1] == "--extract-csv" {
        // Extract all CSV files from nested ZIPs into a single csv folder
        if args.len() > 2 {
            let input_dir = PathBuf::from(&args[2]);
            csv_extractor::extract_csv_from_directory(input_dir)?;
        } else {
            println!("Usage: --extract-csv <directory>");
            println!("Example: --extract-csv /path/to/ERCOT_data");
        }
    } else if args.len() > 1 && args[1] == "--extract-all-ercot" {
        // Extract all ERCOT directories listed in ercot_directories.csv
        if args.len() > 2 {
            let base_dir = PathBuf::from(&args[2]);
            csv_extractor::extract_all_ercot_directories(base_dir)?;
        } else {
            println!("Usage: --extract-all-ercot <base_directory>");
            println!("Example: --extract-all-ercot $ERCOT_DATA_DIR");
        }
    } else if args.len() > 1 && args[1] == "--process-annual" {
        // Process extracted CSV files into annual CSV, Parquet, and Arrow files
        annual_processor::process_all_annual_data()?;
    } else if args.len() > 1 && args[1] == "--process-ercot-prices" {
        // Process ERCOT price data into annual parquet files
        use ercot_price_processor::ErcotPriceProcessor;
        if args.len() > 2 {
            let base_path = PathBuf::from(&args[2]);
            let processor = ErcotPriceProcessor::new(base_path);
            processor.process_all()?;
        } else {
            println!("Usage: --process-ercot-prices <base_directory>");
            println!("Example: --process-ercot-prices $ERCOT_DATA_DIR");
        }
        return Ok(());
    } else if args.len() > 1 && args[1] == "--detect-schema" {
        // Run schema detection to create type registry
        use schema_detector::SchemaDetector;
        let base_dir = if args.len() > 2 {
            PathBuf::from(&args[2])
        } else {
            get_ercot_data_dir()
        };
        let detector = SchemaDetector::new(base_dir);
        detector.generate_and_save_schema()?;
        return Ok(());
    } else if args.len() > 1 && args[1] == "--test-schema" {
        // Test schema registry loading
        use schema_detector::SchemaRegistry;
        let base_dir = if args.len() > 2 {
            PathBuf::from(&args[2])
        } else {
            get_ercot_data_dir()
        };
        let registry_path = base_dir.join("ercot_schema_registry.json");
        println!("Loading schema registry from: {:?}", registry_path);
        let registry = SchemaRegistry::load_from_file(&registry_path)?;
        println!("✅ Successfully loaded schema registry with {} patterns", registry.schemas.len());
        for schema in &registry.schemas {
            println!("  - {}: {} columns", schema.file_pattern, schema.columns.len());
        }
        return Ok(());
    } else if args.len() > 1 && args[1] == "--bess-historical" {
        // Run BESS historical revenue analysis from actual operations
        bess_historical_analyzer::analyze_bess_historical()?;
    } else if args.len() > 1 && args[1] == "--calculate-tbx-all-nodes" {
        // Calculate TBX for ALL nodes from raw DA price data
        println!("🔋 Running TBX Calculator V2 - ALL NODES");
        use tbx_calculator_v2::TbxCalculatorV2;
        
        let data_dir = PathBuf::from(std::env::var("ERCOT_DATA_DIR")
            .unwrap_or_else(|_| {
                eprintln!("⚠️  ERCOT_DATA_DIR not set, using default");
                "/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data".to_string()
            }));
        let output_dir = data_dir.join("tbx_results_all_nodes");
        
        let calculator = TbxCalculatorV2::new(data_dir, output_dir)?;
        calculator.process_all_years(2021, 2025)?;
        
    } else if args.len() > 1 && args[1] == "--calculate-tbx" {
        // Calculate TBX (TB2/TB4) battery arbitrage values
        use tbx_calculator_polars::{TbxCalculator, TbxConfig};
        
        let mut config = TbxConfig::default();
        
        // Parse optional arguments
        let mut i = 2;
        while i < args.len() {
            match args[i].as_str() {
                "--efficiency" => {
                    if i + 1 < args.len() {
                        config.efficiency = args[i + 1].parse().unwrap_or(0.9);
                        i += 2;
                    } else {
                        i += 1;
                    }
                },
                "--years" => {
                    if i + 1 < args.len() {
                        config.years = args[i + 1]
                            .split(',')
                            .filter_map(|y| y.parse().ok())
                            .collect();
                        i += 2;
                    } else {
                        i += 1;
                    }
                },
                "--data-dir" => {
                    if i + 1 < args.len() {
                        config.data_dir = PathBuf::from(&args[i + 1]);
                        i += 2;
                    } else {
                        i += 1;
                    }
                },
                "--output-dir" => {
                    if i + 1 < args.len() {
                        config.output_dir = PathBuf::from(&args[i + 1]);
                        i += 2;
                    } else {
                        i += 1;
                    }
                },
                _ => i += 1,
            }
        }
        
        println!("⚡ TBX Calculator Configuration:");
        println!("  Efficiency: {:.1}%", config.efficiency * 100.0);
        println!("  Years: {:?}", config.years);
        println!("  Data directory: {:?}", config.data_dir);
        println!("  Output directory: {:?}", config.output_dir);
        
        let calculator = TbxCalculator::new(config);
        calculator.calculate_all()
            .map_err(|e| anyhow::anyhow!("TBX calculation failed: {}", e))?;
    } else if args.len() > 1 && args[1] == "--annual-rollup-validated" {
        // Run rollup with schema validation
        use enhanced_annual_processor_validated::ValidatedAnnualProcessor;
        let base_dir = if args.len() > 2 {
            PathBuf::from(&args[2])
        } else {
            get_ercot_data_dir()
        };
        let processor = ValidatedAnnualProcessor::new(base_dir)?;
        processor.process_all_with_validation()?;
        return Ok(());
    } else if args.len() > 1 && args[1] == "--verify-parquet" {
        // Verify parquet files for data integrity
        use parquet_verifier::ParquetVerifier;
        let base_dir = if args.len() > 2 {
            PathBuf::from(&args[2])
        } else {
            get_ercot_data_dir()
        };
        
        println!("🔍 Starting parquet verification...");
        let mut verifier = ParquetVerifier::new(base_dir);
        verifier.verify_all_datasets()?;
        return Ok(());
    } else if args.len() > 1 && args[1] == "--annual-rollup" {
        // Enhanced annual rollup with gap tracking and schema normalization
        use enhanced_annual_processor::EnhancedAnnualProcessor;
        
        // Parse arguments
        let mut base_dir = get_ercot_data_dir();
        let mut dataset = None;
        
        let mut i = 2;
        while i < args.len() {
            if args[i] == "--dataset" && i + 1 < args.len() {
                dataset = Some(args[i + 1].clone());
                i += 2;
            } else if !args[i].starts_with("--") {
                base_dir = PathBuf::from(&args[i]);
                i += 1;
            } else {
                i += 1;
            }
        }
        
        let mut processor = EnhancedAnnualProcessor::new(base_dir);
        if let Some(ds) = dataset {
            processor = processor.with_dataset(ds);
        }
        processor.process_all_data()?;
        return Ok(());
    } else if args.len() > 1 && args[1] == "--bess-parquet" {
        // Analyze BESS revenues using parquet files
        bess_parquet_analyzer::analyze_bess_from_parquet()?;
        return Ok(());
    } else if args.len() > 1 && args[1] == "--bess-parquet-revenue" {
        // High-performance BESS revenue processor using parquet files
        bess_parquet_revenue_processor::process_bess_revenues_from_parquet()?;
        return Ok(());
    } else if args.len() > 1 && args[1] == "--bess-unified" {
        // Unified BESS calculator - high-performance version with all revenue streams
        unified_bess_calculator::run_unified_bess_analysis()?;
        return Ok(());
    } else if args.len() > 1 && args[1] == "--bess-simple" {
        // Run simple BESS test for comparison
        simple_bess_calculator::run_simple_bess_test()?;
        return Ok(());
    } else if args.len() > 1 && args[1] == "--tbx" {
        // Run TBX battery arbitrage calculator (Rust high-performance version)
        use ercot_data_processor::tbx_calculator;
        tbx_calculator::run_tbx_calculation()?;
        return Ok(());
    } else if args.len() > 1 && args[1] == "--verify-results" {
        // Verify data quality of processed files
        verify_data_quality(&PathBuf::from("."))?;
    } else {
        // Default to annual rollup with enhanced processor
        println!("Usage: {} [command] [options]", args[0]);
        println!("\nCommands:");
        println!("  --annual-rollup [dir] [--dataset NAME]  Process ERCOT data (optional: specific dataset)");
        println!("  --verify-parquet [dir]                  Verify parquet files for integrity & duplicates");
        println!("  --bess-parquet                          Analyze BESS revenues from parquet files");
        println!("  --bess-parquet-revenue                  High-performance BESS revenue processor (parallel)");
        println!("  --extract-all-ercot dir                 Extract all ERCOT CSV files from zips");
        println!("  --process-annual                        Process extracted CSV to annual parquet");
        println!("  --verify-results                        Verify data quality of processed files");
        println!("\nDataset options for --annual-rollup:");
        println!("  DA_prices          Day-Ahead Settlement Point Prices");
        println!("  AS_prices          Ancillary Services Clearing Prices");
        println!("  DAM_Gen_Resources  60-Day DAM Generation Resources");
        println!("  SCED_Gen_Resources 60-Day SCED Generation Resources");
        println!("  COP_Snapshots      60-Day COP Adjustment Period Snapshots");
        println!("  RT_prices          Real-Time Settlement Point Prices");
        println!("\nExamples:");
        println!("  {} --annual-rollup", args[0]);
        println!("  {} --annual-rollup --dataset DA_prices", args[0]);
        println!("  {} --annual-rollup /path/to/data --dataset COP_Snapshots", args[0]);
        println!("\nRunning default: --annual-rollup");
        
        // Run enhanced annual processor as default
        use enhanced_annual_processor::EnhancedAnnualProcessor;
        let base_dir = PathBuf::from("/Users/enrico/data/ERCOT_data");
        let processor = EnhancedAnnualProcessor::new(base_dir);
        processor.process_all_data()?;
    }
    
    Ok(())
}