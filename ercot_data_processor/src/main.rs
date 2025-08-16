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

// Removed old broken functions - now using enhanced_annual_processor exclusively

fn main() -> Result<()> {
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
            println!("Example: --extract-all-ercot /Users/enrico/data/ERCOT_data");
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
            println!("Example: --process-ercot-prices /Users/enrico/data/ERCOT_data");
        }
        return Ok(());
    } else if args.len() > 1 && args[1] == "--detect-schema" {
        // Run schema detection to create type registry
        use schema_detector::SchemaDetector;
        let base_dir = if args.len() > 2 {
            PathBuf::from(&args[2])
        } else {
            PathBuf::from("/Users/enrico/data/ERCOT_data")
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
            PathBuf::from("/Users/enrico/data/ERCOT_data")
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
    } else if args.len() > 1 && args[1] == "--annual-rollup-validated" {
        // Run rollup with schema validation
        use enhanced_annual_processor_validated::ValidatedAnnualProcessor;
        let base_dir = if args.len() > 2 {
            PathBuf::from(&args[2])
        } else {
            PathBuf::from("/Users/enrico/data/ERCOT_data")
        };
        let processor = ValidatedAnnualProcessor::new(base_dir)?;
        processor.process_all_with_validation()?;
        return Ok(());
    } else if args.len() > 1 && args[1] == "--annual-rollup" {
        // Enhanced annual rollup with gap tracking and schema normalization
        use enhanced_annual_processor::EnhancedAnnualProcessor;
        let base_dir = if args.len() > 2 {
            PathBuf::from(&args[2])
        } else {
            PathBuf::from("/Users/enrico/data/ERCOT_data")
        };
        let processor = EnhancedAnnualProcessor::new(base_dir);
        processor.process_all_data()?;
        return Ok(());
    } else if args.len() > 1 && args[1] == "--bess-parquet" {
        // Analyze BESS revenues using parquet files
        bess_parquet_analyzer::analyze_bess_from_parquet()?;
        return Ok(());
    } else if args.len() > 1 && args[1] == "--verify-results" {
        // Verify data quality of processed files
        verify_data_quality(&PathBuf::from("."))?;
    } else {
        // Default to annual rollup with enhanced processor
        println!("Usage: {} [command] [options]", args[0]);
        println!("\nCommands:");
        println!("  --annual-rollup [dir]     Process annual ERCOT data with gap tracking");
        println!("  --bess-parquet           Analyze BESS revenues from parquet files");
        println!("  --extract-all-ercot dir  Extract all ERCOT CSV files from zips");
        println!("  --process-annual         Process extracted CSV to annual parquet");
        println!("  --verify-results         Verify data quality of processed files");
        println!("\nRunning default: --annual-rollup");
        
        // Run enhanced annual processor as default
        use enhanced_annual_processor::EnhancedAnnualProcessor;
        let base_dir = PathBuf::from("/Users/enrico/data/ERCOT_data");
        let processor = EnhancedAnnualProcessor::new(base_dir);
        processor.process_all_data()?;
    }
    
    Ok(())
}