use anyhow::Result;
use polars::prelude::*;
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use indicatif::{ProgressBar, ProgressStyle, MultiProgress};
use std::fs;

pub struct RTParallelProcessor {
    base_dir: PathBuf,
    output_dir: PathBuf,
}

impl RTParallelProcessor {
    pub fn new(base_dir: PathBuf) -> Self {
        let output_dir = base_dir.join("rollup_files").join("RT_prices");
        Self { base_dir, output_dir }
    }

    pub fn process_all_years_parallel(&self) -> Result<()> {
        println!("🚀 RT Parallel Processor - Maximum Performance Mode");
        println!("💪 Using all 32 threads for true parallel processing");
        
        // Create output directory
        fs::create_dir_all(&self.output_dir)?;
        
        // Get all CSV files
        let csv_dir = self.base_dir
            .join("Settlement_Point_Prices_at_Resource_Nodes,_Hubs_and_Load_Zones")
            .join("csv");
        
        let pattern = csv_dir.join("*.csv");
        let all_files: Vec<PathBuf> = glob::glob(pattern.to_str().unwrap())?
            .filter_map(Result::ok)
            .collect();
        
        println!("📊 Found {} RT price files", all_files.len());
        
        // Group by year
        let mut files_by_year: BTreeMap<i32, Vec<PathBuf>> = BTreeMap::new();
        for file in all_files {
            if let Some(year) = extract_year_from_filename(&file) {
                files_by_year.entry(year).or_default().push(file);
            }
        }
        
        println!("📅 Processing {} years in parallel", files_by_year.len());
        
        // Create multi-progress bar for parallel year processing
        let multi_progress = Arc::new(MultiProgress::new());
        
        // Process ALL years in parallel using all available threads
        let results: Vec<_> = files_by_year
            .into_par_iter()
            .map(|(year, year_files)| {
                let file_count = year_files.len();
                let pb = multi_progress.add(ProgressBar::new(file_count as u64));
                pb.set_style(
                    ProgressStyle::default_bar()
                        .template(&format!("[{{elapsed_precise}}] Year {}: {{bar:40}} {{pos}}/{{len}} {{msg}}", year))
                        .unwrap()
                        .progress_chars("█▓░")
                );
                
                // Process this year's files in parallel chunks
                // Use smaller chunks to allow better work distribution
                let chunk_size = 100;
                let mut year_dataframes = Vec::new();
                
                for chunk in year_files.chunks(chunk_size) {
                    // Process chunk in parallel
                    let chunk_dfs: Vec<DataFrame> = chunk
                        .par_iter()
                        .filter_map(|file| {
                            pb.inc(1);
                            match read_rt_file_optimized(file) {
                                Ok(df) => Some(df),
                                Err(_) => None,
                            }
                        })
                        .collect();
                    
                    year_dataframes.extend(chunk_dfs);
                }
                
                pb.finish_with_message("✓");
                
                // Combine and save
                if !year_dataframes.is_empty() {
                    match combine_dataframes_fast(year_dataframes) {
                        Ok(mut combined) => {
                            let output_file = self.output_dir.join(format!("{}.parquet", year));
                            match std::fs::File::create(&output_file) {
                                Ok(mut file) => {
                                    if let Ok(_) = ParquetWriter::new(&mut file).finish(&mut combined) {
                                        Ok((year, combined.height()))
                                    } else {
                                        Err(anyhow::anyhow!("Failed to write parquet"))
                                    }
                                }
                                Err(e) => Err(anyhow::anyhow!("Failed to create file: {}", e))
                            }
                        }
                        Err(e) => Err(anyhow::anyhow!("Failed to combine: {}", e))
                    }
                } else {
                    Ok((year, 0))
                }
            })
            .collect();
        
        // Report results
        println!("\n📊 Processing Results:");
        for result in results {
            match result {
                Ok((year, rows)) => println!("  ✅ Year {}: {} rows", year, rows),
                Err(e) => eprintln!("  ❌ Error: {}", e),
            }
        }
        
        Ok(())
    }
}

// Optimized file reader with minimal overhead
fn read_rt_file_optimized(file: &Path) -> Result<DataFrame> {
    // Use lazy reading with optimizations
    let df = CsvReader::from_path(file)?
        .has_header(true)
        .with_parse_options(
            CsvParseOptions::default()
                .with_truncate_ragged_lines(true)
        )
        .with_rechunk(false)  // Don't rechunk, we'll do it later
        .low_memory(false)     // Use more memory for speed
        .finish()?;
    
    Ok(df)
}

// Fast DataFrame combination using vertical concatenation
fn combine_dataframes_fast(dfs: Vec<DataFrame>) -> Result<DataFrame> {
    if dfs.is_empty() {
        return Err(anyhow::anyhow!("No dataframes to combine"));
    }
    
    // Use vstack for fast concatenation
    let mut base = dfs[0].clone();
    for df in dfs.into_iter().skip(1) {
        base = base.vstack(&df)?;
    }
    
    Ok(base)
}

fn extract_year_from_filename(path: &Path) -> Option<i32> {
    let name = path.file_name()?.to_str()?;
    
    // Look for YYYYMMDD pattern
    for part in name.split('.') {
        if part.len() >= 8 && part.starts_with("20") {
            if let Ok(year) = part[0..4].parse::<i32>() {
                if (2000..=2100).contains(&year) {
                    return Some(year);
                }
            }
        }
    }
    None
}