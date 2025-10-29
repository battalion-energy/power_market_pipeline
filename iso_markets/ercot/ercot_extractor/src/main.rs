use anyhow::{Context, Result};
use clap::Parser;
use crossbeam::channel::{bounded, unbounded};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use walkdir::WalkDir;
use zip::ZipArchive;

#[derive(Parser, Debug)]
#[command(author, version, about = "High-performance ERCOT data extractor", long_about = None)]
struct Args {
    /// Directory to process
    #[arg(short, long)]
    directory: PathBuf,

    /// Number of threads to use (default: all CPU cores)
    #[arg(short, long)]
    threads: Option<usize>,

    /// Skip already extracted files
    #[arg(short, long, default_value = "true")]
    skip_existing: bool,
}

struct Stats {
    zips_processed: AtomicUsize,
    csvs_moved: AtomicUsize,
    bytes_extracted: AtomicUsize,
}

impl Stats {
    fn new() -> Self {
        Self {
            zips_processed: AtomicUsize::new(0),
            csvs_moved: AtomicUsize::new(0),
            bytes_extracted: AtomicUsize::new(0),
        }
    }
}

fn extract_zip(
    zip_path: &Path,
    skip_existing: bool,
) -> Result<Vec<PathBuf>> {
    let extract_dir = zip_path.with_extension("");

    // Skip if already extracted
    if skip_existing && extract_dir.exists() {
        return Ok(Vec::new());
    }

    // Create extraction directory
    fs::create_dir_all(&extract_dir)?;

    let file = File::open(zip_path)
        .with_context(|| format!("Failed to open zip file: {:?}", zip_path))?;

    let mut archive = ZipArchive::new(file)
        .with_context(|| format!("Failed to read zip archive: {:?}", zip_path))?;

    let mut extracted_paths = Vec::new();

    for i in 0..archive.len() {
        let mut file = archive.by_index(i)?;
        let outpath = extract_dir.join(file.name());

        if file.name().ends_with('/') {
            fs::create_dir_all(&outpath)?;
        } else {
            if let Some(parent) = outpath.parent() {
                fs::create_dir_all(parent)?;
            }

            let mut outfile = File::create(&outpath)?;
            io::copy(&mut file, &mut outfile)?;
            extracted_paths.push(outpath);
        }
    }

    Ok(extracted_paths)
}

fn find_csv_files(dir: &Path) -> Vec<PathBuf> {
    WalkDir::new(dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path().extension()
                .and_then(|s| s.to_str())
                .map(|s| s.eq_ignore_ascii_case("csv"))
                .unwrap_or(false)
        })
        .map(|e| e.path().to_path_buf())
        .collect()
}

fn move_csv_to_target(csv_path: &Path, target_dir: &Path) -> Result<()> {
    let file_name = csv_path.file_name()
        .context("Failed to get filename")?;

    let target_path = target_dir.join(file_name);

    // Handle duplicate filenames
    let mut counter = 1;
    let mut final_path = target_path.clone();

    while final_path.exists() {
        let stem = csv_path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("file");
        let new_name = format!("{}_{}.csv", stem, counter);
        final_path = target_dir.join(new_name);
        counter += 1;
    }

    fs::rename(csv_path, &final_path)
        .or_else(|_| fs::copy(csv_path, &final_path).and_then(|_| fs::remove_file(csv_path)))?;

    Ok(())
}

fn process_directory(
    dir: &Path,
    skip_existing: bool,
    stats: Arc<Stats>,
    progress: ProgressBar,
) -> Result<()> {
    // Create CSV directory
    let csv_dir = dir.join("CSV");
    fs::create_dir_all(&csv_dir)?;

    // Find all top-level zip files
    let top_level_zips: Vec<_> = fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path().extension()
                .and_then(|s| s.to_str())
                .map(|s| s.eq_ignore_ascii_case("zip"))
                .unwrap_or(false)
        })
        .map(|e| e.path())
        .collect();

    progress.set_length(top_level_zips.len() as u64);
    progress.set_message("Processing top-level zips");

    // Process top-level zips in parallel
    let results: Vec<_> = top_level_zips
        .par_iter()
        .map(|zip_path| {
            // Extract first level
            let extracted_files = extract_zip(zip_path, skip_existing)?;
            stats.zips_processed.fetch_add(1, Ordering::Relaxed);
            progress.inc(1);

            let zip_dir = zip_path.with_extension("");

            // Check for second-level zips
            let second_level_zips: Vec<_> = WalkDir::new(&zip_dir)
                .into_iter()
                .filter_map(|e| e.ok())
                .filter(|e| {
                    e.path().extension()
                        .and_then(|s| s.to_str())
                        .map(|s| s.eq_ignore_ascii_case("zip"))
                        .unwrap_or(false)
                })
                .map(|e| e.path().to_path_buf())
                .collect();

            // Process second-level zips in parallel
            if !second_level_zips.is_empty() {
                second_level_zips.par_iter().for_each(|subzip| {
                    if let Err(e) = extract_zip(subzip, skip_existing) {
                        eprintln!("Error extracting {:?}: {}", subzip, e);
                    } else {
                        stats.zips_processed.fetch_add(1, Ordering::Relaxed);
                    }
                });
            }

            // Find all CSV files in the extracted directory
            let csv_files = find_csv_files(&zip_dir);

            // Move CSV files to target directory
            let moved = csv_files.par_iter()
                .filter_map(|csv_path| {
                    match move_csv_to_target(csv_path, &csv_dir) {
                        Ok(_) => {
                            stats.csvs_moved.fetch_add(1, Ordering::Relaxed);
                            Some(1)
                        }
                        Err(e) => {
                            eprintln!("Error moving {:?}: {}", csv_path, e);
                            None
                        }
                    }
                })
                .sum::<usize>();

            Ok(moved)
        })
        .collect();

    progress.finish_with_message("Complete");

    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Set thread pool size
    if let Some(threads) = args.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .context("Failed to build thread pool")?;
    }

    let num_threads = rayon::current_num_threads();

    println!("========================================");
    println!("ERCOT High-Performance Data Extractor");
    println!("========================================");
    println!("Directory: {:?}", args.directory);
    println!("Threads: {}", num_threads);
    println!("Skip existing: {}", args.skip_existing);
    println!();

    if !args.directory.exists() {
        anyhow::bail!("Directory does not exist: {:?}", args.directory);
    }

    let stats = Arc::new(Stats::new());

    let multi_progress = MultiProgress::new();
    let style = ProgressStyle::default_bar()
        .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
        .unwrap()
        .progress_chars("##-");

    let progress = multi_progress.add(ProgressBar::new(0));
    progress.set_style(style);

    let start = std::time::Instant::now();

    // Process the directory
    process_directory(&args.directory, args.skip_existing, stats.clone(), progress)?;

    let duration = start.elapsed();

    println!();
    println!("========================================");
    println!("EXTRACTION COMPLETE");
    println!("========================================");
    println!("Time elapsed: {:.2}s", duration.as_secs_f64());
    println!("Zip files processed: {}", stats.zips_processed.load(Ordering::Relaxed));
    println!("CSV files moved: {}", stats.csvs_moved.load(Ordering::Relaxed));
    println!("Average speed: {:.0} files/sec",
        stats.csvs_moved.load(Ordering::Relaxed) as f64 / duration.as_secs_f64());

    Ok(())
}
