#!/usr/bin/env python3
"""
Archive Complete Years Script
- Organize CSV files by year into complete/ subfolders
- Zip complete years
- Move to archive location
- Keep parquet files for production
"""

import os
import shutil
import zipfile
from pathlib import Path
from collections import defaultdict
import re
from datetime import datetime
import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

BASE_DIR = Path("/Users/enrico/data/ERCOT_data")


def extract_date_from_filename(filename):
    """Extract date from ERCOT filename."""
    patterns = [
        r'\.(\d{4})(\d{2})(\d{2})\.',
        r'_(\d{4})(\d{2})(\d{2})_',
        r'_(\d{4})(\d{2})(\d{2})\.',
        r'(\d{4})-(\d{2})-(\d{2})',
    ]

    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            try:
                year, month, day = map(int, match.groups())
                if 2010 <= year <= 2030 and 1 <= month <= 12 and 1 <= day <= 31:
                    return datetime(year, month, day)
            except:
                pass

    return None


def organize_csvs_by_year(csv_dir, complete_dir):
    """Organize CSV files by year into complete/ subfolders."""
    logging.info(f"Organizing CSVs from {csv_dir}")

    if not csv_dir.exists():
        logging.warning(f"CSV directory not found: {csv_dir}")
        return {}

    csv_files = list(csv_dir.glob("*.csv"))
    logging.info(f"Found {len(csv_files)} CSV files")

    # Group files by year
    files_by_year = defaultdict(list)

    for csv_file in csv_files:
        date = extract_date_from_filename(csv_file.name)
        if date:
            files_by_year[date.year].append(csv_file)

    # Create year directories and move files
    year_counts = {}

    for year, files in sorted(files_by_year.items()):
        year_dir = complete_dir / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"Moving {len(files)} files to {year_dir}")

        for csv_file in files:
            try:
                dest = year_dir / csv_file.name
                shutil.copy2(csv_file, dest)
            except Exception as e:
                logging.error(f"Error copying {csv_file}: {e}")

        year_counts[year] = len(files)

    return year_counts


def zip_year_directory(year_dir, output_zip):
    """Zip a year directory."""
    logging.info(f"Creating zip archive: {output_zip}")

    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
        for file_path in year_dir.rglob("*.csv"):
            arcname = file_path.relative_to(year_dir.parent)
            zipf.write(file_path, arcname)

    # Get file size
    size_mb = output_zip.stat().st_size / (1024 * 1024)
    logging.info(f"Created {output_zip.name}: {size_mb:.1f} MB")

    return size_mb


def process_dataset(dataset_dir, dataset_name, archive_dir):
    """Process a single dataset for archival."""
    logging.info("=" * 80)
    logging.info(f"Processing: {dataset_name}")
    logging.info("=" * 80)

    csv_dir = dataset_dir / "CSV"
    complete_dir = dataset_dir / "complete"

    # Step 1: Organize CSVs by year
    year_counts = organize_csvs_by_year(csv_dir, complete_dir)

    if not year_counts:
        logging.warning(f"No files organized for {dataset_name}")
        return

    # Step 2: Zip each year
    zip_dir = complete_dir / "zips"
    zip_dir.mkdir(exist_ok=True)

    for year, count in sorted(year_counts.items()):
        year_dir = complete_dir / str(year)

        if year_dir.exists():
            zip_filename = f"{dataset_name}_{year}.zip"
            zip_path = zip_dir / zip_filename

            try:
                size_mb = zip_year_directory(year_dir, zip_path)
                logging.info(f"Year {year}: {count} files -> {zip_filename} ({size_mb:.1f} MB)")
            except Exception as e:
                logging.error(f"Error zipping year {year}: {e}")

    # Step 3: Move zips to archive directory
    if archive_dir:
        dataset_archive_dir = archive_dir / dataset_name
        dataset_archive_dir.mkdir(parents=True, exist_ok=True)

        for zip_file in zip_dir.glob("*.zip"):
            try:
                dest = dataset_archive_dir / zip_file.name
                shutil.copy2(zip_file, dest)
                logging.info(f"Archived: {zip_file.name} -> {dest}")
            except Exception as e:
                logging.error(f"Error archiving {zip_file}: {e}")

    logging.info(f"Completed processing {dataset_name}")


def generate_archival_report(base_dir, datasets):
    """Generate a report of what was archived."""
    report_file = base_dir / "archival_report.txt"

    with open(report_file, 'w') as f:
        f.write("ERCOT DATA ARCHIVAL REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        for dataset in datasets:
            dataset_dir = base_dir / dataset
            complete_dir = dataset_dir / "complete"

            if complete_dir.exists():
                f.write(f"\nDataset: {dataset}\n")
                f.write("-" * 80 + "\n")

                # List years
                year_dirs = sorted([d for d in complete_dir.iterdir() if d.is_dir() and d.name.isdigit()])

                for year_dir in year_dirs:
                    csv_count = len(list(year_dir.glob("*.csv")))
                    f.write(f"  Year {year_dir.name}: {csv_count} CSV files\n")

                # List zips
                zip_dir = complete_dir / "zips"
                if zip_dir.exists():
                    f.write(f"\n  Zip Archives:\n")
                    for zip_file in sorted(zip_dir.glob("*.zip")):
                        size_mb = zip_file.stat().st_size / (1024 * 1024)
                        f.write(f"    {zip_file.name}: {size_mb:.1f} MB\n")

    logging.info(f"Report generated: {report_file}")


def main():
    """Main archival process."""
    import argparse

    parser = argparse.ArgumentParser(description='Archive complete years of ERCOT data')
    parser.add_argument('--dataset', help='Specific dataset to process (optional)')
    parser.add_argument('--archive-dir', help='Archive directory path', default=None)
    parser.add_argument('--dry-run', action='store_true', help='Simulate without making changes')

    args = parser.parse_args()

    logging.info("=" * 80)
    logging.info("ERCOT DATA ARCHIVAL PROCESS")
    logging.info("=" * 80)
    logging.info(f"Base directory: {BASE_DIR}")
    logging.info(f"Archive directory: {args.archive_dir or 'Not specified'}")
    logging.info(f"Dry run: {args.dry_run}")
    logging.info("")

    # Get list of datasets
    critical_datasets = [
        "Settlement_Point_Prices_at_Resource_Nodes,_Hubs_and_Load_Zones",
        "DAM_Settlement_Point_Prices",
        "LMPs_by_Resource_Nodes,_Load_Zones_and_Trading_Hubs",
        "Real-Time_ORDC_and_Reliability_Deployment_Price_Adders_and_Reserves_by_SCED_Interval",
        "SCED_System_Lambda",
        "SCED_Shadow_Prices_and_Binding_Transmission_Constraints",
        "Wind_Power_Production_-_Actual_5-Minute_Averaged_Values",
        "Solar_Power_Production_-_Actual_5-Minute_Averaged_Values",
        "Hourly_Resource_Outage_Capacity",
        "State_Estimator_Load_Report_-_Total_ERCOT_Generation",
        "Actual_System_Load_by_Forecast_Zone",
        "Actual_System_Load_by_Weather_Zone",
        "System-Wide_Demand",
        "Intra-Hour_Load_Forecast_by_Weather_Zone",
        "2-Day_DAM_and_SCED_Energy_Curves_Reports",
        "2-Day_Real_Time_Gen_and_Load_Data_Reports",
    ]

    archive_dir = Path(args.archive_dir) if args.archive_dir else None

    if args.dataset:
        # Process single dataset
        dataset_dir = BASE_DIR / args.dataset
        if dataset_dir.exists():
            if not args.dry_run:
                process_dataset(dataset_dir, args.dataset, archive_dir)
            else:
                logging.info(f"DRY RUN: Would process {args.dataset}")
        else:
            logging.error(f"Dataset not found: {args.dataset}")
    else:
        # Process all datasets
        for dataset in critical_datasets:
            dataset_dir = BASE_DIR / dataset

            if dataset_dir.exists():
                if not args.dry_run:
                    process_dataset(dataset_dir, dataset, archive_dir)
                else:
                    logging.info(f"DRY RUN: Would process {dataset}")
            else:
                logging.warning(f"Dataset directory not found: {dataset}")

    # Generate report
    if not args.dry_run:
        generate_archival_report(BASE_DIR, critical_datasets)

    logging.info("")
    logging.info("=" * 80)
    logging.info("ARCHIVAL PROCESS COMPLETE")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()
