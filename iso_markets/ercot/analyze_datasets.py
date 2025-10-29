#!/usr/bin/env python3
"""
Analyze ERCOT Datasets
- Count zip files and CSV files in each directory
- Identify complete years (no gaps)
- Generate report for archival planning
"""

import os
import csv
import re
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timedelta
import pandas as pd

BASE_DIR = Path("/Users/enrico/data/ERCOT_data")
INVENTORY_FILE = BASE_DIR / "ercot_datasets_inventory.csv"


def extract_date_from_filename(filename):
    """Extract date from ERCOT filename formats."""
    # Try various date patterns
    patterns = [
        r'\.(\d{4})(\d{2})(\d{2})\.',  # .YYYYMMDD.
        r'_(\d{4})(\d{2})(\d{2})_',    # _YYYYMMDD_
        r'_(\d{4})(\d{2})(\d{2})\.',   # _YYYYMMDD.
        r'(\d{4})-(\d{2})-(\d{2})',    # YYYY-MM-DD
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


def analyze_directory(directory):
    """Analyze a single dataset directory."""
    dir_path = BASE_DIR / directory

    if not dir_path.exists():
        return {
            'exists': False,
            'zip_count': 0,
            'csv_count': 0,
            'csv_in_subfolder': 0,
            'dates': [],
            'years': {}
        }

    # Count zip files
    zip_files = list(dir_path.glob("*.zip"))
    zip_count = len(zip_files)

    # Count CSV files in CSV subfolder
    csv_subfolder = dir_path / "CSV"
    csv_in_subfolder = 0
    csv_files_list = []

    if csv_subfolder.exists():
        csv_files = list(csv_subfolder.glob("*.csv"))
        csv_in_subfolder = len(csv_files)
        csv_files_list = csv_files

    # Count total CSV files recursively
    all_csv_files = list(dir_path.rglob("*.csv"))
    csv_count = len(all_csv_files)

    # Extract dates from filenames
    dates = []
    for f in zip_files:
        date = extract_date_from_filename(f.name)
        if date:
            dates.append(date)

    # Also check CSV files for dates
    for f in csv_files_list[:1000]:  # Limit to avoid too much processing
        date = extract_date_from_filename(f.name)
        if date:
            dates.append(date)

    # Group by year
    years = defaultdict(list)
    for date in dates:
        years[date.year].append(date)

    # Check for completeness
    year_completeness = {}
    for year, year_dates in years.items():
        year_dates_sorted = sorted(set(year_dates))

        # Determine expected dates based on data frequency
        if len(year_dates_sorted) > 300:  # Daily or more frequent
            # Check for gaps
            start_date = datetime(year, 1, 1)
            end_date = datetime(year, 12, 31)
            expected_days = (end_date - start_date).days + 1

            # Create set of all days in year
            all_days = set()
            current = start_date
            while current <= end_date:
                all_days.add(current.date())
                current += timedelta(days=1)

            actual_days = set(d.date() for d in year_dates_sorted)
            missing_days = all_days - actual_days

            year_completeness[year] = {
                'count': len(year_dates_sorted),
                'expected': expected_days,
                'missing': len(missing_days),
                'complete': len(missing_days) == 0,
                'coverage_pct': (len(actual_days) / expected_days * 100) if expected_days > 0 else 0
            }
        else:
            # Monthly or less frequent data
            year_completeness[year] = {
                'count': len(year_dates_sorted),
                'expected': 'Variable',
                'missing': 'N/A',
                'complete': True,  # Assume complete for low-frequency data
                'coverage_pct': 100.0
            }

    return {
        'exists': True,
        'zip_count': zip_count,
        'csv_count': csv_count,
        'csv_in_subfolder': csv_in_subfolder,
        'dates': dates,
        'years': years,
        'year_completeness': year_completeness
    }


def main():
    """Main analysis function."""
    print("=" * 80)
    print("ERCOT DATASETS ANALYSIS")
    print("=" * 80)
    print()

    # Read inventory
    datasets = []
    with open(INVENTORY_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            datasets.append(row)

    print(f"Found {len(datasets)} datasets in inventory\n")

    # Analyze each dataset
    results = []

    for dataset in datasets:
        name = dataset['Dataset Name']
        directory = dataset['Directory']

        print(f"Analyzing: {name}")
        analysis = analyze_directory(directory)

        results.append({
            'name': name,
            'directory': directory,
            'url': dataset['URL'],
            'analysis': analysis
        })

        if analysis['exists']:
            print(f"  ✓ Exists")
            print(f"  Zip files: {analysis['zip_count']:,}")
            print(f"  Total CSV files: {analysis['csv_count']:,}")
            print(f"  CSV in CSV subfolder: {analysis['csv_in_subfolder']:,}")
            print(f"  Years found: {sorted(analysis['years'].keys())}")

            # Show completeness
            if analysis['year_completeness']:
                print(f"  Year Completeness:")
                for year in sorted(analysis['year_completeness'].keys()):
                    comp = analysis['year_completeness'][year]
                    status = "✓ COMPLETE" if comp['complete'] else "✗ INCOMPLETE"
                    print(f"    {year}: {status} ({comp['coverage_pct']:.1f}% coverage, {comp['count']} files)")
        else:
            print(f"  ✗ Directory not found")

        print()

    # Generate summary report
    print("\n" + "=" * 80)
    print("SUMMARY REPORT")
    print("=" * 80)
    print()

    # Complete years summary
    print("COMPLETE YEARS BY DATASET:")
    print("-" * 80)

    for result in results:
        if result['analysis']['exists'] and result['analysis']['year_completeness']:
            complete_years = [
                year for year, comp in result['analysis']['year_completeness'].items()
                if comp['complete'] and comp['coverage_pct'] >= 99.0
            ]
            if complete_years:
                print(f"{result['name']}:")
                print(f"  Complete years: {sorted(complete_years)}")

    print()

    # Archival recommendations
    print("\nARCHIVAL RECOMMENDATIONS:")
    print("-" * 80)

    for result in results:
        if result['analysis']['exists']:
            complete_years = []
            if result['analysis']['year_completeness']:
                complete_years = [
                    year for year, comp in result['analysis']['year_completeness'].items()
                    if comp['complete'] and comp['coverage_pct'] >= 99.0
                ]

            if complete_years:
                print(f"\n{result['name']}:")
                print(f"  Recommended for archival: Years {sorted(complete_years)}")
                print(f"  Estimated CSV files: {result['analysis']['csv_in_subfolder']:,}")
                print(f"  Action: Create complete/{year}/ subfolders, convert to parquet, archive zips")

    # Export detailed report
    report_file = BASE_DIR / "dataset_analysis_report.csv"
    with open(report_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Dataset Name', 'Directory', 'Exists', 'Zip Count', 'CSV Count',
            'CSV in Subfolder', 'Years', 'Complete Years', 'Incomplete Years'
        ])

        for result in results:
            analysis = result['analysis']

            if analysis['exists']:
                complete_years = []
                incomplete_years = []

                if analysis['year_completeness']:
                    for year, comp in analysis['year_completeness'].items():
                        if comp['complete'] and comp['coverage_pct'] >= 99.0:
                            complete_years.append(year)
                        else:
                            incomplete_years.append(year)

                writer.writerow([
                    result['name'],
                    result['directory'],
                    'Yes',
                    analysis['zip_count'],
                    analysis['csv_count'],
                    analysis['csv_in_subfolder'],
                    ','.join(map(str, sorted(analysis['years'].keys()))),
                    ','.join(map(str, sorted(complete_years))),
                    ','.join(map(str, sorted(incomplete_years)))
                ])
            else:
                writer.writerow([
                    result['name'],
                    result['directory'],
                    'No',
                    0, 0, 0, '', '', ''
                ])

    print(f"\n\nDetailed report exported to: {report_file}")
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
