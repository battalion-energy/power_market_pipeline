"""
ERCOT Historical Data Downloader - Downloads all historical data from ERCOT archives.

This script:
1. Reads the CSV catalog of all ERCOT datasets
2. Maintains a JSON tracking file with download status for each dataset
3. Supports two modes:
   - update (default): Download from current date back to last downloaded date
   - extend: Download from last downloaded date back to a historical cutoff (default 2019-01-01)
"""

import csv
import json
import os
import argparse
from datetime import datetime, timedelta
from ercot_download_historical_batch import scrape_and_download_by_date

# Configuration
CATALOG_CSV = "ercot_datasets_catalog.csv"
TRACKING_JSON = "ercot_download_tracking.json"
BASE_DOWNLOAD_DIR = "/Users/enrico/data/ERCOT_data_clean_archive"
DEFAULT_HISTORICAL_CUTOFF = "2019-01-01"


def load_catalog(catalog_file):
    """Load the dataset catalog from CSV file."""
    datasets = []
    with open(catalog_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            datasets.append(row)
    return datasets


def load_tracking(tracking_file):
    """Load the download tracking data from JSON file."""
    if os.path.exists(tracking_file):
        with open(tracking_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_tracking(tracking_file, tracking_data):
    """Save the download tracking data to JSON file."""
    with open(tracking_file, 'w', encoding='utf-8') as f:
        json.dump(tracking_data, f, indent=2)


def sanitize_directory_name(name):
    """Sanitize dataset name for use as directory name."""
    # Replace problematic characters
    sanitized = name.replace('/', '_').replace('\\', '_').replace(':', '_')
    sanitized = sanitized.replace('"', '').replace("'", "")
    # Remove other special characters that might cause issues
    sanitized = ''.join(c for c in sanitized if c.isalnum() or c in (' ', '-', '_', ','))
    return sanitized.strip()


def get_dataset_key(url):
    """Extract a unique key from the dataset URL."""
    # Use the last part of the URL as the key (e.g., NP3-911-ER)
    return url.rstrip('/').split('/')[-1]


def download_dataset_update_mode(dataset, tracking_data, tracking_file):
    """
    Download missing data from current date back to last downloaded date.

    Args:
        dataset: Dictionary with 'url' and 'name' keys
        tracking_data: Dictionary containing tracking information
        tracking_file: Path to the tracking JSON file
    """
    url = dataset['url']
    name = dataset['name']
    key = get_dataset_key(url)

    print(f"\n{'='*80}")
    print(f"Processing dataset: {name}")
    print(f"URL: {url}")
    print(f"Key: {key}")

    # Get tracking info for this dataset
    if key not in tracking_data:
        tracking_data[key] = {
            'name': name,
            'url': url,
            'start_date': None,  # Will be set after first successful download
            'end_date': None,    # Latest date downloaded
            'last_updated': None
        }

    dataset_info = tracking_data[key]

    # Determine date range to download
    end_date = datetime.now().strftime("%Y-%m-%d")

    if dataset_info['end_date']:
        # We have downloaded data before, download from last end_date to today
        start_date = dataset_info['end_date']
        print(f"Updating from {start_date} to {end_date}")
    else:
        # First time download, use default historical cutoff
        start_date = DEFAULT_HISTORICAL_CUTOFF
        print(f"First time download: {start_date} to {end_date}")

    # Create download directory
    dir_name = sanitize_directory_name(name)
    download_dir = os.path.join(BASE_DOWNLOAD_DIR, dir_name)

    try:
        # Download the data
        scrape_and_download_by_date(
            url=url,
            download_dir=download_dir,
            end_date=end_date,
            start_date=start_date
        )

        # Update tracking data
        if dataset_info['start_date'] is None:
            dataset_info['start_date'] = start_date

        dataset_info['end_date'] = end_date
        dataset_info['last_updated'] = datetime.now().isoformat()

        # Save tracking data after each successful download
        save_tracking(tracking_file, tracking_data)

        print(f"Successfully downloaded {name}")

    except Exception as e:
        print(f"ERROR downloading {name}: {str(e)}")
        # Continue with next dataset even if this one fails


def download_dataset_extend_mode(dataset, tracking_data, tracking_file, historical_cutoff):
    """
    Download historical data from last downloaded date back to historical cutoff.

    Args:
        dataset: Dictionary with 'url' and 'name' keys
        tracking_data: Dictionary containing tracking information
        tracking_file: Path to the tracking JSON file
        historical_cutoff: Date string (YYYY-MM-DD) for how far back to download
    """
    url = dataset['url']
    name = dataset['name']
    key = get_dataset_key(url)

    print(f"\n{'='*80}")
    print(f"Processing dataset (EXTEND mode): {name}")
    print(f"URL: {url}")
    print(f"Key: {key}")

    # Get tracking info for this dataset
    if key not in tracking_data:
        print(f"No tracking data found for {name}. Run update mode first.")
        return

    dataset_info = tracking_data[key]

    if dataset_info['start_date'] is None:
        print(f"No start_date found for {name}. Run update mode first.")
        return

    # Determine date range to download
    # We want to download from historical_cutoff to the current start_date
    end_date = dataset_info['start_date']
    start_date = historical_cutoff

    print(f"Extending historical data from {start_date} to {end_date}")

    # Create download directory
    dir_name = sanitize_directory_name(name)
    download_dir = os.path.join(BASE_DOWNLOAD_DIR, dir_name)

    try:
        # Download the data
        scrape_and_download_by_date(
            url=url,
            download_dir=download_dir,
            end_date=end_date,
            start_date=start_date
        )

        # Update tracking data - move start_date back
        dataset_info['start_date'] = start_date
        dataset_info['last_updated'] = datetime.now().isoformat()

        # Save tracking data after each successful download
        save_tracking(tracking_file, tracking_data)

        print(f"Successfully extended historical data for {name}")

    except Exception as e:
        print(f"ERROR extending historical data for {name}: {str(e)}")


def main():
    parser = argparse.ArgumentParser(
        description='Download all historical data from ERCOT archives',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Update all datasets with latest data (from last download to today)
  python ercot_download_all_historical.py

  # Extend historical data back to 2019-01-01 (default)
  python ercot_download_all_historical.py --mode extend

  # Extend historical data back to 2010-01-01
  python ercot_download_all_historical.py --mode extend --historical-cutoff 2010-01-01

  # Process only specific datasets (by key, e.g., NP4-732-CD)
  python ercot_download_all_historical.py --datasets NP4-732-CD NP4-737-CD
        """
    )

    parser.add_argument(
        '--mode',
        choices=['update', 'extend'],
        default='update',
        help='Download mode: update (fill gaps from last download to today) or extend (download historical data)'
    )

    parser.add_argument(
        '--historical-cutoff',
        default=DEFAULT_HISTORICAL_CUTOFF,
        help=f'Historical cutoff date for extend mode (default: {DEFAULT_HISTORICAL_CUTOFF})'
    )

    parser.add_argument(
        '--datasets',
        nargs='+',
        help='Specific dataset keys to process (e.g., NP4-732-CD). If not specified, processes all datasets.'
    )

    parser.add_argument(
        '--catalog',
        default=CATALOG_CSV,
        help=f'Path to catalog CSV file (default: {CATALOG_CSV})'
    )

    parser.add_argument(
        '--tracking',
        default=TRACKING_JSON,
        help=f'Path to tracking JSON file (default: {TRACKING_JSON})'
    )

    args = parser.parse_args()

    # Resolve paths relative to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    catalog_path = os.path.join(script_dir, args.catalog)
    tracking_path = os.path.join(script_dir, args.tracking)

    # Load catalog and tracking data
    print(f"Loading catalog from: {catalog_path}")
    datasets = load_catalog(catalog_path)
    print(f"Found {len(datasets)} datasets in catalog")

    print(f"Loading tracking data from: {tracking_path}")
    tracking_data = load_tracking(tracking_path)
    print(f"Loaded tracking data for {len(tracking_data)} datasets")

    # Filter datasets if specific ones requested
    if args.datasets:
        datasets = [d for d in datasets if get_dataset_key(d['url']) in args.datasets]
        print(f"Filtered to {len(datasets)} requested datasets")

    # Create base download directory
    os.makedirs(BASE_DOWNLOAD_DIR, exist_ok=True)

    # Process each dataset
    print(f"\n{'='*80}")
    print(f"Starting downloads in {args.mode.upper()} mode")
    print(f"{'='*80}\n")

    for i, dataset in enumerate(datasets, 1):
        print(f"\n[{i}/{len(datasets)}]")

        if args.mode == 'update':
            download_dataset_update_mode(dataset, tracking_data, tracking_path)
        else:  # extend mode
            download_dataset_extend_mode(dataset, tracking_data, tracking_path, args.historical_cutoff)

    print(f"\n{'='*80}")
    print("All downloads complete!")
    print(f"{'='*80}")

    # Print summary
    print("\nSummary:")
    for key, info in tracking_data.items():
        print(f"  {key}: {info['name']}")
        print(f"    Date range: {info['start_date']} to {info['end_date']}")
        print(f"    Last updated: {info['last_updated']}")


if __name__ == "__main__":
    main()
