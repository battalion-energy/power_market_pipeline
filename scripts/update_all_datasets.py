#!/usr/bin/env python3
"""
Unified ERCOT Data Updater - Updates all datasets from Web Service API

This script updates:
1. DA_prices (Day-Ahead Settlement Point Prices)
2. AS_prices (Ancillary Service Prices)
3. DAM_Gen_Resources (60-day disclosure)
4. SCED_Gen_Resources (60-day disclosure)

For each dataset:
- Checks latest date in existing parquet files
- Downloads missing data from Web Service API
- Converts API format to ZIP CSV format
- Safely regenerates parquet files with verification

Usage:
    python update_all_datasets.py                    # Update all datasets
    python update_all_datasets.py --datasets DA_prices AS_prices
    python update_all_datasets.py --dry-run          # Check without updating
    python update_all_datasets.py --start-date 2025-08-20  # Force date range
"""

import asyncio
import argparse
import sys
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import pyarrow.parquet as pq
import logging

# Add ercot_ws_downloader to path
sys.path.insert(0, str(Path(__file__).parent))

from ercot_ws_downloader.downloaders import (
    DAMPriceDownloader,
    ASPriceDownloader,
    DAMDisclosureDownloader,
    SCEDDisclosureDownloader,
)
from ercot_ws_downloader.client import ERCOTWebServiceClient
from ercot_ws_downloader.state_manager import StateManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
DATA_DIR = Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data")
ROLLUP_DIR = DATA_DIR / "rollup_files"
CURRENT_YEAR = datetime.now().year

# Dataset configurations
DATASET_CONFIG = {
    "DA_prices": {
        "name": "DA Prices",
        "downloader_class": DAMPriceDownloader,
        "csv_dir": DATA_DIR / "DAM_Settlement_Point_Prices" / "csv",
        "parquet_dir": ROLLUP_DIR / "DA_prices",
        "date_column": "DeliveryDate",
        "rust_dataset": "DA_prices",
        "filename_pattern": "cdr.00012331.0000000000000000.{date}.{time}.DAMSPNP4190.csv",
        "output_columns": ["DeliveryDate", "HourEnding", "SettlementPoint", "SettlementPointPrice", "DSTFlag"],
    },
    "AS_prices": {
        "name": "AS Prices",
        "downloader_class": ASPriceDownloader,
        "csv_dir": DATA_DIR / "DAM_Clearing_Prices_for_Capacity" / "csv",
        "parquet_dir": ROLLUP_DIR / "AS_prices",
        "date_column": "DeliveryDate",
        "rust_dataset": "AS_prices",
        "filename_pattern": "cdr.00012329.0000000000000000.{date}.{time}.DAMCPCNP4188.csv",
        "output_columns": ["DeliveryDate", "HourEnding", "AncillaryType", "MCPC", "DSTFlag"],
    },
    "DAM_Gen_Resources": {
        "name": "DAM Gen Resources (60-day)",
        "downloader_class": DAMDisclosureDownloader,
        "csv_dir": DATA_DIR / "60-Day_DAM_Disclosure_Reports" / "csv",
        "parquet_dir": ROLLUP_DIR / "DAM_Gen_Resources",
        "date_column": "DeliveryDate",
        "rust_dataset": "DAM_Gen_Resources",
        "filename_pattern": "60d_DAM_Gen_Resource_Data-{date_fmt}.csv",
        "output_columns": None,  # Keep all columns
    },
    "SCED_Gen_Resources": {
        "name": "SCED Gen Resources (60-day)",
        "downloader_class": SCEDDisclosureDownloader,
        "csv_dir": DATA_DIR / "60-Day_SCED_Disclosure_Reports" / "csv",
        "parquet_dir": ROLLUP_DIR / "SCED_Gen_Resources",
        "date_column": "SCEDTimeStamp",
        "rust_dataset": "SCED_Gen_Resources",
        "filename_pattern": "60d_SCED_Gen_Resource_Data-{date_fmt}.csv",
        "output_columns": None,  # Keep all columns
    },
}


def get_latest_date_from_parquet(dataset: str) -> pd.Timestamp:
    """Get the latest date in the existing parquet file."""
    config = DATASET_CONFIG[dataset]
    parquet_file = config["parquet_dir"] / f"{CURRENT_YEAR}.parquet"

    if not parquet_file.exists():
        logger.warning(f"{dataset}: No {CURRENT_YEAR}.parquet file found")
        return pd.Timestamp("2023-12-11")  # API earliest date

    date_col = config["date_column"]

    try:
        df = pq.read_table(str(parquet_file)).to_pandas()
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        latest_date = df[date_col].max()

        logger.info(f"{dataset}: Latest date = {latest_date.date()}, Rows = {len(df):,}")
        return latest_date
    except Exception as e:
        logger.error(f"{dataset}: Error reading parquet: {e}")
        return pd.Timestamp("2023-12-11")


def convert_date_format(date_str: str) -> str:
    """Convert YYYY-MM-DD to MM/DD/YYYY"""
    if isinstance(date_str, str):
        dt = pd.to_datetime(date_str)
        return dt.strftime("%m/%d/%Y")
    return date_str


def get_output_filename(dataset: str, date_str: str) -> str:
    """Generate output filename matching ZIP pattern."""
    config = DATASET_CONFIG[dataset]
    pattern = config["filename_pattern"]
    now = datetime.now()

    if dataset in ["DA_prices", "AS_prices"]:
        date_part = datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y%m%d")
        time_part = now.strftime("%H%M%S")
        return pattern.format(date=date_part, time=time_part)
    else:
        # 60-day disclosure format: 60d_DAM_Gen_Resource_Data-01-DEC-24.csv
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        date_fmt = dt.strftime("%d-%b-%y").upper()
        return pattern.format(date_fmt=date_fmt)


async def download_missing_data(
    dataset: str,
    start_date: datetime,
    end_date: datetime,
    client: ERCOTWebServiceClient,
    state_manager: StateManager
) -> bool:
    """Download missing data from API."""
    config = DATASET_CONFIG[dataset]
    logger.info(f"{dataset}: Downloading {start_date.date()} to {end_date.date()}")

    # Create downloader
    downloader_class = config["downloader_class"]
    downloader = downloader_class(
        client=client,
        state_manager=state_manager,
        output_dir=DATA_DIR
    )

    # Download in chunks
    all_data = []
    current_date = start_date
    chunk_size = 7  # days

    while current_date <= end_date:
        chunk_end = min(current_date + timedelta(days=chunk_size - 1), end_date)

        logger.info(f"  Chunk: {current_date.date()} to {chunk_end.date()}")
        data = await downloader.download_chunk(current_date, chunk_end)

        if data:
            all_data.extend(data)
            logger.info(f"    Retrieved {len(data)} records")
        else:
            logger.warning(f"    No data returned")

        current_date = chunk_end + timedelta(days=1)

    if not all_data:
        logger.error(f"{dataset}: No data downloaded")
        return False

    logger.info(f"{dataset}: Total records downloaded: {len(all_data)}")

    # Convert to DataFrame and transform
    df = pd.DataFrame(all_data)

    # Transform based on dataset type
    if dataset == "DA_prices":
        # Rename numeric columns
        if df.columns[0] == 0:
            df.columns = ["DeliveryDate", "HourEnding", "SettlementPointName", "SettlementPointPrice", "DSTFlag"]
        df = df.rename(columns={"SettlementPointName": "SettlementPoint"})
        df["DSTFlag"] = df["DSTFlag"].apply(lambda x: "Y" if x else "N")
        df = df[config["output_columns"]]
        df["DeliveryDate"] = df["DeliveryDate"].apply(convert_date_format)

    elif dataset == "AS_prices":
        # Rename numeric columns
        if df.columns[0] == 0:
            df.columns = ["DeliveryDate", "HourEnding", "AncillaryServiceType", "MCPC", "DSTFlag"]
        df = df.rename(columns={"AncillaryServiceType": "AncillaryType"})
        df["DSTFlag"] = df["DSTFlag"].apply(lambda x: "Y" if x else "N")
        df = df[config["output_columns"]]
        df["DeliveryDate"] = df["DeliveryDate"].apply(convert_date_format)

    elif dataset in ["DAM_Gen_Resources", "SCED_Gen_Resources"]:
        # 60-day disclosure - convert date formats
        if "DeliveryDate" in df.columns:
            df["DeliveryDate"] = df["DeliveryDate"].apply(convert_date_format)

    # Create output directory
    output_dir = config["csv_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group by delivery date and save
    date_col = config["date_column"]
    if date_col in df.columns:
        grouped = df.groupby(df[date_col].apply(lambda x: x.split()[0] if isinstance(x, str) and "/" in x else x))
    else:
        grouped = [(start_date.strftime("%m/%d/%Y"), df)]

    saved_files = []
    for delivery_date_str, group_df in grouped:
        # Convert back to YYYY-MM-DD for filename
        if isinstance(delivery_date_str, str) and "/" in delivery_date_str:
            dt = datetime.strptime(delivery_date_str, "%m/%d/%Y")
        else:
            dt = pd.to_datetime(delivery_date_str)
        date_for_filename = dt.strftime("%Y-%m-%d")

        output_filename = get_output_filename(dataset, date_for_filename)
        output_path = output_dir / output_filename

        group_df.to_csv(output_path, index=False)
        saved_files.append(output_filename)

    logger.info(f"{dataset}: Saved {len(saved_files)} CSV files")
    return True


def regenerate_parquet(dataset: str) -> bool:
    """Regenerate parquet file using Rust processor."""
    config = DATASET_CONFIG[dataset]
    rust_dataset = config["rust_dataset"]

    logger.info(f"{dataset}: Regenerating parquet...")

    cmd = [
        str(Path.home() / ".cargo/bin/cargo"),
        "run", "--release",
        "--manifest-path", "ercot_data_processor/Cargo.toml",
        "--bin", "ercot_data_processor", "--",
        "--annual-rollup", "--dataset", rust_dataset
    ]

    env = {
        "ERCOT_DATA_DIR": str(DATA_DIR),
        "SKIP_CSV": "1",
        "PATH": subprocess.os.environ["PATH"]
    }

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    if result.returncode != 0:
        logger.error(f"{dataset}: Parquet generation failed")
        logger.error(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
        return False

    logger.info(f"{dataset}: ✅ Parquet regenerated successfully")
    return True


async def update_dataset(
    dataset: str,
    client: ERCOTWebServiceClient,
    state_manager: StateManager,
    force_start_date: datetime = None,
    dry_run: bool = False
) -> bool:
    """Complete update workflow for a dataset."""
    logger.info(f"{'='*80}")
    logger.info(f"Updating {DATASET_CONFIG[dataset]['name']}")
    logger.info(f"{'='*80}")

    # Step 1: Check latest date
    latest_date = get_latest_date_from_parquet(dataset)
    today = pd.Timestamp.now()

    # Determine date range
    if force_start_date:
        start_date = force_start_date
    else:
        start_date = latest_date + timedelta(days=1)

    end_date = today

    days_to_download = (end_date - start_date).days + 1

    if days_to_download <= 0:
        logger.info(f"{dataset}: Already up to date")
        return True

    logger.info(f"{dataset}: Need to download {days_to_download} days")

    if dry_run:
        logger.info(f"{dataset}: [DRY RUN] Would download {start_date.date()} to {end_date.date()}")
        return True

    # Step 2: Download missing data
    success = await download_missing_data(dataset, start_date, end_date, client, state_manager)
    if not success:
        return False

    # Step 3: Regenerate parquet
    success = regenerate_parquet(dataset)
    if not success:
        return False

    # Step 4: Verify final result
    new_latest = get_latest_date_from_parquet(dataset)
    logger.info(f"{dataset}: ✅ Update complete! New latest date: {new_latest.date()}")

    return True


async def main():
    parser = argparse.ArgumentParser(description="Update all ERCOT datasets")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASET_CONFIG.keys()),
        help="Specific datasets to update (default: all)"
    )
    parser.add_argument(
        "--start-date",
        help="Force start date (YYYY-MM-DD), otherwise use latest date from parquet"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check what would be updated without making changes"
    )

    args = parser.parse_args()

    # Determine which datasets to update
    if args.datasets:
        datasets = args.datasets
    else:
        datasets = list(DATASET_CONFIG.keys())

    # Parse start date if provided
    force_start_date = None
    if args.start_date:
        force_start_date = datetime.strptime(args.start_date, "%Y-%m-%d")

    # Create API client and state manager
    client = ERCOTWebServiceClient()
    state_manager = StateManager(state_file=Path("unified_update_state.json"))

    # Update each dataset
    logger.info(f"{'='*80}")
    logger.info(f"ERCOT Unified Data Updater")
    logger.info(f"Datasets: {', '.join(datasets)}")
    logger.info(f"Dry Run: {args.dry_run}")
    logger.info(f"{'='*80}\n")

    success_count = 0
    for dataset in datasets:
        try:
            success = await update_dataset(
                dataset,
                client,
                state_manager,
                force_start_date,
                args.dry_run
            )
            if success:
                success_count += 1
        except Exception as e:
            logger.error(f"{dataset}: Update failed with exception: {e}", exc_info=True)

    # Summary
    logger.info(f"\n{'='*80}")
    logger.info(f"Update Summary: {success_count}/{len(datasets)} datasets updated successfully")
    logger.info(f"{'='*80}")

    return 0 if success_count == len(datasets) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
