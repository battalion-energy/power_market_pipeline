#!/usr/bin/env python3
"""
Download DA_prices and AS_prices data from ERCOT Web Service API
and save in the correct format for Rust processor.

This script:
1. Downloads data from API
2. Transforms to match ZIP CSV format
3. Saves to correct directory with proper naming

Usage:
    python download_price_data_api.py --dataset DAM_Prices --start-date 2025-08-20 --end-date 2025-10-08
    python download_price_data_api.py --dataset AS_Prices --start-date 2025-08-20 --end-date 2025-10-08
"""

import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import logging
from dotenv import load_dotenv

load_dotenv()

# Add ercot_ws_downloader to path
sys.path.insert(0, str(Path(__file__).parent))

from ercot_ws_downloader.downloaders import DAMPriceDownloader, ASPriceDownloader
from ercot_ws_downloader.client import ERCOTWebServiceClient
from ercot_ws_downloader.state_manager import StateManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
DATA_DIR = Path(os.getenv("ERCOT_DATA_DIR", "/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data"))

DATASET_CONFIG = {
    "DAM_Prices": {
        "downloader_class": DAMPriceDownloader,
        "csv_dir": DATA_DIR / "DAM_Settlement_Point_Prices" / "csv",
        "api_columns": ["DeliveryDate", "HourEnding", "SettlementPointName", "SettlementPointPrice"],
        "output_columns": ["DeliveryDate", "HourEnding", "SettlementPoint", "SettlementPointPrice", "DSTFlag"],
        "filename_pattern": "cdr.00012331.0000000000000000.{date}.{time}.DAMSPNP4190.csv",
    },
    "AS_Prices": {
        "downloader_class": ASPriceDownloader,
        "csv_dir": DATA_DIR / "DAM_Clearing_Prices_for_Capacity" / "csv",
        "api_columns": ["DeliveryDate", "HourEnding", "AncillaryServiceType", "MCPC"],
        "output_columns": ["DeliveryDate", "HourEnding", "AncillaryType", "MCPC", "DSTFlag"],
        "filename_pattern": "cdr.00012329.0000000000000000.{date}.{time}.DAMCPCNP4188.csv",
    }
}


def convert_date_format(date_str: str) -> str:
    """Convert YYYY-MM-DD to MM/DD/YYYY"""
    if isinstance(date_str, str):
        dt = pd.to_datetime(date_str)
        return dt.strftime("%m/%d/%Y")
    return date_str


def get_output_filename(dataset: str, date_str: str) -> str:
    """
    Generate output filename matching ZIP pattern.

    Args:
        dataset: "DAM_Prices" or "AS_Prices"
        date_str: Date as YYYY-MM-DD

    Returns:
        Filename like cdr.00012331.0000000000000000.20250820.123456.DAMSPNP4190.csv
    """
    config = DATASET_CONFIG[dataset]
    pattern = config["filename_pattern"]

    # Use current time for timestamp part
    now = datetime.now()
    date_part = datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y%m%d")
    time_part = now.strftime("%H%M%S")

    return pattern.format(date=date_part, time=time_part)


async def download_and_save_dataset(
    dataset: str,
    start_date: datetime,
    end_date: datetime
) -> bool:
    """
    Download dataset from API and save in correct format.

    Args:
        dataset: "DAM_Prices" or "AS_Prices"
        start_date: Start date
        end_date: End date

    Returns:
        True if successful
    """
    config = DATASET_CONFIG[dataset]
    logger.info(f"Downloading {dataset} from {start_date.date()} to {end_date.date()}")

    # Create downloader
    client = ERCOTWebServiceClient()
    state_manager = StateManager(state_file=Path("price_download_state.json"))
    downloader_class = config["downloader_class"]
    downloader = downloader_class(
        client=client,
        state_manager=state_manager,
        output_dir=DATA_DIR
    )

    # Download data in chunks (7 days at a time to avoid huge files)
    all_data = []
    current_date = start_date
    chunk_size = 7  # days

    while current_date <= end_date:
        chunk_end = min(current_date + timedelta(days=chunk_size - 1), end_date)

        logger.info(f"  Downloading chunk: {current_date.date()} to {chunk_end.date()}")
        data = await downloader.download_chunk(current_date, chunk_end)

        if data:
            all_data.extend(data)
            logger.info(f"    Retrieved {len(data)} records")
        else:
            logger.warning(f"    No data returned")

        current_date = chunk_end + timedelta(days=1)

    if not all_data:
        logger.error(f"No data downloaded for {dataset}")
        return False

    logger.info(f"Total records downloaded: {len(all_data)}")

    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    logger.info(f"DataFrame columns: {list(df.columns)}")
    logger.info(f"Sample row:\n{df.iloc[0]}")

    # Transform to ZIP CSV format
    if dataset == "DAM_Prices":
        # API returns numeric column names, need to rename them
        # Column mapping: 0=DeliveryDate, 1=HourEnding, 2=SettlementPointName, 3=SettlementPointPrice, 4=DSTFlag
        if df.columns[0] == 0:
            df.columns = ["DeliveryDate", "HourEnding", "SettlementPointName", "SettlementPointPrice", "DSTFlag"]

        # Rename SettlementPointName to SettlementPoint
        df = df.rename(columns={"SettlementPointName": "SettlementPoint"})

        # Convert DSTFlag boolean to N/Y
        df["DSTFlag"] = df["DSTFlag"].apply(lambda x: "Y" if x else "N")

        # Select columns in correct order
        df = df[config["output_columns"]]

        # Convert date format: YYYY-MM-DD -> MM/DD/YYYY
        df["DeliveryDate"] = df["DeliveryDate"].apply(convert_date_format)

    elif dataset == "AS_Prices":
        # API returns numeric column names, need to rename them
        # Column mapping: 0=DeliveryDate, 1=HourEnding, 2=AncillaryServiceType, 3=MCPC, 4=DSTFlag
        if df.columns[0] == 0:
            df.columns = ["DeliveryDate", "HourEnding", "AncillaryServiceType", "MCPC", "DSTFlag"]

        # Rename AncillaryServiceType to AncillaryType (to match ZIP format)
        df = df.rename(columns={"AncillaryServiceType": "AncillaryType"})

        # Convert DSTFlag boolean to N/Y
        df["DSTFlag"] = df["DSTFlag"].apply(lambda x: "Y" if x else "N")

        # Select columns in correct order
        df = df[config["output_columns"]]

        # Convert date format: YYYY-MM-DD -> MM/DD/YYYY
        df["DeliveryDate"] = df["DeliveryDate"].apply(convert_date_format)

    # Create output directory
    output_dir = config["csv_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group by delivery date and save one file per date
    grouped = df.groupby("DeliveryDate")
    saved_files = []

    for delivery_date_str, group_df in grouped:
        # Convert MM/DD/YYYY back to YYYY-MM-DD for filename
        dt = datetime.strptime(delivery_date_str, "%m/%d/%Y")
        date_for_filename = dt.strftime("%Y-%m-%d")

        output_filename = get_output_filename(dataset, date_for_filename)
        output_path = output_dir / output_filename

        # Save CSV
        group_df.to_csv(output_path, index=False)
        saved_files.append(output_filename)
        logger.info(f"  Saved {len(group_df)} rows to {output_filename}")

    logger.info(f"✅ {dataset}: Saved {len(saved_files)} CSV files")
    return True


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download price data from ERCOT API")
    parser.add_argument("--dataset", required=True, choices=["DAM_Prices", "AS_Prices"],
                        help="Dataset to download")
    parser.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="End date (YYYY-MM-DD)")

    args = parser.parse_args()

    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")

    success = await download_and_save_dataset(args.dataset, start_date, end_date)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
