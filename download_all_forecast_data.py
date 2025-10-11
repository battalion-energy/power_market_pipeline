#!/usr/bin/env python3
"""
Download ALL ERCOT datasets needed for price forecasting model.

This script downloads:
1. Wind power production (actual + forecasts)
2. Solar power production (actual + forecasts)
3. Load forecasts (by forecast zone and weather zone)
4. Actual system load (by forecast zone and weather zone)
5. Fuel mix (actual generation by fuel type)
6. System-wide demand (5-minute actuals)
7. Unplanned resource outages
8. DAM system lambda (shadow prices)

All data is saved as CSV and can be converted to Parquet.

Usage:
    # Download last 30 days of all datasets
    python download_all_forecast_data.py --days 30

    # Download specific date range
    python download_all_forecast_data.py --start-date 2024-01-01 --end-date 2024-12-31

    # Download specific datasets only
    python download_all_forecast_data.py --datasets wind solar load --days 30

    # Convert to Parquet after download
    python download_all_forecast_data.py --days 30 --convert-to-parquet
"""

import asyncio
import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import logging
from typing import List, Dict, Any

# Add ercot_ws_downloader to path
sys.path.insert(0, str(Path(__file__).parent))

from ercot_ws_downloader.forecast_downloaders import (
    WindPowerDownloader,
    SolarPowerDownloader,
    LoadForecastByForecastZoneDownloader,
    LoadForecastByWeatherZoneDownloader,
    ActualSystemLoadByWeatherZoneDownloader,
    ActualSystemLoadByForecastZoneDownloader,
    UnplannedResourceOutagesDownloader,
    DAMSystemLambdaDownloader,
    FuelMixDownloader,
    SystemWideDemandDownloader,
)
from ercot_ws_downloader.downloaders import (
    DAMPriceDownloader,
    RTMPriceDownloader,
    ASPriceDownloader,
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

# Dataset registry with descriptions
DATASET_REGISTRY = {
    # Renewable generation and forecasts
    "wind": {
        "class": WindPowerDownloader,
        "name": "Wind Power Production",
        "report": "NP4-732-CD",
        "description": "Wind generation actual + STWPF forecasts (hourly)",
        "category": "Renewable Generation",
    },
    "solar": {
        "class": SolarPowerDownloader,
        "name": "Solar Power Production",
        "report": "NP4-745-CD",
        "description": "Solar generation actual + STPPF forecasts (hourly)",
        "category": "Renewable Generation",
    },

    # Load forecasts
    "load_fzone": {
        "class": LoadForecastByForecastZoneDownloader,
        "name": "Load Forecast (Forecast Zones)",
        "report": "NP3-565-CD",
        "description": "7-day load forecast by forecast zone (hourly)",
        "category": "Load Forecasts",
    },
    "load_wzone": {
        "class": LoadForecastByWeatherZoneDownloader,
        "name": "Load Forecast (Weather Zones)",
        "report": "NP3-566-CD",
        "description": "7-day load forecast by weather zone (hourly)",
        "category": "Load Forecasts",
    },

    # Actual load
    "actual_load_wzone": {
        "class": ActualSystemLoadByWeatherZoneDownloader,
        "name": "Actual Load (Weather Zones)",
        "report": "NP6-345-CD",
        "description": "Actual system load by weather zone (5-min)",
        "category": "Actual Load",
    },
    "actual_load_fzone": {
        "class": ActualSystemLoadByForecastZoneDownloader,
        "name": "Actual Load (Forecast Zones)",
        "report": "NP6-346-CD",
        "description": "Actual system load by forecast zone (hourly)",
        "category": "Actual Load",
    },

    # Fuel mix and generation
    "fuel_mix": {
        "class": FuelMixDownloader,
        "name": "Fuel Mix",
        "report": "NP6-787-CD",
        "description": "Actual generation by fuel type (15-min)",
        "category": "Generation Mix",
    },
    "system_demand": {
        "class": SystemWideDemandDownloader,
        "name": "System-Wide Demand",
        "report": "NP6-322-CD",
        "description": "System-wide actual demand (5-min)",
        "category": "System Metrics",
    },

    # Outages and constraints
    "outages": {
        "class": UnplannedResourceOutagesDownloader,
        "name": "Unplanned Outages",
        "report": "NP3-233-CD",
        "description": "Unplanned resource outages",
        "category": "Outages",
    },
    "lambda": {
        "class": DAMSystemLambdaDownloader,
        "name": "DAM System Lambda",
        "report": "NP4-191-CD",
        "description": "Day-ahead system lambda (shadow prices)",
        "category": "System Metrics",
    },

    # Prices (already implemented)
    "dam_prices": {
        "class": DAMPriceDownloader,
        "name": "DAM Prices",
        "report": "NP4-190-CD",
        "description": "Day-ahead market settlement point prices",
        "category": "Prices",
    },
    "rtm_prices": {
        "class": RTMPriceDownloader,
        "name": "RTM Prices",
        "report": "NP6-785-CD",
        "description": "Real-time market settlement point prices (5-min)",
        "category": "Prices",
    },
    "as_prices": {
        "class": ASPriceDownloader,
        "name": "Ancillary Services Prices",
        "report": "NP4-188-CD",
        "description": "DAM clearing prices for all AS products",
        "category": "Prices",
    },
}


def print_dataset_summary():
    """Print a summary of all available datasets."""
    print("\n" + "="*80)
    print("ERCOT DATASETS FOR PRICE FORECASTING")
    print("="*80 + "\n")

    # Group by category
    categories = {}
    for key, info in DATASET_REGISTRY.items():
        category = info["category"]
        if category not in categories:
            categories[category] = []
        categories[category].append((key, info))

    # Print by category
    for category in sorted(categories.keys()):
        print(f"\n{category}:")
        print("-" * 80)
        for key, info in categories[category]:
            print(f"  {key:20s} | {info['report']:12s} | {info['description']}")

    print("\n" + "="*80 + "\n")


async def download_dataset(
    dataset_key: str,
    start_date: datetime,
    end_date: datetime,
    client: ERCOTWebServiceClient,
    state_manager: StateManager
) -> Dict[str, Any]:
    """
    Download a single dataset.

    Returns:
        Dictionary with download results
    """
    dataset_info = DATASET_REGISTRY[dataset_key]
    downloader_class = dataset_info["class"]

    logger.info(f"\n{'='*80}")
    logger.info(f"Downloading: {dataset_info['name']} ({dataset_info['report']})")
    logger.info(f"Description: {dataset_info['description']}")
    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
    logger.info(f"{'='*80}\n")

    # Create downloader
    downloader = downloader_class(
        client=client,
        state_manager=state_manager,
        output_dir=DATA_DIR
    )

    try:
        # Download all data
        success = await downloader.download_range(start_date, end_date)

        # Get stats
        output_dir = downloader.get_output_dir()
        csv_files = list(output_dir.glob("*.csv"))

        result = {
            "dataset": dataset_key,
            "name": dataset_info["name"],
            "report": dataset_info["report"],
            "success": success,
            "files_created": len(csv_files),
            "output_dir": str(output_dir),
            "start_date": start_date.date(),
            "end_date": end_date.date(),
        }

        if success:
            logger.info(f"✅ SUCCESS: {dataset_info['name']}")
            logger.info(f"   Files created: {len(csv_files)}")
            logger.info(f"   Output dir: {output_dir}")
        else:
            logger.error(f"❌ FAILED: {dataset_info['name']}")

        return result

    except Exception as e:
        logger.error(f"❌ ERROR downloading {dataset_info['name']}: {str(e)}")
        return {
            "dataset": dataset_key,
            "name": dataset_info["name"],
            "report": dataset_info["report"],
            "success": False,
            "error": str(e),
            "start_date": start_date.date(),
            "end_date": end_date.date(),
        }


def convert_csv_to_parquet(csv_dir: Path) -> bool:
    """
    Convert all CSV files in a directory to a single Parquet file.

    Args:
        csv_dir: Directory containing CSV files

    Returns:
        True if successful
    """
    try:
        csv_files = sorted(csv_dir.glob("*.csv"))
        if not csv_files:
            logger.warning(f"No CSV files found in {csv_dir}")
            return False

        logger.info(f"Converting {len(csv_files)} CSV files to Parquet...")

        # Read all CSVs and combine
        dfs = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                dfs.append(df)
            except Exception as e:
                logger.error(f"Error reading {csv_file}: {e}")

        if not dfs:
            logger.error("No CSV files could be read")
            return False

        # Combine all dataframes
        combined_df = pd.concat(dfs, ignore_index=True)

        # Save as Parquet
        parquet_file = csv_dir.parent / "parquet" / f"{csv_dir.name}.parquet"
        parquet_file.parent.mkdir(parents=True, exist_ok=True)

        combined_df.to_parquet(parquet_file, index=False, engine='pyarrow', compression='snappy')

        logger.info(f"✅ Created Parquet file: {parquet_file}")
        logger.info(f"   Rows: {len(combined_df):,}")
        logger.info(f"   Columns: {len(combined_df.columns)}")

        # Report size savings
        csv_size = sum(f.stat().st_size for f in csv_files)
        parquet_size = parquet_file.stat().st_size
        savings = (1 - parquet_size / csv_size) * 100

        logger.info(f"   CSV size: {csv_size / 1024 / 1024:.1f} MB")
        logger.info(f"   Parquet size: {parquet_size / 1024 / 1024:.1f} MB")
        logger.info(f"   Compression: {savings:.1f}% smaller")

        return True

    except Exception as e:
        logger.error(f"Error converting to Parquet: {e}")
        return False


async def main():
    parser = argparse.ArgumentParser(
        description="Download ERCOT datasets for price forecasting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download last 30 days of all datasets
  python download_all_forecast_data.py --days 30

  # Download specific date range
  python download_all_forecast_data.py --start-date 2024-01-01 --end-date 2024-12-31

  # Download only wind and solar data
  python download_all_forecast_data.py --datasets wind solar --days 90

  # List all available datasets
  python download_all_forecast_data.py --list
        """
    )

    parser.add_argument("--list", action="store_true",
                        help="List all available datasets and exit")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASET_REGISTRY.keys()),
                        help="Specific datasets to download (default: all)")
    parser.add_argument("--days", type=int,
                        help="Number of days to download (from today backwards)")
    parser.add_argument("--start-date", type=str,
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str,
                        help="End date (YYYY-MM-DD)")
    parser.add_argument("--convert-to-parquet", action="store_true",
                        help="Convert downloaded CSV files to Parquet format")

    args = parser.parse_args()

    # List datasets and exit
    if args.list:
        print_dataset_summary()
        return 0

    # Determine date range
    if args.days:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.days)
    elif args.start_date and args.end_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    else:
        parser.error("Must specify either --days or both --start-date and --end-date")

    # Determine which datasets to download
    if args.datasets:
        datasets_to_download = args.datasets
    else:
        datasets_to_download = list(DATASET_REGISTRY.keys())

    logger.info(f"\n{'='*80}")
    logger.info(f"ERCOT DATA DOWNLOAD FOR PRICE FORECASTING")
    logger.info(f"{'='*80}")
    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
    logger.info(f"Datasets: {len(datasets_to_download)}")
    logger.info(f"Output directory: {DATA_DIR}")
    logger.info(f"{'='*80}\n")

    # Initialize client and state manager
    client = ERCOTWebServiceClient()
    state_manager = StateManager(state_file=Path("forecast_download_state.json"))

    # Download all datasets
    results = []
    for dataset_key in datasets_to_download:
        result = await download_dataset(
            dataset_key,
            start_date,
            end_date,
            client,
            state_manager
        )
        results.append(result)

        # Small delay between datasets to be polite to API
        await asyncio.sleep(5)

    # Convert to Parquet if requested
    if args.convert_to_parquet:
        logger.info(f"\n{'='*80}")
        logger.info("CONVERTING CSV TO PARQUET")
        logger.info(f"{'='*80}\n")

        for result in results:
            if result["success"] and "output_dir" in result:
                csv_dir = Path(result["output_dir"])
                convert_csv_to_parquet(csv_dir)

    # Print summary report
    logger.info(f"\n{'='*80}")
    logger.info("DOWNLOAD SUMMARY")
    logger.info(f"{'='*80}\n")

    # Create summary table
    summary_data = []
    for result in results:
        summary_data.append({
            "Dataset": result["name"],
            "Report": result["report"],
            "Status": "✅ Success" if result["success"] else "❌ Failed",
            "Files": result.get("files_created", 0),
            "Start Date": result["start_date"],
            "End Date": result["end_date"],
        })

    df = pd.DataFrame(summary_data)
    print(df.to_string(index=False))

    # Overall statistics
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful

    logger.info(f"\n{'='*80}")
    logger.info(f"Total datasets: {len(results)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"{'='*80}\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
