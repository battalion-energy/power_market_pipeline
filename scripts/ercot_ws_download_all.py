#!/usr/bin/env python3
"""
ERCOT Web Service Downloader - Main CLI

Downloads all required ERCOT data from the Web Service API:
- Day-Ahead Market (DAM) prices
- Real-Time Market (RTM) prices
- Ancillary Services (AS) prices
- 60-day disclosure data (DAM and SCED)

Usage:
    # Download all datasets with default settings
    python ercot_ws_download_all.py

    # Download specific datasets
    python ercot_ws_download_all.py --datasets DAM_Prices RTM_Prices

    # Test mode (dry run)
    python ercot_ws_download_all.py --test

    # Batch mode (download and exit)
    python ercot_ws_download_all.py --batch

    # Continuous mode (keep updating)
    python ercot_ws_download_all.py --continuous --interval 3600
"""

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from ercot_ws_downloader import (
    ERCOTWebServiceClient,
    StateManager,
    DAMPriceDownloader,
    RTMPriceDownloader,
    ASPriceDownloader,
    DAMDisclosureDownloader,
    DAMLoadResourceDownloader,
    SCEDDisclosureDownloader,
    SCEDLoadResourceDownloader,
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ercot_ws_download.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


async def run_downloader(
    downloader_class,
    client: ERCOTWebServiceClient,
    state_manager: StateManager,
    output_dir: Path,
    start_date=None,
    end_date=None,
):
    """Run a single downloader."""
    try:
        downloader = downloader_class(client, state_manager, output_dir)
        logger.info(f"Starting {downloader.dataset_name} download...")

        success = await downloader.download_range(start_date, end_date)

        if success:
            logger.info(f"✓ {downloader.dataset_name} download completed successfully")
        else:
            logger.error(f"✗ {downloader.dataset_name} download had errors")

        return success
    except Exception as e:
        logger.error(f"✗ {downloader_class.__name__} failed: {e}", exc_info=True)
        return False


async def download_all(
    datasets: list = None,
    output_dir: Path = None,
    state_file: Path = None,
    test_mode: bool = False,
):
    """Download all specified datasets."""

    # Default output directory
    if output_dir is None:
        ercot_base = Path(os.getenv(
            "ERCOT_DATA_DIR",
            "/pool/ssd8tb/data/iso/ERCOT"
        ))
        # If ERCOT_DATA_DIR already points to ERCOT_data, use it directly
        if ercot_base.name == "ERCOT_data" or (ercot_base / "ERCOT_data").exists():
            if ercot_base.name != "ERCOT_data":
                output_dir = ercot_base / "ercot_market_data" / "ERCOT_data"
            else:
                output_dir = ercot_base
        else:
            output_dir = ercot_base / "ercot_market_data" / "ERCOT_data"

    # Default state file
    if state_file is None:
        state_file = Path("ercot_download_state.json")

    logger.info("=" * 80)
    logger.info("ERCOT Web Service Downloader")
    logger.info("=" * 80)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"State file: {state_file}")
    logger.info(f"Test mode: {test_mode}")
    logger.info("=" * 80)

    # Initialize client and state manager
    try:
        client = ERCOTWebServiceClient()
        state_manager = StateManager(state_file)
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        return False

    # Test connection
    if test_mode or not await client.test_connection():
        logger.error("Connection test failed!")
        if not test_mode:
            return False
        logger.info("Test mode: Connection test skipped")

    # Define all available downloaders in priority order
    all_downloaders = [
        ("DAM_Prices", DAMPriceDownloader),
        ("AS_Prices", ASPriceDownloader),
        ("RTM_Prices", RTMPriceDownloader),
        ("60d_DAM_Gen_Resources", DAMDisclosureDownloader),
        ("60d_DAM_Load_Resources", DAMLoadResourceDownloader),
        ("60d_SCED_Gen_Resources", SCEDDisclosureDownloader),
        ("60d_SCED_Load_Resources", SCEDLoadResourceDownloader),
    ]

    # Filter datasets if specified
    if datasets:
        downloaders_to_run = [
            (name, cls) for name, cls in all_downloaders
            if name in datasets
        ]
    else:
        downloaders_to_run = all_downloaders

    logger.info(f"Will download {len(downloaders_to_run)} datasets:")
    for name, _ in downloaders_to_run:
        logger.info(f"  - {name}")
    logger.info("")

    if test_mode:
        logger.info("Test mode: Exiting without downloading")
        return True

    # Run all downloaders
    results = {}
    for name, downloader_class in downloaders_to_run:
        success = await run_downloader(
            downloader_class,
            client,
            state_manager,
            output_dir
        )
        results[name] = success
        logger.info("")

    # Print summary
    logger.info("=" * 80)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("=" * 80)

    successful = sum(1 for v in results.values() if v)
    failed = len(results) - successful

    for name, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        logger.info(f"{status}: {name}")

    logger.info("")
    logger.info(f"Total: {successful} successful, {failed} failed")
    logger.info("=" * 80)

    return failed == 0


async def continuous_download(
    datasets: list = None,
    output_dir: Path = None,
    state_file: Path = None,
    interval_seconds: int = 3600,
):
    """Run downloads continuously at specified interval."""

    logger.info("=" * 80)
    logger.info("CONTINUOUS DOWNLOAD MODE")
    logger.info(f"Update interval: {interval_seconds} seconds ({interval_seconds/3600:.1f} hours)")
    logger.info("Press Ctrl+C to stop")
    logger.info("=" * 80)

    iteration = 1
    while True:
        try:
            logger.info(f"\n{'='*80}")
            logger.info(f"Starting download iteration {iteration} at {datetime.now()}")
            logger.info(f"{'='*80}\n")

            await download_all(datasets, output_dir, state_file, test_mode=False)

            logger.info(f"\nIteration {iteration} complete. Sleeping for {interval_seconds} seconds...")
            iteration += 1

            await asyncio.sleep(interval_seconds)

        except KeyboardInterrupt:
            logger.info("\n\nReceived interrupt signal. Shutting down gracefully...")
            break
        except Exception as e:
            logger.error(f"Error in continuous download: {e}", exc_info=True)
            logger.info(f"Sleeping for {interval_seconds} seconds before retry...")
            await asyncio.sleep(interval_seconds)


def main():
    """Main entry point."""

    parser = argparse.ArgumentParser(
        description="Download ERCOT data from Web Service API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Specific datasets to download (default: all)",
        choices=[
            "DAM_Prices",
            "AS_Prices",
            "RTM_Prices",
            "60d_DAM_Gen_Resources",
            "60d_DAM_Load_Resources",
            "60d_SCED_Gen_Resources",
            "60d_SCED_Load_Resources",
        ]
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for CSV files (default: ERCOT_DATA_DIR env var)"
    )

    parser.add_argument(
        "--state-file",
        type=Path,
        default=Path("ercot_download_state.json"),
        help="State file for tracking downloads (default: ercot_download_state.json)"
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: verify credentials and show what would be downloaded"
    )

    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Run continuously, updating at regular intervals"
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=3600,
        help="Update interval in seconds for continuous mode (default: 3600 = 1 hour)"
    )

    parser.add_argument(
        "--batch",
        action="store_true",
        help="Batch mode: download and exit (default)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Run downloader
    if args.continuous:
        asyncio.run(continuous_download(
            datasets=args.datasets,
            output_dir=args.output_dir,
            state_file=args.state_file,
            interval_seconds=args.interval,
        ))
    else:
        success = asyncio.run(download_all(
            datasets=args.datasets,
            output_dir=args.output_dir,
            state_file=args.state_file,
            test_mode=args.test,
        ))

        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
