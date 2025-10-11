#!/usr/bin/env python3
"""
Parallel ISO data downloader orchestration script.

Downloads energy market data from all North American ISOs:
- NYISO (New York)
- CAISO (California)
- IESO (Ontario)
- AESO (Alberta)
- SPP (Southwest Power Pool) - requires certificates

Usage:
    python download_all_isos.py --start-date 2019-01-01 --end-date 2025-10-10
    python download_all_isos.py --start-date 2024-01-01 --end-date 2024-01-31 --isos NYISO CAISO
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import structlog

from downloaders.base_v2 import DownloadConfig
from downloaders.nyiso import NYISODownloaderV2
from downloaders.caiso import CAISODownloaderV2
from downloaders.ieso import IESODownloaderV2
from downloaders.aeso import AESODownloaderV2
from downloaders.spp import SPPDownloaderV2


# Configure structured logging
logging.basicConfig(
    format="%(message)s",
    stream=sys.stdout,
    level=logging.INFO,
)
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.dev.ConsoleRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


async def download_nyiso(config: DownloadConfig) -> dict:
    """Download all NYISO data."""
    logger.info("Starting NYISO downloads", start=config.start_date, end=config.end_date)

    try:
        downloader = NYISODownloaderV2(config)
        results = {}

        # Download DAM LMP (zone and gen)
        results['dam_lmp'] = await downloader.download_lmp(
            "DAM", config.start_date, config.end_date
        )

        # Download RT LMP (zone and gen)
        results['rt_lmp'] = await downloader.download_lmp(
            "RT5M", config.start_date, config.end_date
        )

        # Download DAM ancillary services
        results['dam_as'] = await downloader.download_ancillary_services(
            "ALL", "DAM", config.start_date, config.end_date
        )

        # Download RT ancillary services
        results['rt_as'] = await downloader.download_ancillary_services(
            "ALL", "RTM", config.start_date, config.end_date
        )

        # Download load data
        results['actual_load'] = await downloader.download_load(
            "actual", config.start_date, config.end_date
        )
        results['forecast_load'] = await downloader.download_load(
            "forecast", config.start_date, config.end_date
        )

        logger.info("NYISO downloads complete", results=results)
        return {"iso": "NYISO", "status": "success", "results": results}

    except Exception as e:
        logger.error("NYISO download failed", error=str(e))
        return {"iso": "NYISO", "status": "failed", "error": str(e)}


async def download_caiso(config: DownloadConfig) -> dict:
    """Download all CAISO data."""
    logger.info("Starting CAISO downloads", start=config.start_date, end=config.end_date)

    try:
        downloader = CAISODownloaderV2(config)
        results = {}

        # Download DAM LMP
        results['dam_lmp'] = await downloader.download_lmp(
            "DAM", config.start_date, config.end_date
        )

        # Download RT LMP
        results['rt_lmp'] = await downloader.download_lmp(
            "RT5M", config.start_date, config.end_date
        )

        # Download DAM ancillary services
        results['dam_as'] = await downloader.download_ancillary_services(
            "ALL", "DAM", config.start_date, config.end_date
        )

        # Download RT ancillary services
        results['rt_as'] = await downloader.download_ancillary_services(
            "ALL", "RTM", config.start_date, config.end_date
        )

        # Download load forecast
        results['load_forecast'] = await downloader.download_load(
            "forecast", config.start_date, config.end_date
        )

        logger.info("CAISO downloads complete", results=results)
        return {"iso": "CAISO", "status": "success", "results": results}

    except Exception as e:
        logger.error("CAISO download failed", error=str(e))
        return {"iso": "CAISO", "status": "failed", "error": str(e)}


async def download_ieso(config: DownloadConfig) -> dict:
    """Download all IESO data."""
    logger.info("Starting IESO downloads", start=config.start_date, end=config.end_date)

    try:
        downloader = IESODownloaderV2(config)
        results = {}

        # Download DAM LMP (post-May 2025)
        results['dam_lmp'] = await downloader.download_lmp(
            "DAM", config.start_date, config.end_date
        )

        # Download RT LMP (post-May 2025)
        results['rt_lmp'] = await downloader.download_lmp(
            "RT5M", config.start_date, config.end_date
        )

        # Download zonal prices
        results['zonal_prices'] = await downloader.download_ontario_zonal_prices(
            config.start_date, config.end_date
        )

        # Download OEMP
        results['oemp'] = await downloader.download_oemp(
            config.start_date, config.end_date
        )

        # Download legacy HOEP (pre-May 2025)
        results['hoep'] = await downloader.download_legacy_hoep(
            config.start_date, config.end_date
        )

        # Download ancillary services
        for product in ["10S", "10NS", "30OR"]:
            results[f'as_{product.lower()}'] = await downloader.download_ancillary_services(
                product, "RTM", config.start_date, config.end_date
            )

        logger.info("IESO downloads complete", results=results)
        return {"iso": "IESO", "status": "success", "results": results}

    except Exception as e:
        logger.error("IESO download failed", error=str(e))
        return {"iso": "IESO", "status": "failed", "error": str(e)}


async def download_aeso(config: DownloadConfig) -> dict:
    """Download all AESO data."""
    logger.info("Starting AESO downloads", start=config.start_date, end=config.end_date)

    try:
        downloader = AESODownloaderV2(config)

        # AESO has a convenience method to download all data types
        results = await downloader.download_all_data_types(
            config.start_date, config.end_date
        )

        logger.info("AESO downloads complete", results=results)
        return {"iso": "AESO", "status": "success", "results": results}

    except Exception as e:
        logger.error("AESO download failed", error=str(e))
        return {"iso": "AESO", "status": "failed", "error": str(e)}


async def download_spp(config: DownloadConfig) -> dict:
    """Download all SPP data (requires certificates)."""
    logger.info("Starting SPP downloads", start=config.start_date, end=config.end_date)

    try:
        downloader = SPPDownloaderV2(config)
        results = {}

        # SPP requires digital certificates - will raise NotImplementedError
        results['dam_lmp'] = await downloader.download_lmp(
            "DAM", config.start_date, config.end_date
        )

        results['rt_lmp'] = await downloader.download_lmp(
            "RTBM", config.start_date, config.end_date
        )

        results['as'] = await downloader.download_ancillary_services(
            "ALL", "DAM", config.start_date, config.end_date
        )

        logger.info("SPP downloads complete", results=results)
        return {"iso": "SPP", "status": "success", "results": results}

    except NotImplementedError as e:
        logger.warning("SPP requires digital certificates", error=str(e))
        return {"iso": "SPP", "status": "skipped", "reason": "certificates_required"}

    except Exception as e:
        logger.error("SPP download failed", error=str(e))
        return {"iso": "SPP", "status": "failed", "error": str(e)}


async def run_parallel_downloads(
    isos: List[str],
    start_date: datetime,
    end_date: datetime,
    output_dir: Path
) -> List[dict]:
    """Run all ISO downloads in parallel."""

    # Create download config
    config = DownloadConfig(
        start_date=start_date,
        end_date=end_date,
        data_types=["lmp", "ancillary_services", "load"],
        output_dir=str(output_dir),
        batch_size=1000,
        retry_attempts=3,
        retry_delay=60
    )

    # Map ISO names to download functions
    download_functions = {
        "NYISO": download_nyiso,
        "CAISO": download_caiso,
        "IESO": download_ieso,
        "AESO": download_aeso,
        "SPP": download_spp,
    }

    # Create tasks for selected ISOs
    tasks = []
    for iso in isos:
        if iso in download_functions:
            tasks.append(download_functions[iso](config))
        else:
            logger.warning(f"Unknown ISO: {iso}")

    # Run all downloads in parallel
    logger.info(f"Starting parallel downloads for {len(tasks)} ISOs")
    results = await asyncio.gather(*tasks, return_exceptions=True)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Download energy market data from all North American ISOs"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--isos",
        nargs="+",
        default=["NYISO", "CAISO", "IESO", "AESO", "SPP"],
        help="ISOs to download (default: all)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/pool/ssd8tb/data/iso",
        help="Output directory for CSV files"
    )

    args = parser.parse_args()

    # Parse dates
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    output_dir = Path(args.output_dir)

    logger.info(
        "Starting ISO data downloads",
        isos=args.isos,
        start_date=start_date,
        end_date=end_date,
        output_dir=output_dir
    )

    # Run parallel downloads
    results = asyncio.run(run_parallel_downloads(
        isos=args.isos,
        start_date=start_date,
        end_date=end_date,
        output_dir=output_dir
    ))

    # Print summary
    print("\n" + "="*80)
    print("DOWNLOAD SUMMARY")
    print("="*80)

    for result in results:
        if isinstance(result, dict):
            iso = result.get("iso", "Unknown")
            status = result.get("status", "unknown")

            print(f"\n{iso}: {status.upper()}")

            if status == "success":
                for key, value in result.get("results", {}).items():
                    print(f"  {key}: {value} files")
            elif status == "failed":
                print(f"  Error: {result.get('error', 'Unknown error')}")
            elif status == "skipped":
                print(f"  Reason: {result.get('reason', 'Unknown')}")
        else:
            print(f"\nError: {result}")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
