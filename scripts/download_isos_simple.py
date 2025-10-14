#!/usr/bin/env python3
"""
Simplified parallel ISO data downloader - CSV only, no database.

Downloads energy market data from all North American ISOs directly to CSV files.
"""

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List
import aiohttp


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def download_nyiso_csv(
    url: str,
    output_path: Path,
    session: aiohttp.ClientSession,
    retry_attempts: int = 3
) -> bool:
    """Download a single NYISO CSV file."""
    for attempt in range(retry_attempts):
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=300)) as response:
                if response.status == 200:
                    content = await response.read()
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_bytes(content)
                    return True
                elif response.status == 404:
                    return False
                else:
                    logger.warning(f"HTTP {response.status} for {url}")
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
            if attempt < retry_attempts - 1:
                await asyncio.sleep(60)
    return False


async def download_nyiso(start_date: datetime, end_date: datetime, output_dir: Path):
    """Download all NYISO data."""
    logger.info(f"Starting NYISO downloads from {start_date} to {end_date}")

    base_url = "http://mis.nyiso.com/public/csv"
    csv_dir = output_dir / "NYISO_data" / "csv_files"
    downloaded = 0

    async with aiohttp.ClientSession() as session:
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime("%Y%m%d")

            # Download DAM zonal LMP
            for loc_type in ["zone", "gen"]:
                filename = f"{date_str}damlbmp_{loc_type}.csv"
                url = f"{base_url}/damlbmp/{filename}"
                output_path = csv_dir / "dam" / loc_type / filename

                if not output_path.exists():
                    if await download_nyiso_csv(url, output_path, session):
                        downloaded += 1
                        logger.info(f"Downloaded: {filename}")

                # RT zonal LMP
                filename = f"{date_str}realtime_{loc_type}.csv"
                url = f"{base_url}/realtime/{filename}"
                output_path = csv_dir / "rt5m" / loc_type / filename

                if not output_path.exists():
                    if await download_nyiso_csv(url, output_path, session):
                        downloaded += 1
                        logger.info(f"Downloaded: {filename}")

            # DAM ancillary services
            filename = f"{date_str}damasp.csv"
            url = f"{base_url}/damasp/{filename}"
            output_path = csv_dir / "ancillary_services" / "dam" / filename

            if not output_path.exists():
                if await download_nyiso_csv(url, output_path, session):
                    downloaded += 1
                    logger.info(f"Downloaded: {filename}")

            # RT ancillary services
            filename = f"{date_str}rtasp.csv"
            url = f"{base_url}/rtasp/{filename}"
            output_path = csv_dir / "ancillary_services" / "rtm" / filename

            if not output_path.exists():
                if await download_nyiso_csv(url, output_path, session):
                    downloaded += 1
                    logger.info(f"Downloaded: {filename}")

            current_date += timedelta(days=1)

    logger.info(f"NYISO complete: {downloaded} files downloaded")
    return {"iso": "NYISO", "downloaded": downloaded}


async def download_ieso(start_date: datetime, end_date: datetime, output_dir: Path):
    """Download all IESO data."""
    logger.info(f"Starting IESO downloads from {start_date} to {end_date}")

    base_url = "https://reports-public.ieso.ca/public"
    csv_dir = output_dir / "IESO_data" / "csv_files"
    downloaded = 0

    # Report codes (these may need verification with actual IESO API)
    reports = {
        "da_lmp": "PUB_DALMPEnergy",
        "rt_lmp": "PUB_RTLMPEnergy",
        "zonal_prices": "PUB_OntarioZonalPrice",
        "oemp": "PUB_OEMP",
    }

    async with aiohttp.ClientSession() as session:
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime("%Y%m%d")

            for report_type, report_code in reports.items():
                filename = f"{report_code}_{date_str}.csv"
                url = f"{base_url}/{filename}"
                output_path = csv_dir / report_type / filename

                if not output_path.exists():
                    if await download_nyiso_csv(url, output_path, session):
                        downloaded += 1
                        logger.info(f"Downloaded: {filename}")

            current_date += timedelta(days=1)

    logger.info(f"IESO complete: {downloaded} files downloaded")
    return {"iso": "IESO", "downloaded": downloaded}


async def download_aeso(start_date: datetime, end_date: datetime, output_dir: Path):
    """Download AESO data."""
    logger.info(f"Starting AESO downloads from {start_date} to {end_date}")

    csv_dir = output_dir / "AESO_data" / "csv_files"
    csv_dir.mkdir(parents=True, exist_ok=True)

    logger.info("AESO requires manual download from http://ets.aeso.ca/")
    logger.info("Download hourly pool price files and place in: " + str(csv_dir))

    return {"iso": "AESO", "downloaded": 0, "status": "manual_required"}


async def run_parallel_downloads(
    isos: List[str],
    start_date: datetime,
    end_date: datetime,
    output_dir: Path
):
    """Run all ISO downloads in parallel."""

    download_functions = {
        "NYISO": download_nyiso,
        "IESO": download_ieso,
        "AESO": download_aeso,
    }

    tasks = []
    for iso in isos:
        if iso in download_functions:
            tasks.append(download_functions[iso](start_date, end_date, output_dir))
        else:
            logger.warning(f"Unknown ISO: {iso}")

    logger.info(f"Starting parallel downloads for {len(tasks)} ISOs")
    results = await asyncio.gather(*tasks, return_exceptions=True)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Simplified ISO data downloader - CSV only"
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
        default=["NYISO", "IESO"],
        help="ISOs to download (default: NYISO IESO)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/pool/ssd8tb/data/iso",
        help="Output directory for CSV files"
    )

    args = parser.parse_args()

    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    output_dir = Path(args.output_dir)

    logger.info(f"ISO Data Download Tool")
    logger.info(f"ISOs: {args.isos}")
    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
    logger.info(f"Output: {output_dir}")
    logger.info("="*80)

    results = asyncio.run(run_parallel_downloads(
        isos=args.isos,
        start_date=start_date,
        end_date=end_date,
        output_dir=output_dir
    ))

    print("\n" + "="*80)
    print("DOWNLOAD SUMMARY")
    print("="*80)

    for result in results:
        if isinstance(result, dict):
            iso = result.get("iso", "Unknown")
            downloaded = result.get("downloaded", 0)
            status = result.get("status", "success")
            print(f"{iso}: {downloaded} files downloaded ({status})")
        else:
            print(f"Error: {result}")

    print("="*80)


if __name__ == "__main__":
    main()
