"""NYISO downloader using standardized schema."""

import asyncio
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
import pandas as pd

from database import AncillaryServices, LMP, Load, get_db
from downloaders.base_v2 import BaseDownloaderV2, DownloadConfig


class NYISODownloaderV2(BaseDownloaderV2):
    """NYISO data downloader using standardized schema."""

    def __init__(self, config: DownloadConfig):
        super().__init__("NYISO", config)

        # NYISO provides public CSV files
        self.base_url = "http://mis.nyiso.com/public/csv"

        # Create output directories
        self.csv_dir = Path(config.output_dir) / "NYISO_data" / "csv_files"
        self.csv_dir.mkdir(parents=True, exist_ok=True)

    async def _download_csv(
        self,
        url: str,
        output_path: Path,
        session: aiohttp.ClientSession
    ) -> bool:
        """Download a single CSV file with retry logic."""
        for attempt in range(self.config.retry_attempts):
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=300)) as response:
                    if response.status == 200:
                        content = await response.read()
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        output_path.write_bytes(content)
                        self.logger.debug(f"Downloaded {url} -> {output_path}")
                        return True
                    elif response.status == 404:
                        self.logger.warning(f"File not found (404): {url}")
                        return False
                    else:
                        self.logger.warning(f"HTTP {response.status} for {url}")
            except Exception as e:
                self.logger.warning(
                    f"Download attempt {attempt + 1} failed for {url}: {e}"
                )
                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(self.config.retry_delay)

        return False

    async def download_lmp(
        self,
        market: str,
        start_date: datetime,
        end_date: datetime,
        locations: Optional[List[str]] = None
    ) -> int:
        """Download LMP data for NYISO.

        Args:
            market: 'DAM' (day-ahead) or 'RT5M' (real-time)
            start_date: Start date for download
            end_date: End date for download
            locations: 'zone' or 'gen' - if None, downloads both
        """
        self.logger.info(
            f"Downloading NYISO {market} LMP data",
            start=start_date,
            end=end_date
        )

        # Market mapping
        if market == "DAM":
            url_prefix = "damlbmp"
            file_prefix = "damlbmp"
        elif market == "RT5M":
            url_prefix = "realtime"
            file_prefix = "realtime"
        else:
            raise ValueError(f"Unknown market: {market}")

        # Location types to download
        location_types = locations or ["zone", "gen"]

        downloaded = 0
        async with aiohttp.ClientSession() as session:
            # Generate all dates
            current_date = start_date
            while current_date <= end_date:
                date_str = current_date.strftime("%Y%m%d")

                for loc_type in location_types:
                    # URL format: http://mis.nyiso.com/public/csv/damlbmp/20240101damlbmp_zone.csv
                    filename = f"{date_str}{file_prefix}_{loc_type}.csv"
                    url = f"{self.base_url}/{url_prefix}/{filename}"

                    output_dir = self.csv_dir / market.lower() / loc_type
                    output_path = output_dir / filename

                    # Skip if already downloaded
                    if output_path.exists():
                        self.logger.debug(f"Skipping existing file: {output_path}")
                        downloaded += 1
                        continue

                    success = await self._download_csv(url, output_path, session)
                    if success:
                        downloaded += 1

                current_date += timedelta(days=1)

        self.logger.info(f"Downloaded {downloaded} NYISO {market} LMP files")
        return downloaded

    async def download_ancillary_services(
        self,
        product: str,
        market: str,
        start_date: datetime,
        end_date: datetime
    ) -> int:
        """Download ancillary services data.

        Args:
            product: 'ALL' or specific product (not used - NYISO provides all in one file)
            market: 'DAM' or 'RTM'
        """
        self.logger.info(
            f"Downloading NYISO {market} ancillary services",
            start=start_date,
            end=end_date
        )

        # Market mapping
        if market == "DAM":
            url_prefix = "damasp"
            file_prefix = "damasp"
        elif market == "RTM":
            url_prefix = "rtasp"
            file_prefix = "rtasp"
        else:
            raise ValueError(f"Unknown market: {market}")

        downloaded = 0
        async with aiohttp.ClientSession() as session:
            current_date = start_date
            while current_date <= end_date:
                date_str = current_date.strftime("%Y%m%d")

                # URL format: http://mis.nyiso.com/public/csv/damasp/20240101damasp.csv
                filename = f"{date_str}{file_prefix}.csv"
                url = f"{self.base_url}/{url_prefix}/{filename}"

                output_dir = self.csv_dir / "ancillary_services" / market.lower()
                output_path = output_dir / filename

                if output_path.exists():
                    self.logger.debug(f"Skipping existing file: {output_path}")
                    downloaded += 1
                    current_date += timedelta(days=1)
                    continue

                success = await self._download_csv(url, output_path, session)
                if success:
                    downloaded += 1

                current_date += timedelta(days=1)

        self.logger.info(f"Downloaded {downloaded} NYISO AS files")
        return downloaded

    async def download_load(
        self,
        forecast_type: str,
        start_date: datetime,
        end_date: datetime
    ) -> int:
        """Download load data.

        Args:
            forecast_type: 'actual' (pal) or 'forecast' (isolf)
        """
        self.logger.info(
            f"Downloading NYISO {forecast_type} load data",
            start=start_date,
            end=end_date
        )

        # Forecast type mapping
        if forecast_type == "actual":
            url_prefix = "pal"
            file_prefix = "pal"
        elif forecast_type == "forecast":
            url_prefix = "isolf"
            file_prefix = "isolf"
        else:
            raise ValueError(f"Unknown forecast type: {forecast_type}")

        downloaded = 0
        async with aiohttp.ClientSession() as session:
            current_date = start_date
            while current_date <= end_date:
                date_str = current_date.strftime("%Y%m%d")

                # URL format: http://mis.nyiso.com/public/csv/pal/20240101pal.csv
                filename = f"{date_str}{file_prefix}.csv"
                url = f"{self.base_url}/{url_prefix}/{filename}"

                output_dir = self.csv_dir / "load" / forecast_type
                output_path = output_dir / filename

                if output_path.exists():
                    self.logger.debug(f"Skipping existing file: {output_path}")
                    downloaded += 1
                    current_date += timedelta(days=1)
                    continue

                success = await self._download_csv(url, output_path, session)
                if success:
                    downloaded += 1

                current_date += timedelta(days=1)

        self.logger.info(f"Downloaded {downloaded} NYISO load files")
        return downloaded

    async def get_available_locations(self) -> List[Dict[str, Any]]:
        """Get available NYISO locations."""
        # Major zones
        locations = [
            {"location_id": "CAPITL", "location_name": "Capital Zone", "location_type": "zone"},
            {"location_id": "CENTRL", "location_name": "Central Zone", "location_type": "zone"},
            {"location_id": "DUNWOD", "location_name": "Dunwoodie Zone", "location_type": "zone"},
            {"location_id": "GENESE", "location_name": "Genesee Zone", "location_type": "zone"},
            {"location_id": "HUD VL", "location_name": "Hudson Valley Zone", "location_type": "zone"},
            {"location_id": "LONGIL", "location_name": "Long Island Zone", "location_type": "zone"},
            {"location_id": "MHK VL", "location_name": "Mohawk Valley Zone", "location_type": "zone"},
            {"location_id": "MILLWD", "location_name": "Millwood Zone", "location_type": "zone"},
            {"location_id": "N.Y.C.", "location_name": "New York City Zone", "location_type": "zone"},
            {"location_id": "NORTH", "location_name": "North Zone", "location_type": "zone"},
            {"location_id": "WEST", "location_name": "West Zone", "location_type": "zone"},
        ]

        return locations

    def _infer_location_type(self, location_id: str) -> str:
        """Infer NYISO location type from ID."""
        # NYISO primarily uses zones
        return "zone"