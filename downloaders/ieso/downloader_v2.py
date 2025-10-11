"""IESO (Ontario) downloader using standardized schema.

IMPORTANT: IESO transitioned from HOEP to LMP pricing on May 1, 2025.
- Pre-May 2025: HOEP (Hourly Ontario Energy Price) - single price for province
- Post-May 2025: LMP (Locational Marginal Pricing) - ~1000 nodes
- Post-May 2025: OEMP (Ontario Energy Market Price) - replaces HOEP
"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
import pandas as pd

from database import AncillaryServices, LMP, Load, get_db
from downloaders.base_v2 import BaseDownloaderV2, DownloadConfig


class IESODownloaderV2(BaseDownloaderV2):
    """IESO data downloader using standardized schema.

    Downloads public data from IESO REST API endpoints:
    - Day-Ahead LMP (post-May 2025)
    - Real-Time LMP (post-May 2025)
    - Ontario Zonal Prices
    - OEMP (Ontario Energy Market Price)
    - HOEP (legacy, pre-May 2025)
    - Operating Reserve prices (10S, 10NS, 30OR)
    """

    # Market transition date
    LMP_TRANSITION_DATE = datetime(2025, 5, 1)

    def __init__(self, config: DownloadConfig):
        super().__init__("IESO", config)

        # IESO provides public CSV files via REST API
        self.base_url = "https://reports-public.ieso.ca/public"

        # Create output directories
        self.csv_dir = Path(config.output_dir) / "IESO_data" / "csv_files"
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
        """Download LMP data for IESO.

        Args:
            market: 'DAM' (day-ahead) or 'RT5M' (real-time)
            start_date: Start date for download
            end_date: End date for download
            locations: Not used - IESO provides all nodes in single files

        Note:
            - LMP data only available from May 1, 2025 onwards
            - Pre-May 2025: Use HOEP (download via download_legacy_hoep)
            - ~1000 LMP nodes in Ontario
        """
        self.logger.info(
            f"Downloading IESO {market} LMP data",
            start=start_date,
            end=end_date
        )

        # Check if date range is valid for LMP
        if end_date < self.LMP_TRANSITION_DATE:
            self.logger.warning(
                f"LMP data not available before {self.LMP_TRANSITION_DATE}. "
                "Use download_legacy_hoep() for historical data."
            )
            return 0

        # Adjust start date if needed
        effective_start = max(start_date, self.LMP_TRANSITION_DATE)

        # Report code mapping
        if market == "DAM":
            report_code = "PUB_DALMPEnergy"
            subdir = "da_lmp"
        elif market == "RT5M":
            report_code = "PUB_RTLMPEnergy"
            subdir = "rt_lmp"
        else:
            raise ValueError(f"Unknown market: {market}. Use 'DAM' or 'RT5M'")

        downloaded = 0
        async with aiohttp.ClientSession() as session:
            current_date = effective_start
            while current_date <= end_date:
                date_str = current_date.strftime("%Y%m%d")

                # URL format: https://reports-public.ieso.ca/public/PUB_DALMPEnergy_20250501.csv
                filename = f"{report_code}_{date_str}.csv"
                url = f"{self.base_url}/{filename}"

                output_dir = self.csv_dir / subdir
                output_path = output_dir / filename

                # Skip if already downloaded
                if output_path.exists():
                    self.logger.debug(f"Skipping existing file: {output_path}")
                    downloaded += 1
                    current_date += timedelta(days=1)
                    continue

                success = await self._download_csv(url, output_path, session)
                if success:
                    downloaded += 1

                current_date += timedelta(days=1)

        self.logger.info(f"Downloaded {downloaded} IESO {market} LMP files")
        return downloaded

    async def download_ontario_zonal_prices(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> int:
        """Download Ontario Zonal Prices.

        Ontario has 10 pricing zones. This data is available post-May 2025.

        Args:
            start_date: Start date for download
            end_date: End date for download
        """
        self.logger.info(
            f"Downloading IESO Ontario Zonal Prices",
            start=start_date,
            end=end_date
        )

        # Check if date range is valid
        if end_date < self.LMP_TRANSITION_DATE:
            self.logger.warning(
                f"Ontario Zonal Prices not available before {self.LMP_TRANSITION_DATE}"
            )
            return 0

        effective_start = max(start_date, self.LMP_TRANSITION_DATE)
        report_code = "PUB_OntarioZonalPrice"

        downloaded = 0
        async with aiohttp.ClientSession() as session:
            current_date = effective_start
            while current_date <= end_date:
                date_str = current_date.strftime("%Y%m%d")

                filename = f"{report_code}_{date_str}.csv"
                url = f"{self.base_url}/{filename}"

                output_dir = self.csv_dir / "zonal_prices"
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

        self.logger.info(f"Downloaded {downloaded} IESO Zonal Price files")
        return downloaded

    async def download_oemp(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> int:
        """Download OEMP (Ontario Energy Market Price).

        OEMP replaces HOEP starting May 1, 2025.
        This is the reference price for Ontario's wholesale market.

        Args:
            start_date: Start date for download
            end_date: End date for download
        """
        self.logger.info(
            f"Downloading IESO OEMP",
            start=start_date,
            end=end_date
        )

        if end_date < self.LMP_TRANSITION_DATE:
            self.logger.warning(
                f"OEMP not available before {self.LMP_TRANSITION_DATE}. "
                "Use download_legacy_hoep() for historical data."
            )
            return 0

        effective_start = max(start_date, self.LMP_TRANSITION_DATE)
        # Note: Report code TBD - IESO documentation doesn't specify exact code yet
        # Using likely naming convention
        report_code = "PUB_OEMP"

        downloaded = 0
        async with aiohttp.ClientSession() as session:
            current_date = effective_start
            while current_date <= end_date:
                date_str = current_date.strftime("%Y%m%d")

                filename = f"{report_code}_{date_str}.csv"
                url = f"{self.base_url}/{filename}"

                output_dir = self.csv_dir / "oemp"
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

        self.logger.info(f"Downloaded {downloaded} IESO OEMP files")
        return downloaded

    async def download_legacy_hoep(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> int:
        """Download legacy HOEP (Hourly Ontario Energy Price).

        HOEP was the single wholesale price for Ontario until April 30, 2025.

        Args:
            start_date: Start date for download
            end_date: End date for download (max: 2025-04-30)
        """
        self.logger.info(
            f"Downloading IESO legacy HOEP",
            start=start_date,
            end=end_date
        )

        if start_date >= self.LMP_TRANSITION_DATE:
            self.logger.warning(
                f"HOEP not available from {self.LMP_TRANSITION_DATE} onwards. "
                "Use download_oemp() for new data."
            )
            return 0

        # Cap end date at transition
        effective_end = min(end_date, self.LMP_TRANSITION_DATE - timedelta(days=1))

        # Note: Report code TBD - need to confirm actual IESO report code
        report_code = "PUB_HOEP"

        downloaded = 0
        async with aiohttp.ClientSession() as session:
            current_date = start_date
            while current_date <= effective_end:
                date_str = current_date.strftime("%Y%m%d")

                filename = f"{report_code}_{date_str}.csv"
                url = f"{self.base_url}/{filename}"

                output_dir = self.csv_dir / "hoep_legacy"
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

        self.logger.info(f"Downloaded {downloaded} IESO HOEP files")
        return downloaded

    async def download_ancillary_services(
        self,
        product: str,
        market: str,
        start_date: datetime,
        end_date: datetime
    ) -> int:
        """Download ancillary services data.

        IESO operates three operating reserve markets:
        - 10S: 10-minute Synchronized Reserve
        - 10NS: 10-minute Non-Synchronized Reserve
        - 30OR: 30-minute Operating Reserve

        Other ancillary services (Regulation, Black Start, etc.) are contracted
        and not publicly available via market pricing.

        Args:
            product: '10S', '10NS', '30OR', or 'ALL' for all three
            market: 'DAM' or 'RTM' (IESO may only have RT pricing)
            start_date: Start date for download
            end_date: End date for download
        """
        self.logger.info(
            f"Downloading IESO {product} ancillary services",
            start=start_date,
            end=end_date
        )

        # Determine which products to download
        if product == "ALL":
            products = ["10S", "10NS", "30OR"]
        else:
            products = [product]

        # Validate products
        valid_products = {"10S", "10NS", "30OR"}
        for prod in products:
            if prod not in valid_products:
                raise ValueError(
                    f"Unknown product: {prod}. Valid: {valid_products}"
                )

        downloaded = 0
        async with aiohttp.ClientSession() as session:
            for prod in products:
                # Note: Report codes TBD - need to confirm actual IESO codes
                # Using likely naming convention
                report_code = f"PUB_OR_{prod}"

                current_date = start_date
                while current_date <= end_date:
                    date_str = current_date.strftime("%Y%m%d")

                    filename = f"{report_code}_{date_str}.csv"
                    url = f"{self.base_url}/{filename}"

                    output_dir = self.csv_dir / "ancillary_services" / prod.lower()
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

                # Reset for next product
                current_date = start_date

        self.logger.info(f"Downloaded {downloaded} IESO AS files")
        return downloaded

    async def download_load(
        self,
        forecast_type: str,
        start_date: datetime,
        end_date: datetime
    ) -> int:
        """Download load data.

        Args:
            forecast_type: 'actual' or 'forecast'
            start_date: Start date for download
            end_date: End date for download
        """
        self.logger.info(
            f"Downloading IESO {forecast_type} load data",
            start=start_date,
            end=end_date
        )

        # Note: Report codes TBD - need to confirm actual IESO codes
        if forecast_type == "actual":
            report_code = "PUB_Load_Actual"
        elif forecast_type == "forecast":
            report_code = "PUB_Load_Forecast"
        else:
            raise ValueError(f"Unknown forecast type: {forecast_type}")

        downloaded = 0
        async with aiohttp.ClientSession() as session:
            current_date = start_date
            while current_date <= end_date:
                date_str = current_date.strftime("%Y%m%d")

                filename = f"{report_code}_{date_str}.csv"
                url = f"{self.base_url}/{filename}"

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

        self.logger.info(f"Downloaded {downloaded} IESO load files")
        return downloaded

    async def get_available_locations(self) -> List[Dict[str, Any]]:
        """Get available IESO locations.

        Returns:
            List of location dictionaries with keys:
            - location_id: Unique identifier
            - location_name: Human-readable name
            - location_type: 'zone' or 'node'

        Note:
            - Pre-May 2025: Single province-wide HOEP price
            - Post-May 2025: 10 zones + ~1000 LMP nodes
            - Actual node list must be extracted from LMP CSV files
        """
        # Ontario zones (10 zones post-May 2025)
        # Note: Exact zone names TBD - need to verify from actual data files
        locations = [
            {"location_id": "ONTARIO", "location_name": "Ontario (HOEP)", "location_type": "zone"},
            {"location_id": "ZONE_1", "location_name": "Ontario Zone 1", "location_type": "zone"},
            {"location_id": "ZONE_2", "location_name": "Ontario Zone 2", "location_type": "zone"},
            {"location_id": "ZONE_3", "location_name": "Ontario Zone 3", "location_type": "zone"},
            {"location_id": "ZONE_4", "location_name": "Ontario Zone 4", "location_type": "zone"},
            {"location_id": "ZONE_5", "location_name": "Ontario Zone 5", "location_type": "zone"},
            {"location_id": "ZONE_6", "location_name": "Ontario Zone 6", "location_type": "zone"},
            {"location_id": "ZONE_7", "location_name": "Ontario Zone 7", "location_type": "zone"},
            {"location_id": "ZONE_8", "location_name": "Ontario Zone 8", "location_type": "zone"},
            {"location_id": "ZONE_9", "location_name": "Ontario Zone 9", "location_type": "zone"},
            {"location_id": "ZONE_10", "location_name": "Ontario Zone 10", "location_type": "zone"},
        ]

        return locations

    def _infer_location_type(self, location_id: str) -> str:
        """Infer IESO location type from ID.

        Args:
            location_id: Location identifier

        Returns:
            'zone' for zones, 'node' for LMP nodes
        """
        # ONTARIO and ZONE_* are zones
        if location_id.startswith("ZONE") or location_id == "ONTARIO":
            return "zone"
        # Everything else is a node
        return "node"

    async def download_all_markets(
        self,
        start_date: datetime,
        end_date: datetime,
        include_legacy: bool = True
    ) -> Dict[str, int]:
        """Convenience method to download all IESO markets.

        Args:
            start_date: Start date for download
            end_date: End date for download
            include_legacy: Whether to download legacy HOEP data

        Returns:
            Dictionary with counts for each dataset
        """
        results = {}

        # Split date range at transition
        if start_date < self.LMP_TRANSITION_DATE and include_legacy:
            legacy_end = min(end_date, self.LMP_TRANSITION_DATE - timedelta(days=1))
            results["hoep"] = await self.download_legacy_hoep(start_date, legacy_end)

        # Post-transition data
        if end_date >= self.LMP_TRANSITION_DATE:
            lmp_start = max(start_date, self.LMP_TRANSITION_DATE)

            results["da_lmp"] = await self.download_lmp("DAM", lmp_start, end_date)
            results["rt_lmp"] = await self.download_lmp("RT5M", lmp_start, end_date)
            results["zonal"] = await self.download_ontario_zonal_prices(lmp_start, end_date)
            results["oemp"] = await self.download_oemp(lmp_start, end_date)

        # Ancillary services (all periods)
        results["as_10s"] = await self.download_ancillary_services("10S", "RTM", start_date, end_date)
        results["as_10ns"] = await self.download_ancillary_services("10NS", "RTM", start_date, end_date)
        results["as_30or"] = await self.download_ancillary_services("30OR", "RTM", start_date, end_date)

        # Load data
        results["load_actual"] = await self.download_load("actual", start_date, end_date)
        results["load_forecast"] = await self.download_load("forecast", start_date, end_date)

        self.logger.info(f"Download complete", results=results)
        return results
