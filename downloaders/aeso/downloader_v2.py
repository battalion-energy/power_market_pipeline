"""AESO downloader using standardized schema.

Alberta Electric System Operator (AESO) operates a single-price market (pool price).
Unlike other ISOs, AESO does not have nodal/zonal LMP pricing.

Data is available through file downloads from http://ets.aeso.ca/
"""

import asyncio
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from io import BytesIO

import aiohttp
import pandas as pd
from bs4 import BeautifulSoup

from database import AncillaryServices, LMP, Load, get_db
from downloaders.base_v2 import BaseDownloaderV2, DownloadConfig


class AESODownloaderV2(BaseDownloaderV2):
    """AESO data downloader using standardized schema.

    AESO provides:
    - Hourly pool price (single price for entire Alberta market)
    - Generation data
    - Operating reserve prices
    - Load data

    Note: AESO is a single-price market, not nodal/zonal like most other ISOs.
    """

    def __init__(self, config: DownloadConfig):
        super().__init__("AESO", config)

        # AESO base URLs
        self.base_url = "http://ets.aeso.ca"
        self.reports_url = f"{self.base_url}/ets_web/ip/Market/Reports"

        # Create output directories
        self.csv_dir = Path(config.output_dir) / "AESO_data" / "csv_files"
        self.csv_dir.mkdir(parents=True, exist_ok=True)

        # Multi-year file mappings (based on historical data availability)
        self.historical_files = {
            "2001-2009": "PoolPrice2001-2009.csv",  # Example naming
            "2010-2019": "PoolPrice2010-2019.csv",
            "2020-2025": "PoolPrice2020-current.csv"
        }

    async def _download_file(
        self,
        url: str,
        output_path: Path,
        session: aiohttp.ClientSession
    ) -> bool:
        """Download a single file with retry logic."""
        for attempt in range(self.config.retry_attempts):
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=600)) as response:
                    if response.status == 200:
                        content = await response.read()
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        output_path.write_bytes(content)
                        self.logger.info(f"Downloaded {url} -> {output_path}")
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

    async def _scrape_available_reports(
        self,
        session: aiohttp.ClientSession,
        report_type: str = "pool_price"
    ) -> List[Dict[str, str]]:
        """Scrape AESO website to find available downloadable reports.

        Args:
            session: aiohttp session
            report_type: Type of report to look for

        Returns:
            List of dicts with 'url', 'filename', 'date_range' keys
        """
        available_files = []

        try:
            # Try common data download pages
            potential_urls = [
                f"{self.base_url}/ets_web/ip/Market/Reports/DataDownloadServlet",
                f"{self.base_url}/ets_web/ip/Market/Reports/HistoricalDataDownloadServlet",
                f"{self.reports_url}/DailyAveragePoolPriceReportServlet",
            ]

            for page_url in potential_urls:
                try:
                    async with session.get(page_url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                        if response.status == 200:
                            html = await response.text()
                            soup = BeautifulSoup(html, 'html.parser')

                            # Look for links to CSV/Excel files
                            for link in soup.find_all('a', href=True):
                                href = link['href']
                                if any(ext in href.lower() for ext in ['.csv', '.xls', '.xlsx', '.zip']):
                                    # Construct full URL
                                    if href.startswith('http'):
                                        file_url = href
                                    else:
                                        file_url = f"{self.base_url}{href}" if href.startswith('/') else f"{page_url}/{href}"

                                    filename = href.split('/')[-1]
                                    available_files.append({
                                        'url': file_url,
                                        'filename': filename,
                                        'source_page': page_url
                                    })
                                    self.logger.debug(f"Found file: {filename} at {file_url}")
                except Exception as e:
                    self.logger.debug(f"Could not access {page_url}: {e}")

        except Exception as e:
            self.logger.warning(f"Error scraping AESO reports: {e}")

        return available_files

    async def download_lmp(
        self,
        market: str,
        start_date: datetime,
        end_date: datetime,
        locations: Optional[List[str]] = None
    ) -> int:
        """Download pool price data for AESO.

        AESO operates a single-price market (pool price), not nodal/zonal LMP.
        The 'market' parameter is included for compatibility but AESO only has
        one market type.

        Args:
            market: 'DAM' or 'POOL' (AESO uses pool pricing)
            start_date: Start date for download
            end_date: End date for download
            locations: Not used for AESO (single price for all of Alberta)
        """
        self.logger.info(
            f"Downloading AESO pool price data",
            start=start_date,
            end=end_date
        )

        downloaded = 0
        async with aiohttp.ClientSession() as session:
            # Strategy 1: Try to download multi-year historical files
            output_dir = self.csv_dir / "pool_price"
            output_dir.mkdir(parents=True, exist_ok=True)

            # First, try to scrape and find available files
            available_reports = await self._scrape_available_reports(session, "pool_price")

            if available_reports:
                self.logger.info(f"Found {len(available_reports)} available reports on AESO website")
                for report in available_reports:
                    output_path = output_dir / report['filename']

                    if output_path.exists():
                        self.logger.debug(f"Skipping existing file: {output_path}")
                        downloaded += 1
                        continue

                    success = await self._download_file(report['url'], output_path, session)
                    if success:
                        downloaded += 1

            # Strategy 2: Try known historical file URLs (if scraping didn't work)
            if downloaded == 0:
                self.logger.info("Trying known historical file patterns")

                # Common patterns for AESO data files
                base_data_url = f"{self.base_url}/ets_web/ip/Market/Reports"
                potential_files = [
                    f"{base_data_url}/PoolPriceReportServlet?contentType=csv",
                    f"{base_data_url}/DailyAveragePoolPriceReportServlet?contentType=csv",
                ]

                for url in potential_files:
                    filename = f"aeso_pool_price_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
                    output_path = output_dir / filename

                    if output_path.exists():
                        self.logger.debug(f"Skipping existing file: {output_path}")
                        downloaded += 1
                        continue

                    success = await self._download_file(url, output_path, session)
                    if success:
                        downloaded += 1

            # Strategy 3: Try daily download approach
            if downloaded == 0:
                self.logger.info("Attempting daily download approach")
                downloaded = await self._download_daily_pool_prices(
                    session, start_date, end_date, output_dir
                )

        self.logger.info(f"Downloaded {downloaded} AESO pool price files")
        return downloaded

    async def _download_daily_pool_prices(
        self,
        session: aiohttp.ClientSession,
        start_date: datetime,
        end_date: datetime,
        output_dir: Path
    ) -> int:
        """Download daily pool price reports.

        Args:
            session: aiohttp session
            start_date: Start date
            end_date: End date
            output_dir: Output directory

        Returns:
            Number of files downloaded
        """
        downloaded = 0
        current_date = start_date

        while current_date <= end_date:
            date_str = current_date.strftime("%Y%m%d")

            # Try various date formats AESO might use
            potential_urls = [
                f"{self.reports_url}/DailyAveragePoolPriceReportServlet?contentType=csv&date={date_str}",
                f"{self.reports_url}/DailyAveragePoolPriceReportServlet?beginDate={date_str}&endDate={date_str}",
                f"{self.base_url}/ets_web/ip/Market/Reports/DailyPoolPrice_{date_str}.csv",
            ]

            success = False
            for url in potential_urls:
                filename = f"daily_pool_price_{date_str}.csv"
                output_path = output_dir / filename

                if output_path.exists():
                    self.logger.debug(f"Skipping existing file: {output_path}")
                    downloaded += 1
                    success = True
                    break

                if await self._download_file(url, output_path, session):
                    downloaded += 1
                    success = True
                    break

            if not success:
                self.logger.debug(f"No data available for {date_str}")

            current_date += timedelta(days=1)

        return downloaded

    async def download_ancillary_services(
        self,
        product: str,
        market: str,
        start_date: datetime,
        end_date: datetime
    ) -> int:
        """Download ancillary services data.

        AESO provides operating reserve prices and other ancillary services.

        Args:
            product: 'OPERATING_RESERVE', 'REGULATING_RESERVE', etc.
            market: Not really applicable for AESO (included for compatibility)
        """
        self.logger.info(
            f"Downloading AESO ancillary services data",
            product=product,
            start=start_date,
            end=end_date
        )

        downloaded = 0
        async with aiohttp.ClientSession() as session:
            output_dir = self.csv_dir / "ancillary_services" / product.lower()
            output_dir.mkdir(parents=True, exist_ok=True)

            # Scrape for ancillary services files
            available_reports = await self._scrape_available_reports(session, "ancillary_services")

            for report in available_reports:
                # Filter for relevant product type
                if product.lower() in report['filename'].lower() or product == 'ALL':
                    output_path = output_dir / report['filename']

                    if output_path.exists():
                        self.logger.debug(f"Skipping existing file: {output_path}")
                        downloaded += 1
                        continue

                    success = await self._download_file(report['url'], output_path, session)
                    if success:
                        downloaded += 1

            # Try specific ancillary service URLs
            if downloaded == 0:
                as_urls = [
                    f"{self.reports_url}/OperatingReserveReportServlet?contentType=csv",
                    f"{self.reports_url}/RegulatingReserveReportServlet?contentType=csv",
                ]

                for url in as_urls:
                    filename = f"aeso_as_{product.lower()}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
                    output_path = output_dir / filename

                    if output_path.exists():
                        downloaded += 1
                        continue

                    success = await self._download_file(url, output_path, session)
                    if success:
                        downloaded += 1

        self.logger.info(f"Downloaded {downloaded} AESO ancillary services files")
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
        """
        self.logger.info(
            f"Downloading AESO {forecast_type} load data",
            start=start_date,
            end=end_date
        )

        downloaded = 0
        async with aiohttp.ClientSession() as session:
            output_dir = self.csv_dir / "load" / forecast_type
            output_dir.mkdir(parents=True, exist_ok=True)

            # Try load report URLs
            load_urls = [
                f"{self.reports_url}/ActualForecastWMRReportServlet?contentType=csv",
                f"{self.reports_url}/SystemMarginalPriceReportServlet?contentType=csv",
            ]

            for url in load_urls:
                filename = f"aeso_load_{forecast_type}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
                output_path = output_dir / filename

                if output_path.exists():
                    self.logger.debug(f"Skipping existing file: {output_path}")
                    downloaded += 1
                    continue

                success = await self._download_file(url, output_path, session)
                if success:
                    downloaded += 1

            # Try scraping for load data files
            available_reports = await self._scrape_available_reports(session, "load")
            for report in available_reports:
                if 'load' in report['filename'].lower() or 'demand' in report['filename'].lower():
                    output_path = output_dir / report['filename']

                    if output_path.exists():
                        downloaded += 1
                        continue

                    success = await self._download_file(report['url'], output_path, session)
                    if success:
                        downloaded += 1

        self.logger.info(f"Downloaded {downloaded} AESO load files")
        return downloaded

    async def download_generation(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> int:
        """Download generation data.

        AESO provides historical generation data by fuel type.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            Number of files downloaded
        """
        self.logger.info(
            f"Downloading AESO generation data",
            start=start_date,
            end=end_date
        )

        downloaded = 0
        async with aiohttp.ClientSession() as session:
            output_dir = self.csv_dir / "generation"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Scrape for generation files
            available_reports = await self._scrape_available_reports(session, "generation")

            for report in available_reports:
                if 'gen' in report['filename'].lower() or 'supply' in report['filename'].lower():
                    output_path = output_dir / report['filename']

                    if output_path.exists():
                        self.logger.debug(f"Skipping existing file: {output_path}")
                        downloaded += 1
                        continue

                    success = await self._download_file(report['url'], output_path, session)
                    if success:
                        downloaded += 1

            # Try specific generation URLs
            gen_urls = [
                f"{self.reports_url}/CSDReportServlet?contentType=csv",
            ]

            for url in gen_urls:
                filename = f"aeso_generation_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
                output_path = output_dir / filename

                if output_path.exists():
                    downloaded += 1
                    continue

                success = await self._download_file(url, output_path, session)
                if success:
                    downloaded += 1

        self.logger.info(f"Downloaded {downloaded} AESO generation files")
        return downloaded

    async def get_available_locations(self) -> List[Dict[str, Any]]:
        """Get available AESO locations.

        AESO operates a single-price market for all of Alberta.
        """
        locations = [
            {
                "location_id": "ALBERTA",
                "location_name": "Alberta Pool Price",
                "location_type": "system"
            }
        ]

        return locations

    def _infer_location_type(self, location_id: str) -> str:
        """Infer AESO location type from ID.

        AESO has a single-price system market.
        """
        return "system"

    async def download_all_data_types(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, int]:
        """Download all available data types for AESO.

        Convenience method to download all data in one call.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            Dictionary with counts of files downloaded per data type
        """
        results = {}

        # Pool prices (equivalent to LMP for other ISOs)
        results['pool_price'] = await self.download_lmp(
            market='POOL',
            start_date=start_date,
            end_date=end_date
        )

        # Ancillary services
        results['ancillary_services'] = await self.download_ancillary_services(
            product='ALL',
            market='POOL',
            start_date=start_date,
            end_date=end_date
        )

        # Load data
        results['load_actual'] = await self.download_load(
            forecast_type='actual',
            start_date=start_date,
            end_date=end_date
        )

        results['load_forecast'] = await self.download_load(
            forecast_type='forecast',
            start_date=start_date,
            end_date=end_date
        )

        # Generation data
        results['generation'] = await self.download_generation(
            start_date=start_date,
            end_date=end_date
        )

        self.logger.info("AESO download summary", **results)
        return results
