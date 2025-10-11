"""CAISO downloader using standardized schema."""

import asyncio
import io
import os
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
import pandas as pd

from database import AncillaryServices, LMP, Load, get_db
from downloaders.base_v2 import BaseDownloaderV2, DownloadConfig


class CAISODownloaderV2(BaseDownloaderV2):
    """CAISO data downloader using OASIS API."""

    def __init__(self, config: DownloadConfig):
        super().__init__("CAISO", config)

        # CAISO OASIS API
        self.base_url = "http://oasis.caiso.com/oasisapi/SingleZip"
        self.username = os.getenv("CAISO_USERNAME")
        self.password = os.getenv("CAISO_PASSWORD")

        if not self.username or not self.password:
            raise ValueError("CAISO_USERNAME and CAISO_PASSWORD must be set in environment")

        # Create output directories
        self.csv_dir = Path(config.output_dir) / "CAISO_data" / "csv_files"
        self.csv_dir.mkdir(parents=True, exist_ok=True)

        # Query names and versions
        self.query_config = {
            "PRC_LMP": {"version": "12", "name": "PRC_LMP"},  # LMP prices
            "PRC_AS": {"version": "1", "name": "PRC_AS"},     # Ancillary services
        }

        # Market mappings
        self.market_mapping = {
            "DAM": "DAM",   # Day-ahead market
            "RT5M": "RTM",  # Real-time market (15-min in CAISO, but called RT5M for consistency)
            "RT15M": "RTM", # Real-time market
            "HASP": "HASP", # Hour-ahead scheduling process
        }

        # AS product mappings
        self.as_product_mapping = {
            "REG_UP": "RU",
            "REG_DOWN": "RD",
            "SPIN": "SR",
            "NON_SPIN": "NR",
        }

    async def _download_oasis_data(
        self,
        query_name: str,
        version: str,
        market_run_id: str,
        node: str,
        start_date: datetime,
        end_date: datetime,
        output_path: Path,
        session: aiohttp.ClientSession,
        additional_params: Optional[Dict[str, str]] = None
    ) -> bool:
        """Download data from CAISO OASIS API with retry logic."""

        # Format dates for OASIS API (YYYYMMDDTHH:MM-0000)
        start_str = start_date.strftime("%Y%m%dT00:00-0000")
        end_str = end_date.strftime("%Y%m%dT23:59-0000")

        # Build query parameters
        params = {
            "resultformat": "6",  # CSV in ZIP format
            "queryname": query_name,
            "version": version,
            "market_run_id": market_run_id,
            "node": node,
            "startdatetime": start_str,
            "enddatetime": end_str,
        }

        # Add any additional parameters (e.g., anc_type for ancillary services)
        if additional_params:
            params.update(additional_params)

        for attempt in range(self.config.retry_attempts):
            try:
                auth = aiohttp.BasicAuth(self.username, self.password)
                async with session.get(
                    self.base_url,
                    params=params,
                    auth=auth,
                    timeout=aiohttp.ClientTimeout(total=300)
                ) as response:
                    if response.status == 200:
                        content = await response.read()

                        # OASIS returns ZIP files - extract CSV
                        try:
                            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                                # Extract all CSV files from ZIP
                                output_path.parent.mkdir(parents=True, exist_ok=True)

                                for filename in zf.namelist():
                                    if filename.endswith('.csv'):
                                        csv_data = zf.read(filename)
                                        # Save with the original filename from the ZIP
                                        csv_path = output_path.parent / filename
                                        csv_path.write_bytes(csv_data)
                                        self.logger.debug(f"Extracted {filename} -> {csv_path}")

                                return True
                        except zipfile.BadZipFile:
                            self.logger.warning(f"Invalid ZIP file received from OASIS")
                            return False

                    elif response.status == 404:
                        self.logger.warning(f"Data not found (404) for {query_name} {market_run_id} {node}")
                        return False
                    else:
                        self.logger.warning(f"HTTP {response.status} from OASIS API")

            except asyncio.TimeoutError:
                self.logger.warning(f"Timeout on attempt {attempt + 1} for {query_name}")
            except Exception as e:
                self.logger.warning(f"Download attempt {attempt + 1} failed: {e}")

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
        """Download LMP data from CAISO OASIS API.

        Args:
            market: 'DAM', 'RT5M', 'RT15M', or 'HASP'
            start_date: Start date for download
            end_date: End date for download
            locations: List of node IDs (if None, downloads major hubs/zones)
        """
        self.logger.info(
            f"Downloading CAISO {market} LMP data",
            start=start_date,
            end=end_date
        )

        # Map to OASIS market_run_id
        if market not in self.market_mapping:
            raise ValueError(f"Unknown market: {market}. Use DAM, RT5M, RT15M, or HASP")

        market_run_id = self.market_mapping[market]

        # Default to major trading hubs and zones if no locations specified
        if not locations:
            locations = [
                "TH_NP15_GEN-APND",  # NP15 hub
                "TH_SP15_GEN-APND",  # SP15 hub
                "TH_ZP26_GEN-APND",  # ZP26 hub
                "PGAE-APND",         # PG&E zone
                "SCE-APND",          # SCE zone
                "SDGE-APND",         # SDG&E zone
            ]

        query_info = self.query_config["PRC_LMP"]
        downloaded = 0

        async with aiohttp.ClientSession() as session:
            # Download by chunks (OASIS has rate limits)
            chunks = self.chunk_date_range(start_date, end_date, days_per_chunk=7)

            for chunk_start, chunk_end in chunks:
                for node in locations:
                    # Create output filename
                    date_str = chunk_start.strftime("%Y%m%d")
                    node_safe = node.replace("/", "_").replace(" ", "_")
                    filename = f"PRC_LMP_{market_run_id}_{node_safe}_{date_str}.csv"

                    output_dir = self.csv_dir / market.lower() / "lmp"
                    output_path = output_dir / filename

                    # Skip if already downloaded
                    if output_path.exists():
                        self.logger.debug(f"Skipping existing file: {output_path}")
                        downloaded += 1
                        continue

                    success = await self._download_oasis_data(
                        query_name=query_info["name"],
                        version=query_info["version"],
                        market_run_id=market_run_id,
                        node=node,
                        start_date=chunk_start,
                        end_date=chunk_end,
                        output_path=output_path,
                        session=session
                    )

                    if success:
                        downloaded += 1

                    # Rate limiting - OASIS recommends ~10 requests/minute
                    await asyncio.sleep(6)

        self.logger.info(f"Downloaded {downloaded} CAISO {market} LMP files")
        return downloaded

    async def download_ancillary_services(
        self,
        product: str,
        market: str,
        start_date: datetime,
        end_date: datetime
    ) -> int:
        """Download ancillary services data from CAISO OASIS API.

        Args:
            product: 'REG_UP', 'REG_DOWN', 'SPIN', 'NON_SPIN', or 'ALL'
            market: 'DAM' or 'RTM'
        """
        self.logger.info(
            f"Downloading CAISO {market} ancillary services",
            product=product,
            start=start_date,
            end=end_date
        )

        # Map to OASIS market_run_id
        if market not in ["DAM", "RTM"]:
            raise ValueError(f"Unknown market: {market}. Use DAM or RTM")

        market_run_id = market

        # Determine which products to download
        if product == "ALL":
            products = ["REG_UP", "REG_DOWN", "SPIN", "NON_SPIN"]
        else:
            products = [product]

        query_info = self.query_config["PRC_AS"]
        downloaded = 0

        async with aiohttp.ClientSession() as session:
            chunks = self.chunk_date_range(start_date, end_date, days_per_chunk=7)

            for chunk_start, chunk_end in chunks:
                for prod in products:
                    # Map to OASIS ancillary service type
                    oasis_anc_type = self.as_product_mapping.get(prod)
                    if not oasis_anc_type:
                        self.logger.warning(f"Unknown AS product: {prod}")
                        continue

                    # Create output filename
                    date_str = chunk_start.strftime("%Y%m%d")
                    filename = f"PRC_AS_{market_run_id}_{prod}_{date_str}.csv"

                    output_dir = self.csv_dir / "ancillary_services" / market.lower()
                    output_path = output_dir / filename

                    if output_path.exists():
                        self.logger.debug(f"Skipping existing file: {output_path}")
                        downloaded += 1
                        continue

                    # Additional parameters for AS query
                    additional_params = {
                        "anc_type": oasis_anc_type,
                        "anc_region": "AS_CAISO",  # System-wide region
                    }

                    success = await self._download_oasis_data(
                        query_name=query_info["name"],
                        version=query_info["version"],
                        market_run_id=market_run_id,
                        node="ALL",  # AS prices are system-wide
                        start_date=chunk_start,
                        end_date=chunk_end,
                        output_path=output_path,
                        session=session,
                        additional_params=additional_params
                    )

                    if success:
                        downloaded += 1

                    # Rate limiting
                    await asyncio.sleep(6)

        self.logger.info(f"Downloaded {downloaded} CAISO AS files")
        return downloaded

    async def download_load(
        self,
        forecast_type: str,
        start_date: datetime,
        end_date: datetime
    ) -> int:
        """Download load data from CAISO OASIS API.

        Args:
            forecast_type: 'forecast' for system load forecast
        """
        self.logger.info(
            f"Downloading CAISO {forecast_type} load data",
            start=start_date,
            end=end_date
        )

        # CAISO load data uses SLD_FCST query
        query_name = "SLD_FCST"
        version = "1"

        downloaded = 0

        async with aiohttp.ClientSession() as session:
            chunks = self.chunk_date_range(start_date, end_date, days_per_chunk=7)

            for chunk_start, chunk_end in chunks:
                date_str = chunk_start.strftime("%Y%m%d")
                filename = f"SLD_FCST_{date_str}.csv"

                output_dir = self.csv_dir / "load" / forecast_type
                output_path = output_dir / filename

                if output_path.exists():
                    self.logger.debug(f"Skipping existing file: {output_path}")
                    downloaded += 1
                    continue

                success = await self._download_oasis_data(
                    query_name=query_name,
                    version=version,
                    market_run_id="DAM",  # Day-ahead forecast
                    node="ALL",
                    start_date=chunk_start,
                    end_date=chunk_end,
                    output_path=output_path,
                    session=session
                )

                if success:
                    downloaded += 1

                # Rate limiting
                await asyncio.sleep(6)

        self.logger.info(f"Downloaded {downloaded} CAISO load files")
        return downloaded

    async def get_available_locations(self) -> List[Dict[str, Any]]:
        """Get available CAISO locations (trading hubs and zones)."""
        locations = [
            {"location_id": "TH_NP15_GEN-APND", "location_name": "NP15 Trading Hub", "location_type": "hub"},
            {"location_id": "TH_SP15_GEN-APND", "location_name": "SP15 Trading Hub", "location_type": "hub"},
            {"location_id": "TH_ZP26_GEN-APND", "location_name": "ZP26 Trading Hub", "location_type": "hub"},
            {"location_id": "PGAE-APND", "location_name": "PG&E Aggregate", "location_type": "zone"},
            {"location_id": "SCE-APND", "location_name": "SCE Aggregate", "location_type": "zone"},
            {"location_id": "SDGE-APND", "location_name": "SDG&E Aggregate", "location_type": "zone"},
            {"location_id": "VEA-APND", "location_name": "Valley Electric Association", "location_type": "zone"},
        ]

        return locations

    def _infer_location_type(self, location_id: str) -> str:
        """Infer CAISO location type from ID."""
        if location_id.startswith("TH_"):
            return "hub"
        elif location_id.endswith("-APND"):
            return "zone"
        else:
            return "node"