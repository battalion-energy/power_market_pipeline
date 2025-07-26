"""Comprehensive data fetcher service for all ISO data types."""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import structlog

from database import get_db
from downloaders.caiso.downloader_v2 import CAISODownloaderV2
from downloaders.ercot.downloader_v2 import ERCOTDownloaderV2
from downloaders.isone.downloader_v2 import ISONEDownloaderV2
from downloaders.nyiso.downloader_v2 import NYISODownloaderV2
from downloaders.base_v2 import DownloadConfig


class DataFetcher:
    """Service to coordinate downloading all data types from ISOs."""
    
    def __init__(self, config: DownloadConfig):
        self.config = config
        self.logger = structlog.get_logger()
        
        # Initialize downloaders
        self.downloaders = {
            "ERCOT": ERCOTDownloaderV2(config),
            "CAISO": CAISODownloaderV2(config),
            "ISONE": ISONEDownloaderV2(config),
            "NYISO": NYISODownloaderV2(config),
        }
    
    async def fetch_all_data(
        self,
        isos: List[str],
        start_date: datetime,
        end_date: datetime,
        data_types: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, int]]:
        """Fetch all requested data types for specified ISOs.
        
        Returns dict of {iso: {data_type: record_count}}
        """
        if not data_types:
            data_types = self._get_all_data_types()
        
        results = {}
        
        for iso in isos:
            if iso not in self.downloaders:
                self.logger.warning(f"No downloader available for {iso}")
                continue
            
            iso_results = {}
            downloader = self.downloaders[iso]
            
            # Energy prices (LMP)
            if "lmp" in data_types:
                self.logger.info(f"Fetching LMP data for {iso}")
                
                # Day-ahead
                count = await downloader.download_lmp("DAM", start_date, end_date)
                iso_results["lmp_dam"] = count
                
                # Real-time
                count = await downloader.download_lmp("RT5M", start_date, end_date)
                iso_results["lmp_rt5m"] = count
            
            # Ancillary services
            if "ancillary" in data_types:
                self.logger.info(f"Fetching ancillary services data for {iso}")
                
                as_products = self._get_ancillary_products(iso)
                for product in as_products:
                    count = await downloader.download_ancillary_services(
                        product, "DAM", start_date, end_date
                    )
                    iso_results[f"as_{product.lower()}_dam"] = count
            
            # Load and forecast
            if "load" in data_types:
                self.logger.info(f"Fetching load data for {iso}")
                
                # Actual load
                count = await downloader.download_load("actual", start_date, end_date)
                iso_results["load_actual"] = count
                
                # Forecasts
                for forecast_type in ["forecast_1h", "forecast_dam"]:
                    count = await downloader.download_load(forecast_type, start_date, end_date)
                    iso_results[f"load_{forecast_type}"] = count
            
            # Generation by fuel
            if "generation" in data_types:
                self.logger.info(f"Fetching generation data for {iso}")
                count = await self._download_generation_fuel(downloader, start_date, end_date)
                iso_results["generation_fuel"] = count
            
            # Transmission constraints
            if "constraints" in data_types:
                self.logger.info(f"Fetching transmission constraints for {iso}")
                count = await self._download_constraints(downloader, start_date, end_date)
                iso_results["constraints"] = count
            
            # Weather
            if "weather" in data_types:
                self.logger.info(f"Fetching weather data for {iso}")
                count = await self._download_weather(downloader, start_date, end_date)
                iso_results["weather"] = count
            
            # Renewable forecasts
            if "renewable_forecast" in data_types:
                self.logger.info(f"Fetching renewable forecasts for {iso}")
                count = await self._download_renewable_forecasts(downloader, start_date, end_date)
                iso_results["renewable_forecast"] = count
            
            # Storage operations
            if "storage" in data_types:
                self.logger.info(f"Fetching storage data for {iso}")
                count = await self._download_storage_operations(downloader, start_date, end_date)
                iso_results["storage"] = count
            
            # Emissions
            if "emissions" in data_types:
                self.logger.info(f"Fetching emissions data for {iso}")
                count = await self._download_emissions(downloader, start_date, end_date)
                iso_results["emissions"] = count
            
            results[iso] = iso_results
        
        return results
    
    def _get_all_data_types(self) -> List[str]:
        """Get list of all supported data types."""
        return [
            "lmp",
            "ancillary",
            "load",
            "generation",
            "constraints",
            "weather",
            "renewable_forecast",
            "storage",
            "emissions",
            "curtailment",
            "capacity",
            "demand_response"
        ]
    
    def _get_ancillary_products(self, iso: str) -> List[str]:
        """Get ancillary service products for each ISO."""
        products_by_iso = {
            "ERCOT": ["REGUP", "REGDOWN", "SPIN", "NON_SPIN", "RRS", "ECRS"],
            "CAISO": ["SPIN", "NON_SPIN", "REG_UP", "REG_DOWN"],
            "ISONE": ["TMSR", "TMNSR", "TMOR", "REG_CAPACITY", "REG_SERVICE"],
            "NYISO": ["SPIN_10", "NON_SYNC_10", "OPER_30", "REG_CAPACITY", "REG_MOVEMENT"]
        }
        return products_by_iso.get(iso, [])
    
    async def _download_generation_fuel(self, downloader, start_date: datetime, end_date: datetime) -> int:
        """Download generation by fuel type data."""
        # This would be implemented per ISO
        # For now, placeholder
        return 0
    
    async def _download_constraints(self, downloader, start_date: datetime, end_date: datetime) -> int:
        """Download transmission constraints data."""
        # This would be implemented per ISO
        return 0
    
    async def _download_weather(self, downloader, start_date: datetime, end_date: datetime) -> int:
        """Download weather data."""
        # This would be implemented using weather APIs
        return 0
    
    async def _download_renewable_forecasts(self, downloader, start_date: datetime, end_date: datetime) -> int:
        """Download renewable generation forecasts."""
        # This would be implemented per ISO
        return 0
    
    async def _download_storage_operations(self, downloader, start_date: datetime, end_date: datetime) -> int:
        """Download storage operations data."""
        # This would be implemented per ISO
        return 0
    
    async def _download_emissions(self, downloader, start_date: datetime, end_date: datetime) -> int:
        """Download emissions data."""
        # This would be implemented per ISO
        return 0
    
    async def run_historical_backfill(
        self,
        isos: List[str],
        start_date: datetime = datetime(2019, 1, 1),
        end_date: datetime = None
    ):
        """Run historical data backfill."""
        if not end_date:
            end_date = datetime.now()
        
        self.logger.info(
            "Starting historical backfill",
            isos=isos,
            start_date=start_date.date(),
            end_date=end_date.date()
        )
        
        # Process in monthly chunks to avoid overwhelming the system
        current_date = start_date
        while current_date < end_date:
            chunk_end = min(
                current_date + timedelta(days=30),
                end_date
            )
            
            self.logger.info(
                "Processing chunk",
                start=current_date.date(),
                end=chunk_end.date()
            )
            
            results = await self.fetch_all_data(
                isos=isos,
                start_date=current_date,
                end_date=chunk_end,
                data_types=["lmp", "ancillary", "load"]  # Start with core data types
            )
            
            # Log results
            for iso, iso_results in results.items():
                total_records = sum(iso_results.values())
                self.logger.info(
                    f"{iso} chunk complete",
                    records=total_records,
                    details=iso_results
                )
            
            current_date = chunk_end
            
            # Add delay between chunks
            await asyncio.sleep(5)
    
    async def run_real_time_updates(self, isos: List[str]):
        """Run real-time data updates."""
        while True:
            try:
                # Get data for the last hour
                end_date = datetime.now()
                start_date = end_date - timedelta(hours=1)
                
                results = await self.fetch_all_data(
                    isos=isos,
                    start_date=start_date,
                    end_date=end_date,
                    data_types=["lmp", "load", "generation"]  # Real-time priorities
                )
                
                self.logger.info("Real-time update complete", results=results)
                
                # Wait 5 minutes before next update
                await asyncio.sleep(300)
                
            except Exception as e:
                self.logger.error("Real-time update failed", error=str(e))
                await asyncio.sleep(60)  # Wait 1 minute on error