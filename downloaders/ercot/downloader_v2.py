"""ERCOT downloader using standardized schema."""

import asyncio
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from database import AncillaryServices, LMP, Load, get_db
from processors.ercot import ERCOTProcessor

from ..base_v2 import BaseDownloaderV2, DownloadConfig
from .constants import WEBSERVICE_CUTOFF_DATE
from .selenium_client import ERCOTSeleniumClient
from .webservice_client import ERCOTWebServiceClient


class ERCOTDownloaderV2(BaseDownloaderV2):
    """ERCOT data downloader using standardized schema."""
    
    def __init__(self, config: DownloadConfig):
        super().__init__("ERCOT", config)
        
        # Initialize clients
        self.selenium_client = ERCOTSeleniumClient(
            download_dir=os.path.join(config.output_dir, "ercot", "raw"),
            username=os.getenv("ERCOT_USERNAME"),
            password=os.getenv("ERCOT_PASSWORD")
        )
        
        self.webservice_client = ERCOTWebServiceClient()
        self.processor = ERCOTProcessor()
    
    async def download_lmp(
        self,
        market: str,
        start_date: datetime,
        end_date: datetime,
        locations: Optional[List[str]] = None
    ) -> int:
        """Download LMP data for ERCOT."""
        total_records = 0
        
        # Map market codes
        market_mapping = {
            "DAM": "DAM",
            "RT5M": "RTM",
            "RTM": "RTM"
        }
        ercot_market = market_mapping.get(market, market)
        
        # Get missing intervals
        missing_intervals = self.get_missing_intervals(
            "lmp", market, start_date, end_date, 
            "hourly" if market == "DAM" else "5min"
        )
        
        for interval_start, interval_end in missing_intervals:
            try:
                if interval_end >= WEBSERVICE_CUTOFF_DATE:
                    # Use web service for recent data
                    records = await self._download_lmp_webservice(
                        ercot_market,
                        max(interval_start, WEBSERVICE_CUTOFF_DATE),
                        interval_end,
                        locations
                    )
                    total_records += records
                
                if interval_start < WEBSERVICE_CUTOFF_DATE:
                    # Use Selenium for historical data
                    records = await self._download_lmp_selenium(
                        ercot_market,
                        interval_start,
                        min(interval_end, WEBSERVICE_CUTOFF_DATE - timedelta(days=1)),
                        locations
                    )
                    total_records += records
                
                # Update dataset metadata
                dataset_name = f"ercot_lmp_{market.lower()}"
                self.update_dataset_metadata(dataset_name, interval_end, total_records)
                
            except Exception as e:
                self.logger.error(f"Error downloading LMP data", error=str(e))
                raise
        
        return total_records
    
    async def _download_lmp_webservice(
        self,
        market: str,
        start_date: datetime,
        end_date: datetime,
        locations: Optional[List[str]] = None
    ) -> int:
        """Download LMP data using web service."""
        if market == "DAM":
            df = await self.webservice_client.get_dam_spp_prices(
                start_date, end_date, locations
            )
        else:
            df = await self.webservice_client.get_rtm_spp_prices(
                start_date, end_date, locations
            )
        
        if df.empty:
            return 0
        
        # Process and store data
        return await self._store_lmp_data(df, market)
    
    async def _download_lmp_selenium(
        self,
        market: str,
        start_date: datetime,
        end_date: datetime,
        locations: Optional[List[str]] = None
    ) -> int:
        """Download LMP data using Selenium."""
        product_key = f"{market}_SPP"
        files = self.selenium_client.scrape_data_product(product_key, start_date, end_date)
        
        total_records = 0
        for file_path in files:
            try:
                full_path = Path(self.selenium_client.download_dir) / file_path
                df = self.processor.process_energy_file(full_path, market)
                
                if not df.empty:
                    records = await self._store_lmp_data(df, market)
                    total_records += records
                    
            except Exception as e:
                self.logger.error(f"Error processing file {file_path}", error=str(e))
        
        return total_records
    
    async def _store_lmp_data(self, df: pd.DataFrame, market: str) -> int:
        """Store LMP data in standardized format."""
        records_stored = 0
        
        with get_db() as db:
            # Prepare data for bulk insert
            records = []
            
            for _, row in df.iterrows():
                # Get or create location
                location = self.get_or_create_location(
                    row.get('settlement_point', row.get('node', row.get('location'))),
                    row.get('settlement_point_name'),
                    self._infer_location_type(row.get('settlement_point', ''))
                )
                
                # Calculate intervals
                timestamp = pd.to_datetime(row['timestamp'])
                if market == "DAM":
                    interval_start = timestamp
                    interval_end = timestamp + timedelta(hours=1)
                    market_code = "DAM"
                else:
                    interval_start = timestamp
                    interval_end = timestamp + timedelta(minutes=5)
                    market_code = "RT5M"
                
                # Create LMP record
                lmp_record = {
                    'interval_start': interval_start,
                    'interval_end': interval_end,
                    'iso': self.iso_code,
                    'location': location.location_id,
                    'location_type': location.location_type,
                    'market': market_code,
                    'lmp': row.get('lmp', row.get('spp')),
                    'energy': row.get('energy_component', row.get('mw')),
                    'congestion': row.get('congestion_component', row.get('mcc')),
                    'loss': row.get('loss_component', row.get('mlc'))
                }
                records.append(lmp_record)
            
            # Bulk insert
            if records:
                db.bulk_insert_mappings(LMP, records)
                db.commit()
                records_stored = len(records)
        
        return records_stored
    
    def _infer_location_type(self, location_id: str) -> str:
        """Infer ERCOT location type from ID."""
        if location_id.startswith("HB_"):
            return "hub"
        elif location_id.startswith("LZ_"):
            return "zone"
        else:
            return "node"
    
    async def download_ancillary_services(
        self,
        product: str,
        market: str,
        start_date: datetime,
        end_date: datetime
    ) -> int:
        """Download ancillary services data."""
        total_records = 0
        
        # Map product codes
        product_mapping = {
            "REGUP": "REGUP",
            "REGDOWN": "REGDN",
            "SPIN": "SPIN",
            "NON_SPIN": "NON_SPIN",
            "RRS": "RRS",
            "ECRS": "ECRS"
        }
        ercot_product = product_mapping.get(product, product)
        
        if end_date >= WEBSERVICE_CUTOFF_DATE:
            # Use web service for recent data
            df = await self.webservice_client.get_ancillary_prices(
                ercot_product,
                max(start_date, WEBSERVICE_CUTOFF_DATE),
                end_date
            )
            
            if not df.empty:
                records = await self._store_ancillary_data(df, product, market)
                total_records += records
        
        # Update dataset metadata
        dataset_name = f"ercot_as_{market.lower()}"
        self.update_dataset_metadata(dataset_name, end_date, total_records)
        
        return total_records
    
    async def _store_ancillary_data(self, df: pd.DataFrame, product: str, market: str) -> int:
        """Store ancillary services data."""
        records_stored = 0
        
        with get_db() as db:
            records = []
            
            for _, row in df.iterrows():
                timestamp = pd.to_datetime(row['timestamp'])
                if market == "DAM":
                    interval_start = timestamp
                    interval_end = timestamp + timedelta(hours=1)
                else:
                    interval_start = timestamp
                    interval_end = timestamp + timedelta(minutes=5)
                
                as_record = {
                    'interval_start': interval_start,
                    'interval_end': interval_end,
                    'iso': self.iso_code,
                    'region': row.get('zone', 'ERCOT'),
                    'market': market,
                    'product': product,
                    'clearing_price': row.get('price'),
                    'clearing_quantity': row.get('quantity_mw'),
                    'requirement': row.get('requirement')
                }
                records.append(as_record)
            
            if records:
                db.bulk_insert_mappings(AncillaryServices, records)
                db.commit()
                records_stored = len(records)
        
        return records_stored
    
    async def download_load(
        self,
        forecast_type: str,
        start_date: datetime,
        end_date: datetime
    ) -> int:
        """Download load data - placeholder for future implementation."""
        # ERCOT load data download would be implemented here
        self.logger.info("Load data download not yet implemented for ERCOT")
        return 0
    
    async def get_available_locations(self) -> List[Dict[str, Any]]:
        """Get available ERCOT locations."""
        from .constants import TRADING_HUBS, LOAD_ZONES
        
        locations = []
        
        # Trading hubs
        for hub in TRADING_HUBS:
            locations.append({
                "location_id": hub,
                "location_name": hub.replace("HB_", "").replace("_", " ").title() + " Hub",
                "location_type": "hub"
            })
        
        # Load zones
        for zone in LOAD_ZONES:
            locations.append({
                "location_id": zone,
                "location_name": zone.replace("LZ_", "").replace("_", " ").title() + " Zone",
                "location_type": "zone"
            })
        
        return locations