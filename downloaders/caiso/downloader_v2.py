"""CAISO downloader using standardized schema."""

import asyncio
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

from database import AncillaryServices, LMP, Load, get_db
from downloaders.base_v2 import BaseDownloaderV2, DownloadConfig


class CAISODownloaderV2(BaseDownloaderV2):
    """CAISO data downloader using standardized schema."""
    
    def __init__(self, config: DownloadConfig):
        super().__init__("CAISO", config)
        
        # CAISO uses OASIS API
        self.base_url = "http://oasis.caiso.com/oasisapi/SingleZip"
        self.username = os.getenv("CAISO_USERNAME")
        self.password = os.getenv("CAISO_PASSWORD")
        
    async def download_lmp(
        self,
        market: str,
        start_date: datetime,
        end_date: datetime,
        locations: Optional[List[str]] = None
    ) -> int:
        """Download LMP data for CAISO."""
        self.logger.info(
            f"CAISO LMP download not yet implemented",
            market=market,
            start=start_date,
            end=end_date
        )
        
        # TODO: Implement CAISO OASIS API calls
        # Market mapping:
        # - DAM -> DAM
        # - RT5M -> RTM
        # - HASP -> HASP (Hour Ahead Scheduling Process)
        
        return 0
    
    async def download_ancillary_services(
        self,
        product: str,
        market: str,
        start_date: datetime,
        end_date: datetime
    ) -> int:
        """Download ancillary services data."""
        self.logger.info(
            f"CAISO AS download not yet implemented",
            product=product,
            market=market
        )
        
        # TODO: Implement AS downloads
        # Products: SPIN, NON_SPIN, REG_UP, REG_DOWN
        
        return 0
    
    async def download_load(
        self,
        forecast_type: str,
        start_date: datetime,
        end_date: datetime
    ) -> int:
        """Download load data."""
        self.logger.info(f"CAISO load download not yet implemented")
        
        # TODO: Implement load downloads
        # Types: actual, forecast_dam, forecast_2da, forecast_7da
        
        return 0
    
    async def get_available_locations(self) -> List[Dict[str, Any]]:
        """Get available CAISO locations."""
        # Major trading hubs and zones
        locations = [
            {"location_id": "TH_NP15_GEN-APND", "location_name": "NP15 Trading Hub", "location_type": "hub"},
            {"location_id": "TH_SP15_GEN-APND", "location_name": "SP15 Trading Hub", "location_type": "hub"},
            {"location_id": "TH_ZP26_GEN-APND", "location_name": "ZP26 Trading Hub", "location_type": "hub"},
            {"location_id": "PGAE-APND", "location_name": "PG&E Aggregate", "location_type": "zone"},
            {"location_id": "SCE-APND", "location_name": "SCE Aggregate", "location_type": "zone"},
            {"location_id": "SDGE-APND", "location_name": "SDG&E Aggregate", "location_type": "zone"},
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