"""Main ERCOT downloader that combines Selenium and Web Service methods."""

import asyncio
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy.orm import Session

from database import EnergyPrice, Node, get_db
from processors.ercot import ERCOTProcessor

from ..base import BaseDownloader, DownloadConfig
from .constants import WEBSERVICE_CUTOFF_DATE
from .selenium_client import ERCOTSeleniumClient
from .webservice_client import ERCOTWebServiceClient


class ERCOTDownloader(BaseDownloader):
    """ERCOT data downloader using both Selenium and Web Service methods."""
    
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
    
    async def download_energy_prices(
        self,
        market_type: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Download energy price data using appropriate method based on date."""
        all_data = []
        
        # Check for missing date ranges
        data_type = f"ENERGY_{market_type}"
        missing_ranges = self.get_missing_date_ranges(data_type, start_date, end_date)
        
        for range_start, range_end in missing_ranges:
            # Record download start
            download_id = self.record_download_start(data_type, range_start, range_end)
            
            try:
                if range_end >= WEBSERVICE_CUTOFF_DATE:
                    # Split range if it spans the cutoff date
                    if range_start < WEBSERVICE_CUTOFF_DATE:
                        # Historical part
                        historical_data = await self._download_historical_energy(
                            market_type, range_start, WEBSERVICE_CUTOFF_DATE - timedelta(days=1)
                        )
                        all_data.extend(historical_data)
                        
                        # Recent part
                        recent_data = await self._download_recent_energy(
                            market_type, WEBSERVICE_CUTOFF_DATE, range_end
                        )
                        all_data.extend(recent_data)
                    else:
                        # All recent data
                        recent_data = await self._download_recent_energy(
                            market_type, range_start, range_end
                        )
                        all_data.extend(recent_data)
                else:
                    # All historical data
                    historical_data = await self._download_historical_energy(
                        market_type, range_start, range_end
                    )
                    all_data.extend(historical_data)
                
                # Record success
                self.record_download_complete(
                    download_id,
                    row_count=len(all_data)
                )
                
            except Exception as e:
                self.logger.error(f"Download failed for {data_type}", error=str(e))
                self.record_download_complete(download_id, error_message=str(e))
                raise
        
        return all_data
    
    async def _download_historical_energy(
        self,
        market_type: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Download historical energy data using Selenium."""
        product_key = f"{market_type}_SPP"
        files = self.selenium_client.scrape_data_product(product_key, start_date, end_date)
        
        all_data = []
        for file_path in files:
            try:
                # Process the downloaded file
                full_path = Path(self.selenium_client.download_dir) / file_path
                df = self.processor.process_energy_file(full_path, market_type)
                
                # Convert to list of dicts
                data = df.to_dict('records')
                all_data.extend(data)
                
                # Store in database
                await self._store_energy_data(data)
                
            except Exception as e:
                self.logger.error(f"Error processing file {file_path}", error=str(e))
        
        return all_data
    
    async def _download_recent_energy(
        self,
        market_type: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Download recent energy data using Web Service API."""
        if market_type == "DAM":
            df = await self.webservice_client.get_dam_spp_prices(start_date, end_date)
        elif market_type == "RTM":
            df = await self.webservice_client.get_rtm_spp_prices(start_date, end_date)
        else:
            raise ValueError(f"Unknown market type: {market_type}")
        
        if df.empty:
            return []
        
        # Process and standardize data
        df = self.processor.standardize_energy_data(df, market_type)
        data = df.to_dict('records')
        
        # Store in database
        await self._store_energy_data(data)
        
        return data
    
    async def _store_energy_data(self, data: List[Dict[str, Any]]):
        """Store energy price data in the database."""
        with get_db() as db:
            # Get or create nodes
            node_cache = {}
            
            for record in data:
                # Get node ID
                node_id_key = (self.iso_id, record['settlement_point'])
                if node_id_key not in node_cache:
                    node = db.query(Node).filter(
                        Node.iso_id == self.iso_id,
                        Node.node_id == record['settlement_point']
                    ).first()
                    
                    if not node:
                        # Create new node
                        node = Node(
                            iso_id=self.iso_id,
                            node_id=record['settlement_point'],
                            node_name=record.get('settlement_point_name', record['settlement_point']),
                            node_type=self._determine_node_type(record['settlement_point'])
                        )
                        db.add(node)
                        db.flush()
                    
                    node_cache[node_id_key] = node.id
                
                # Create energy price record
                price = EnergyPrice(
                    timestamp=record['timestamp'],
                    iso_id=self.iso_id,
                    node_id=node_cache[node_id_key],
                    market_type=record['market_type'],
                    lmp=record.get('lmp'),
                    energy_component=record.get('energy_component'),
                    congestion_component=record.get('congestion_component'),
                    loss_component=record.get('loss_component')
                )
                db.add(price)
            
            db.commit()
    
    def _determine_node_type(self, node_id: str) -> str:
        """Determine node type based on ID pattern."""
        if node_id.startswith("HB_"):
            return "HUB"
        elif node_id.startswith("LZ_"):
            return "ZONE"
        else:
            return "NODE"
    
    async def download_ancillary_prices(
        self,
        service_type: str,
        market_type: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Download ancillary service price data."""
        all_data = []
        
        # Check for missing date ranges
        data_type = f"ANCILLARY_{service_type}_{market_type}"
        missing_ranges = self.get_missing_date_ranges(data_type, start_date, end_date)
        
        for range_start, range_end in missing_ranges:
            download_id = self.record_download_start(data_type, range_start, range_end)
            
            try:
                if range_end >= WEBSERVICE_CUTOFF_DATE:
                    # Use web service for recent data
                    df = await self.webservice_client.get_ancillary_prices(
                        service_type,
                        max(range_start, WEBSERVICE_CUTOFF_DATE),
                        range_end
                    )
                    
                    if not df.empty:
                        data = df.to_dict('records')
                        all_data.extend(data)
                        await self._store_ancillary_data(data)
                
                if range_start < WEBSERVICE_CUTOFF_DATE:
                    # Use Selenium for historical data
                    # Implementation would follow similar pattern to energy prices
                    self.logger.info("Historical ancillary data download not yet implemented")
                
                self.record_download_complete(download_id, row_count=len(all_data))
                
            except Exception as e:
                self.logger.error(f"Download failed for {data_type}", error=str(e))
                self.record_download_complete(download_id, error_message=str(e))
                raise
        
        return all_data
    
    async def _store_ancillary_data(self, data: List[Dict[str, Any]]):
        """Store ancillary price data in the database."""
        # Implementation similar to _store_energy_data
        pass
    
    async def get_available_nodes(self) -> List[Dict[str, Any]]:
        """Get list of available nodes/settlement points."""
        # For ERCOT, we can return known hubs and zones
        # In a production system, this would query the actual available nodes
        from .constants import TRADING_HUBS, LOAD_ZONES
        
        nodes = []
        
        # Trading hubs
        for hub in TRADING_HUBS:
            nodes.append({
                "node_id": hub,
                "node_name": hub.replace("HB_", "").replace("_", " ").title() + " Hub",
                "node_type": "HUB"
            })
        
        # Load zones
        for zone in LOAD_ZONES:
            nodes.append({
                "node_id": zone,
                "node_name": zone.replace("LZ_", "").replace("_", " ").title() + " Zone",
                "node_type": "ZONE"
            })
        
        return nodes