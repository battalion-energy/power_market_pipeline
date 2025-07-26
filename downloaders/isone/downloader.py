"""ISONE data downloader using Web Services API."""

import asyncio
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import httpx
import pandas as pd
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

from database import AncillaryPrice, EnergyPrice, Node, get_db
from processors.isone import ISONEProcessor

from ..base import BaseDownloader, DownloadConfig
from .constants import (
    API_TIMEOUT,
    ENDPOINTS,
    ISONE_API_BASE,
    ISONE_HUB,
    ISONE_ZONES,
    RESERVE_ZONES,
)

load_dotenv()


class ISONEDownloader(BaseDownloader):
    """ISONE data downloader using Web Services API."""
    
    def __init__(self, config: DownloadConfig):
        super().__init__("ISONE", config)
        self.processor = ISONEProcessor()
        
        # API credentials from environment
        self.username = os.getenv("ISONE_USERNAME")
        self.password = os.getenv("ISONE_PASSWORD")
        
        if not self.username or not self.password:
            raise ValueError("ISONE API credentials not found in environment variables")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def _make_request(self, endpoint: str) -> Dict:
        """Make authenticated request to ISONE API."""
        url = f"{ISONE_API_BASE}{endpoint}"
        
        async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
            response = await client.get(
                url,
                auth=(self.username, self.password)
            )
            
            if response.status_code == 404:
                return {}  # No data available
            
            response.raise_for_status()
            return response.json()
    
    async def download_energy_prices(
        self,
        market_type: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Download energy price data from ISONE."""
        all_data = []
        
        # Determine endpoint pattern
        if market_type == "DAM":
            endpoint_pattern = ENDPOINTS["DAM_LMP"]
        elif market_type == "RTM":
            endpoint_pattern = ENDPOINTS["RTM_LMP"]
        else:
            raise ValueError(f"Unknown market type: {market_type}")
        
        # Check for missing date ranges
        data_type = f"ENERGY_{market_type}"
        missing_ranges = self.get_missing_date_ranges(data_type, start_date, end_date)
        
        for range_start, range_end in missing_ranges:
            download_id = self.record_download_start(data_type, range_start, range_end)
            
            try:
                # Download locations to download: hub + zones
                locations = [ISONE_HUB] + [f".Z.LOAD_ZONE_{zone}" for zone in ISONE_ZONES]
                
                # Process each day
                current_date = range_start
                while current_date <= range_end:
                    date_str = current_date.strftime("%Y%m%d")
                    
                    for location in locations:
                        endpoint = endpoint_pattern.format(date=date_str, location=location)
                        
                        try:
                            data = await self._make_request(endpoint)
                            
                            if data:
                                # Process the data
                                df = self.processor.process_energy_data(data, market_type, location)
                                
                                if not df.empty:
                                    records = df.to_dict('records')
                                    all_data.extend(records)
                                    
                                    # Store in database
                                    await self._store_energy_data(records)
                                
                        except Exception as e:
                            self.logger.error(f"Error downloading {location} on {date_str}", 
                                            error=str(e))
                    
                    current_date += timedelta(days=1)
                
                self.record_download_complete(download_id, row_count=len(all_data))
                
            except Exception as e:
                self.logger.error(f"Download failed for {data_type}", error=str(e))
                self.record_download_complete(download_id, error_message=str(e))
                raise
        
        return all_data
    
    async def download_ancillary_prices(
        self,
        service_type: str,
        market_type: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Download ancillary service price data from ISONE."""
        all_data = []
        
        # Check for missing date ranges
        data_type = f"ANCILLARY_{service_type}_{market_type}"
        missing_ranges = self.get_missing_date_ranges(data_type, start_date, end_date)
        
        for range_start, range_end in missing_ranges:
            download_id = self.record_download_start(data_type, range_start, range_end)
            
            try:
                current_date = range_start
                while current_date <= range_end:
                    date_str = current_date.strftime("%Y%m%d")
                    
                    if service_type in ["TMSR", "TMNSR", "TMOR"]:
                        # Reserve prices by zone
                        for zone in RESERVE_ZONES:
                            endpoint = ENDPOINTS["RTM_AS"].format(date=date_str, zone=zone)
                            
                            try:
                                data = await self._make_request(endpoint)
                                
                                if data:
                                    df = self.processor.process_reserve_data(
                                        data, service_type, market_type, zone
                                    )
                                    
                                    if not df.empty:
                                        records = df.to_dict('records')
                                        all_data.extend(records)
                                        await self._store_ancillary_data(records)
                                
                            except Exception as e:
                                self.logger.error(f"Error downloading {service_type} for {zone}", 
                                                error=str(e))
                    
                    elif service_type in ["REG", "REG_CAPACITY", "REG_SERVICE"]:
                        # Frequency regulation
                        endpoint = ENDPOINTS["FREQ_REG"].format(date=date_str)
                        
                        try:
                            data = await self._make_request(endpoint)
                            
                            if data:
                                df = self.processor.process_regulation_data(data, market_type)
                                
                                if not df.empty:
                                    records = df.to_dict('records')
                                    all_data.extend(records)
                                    await self._store_ancillary_data(records)
                        
                        except Exception as e:
                            self.logger.error(f"Error downloading regulation data", error=str(e))
                    
                    current_date += timedelta(days=1)
                
                self.record_download_complete(download_id, row_count=len(all_data))
                
            except Exception as e:
                self.logger.error(f"Download failed for {data_type}", error=str(e))
                self.record_download_complete(download_id, error_message=str(e))
                raise
        
        return all_data
    
    async def _store_energy_data(self, data: List[Dict[str, Any]]):
        """Store energy price data in the database."""
        with get_db() as db:
            # Get or create nodes
            node_cache = {}
            
            for record in data:
                # Get node ID
                node_id_key = (self.iso_id, record['location'])
                if node_id_key not in node_cache:
                    node = db.query(Node).filter(
                        Node.iso_id == self.iso_id,
                        Node.node_id == record['location']
                    ).first()
                    
                    if not node:
                        node = Node(
                            iso_id=self.iso_id,
                            node_id=record['location'],
                            node_name=self._format_location_name(record['location']),
                            node_type=self._determine_node_type(record['location'])
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
                    lmp=record.get('lmp_total'),
                    energy_component=record.get('energy_component'),
                    congestion_component=record.get('congestion_component'),
                    loss_component=record.get('loss_component')
                )
                db.add(price)
            
            db.commit()
    
    async def _store_ancillary_data(self, data: List[Dict[str, Any]]):
        """Store ancillary price data in the database."""
        with get_db() as db:
            for record in data:
                price = AncillaryPrice(
                    timestamp=record['timestamp'],
                    iso_id=self.iso_id,
                    service_type=record['service_type'],
                    market_type=record['market_type'],
                    price=record.get('price'),
                    quantity_mw=record.get('quantity_mw'),
                    zone=record.get('zone')
                )
                db.add(price)
            
            db.commit()
    
    def _determine_node_type(self, location: str) -> str:
        """Determine node type from location ID."""
        if location == ISONE_HUB:
            return "HUB"
        elif location.startswith(".Z.LOAD_ZONE"):
            return "ZONE"
        elif location.startswith(".I."):
            return "INTERFACE"
        else:
            return "NODE"
    
    def _format_location_name(self, location: str) -> str:
        """Format location ID into readable name."""
        if location == ISONE_HUB:
            return "ISO-NE Hub"
        elif location.startswith(".Z.LOAD_ZONE_"):
            zone_id = location.replace(".Z.LOAD_ZONE_", "")
            zone_names = {
                "4001": "Maine",
                "4002": "New Hampshire",
                "4003": "Vermont",
                "4004": "Connecticut",
                "4005": "Rhode Island",
                "4006": "Southeast Massachusetts",
                "4007": "Western/Central Massachusetts",
                "4008": "Northeast Massachusetts/Boston"
            }
            return zone_names.get(zone_id, f"Zone {zone_id}")
        else:
            return location.replace(".", " ").replace("_", " ").title()
    
    async def get_available_nodes(self) -> List[Dict[str, Any]]:
        """Get list of available nodes/locations."""
        try:
            # Fetch current locations from API
            data = await self._make_request(ENDPOINTS["LOCATIONS"])
            
            if data and "Locations" in data:
                locations = data["Locations"]["Location"]
                nodes = []
                
                for loc in locations:
                    nodes.append({
                        "node_id": loc["LocationId"],
                        "node_name": loc.get("LocationName", loc["LocationId"]),
                        "node_type": loc.get("LocationType", "NODE")
                    })
                
                return nodes
        
        except Exception as e:
            self.logger.error("Failed to fetch locations", error=str(e))
        
        # Return default locations if API fails
        nodes = []
        
        # Hub
        nodes.append({
            "node_id": ISONE_HUB,
            "node_name": "ISO-NE Hub",
            "node_type": "HUB"
        })
        
        # Zones
        for zone_id in ISONE_ZONES:
            nodes.append({
                "node_id": f".Z.LOAD_ZONE_{zone_id}",
                "node_name": self._format_location_name(f".Z.LOAD_ZONE_{zone_id}"),
                "node_type": "ZONE"
            })
        
        return nodes