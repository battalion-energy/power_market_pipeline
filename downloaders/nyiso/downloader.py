"""NYISO data downloader."""

import asyncio
import io
import os
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from database import AncillaryPrice, EnergyPrice, Node, get_db
from processors.nyiso import NYISOProcessor

from ..base import BaseDownloader, DownloadConfig
from .constants import (
    COLUMN_MAPPINGS,
    DATA_TYPE_MAP,
    FILE_PATTERNS,
    NYISO_DATA_URL,
    NYISO_REF_BUS,
    NYISO_ZONES,
)


class NYISODownloader(BaseDownloader):
    """NYISO data downloader."""
    
    def __init__(self, config: DownloadConfig):
        super().__init__("NYISO", config)
        self.processor = NYISOProcessor()
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def _download_file(self, url: str) -> bytes:
        """Download a file from NYISO."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.content
    
    async def _extract_csv_from_zip(self, zip_content: bytes) -> pd.DataFrame:
        """Extract CSV data from a ZIP file."""
        all_data = []
        
        with zipfile.ZipFile(io.BytesIO(zip_content)) as zf:
            for filename in zf.namelist():
                if filename.endswith('.csv'):
                    with zf.open(filename) as f:
                        df = pd.read_csv(f)
                        all_data.append(df)
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        
        return pd.DataFrame()
    
    async def download_energy_prices(
        self,
        market_type: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Download energy price data from NYISO."""
        all_data = []
        
        # Check for missing date ranges
        data_type = f"ENERGY_{market_type}"
        missing_ranges = self.get_missing_date_ranges(data_type, start_date, end_date)
        
        for range_start, range_end in missing_ranges:
            download_id = self.record_download_start(data_type, range_start, range_end)
            
            try:
                # Download both zone and generator data
                for node_type in ["zone", "gen"]:
                    # Determine file pattern
                    if market_type == "DAM":
                        pattern_key = f"DAM_LMP_{node_type.upper()}"
                    else:
                        pattern_key = f"RTM_LMP_{node_type.upper()}"
                    
                    # Process monthly
                    current_date = range_start
                    while current_date <= range_end:
                        month_str = current_date.strftime("%Y%m")
                        filename = FILE_PATTERNS[pattern_key].format(date=month_str+"01")
                        
                        # Determine URL folder
                        if market_type == "DAM":
                            folder = "damlbmp"
                        else:
                            folder = "realtime"
                        
                        url = NYISO_DATA_URL.format(type=folder, filename=filename)
                        
                        try:
                            self.logger.info(f"Downloading {market_type} {node_type} data", 
                                           month=month_str)
                            
                            # Download and extract
                            zip_content = await self._download_file(url)
                            df = await self._extract_csv_from_zip(zip_content)
                            
                            if not df.empty:
                                # Process data
                                processed_df = self.processor.process_energy_data(
                                    df, market_type, node_type
                                )
                                
                                # Filter for zones only if node_type is zone
                                if node_type == "zone":
                                    processed_df = processed_df[
                                        processed_df['location'].isin(NYISO_ZONES + [NYISO_REF_BUS])
                                    ]
                                
                                records = processed_df.to_dict('records')
                                all_data.extend(records)
                                
                                # Store in database
                                await self._store_energy_data(records)
                        
                        except Exception as e:
                            self.logger.error(f"Error downloading {month_str}", error=str(e))
                        
                        # Move to next month
                        if current_date.month == 12:
                            current_date = current_date.replace(
                                year=current_date.year + 1, month=1
                            )
                        else:
                            current_date = current_date.replace(month=current_date.month + 1)
                
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
        """Download ancillary service price data from NYISO."""
        all_data = []
        
        # Determine data type based on service
        if service_type in ["REG", "REG_CAPACITY", "REG_MOVEMENT"]:
            file_type = "ASP"
        else:
            file_type = "RESERVE"
        
        # Check for missing date ranges
        data_type = f"ANCILLARY_{service_type}_{market_type}"
        missing_ranges = self.get_missing_date_ranges(data_type, start_date, end_date)
        
        for range_start, range_end in missing_ranges:
            download_id = self.record_download_start(data_type, range_start, range_end)
            
            try:
                # Determine file pattern
                pattern_key = f"{market_type}_{file_type}"
                
                # Process monthly
                current_date = range_start
                while current_date <= range_end:
                    month_str = current_date.strftime("%Y%m")
                    filename = FILE_PATTERNS[pattern_key].format(date=month_str+"01")
                    
                    # Determine URL folder
                    folder = pattern_key.lower().replace("_", "")
                    url = NYISO_DATA_URL.format(type=folder, filename=filename)
                    
                    try:
                        self.logger.info(f"Downloading {service_type} data", 
                                       market=market_type, month=month_str)
                        
                        # Download and extract
                        zip_content = await self._download_file(url)
                        df = await self._extract_csv_from_zip(zip_content)
                        
                        if not df.empty:
                            # Process data
                            if file_type == "ASP":
                                processed_df = self.processor.process_regulation_data(
                                    df, market_type
                                )
                            else:
                                processed_df = self.processor.process_reserve_data(
                                    df, service_type, market_type
                                )
                            
                            # Filter for specific service if needed
                            if 'service_type' in processed_df.columns:
                                processed_df = processed_df[
                                    processed_df['service_type'] == service_type
                                ]
                            
                            records = processed_df.to_dict('records')
                            all_data.extend(records)
                            
                            # Store in database
                            await self._store_ancillary_data(records)
                    
                    except Exception as e:
                        self.logger.error(f"Error downloading {month_str}", error=str(e))
                    
                    # Move to next month
                    if current_date.month == 12:
                        current_date = current_date.replace(
                            year=current_date.year + 1, month=1
                        )
                    else:
                        current_date = current_date.replace(month=current_date.month + 1)
                
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
                            node_name=record.get('location_name', record['location']),
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
                    lmp=record.get('lmp'),
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
                    zone=record.get('zone', 'NYISO')
                )
                db.add(price)
            
            db.commit()
    
    def _determine_node_type(self, location: str) -> str:
        """Determine node type from location."""
        if location == NYISO_REF_BUS:
            return "HUB"
        elif location in NYISO_ZONES:
            return "ZONE"
        else:
            return "GEN"  # Generator node
    
    async def get_available_nodes(self) -> List[Dict[str, Any]]:
        """Get list of available nodes."""
        nodes = []
        
        # Reference bus
        nodes.append({
            "node_id": NYISO_REF_BUS,
            "node_name": "NYISO Reference Bus",
            "node_type": "HUB"
        })
        
        # Zones
        for zone in NYISO_ZONES:
            nodes.append({
                "node_id": zone,
                "node_name": zone.title() + " Zone",
                "node_type": "ZONE"
            })
        
        # Note: Generator nodes are numerous and would be retrieved from actual data
        
        return nodes