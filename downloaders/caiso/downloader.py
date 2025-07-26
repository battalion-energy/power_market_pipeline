"""CAISO data downloader using OASIS API."""

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

from database import EnergyPrice, Node, get_db
from processors.caiso import CAISOProcessor

from ..base import BaseDownloader, DownloadConfig
from .constants import (
    DLAPS,
    OASIS_BASE_URL,
    OASIS_ENDPOINTS,
    OASIS_TIMEOUT,
    TRADING_HUBS,
)


class CAISODownloader(BaseDownloader):
    """CAISO data downloader using OASIS API."""
    
    def __init__(self, config: DownloadConfig):
        super().__init__("CAISO", config)
        self.processor = CAISOProcessor()
        self.nodes_file = Path(__file__).parent / "nodes_caiso.csv"
        self._load_nodes()
    
    def _load_nodes(self):
        """Load CAISO nodes from CSV file."""
        if self.nodes_file.exists():
            self.nodes_df = pd.read_csv(self.nodes_file)
        else:
            # Create a basic nodes file with major points
            nodes_data = []
            for hub in TRADING_HUBS:
                nodes_data.append({"Node ID": hub, "Node Type": "TH", "Node Name": hub})
            for dlap in DLAPS:
                nodes_data.append({"Node ID": dlap, "Node Type": "DLAP", "Node Name": dlap})
            
            self.nodes_df = pd.DataFrame(nodes_data)
            # Save for future use
            self.nodes_df.to_csv(self.nodes_file, index=False)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def _download_oasis_data(
        self,
        query_name: str,
        start_date: datetime,
        end_date: datetime,
        node: Optional[str] = None,
        market_run: str = "DAM"
    ) -> pd.DataFrame:
        """Download data from CAISO OASIS API."""
        params = {
            "queryname": query_name,
            "startdatetime": start_date.strftime("%Y%m%dT%H:%M-0000"),
            "enddatetime": end_date.strftime("%Y%m%dT%H:%M-0000"),
            "market_run_id": market_run,
            "version": 1
        }
        
        if node:
            params["node"] = node
        
        async with httpx.AsyncClient(timeout=OASIS_TIMEOUT) as client:
            response = await client.get(OASIS_BASE_URL, params=params)
            response.raise_for_status()
            
            # OASIS returns zip files
            zip_content = io.BytesIO(response.content)
            
            all_data = []
            with zipfile.ZipFile(zip_content) as zf:
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
        """Download energy price data from CAISO."""
        all_data = []
        
        # Get query name for market type
        if market_type not in OASIS_ENDPOINTS["LMP"]:
            raise ValueError(f"Unknown market type: {market_type}")
        
        query_name = OASIS_ENDPOINTS["LMP"][market_type]
        
        # Check for missing date ranges
        data_type = f"ENERGY_{market_type}"
        missing_ranges = self.get_missing_date_ranges(data_type, start_date, end_date)
        
        for range_start, range_end in missing_ranges:
            download_id = self.record_download_start(data_type, range_start, range_end)
            
            try:
                # Download in monthly chunks
                chunks = self.chunk_date_range(range_start, range_end, days_per_chunk=30)
                
                for chunk_start, chunk_end in chunks:
                    # Download major nodes (hubs and DLAPs)
                    nodes_to_download = TRADING_HUBS + DLAPS
                    
                    for node in nodes_to_download:
                        self.logger.info(f"Downloading {market_type} prices for {node}", 
                                       start=chunk_start.date(), end=chunk_end.date())
                        
                        try:
                            df = await self._download_oasis_data(
                                query_name, chunk_start, chunk_end, node, market_type
                            )
                            
                            if not df.empty:
                                # Process and standardize data
                                processed_df = self.processor.process_energy_data(df, market_type)
                                data = processed_df.to_dict('records')
                                all_data.extend(data)
                                
                                # Store in database
                                await self._store_energy_data(data)
                                
                        except Exception as e:
                            self.logger.error(f"Error downloading {node}", error=str(e))
                
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
        """Download ancillary service price data from CAISO."""
        all_data = []
        
        # Get query name
        if market_type not in OASIS_ENDPOINTS["AS_PRICES"]:
            raise ValueError(f"Unknown market type for AS: {market_type}")
        
        query_name = OASIS_ENDPOINTS["AS_PRICES"][market_type]
        
        # Check for missing date ranges
        data_type = f"ANCILLARY_{service_type}_{market_type}"
        missing_ranges = self.get_missing_date_ranges(data_type, start_date, end_date)
        
        for range_start, range_end in missing_ranges:
            download_id = self.record_download_start(data_type, range_start, range_end)
            
            try:
                # Download in monthly chunks
                chunks = self.chunk_date_range(range_start, range_end, days_per_chunk=30)
                
                for chunk_start, chunk_end in chunks:
                    self.logger.info(f"Downloading {service_type} prices", 
                                   market=market_type, start=chunk_start.date(), end=chunk_end.date())
                    
                    try:
                        df = await self._download_oasis_data(
                            query_name, chunk_start, chunk_end, market_run=market_type
                        )
                        
                        if not df.empty:
                            # Process and filter for specific service
                            processed_df = self.processor.process_ancillary_data(
                                df, service_type, market_type
                            )
                            data = processed_df.to_dict('records')
                            all_data.extend(data)
                            
                            # Store in database
                            await self._store_ancillary_data(data)
                            
                    except Exception as e:
                        self.logger.error(f"Error downloading AS prices", error=str(e))
                
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
                node_id_key = (self.iso_id, record['node'])
                if node_id_key not in node_cache:
                    node = db.query(Node).filter(
                        Node.iso_id == self.iso_id,
                        Node.node_id == record['node']
                    ).first()
                    
                    if not node:
                        # Determine node type
                        node_info = self.nodes_df[self.nodes_df['Node ID'] == record['node']]
                        if not node_info.empty:
                            node_type = node_info.iloc[0]['Node Type']
                            node_name = node_info.iloc[0].get('Node Name', record['node'])
                        else:
                            node_type = self._determine_node_type(record['node'])
                            node_name = record['node']
                        
                        node = Node(
                            iso_id=self.iso_id,
                            node_id=record['node'],
                            node_name=node_name,
                            node_type=node_type
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
                    energy_component=record.get('mw'),
                    congestion_component=record.get('mcc'),
                    loss_component=record.get('mlc')
                )
                db.add(price)
            
            db.commit()
    
    def _determine_node_type(self, node_id: str) -> str:
        """Determine node type based on ID pattern."""
        if node_id.startswith("TH_"):
            return "TH"
        elif node_id.startswith("DLAP_"):
            return "DLAP"
        elif "_ASP" in node_id:
            return "ASP"
        else:
            return "PNODE"
    
    async def _store_ancillary_data(self, data: List[Dict[str, Any]]):
        """Store ancillary price data in the database."""
        # Implementation similar to _store_energy_data
        pass
    
    async def get_available_nodes(self) -> List[Dict[str, Any]]:
        """Get list of available nodes."""
        nodes = []
        
        for _, row in self.nodes_df.iterrows():
            nodes.append({
                "node_id": row["Node ID"],
                "node_name": row.get("Node Name", row["Node ID"]),
                "node_type": row["Node Type"]
            })
        
        return nodes