"""Base downloader class for all ISO data downloaders - V2 with standardized schema."""

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import structlog
from pydantic import BaseModel

from database import DataCatalog, ISO, Location, get_db


class DownloadConfig(BaseModel):
    """Configuration for data downloads."""
    
    start_date: datetime
    end_date: datetime
    data_types: List[str]
    output_dir: str
    batch_size: int = 1000
    retry_attempts: int = 3
    retry_delay: int = 60  # seconds


class BaseDownloaderV2(ABC):
    """Abstract base class for ISO data downloaders using standardized schema."""
    
    def __init__(self, iso_code: str, config: DownloadConfig):
        self.iso_code = iso_code
        self.config = config
        self.logger = structlog.get_logger().bind(iso=iso_code)
        
        # Get ISO from database
        with get_db() as db:
            iso = db.query(ISO).filter(ISO.code == iso_code).first()
            if not iso:
                raise ValueError(f"ISO {iso_code} not found in database")
            self.iso = iso
    
    @abstractmethod
    async def download_lmp(
        self, 
        market: str,  # 'DAM', 'RT5M', 'RT15M', 'HASP'
        start_date: datetime, 
        end_date: datetime,
        locations: Optional[List[str]] = None
    ) -> int:
        """Download LMP data for specified market and date range.
        
        Returns number of records downloaded.
        """
        pass
    
    @abstractmethod
    async def download_ancillary_services(
        self,
        product: str,  # 'REGUP', 'REGDOWN', 'SPIN', 'NON_SPIN', etc.
        market: str,  # 'DAM', 'RTM'
        start_date: datetime,
        end_date: datetime
    ) -> int:
        """Download ancillary service data.
        
        Returns number of records downloaded.
        """
        pass
    
    @abstractmethod
    async def download_load(
        self,
        forecast_type: str,  # 'actual', 'forecast_1h', 'forecast_dam'
        start_date: datetime,
        end_date: datetime
    ) -> int:
        """Download load data.
        
        Returns number of records downloaded.
        """
        pass
    
    @abstractmethod
    async def get_available_locations(self) -> List[Dict[str, Any]]:
        """Get list of available locations for this ISO."""
        pass
    
    def get_or_create_location(self, location_id: str, location_name: str = None, 
                             location_type: str = None) -> Location:
        """Get or create a location in the database."""
        with get_db() as db:
            location = db.query(Location).filter(
                Location.iso_id == self.iso.id,
                Location.location_id == location_id
            ).first()
            
            if not location:
                location = Location(
                    iso_id=self.iso.id,
                    location_id=location_id,
                    location_name=location_name or location_id,
                    location_type=location_type or self._infer_location_type(location_id)
                )
                db.add(location)
                db.commit()
                db.refresh(location)
            
            return location
    
    def _infer_location_type(self, location_id: str) -> str:
        """Infer location type from location ID patterns."""
        # Override in subclasses with ISO-specific logic
        return "node"
    
    def update_dataset_metadata(self, dataset_name: str, latest_timestamp: datetime = None,
                              row_count: int = None):
        """Update dataset metadata after download."""
        with get_db() as db:
            dataset = db.query(DataCatalog).filter(
                DataCatalog.dataset_name == dataset_name
            ).first()
            
            if dataset:
                if latest_timestamp:
                    dataset.latest_data = latest_timestamp
                    if not dataset.earliest_data or latest_timestamp < dataset.earliest_data:
                        dataset.earliest_data = latest_timestamp
                
                dataset.last_updated = datetime.utcnow()
                db.commit()
    
    def get_missing_intervals(
        self,
        table_name: str,
        market: str,
        start_date: datetime,
        end_date: datetime,
        temporal_granularity: str
    ) -> List[Tuple[datetime, datetime]]:
        """Get intervals that haven't been downloaded yet."""
        # This is a simplified version - in production, query the actual table
        # to find gaps in the data
        return [(start_date, end_date)]
    
    def chunk_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        days_per_chunk: int = 30
    ) -> List[Tuple[datetime, datetime]]:
        """Split a date range into smaller chunks."""
        chunks = []
        current_start = start_date
        
        while current_start < end_date:
            current_end = min(
                current_start + timedelta(days=days_per_chunk),
                end_date
            )
            chunks.append((current_start, current_end))
            current_start = current_end
            
        return chunks
    
    def calculate_interval_end(self, interval_start: datetime, granularity: str) -> datetime:
        """Calculate interval end based on granularity."""
        if granularity == "5min":
            return interval_start + timedelta(minutes=5)
        elif granularity == "15min":
            return interval_start + timedelta(minutes=15)
        elif granularity == "hourly":
            return interval_start + timedelta(hours=1)
        elif granularity == "daily":
            return interval_start + timedelta(days=1)
        else:
            return interval_start + timedelta(hours=1)