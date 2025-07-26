"""Base downloader class for all ISO data downloaders."""

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import structlog
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import DownloadHistory, ISO, get_db


class DownloadConfig(BaseModel):
    """Configuration for data downloads."""
    
    start_date: datetime
    end_date: datetime
    data_types: List[str]
    output_dir: str
    batch_size: int = 1000
    retry_attempts: int = 3
    retry_delay: int = 60  # seconds


class BaseDownloader(ABC):
    """Abstract base class for ISO data downloaders."""
    
    def __init__(self, iso_code: str, config: DownloadConfig):
        self.iso_code = iso_code
        self.config = config
        self.logger = structlog.get_logger().bind(iso=iso_code)
        
        # Get ISO ID from database
        with get_db() as db:
            iso = db.query(ISO).filter(ISO.code == iso_code).first()
            if not iso:
                raise ValueError(f"ISO {iso_code} not found in database")
            self.iso_id = iso.id
    
    @abstractmethod
    async def download_energy_prices(
        self, 
        market_type: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Download energy price data for specified market type and date range."""
        pass
    
    @abstractmethod
    async def download_ancillary_prices(
        self,
        service_type: str,
        market_type: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Download ancillary service price data."""
        pass
    
    @abstractmethod
    async def get_available_nodes(self) -> List[Dict[str, Any]]:
        """Get list of available nodes/settlement points."""
        pass
    
    def record_download_start(
        self,
        data_type: str,
        start_timestamp: datetime,
        end_timestamp: datetime
    ) -> int:
        """Record the start of a download in the database."""
        with get_db() as db:
            download = DownloadHistory(
                iso_id=self.iso_id,
                data_type=data_type,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                download_started_at=datetime.utcnow(),
                status="IN_PROGRESS"
            )
            db.add(download)
            db.commit()
            return download.id
    
    def record_download_complete(
        self,
        download_id: int,
        file_path: Optional[str] = None,
        row_count: Optional[int] = None,
        error_message: Optional[str] = None
    ):
        """Record the completion or failure of a download."""
        with get_db() as db:
            download = db.query(DownloadHistory).filter(
                DownloadHistory.id == download_id
            ).first()
            
            if download:
                download.download_completed_at = datetime.utcnow()
                download.status = "FAILED" if error_message else "COMPLETED"
                download.error_message = error_message
                download.file_path = file_path
                download.row_count = row_count
                db.commit()
    
    def get_missing_date_ranges(
        self,
        data_type: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Tuple[datetime, datetime]]:
        """Get date ranges that haven't been downloaded yet."""
        with get_db() as db:
            # Get completed downloads
            completed = db.query(DownloadHistory).filter(
                DownloadHistory.iso_id == self.iso_id,
                DownloadHistory.data_type == data_type,
                DownloadHistory.status == "COMPLETED",
                DownloadHistory.start_timestamp >= start_date,
                DownloadHistory.end_timestamp <= end_date
            ).order_by(DownloadHistory.start_timestamp).all()
            
            if not completed:
                return [(start_date, end_date)]
            
            # Find gaps
            missing_ranges = []
            current_date = start_date
            
            for download in completed:
                if current_date < download.start_timestamp:
                    missing_ranges.append((current_date, download.start_timestamp))
                current_date = max(current_date, download.end_timestamp)
            
            if current_date < end_date:
                missing_ranges.append((current_date, end_date))
            
            return missing_ranges
    
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