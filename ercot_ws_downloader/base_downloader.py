"""
Base class for all ERCOT Web Service downloaders.
"""

import asyncio
import csv
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .client import ERCOTWebServiceClient
from .state_manager import StateManager

logger = logging.getLogger(__name__)


class BaseDownloader(ABC):
    """
    Abstract base class for ERCOT data downloaders.

    Each downloader handles a specific dataset type and knows:
    - Which API endpoint to call
    - How to format parameters
    - Where to save CSV files
    - How to track download state
    """

    def __init__(
        self,
        client: ERCOTWebServiceClient,
        state_manager: StateManager,
        output_dir: Path,
    ):
        """
        Initialize downloader.

        Args:
            client: ERCOT Web Service API client
            state_manager: State manager for tracking downloads
            output_dir: Base directory for output CSV files
        """
        self.client = client
        self.state_manager = state_manager
        self.output_dir = output_dir
        self.dataset_name = self.get_dataset_name()
        self.endpoint = self.get_endpoint()

        # Create output directory
        dataset_output_dir = self.get_output_dir()
        dataset_output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized {self.dataset_name} downloader")

    @abstractmethod
    def get_dataset_name(self) -> str:
        """Return the name of this dataset (used for state tracking)."""
        pass

    @abstractmethod
    def get_endpoint(self) -> str:
        """Return the API endpoint path for this dataset."""
        pass

    @abstractmethod
    def get_output_dir(self) -> Path:
        """Return the output directory for this dataset."""
        pass

    @abstractmethod
    def format_params(
        self, start_date: datetime, end_date: datetime
    ) -> Dict:
        """
        Format API parameters for the date range.

        Args:
            start_date: Start of date range
            end_date: End of date range

        Returns:
            Dictionary of API parameters
        """
        pass

    def get_chunk_size(self) -> int:
        """
        Return the number of days to download in each chunk.
        Override this in subclasses based on data volume.
        """
        return 7  # Default: 1 week chunks

    def get_page_size(self) -> int:
        """
        Return the number of records per page.
        Override this in subclasses based on data volume.
        """
        return 50000  # Default page size

    def get_lag_days(self) -> int:
        """
        Return the number of days to lag behind current date.
        Override this for 60-day disclosure data (60 days)
        or near-real-time data (0 days).
        """
        return 0  # Default: no lag

    def get_csv_filename(self, start_date: datetime, end_date: datetime) -> str:
        """
        Generate CSV filename for the date range.

        Args:
            start_date: Start of date range
            end_date: End of date range

        Returns:
            Filename string
        """
        dataset = self.get_dataset_name().lower().replace(" ", "_")
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")
        return f"{dataset}_{start_str}_{end_str}.csv"

    async def download_chunk(
        self, start_date: datetime, end_date: datetime
    ) -> Optional[List[Dict]]:
        """
        Download data for a specific date range chunk.

        Args:
            start_date: Start of date range
            end_date: End of date range

        Returns:
            List of records, or None if error
        """
        try:
            params = self.format_params(start_date, end_date)
            page_size = self.get_page_size()

            logger.info(
                f"{self.dataset_name}: Downloading {start_date.date()} to {end_date.date()}"
            )

            data = await self.client.get_paginated_data(
                endpoint=self.endpoint,
                params=params,
                page_size=page_size,
            )

            if data:
                logger.info(
                    f"{self.dataset_name}: Retrieved {len(data)} records for {start_date.date()} to {end_date.date()}"
                )
            else:
                logger.warning(
                    f"{self.dataset_name}: No data returned for {start_date.date()} to {end_date.date()}"
                )

            return data

        except Exception as e:
            logger.error(
                f"{self.dataset_name}: Error downloading {start_date.date()} to {end_date.date()}: {e}",
                exc_info=True,
            )
            return None

    def save_to_csv(
        self, data: List[Dict], start_date: datetime, end_date: datetime
    ) -> Optional[Path]:
        """
        Save data to CSV file.

        Args:
            data: List of records to save
            start_date: Start of date range (for filename)
            end_date: End of date range (for filename)

        Returns:
            Path to saved CSV file, or None if error
        """
        if not data:
            logger.warning(f"{self.dataset_name}: No data to save")
            return None

        try:
            output_dir = self.get_output_dir()
            filename = self.get_csv_filename(start_date, end_date)
            output_file = output_dir / filename

            # Convert to DataFrame for easier CSV writing
            df = pd.DataFrame(data)

            # Save to CSV
            df.to_csv(output_file, index=False)

            logger.info(
                f"{self.dataset_name}: Saved {len(data)} records to {output_file}"
            )
            return output_file

        except Exception as e:
            logger.error(
                f"{self.dataset_name}: Error saving CSV: {e}",
                exc_info=True,
            )
            return None

    async def download_range(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> bool:
        """
        Download all data for a date range in chunks.

        Args:
            start_date: Start of date range (None = use state manager)
            end_date: End of date range (None = use now minus lag)

        Returns:
            True if successful, False otherwise
        """
        try:
            # Get download range from state manager
            if start_date is None or end_date is None:
                start_date, end_date = self.state_manager.get_download_range(
                    dataset_name=self.dataset_name,
                    default_start=start_date,
                    end_date=end_date,
                    lag_days=self.get_lag_days(),
                )

            # Check if we need to download anything
            if start_date >= end_date:
                logger.info(
                    f"{self.dataset_name}: No new data to download (already up to date)"
                )
                return True

            chunk_size_days = self.get_chunk_size()
            current_start = start_date
            total_records = 0
            successful_chunks = 0
            failed_chunks = 0

            while current_start < end_date:
                # Calculate chunk end date
                chunk_end = min(
                    current_start + timedelta(days=chunk_size_days - 1),
                    end_date
                )

                # Download chunk
                data = await self.download_chunk(current_start, chunk_end)

                if data is not None:
                    # Save to CSV
                    csv_file = self.save_to_csv(data, current_start, chunk_end)

                    if csv_file:
                        # Update state
                        self.state_manager.update_last_timestamp(
                            self.dataset_name,
                            chunk_end,
                            records_count=len(data),
                        )
                        self.state_manager.record_download_attempt(
                            self.dataset_name,
                            current_start,
                            chunk_end,
                            success=True,
                            records_count=len(data),
                        )
                        total_records += len(data)
                        successful_chunks += 1

                        # Save state after each successful chunk
                        self.state_manager.save_state()
                    else:
                        failed_chunks += 1
                        self.state_manager.record_download_attempt(
                            self.dataset_name,
                            current_start,
                            chunk_end,
                            success=False,
                            error="Failed to save CSV",
                        )
                else:
                    failed_chunks += 1
                    self.state_manager.record_download_attempt(
                        self.dataset_name,
                        current_start,
                        chunk_end,
                        success=False,
                        error="Failed to download data",
                    )

                # Move to next chunk
                current_start = chunk_end + timedelta(days=1)

            logger.info(
                f"{self.dataset_name} download complete: "
                f"{successful_chunks} successful chunks, "
                f"{failed_chunks} failed chunks, "
                f"{total_records} total records"
            )

            return failed_chunks == 0

        except Exception as e:
            logger.error(
                f"{self.dataset_name}: Error in download_range: {e}",
                exc_info=True,
            )
            return False
