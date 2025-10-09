"""
State management for tracking downloaded data timestamps and preventing gaps.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class StateManager:
    """
    Manages download state to track last successful download timestamps
    and prevent data gaps.
    """

    def __init__(self, state_file: Path = Path("ercot_download_state.json")):
        """
        Initialize StateManager.

        Args:
            state_file: Path to JSON file storing download state
        """
        self.state_file = state_file
        self.state: Dict = self._load_state()

    def _load_state(self) -> Dict:
        """Load state from JSON file."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                logger.info(f"Loaded state from {self.state_file}")
                return state
            except Exception as e:
                logger.error(f"Error loading state file: {e}")
                return self._initialize_state()
        else:
            logger.info("No existing state file, initializing new state")
            return self._initialize_state()

    def _initialize_state(self) -> Dict:
        """Initialize empty state structure."""
        return {
            "datasets": {},
            "last_updated": datetime.now().isoformat(),
            "version": "1.0",
        }

    def save_state(self):
        """Save current state to JSON file."""
        try:
            self.state["last_updated"] = datetime.now().isoformat()
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2)
            logger.info(f"State saved to {self.state_file}")
        except Exception as e:
            logger.error(f"Error saving state file: {e}")
            raise

    def get_last_timestamp(self, dataset_name: str) -> Optional[datetime]:
        """
        Get the last successfully downloaded timestamp for a dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            Last timestamp as datetime, or None if no previous downloads
        """
        dataset_state = self.state["datasets"].get(dataset_name, {})
        last_ts_str = dataset_state.get("last_timestamp")

        if last_ts_str:
            return datetime.fromisoformat(last_ts_str)
        return None

    def update_last_timestamp(
        self,
        dataset_name: str,
        timestamp: datetime,
        records_count: Optional[int] = None,
    ):
        """
        Update the last successfully downloaded timestamp for a dataset.

        Args:
            dataset_name: Name of the dataset
            timestamp: Last timestamp successfully downloaded
            records_count: Number of records in this download (optional)
        """
        if dataset_name not in self.state["datasets"]:
            self.state["datasets"][dataset_name] = {}

        self.state["datasets"][dataset_name].update({
            "last_timestamp": timestamp.isoformat(),
            "last_download": datetime.now().isoformat(),
        })

        if records_count is not None:
            self.state["datasets"][dataset_name]["last_records_count"] = records_count

        # Don't auto-save here, let caller decide when to save
        logger.info(
            f"Updated {dataset_name} last_timestamp to {timestamp.isoformat()}"
        )

    def get_download_range(
        self,
        dataset_name: str,
        default_start: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        lag_days: int = 0,
    ) -> tuple[datetime, datetime]:
        """
        Calculate the date range to download based on last successful download.

        Args:
            dataset_name: Name of the dataset
            default_start: Default start date if no previous downloads
            end_date: End date for download range (defaults to now minus lag_days)
            lag_days: Days to lag end_date (for 60-day disclosure data)

        Returns:
            Tuple of (start_date, end_date)
        """
        # Get the last downloaded timestamp
        last_ts = self.get_last_timestamp(dataset_name)

        if last_ts:
            # Start from the day after last download
            start_date = last_ts + timedelta(days=1)
            logger.info(f"{dataset_name}: Resuming from {start_date.date()}")
        elif default_start:
            start_date = default_start
            logger.info(
                f"{dataset_name}: No previous downloads, starting from {start_date.date()}"
            )
        else:
            # Default to Web Service cutoff date (Dec 11, 2023)
            start_date = datetime(2023, 12, 11)
            logger.info(
                f"{dataset_name}: No previous downloads, using API earliest date {start_date.date()}"
            )

        # Calculate end date
        if end_date:
            end_date = end_date
        else:
            end_date = datetime.now() - timedelta(days=lag_days)

        # Ensure start is not after end
        if start_date > end_date:
            logger.warning(
                f"{dataset_name}: Start date ({start_date.date()}) is after end date ({end_date.date()}). "
                "No download needed."
            )
            return start_date, start_date  # Return same date to signal no work

        logger.info(
            f"{dataset_name}: Download range: {start_date.date()} to {end_date.date()}"
        )
        return start_date, end_date

    def record_download_attempt(
        self,
        dataset_name: str,
        start_date: datetime,
        end_date: datetime,
        success: bool,
        records_count: Optional[int] = None,
        error: Optional[str] = None,
    ):
        """
        Record a download attempt (success or failure).

        Args:
            dataset_name: Name of the dataset
            start_date: Start of attempted download range
            end_date: End of attempted download range
            success: Whether download was successful
            records_count: Number of records downloaded (if successful)
            error: Error message (if failed)
        """
        if dataset_name not in self.state["datasets"]:
            self.state["datasets"][dataset_name] = {}

        if "download_history" not in self.state["datasets"][dataset_name]:
            self.state["datasets"][dataset_name]["download_history"] = []

        attempt = {
            "timestamp": datetime.now().isoformat(),
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "success": success,
        }

        if records_count is not None:
            attempt["records_count"] = records_count

        if error:
            attempt["error"] = error

        # Keep only last 100 attempts
        history = self.state["datasets"][dataset_name]["download_history"]
        history.append(attempt)
        self.state["datasets"][dataset_name]["download_history"] = history[-100:]

        logger.debug(f"Recorded download attempt for {dataset_name}: {success}")

    def get_dataset_info(self, dataset_name: str) -> Dict:
        """
        Get all information about a dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            Dictionary with dataset information
        """
        return self.state["datasets"].get(dataset_name, {})

    def get_all_datasets(self) -> Dict:
        """
        Get information about all datasets.

        Returns:
            Dictionary mapping dataset names to their info
        """
        return self.state["datasets"]
