#!/usr/bin/env python3
"""
CAISO OASIS API Client
Handles rate limiting and data retrieval from CAISO OASIS public APIs.
No authentication required - CAISO OASIS API is public.
"""

import os
import time
import logging
import zipfile
import io
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RateLimiter:
    """Rate limiter for CAISO OASIS API (requires 5-second delays)."""

    def __init__(self, min_delay_between_requests: float = 5.0):
        """
        Initialize rate limiter.

        Args:
            min_delay_between_requests: Minimum seconds to wait between requests (default 5.0 per CAISO requirement)
        """
        self.min_delay = min_delay_between_requests
        self.last_request_time = None

    def wait_if_needed(self):
        """Wait if necessary to comply with minimum delay requirement."""
        now = time.time()

        # Enforce minimum delay between requests (CAISO requires 5 seconds)
        if self.last_request_time is not None:
            time_since_last = now - self.last_request_time
            if time_since_last < self.min_delay:
                wait_time = self.min_delay - time_since_last
                logger.debug(f"Minimum delay: waiting {wait_time:.2f} seconds...")
                time.sleep(wait_time)
                now = time.time()

        # Record this request
        self.last_request_time = now


class CAISOAPIClient:
    """Client for interacting with CAISO OASIS API."""

    BASE_URL = "http://oasis.caiso.com/oasisapi/SingleZip"

    def __init__(self, min_delay_between_requests: float = 6.0):
        """
        Initialize CAISO API client.

        Args:
            min_delay_between_requests: Minimum seconds between requests (default 6.0, conservative for CAISO 5-sec requirement)
        """
        self.rate_limiter = RateLimiter(
            min_delay_between_requests=min_delay_between_requests
        )
        self.session = self._create_session()

    def _create_session(self) -> requests.Session:
        """Create a requests session with retry logic."""
        session = requests.Session()

        # Configure retry strategy
        # Don't retry 429 here - handle it in _make_request for better control
        retry_strategy = Retry(
            total=3,
            backoff_factor=2,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["GET"]
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests."""
        return {
            'Accept': 'application/zip',
            'User-Agent': 'CAISO-Data-Downloader/1.0'
        }

    def _make_request(self, params: Dict) -> bytes:
        """
        Make a rate-limited API request.

        Args:
            params: Query parameters

        Returns:
            ZIP file content as bytes
        """
        self.rate_limiter.wait_if_needed()

        logger.info(f"Requesting: {params.get('queryname')} for {params.get('startdatetime')} to {params.get('enddatetime')}")

        try:
            response = self.session.get(
                self.BASE_URL,
                headers=self._get_headers(),
                params=params,
                timeout=120  # Longer timeout for ZIP files
            )

            response.raise_for_status()
            return response.content

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                # Don't retry recursively - let caller handle retries
                logger.warning(f"Rate limit exceeded (429). Not retrying - caller should implement exponential backoff.")
                raise
            else:
                logger.error(f"HTTP error: {e}")
                raise

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise

    def _extract_csv_from_zip(self, zip_content: bytes) -> pd.DataFrame:
        """
        Extract data from ZIP file content (handles both CSV and XML formats).

        Args:
            zip_content: ZIP file content as bytes

        Returns:
            DataFrame with the data
        """
        with zipfile.ZipFile(io.BytesIO(zip_content)) as zip_file:
            # Get list of files in ZIP
            file_list = zip_file.namelist()

            # Try CSV first (newer data format)
            csv_files = [f for f in file_list if f.endswith('.csv')]

            if csv_files:
                if len(csv_files) > 1:
                    logger.warning(f"Multiple CSV files found in ZIP: {csv_files}. Using first one.")
                # Read the CSV file
                csv_file = csv_files[0]
                with zip_file.open(csv_file) as f:
                    df = pd.read_csv(f)
                return df

            # Fall back to XML (older data format)
            xml_files = [f for f in file_list if f.endswith('.xml')]

            if not xml_files:
                raise ValueError(f"No CSV or XML file found in ZIP. Files: {file_list}")

            if len(xml_files) > 1:
                logger.warning(f"Multiple XML files found in ZIP: {xml_files}. Using first one.")

            # Read the XML file
            xml_file = xml_files[0]
            with zip_file.open(xml_file) as f:
                df = pd.read_xml(f)

            return df

    def get_day_ahead_lmps(self, start_datetime: str, end_datetime: str,
                          node: str = "ALL") -> pd.DataFrame:
        """
        Get day-ahead LMPs (Locational Marginal Prices).

        Args:
            start_datetime: Start datetime in format 'YYYYMMDDTHH:MM-0000'
            end_datetime: End datetime in format 'YYYYMMDDTHH:MM-0000'
            node: Node ID or 'ALL' for all nodes (default)

        Returns:
            DataFrame with price records
        """
        params = {
            'queryname': 'PRC_LMP',
            'startdatetime': start_datetime,
            'enddatetime': end_datetime,
            'version': '1',
            'market_run_id': 'DAM',
            'node': node,
            'resultformat': '6'  # CSV in ZIP
        }

        zip_content = self._make_request(params)
        return self._extract_csv_from_zip(zip_content)

    def get_rt_5min_lmps(self, start_datetime: str, end_datetime: str,
                        node: str = "ALL") -> pd.DataFrame:
        """
        Get real-time 5-minute LMPs.

        Args:
            start_datetime: Start datetime in format 'YYYYMMDDTHH:MM-0000'
            end_datetime: End datetime in format 'YYYYMMDDTHH:MM-0000'
            node: Node ID or 'ALL' for all nodes (default)

        Returns:
            DataFrame with price records
        """
        params = {
            'queryname': 'PRC_INTVL_LMP',
            'startdatetime': start_datetime,
            'enddatetime': end_datetime,
            'version': '1',
            'market_run_id': 'RTM',
            'node': node,
            'resultformat': '6'  # CSV in ZIP
        }

        zip_content = self._make_request(params)
        return self._extract_csv_from_zip(zip_content)

    def get_ancillary_services(self, start_datetime: str, end_datetime: str,
                              anc_type: str = "ALL", anc_region: str = "ALL") -> pd.DataFrame:
        """
        Get ancillary services pricing data.

        Args:
            start_datetime: Start datetime in format 'YYYYMMDDTHH:MM-0000'
            end_datetime: End datetime in format 'YYYYMMDDTHH:MM-0000'
            anc_type: Ancillary service type (RU, RD, SR, NR, or ALL)
            anc_region: Region (default ALL)

        Returns:
            DataFrame with ancillary service price records
        """
        params = {
            'queryname': 'PRC_AS',
            'startdatetime': start_datetime,
            'enddatetime': end_datetime,
            'version': '1',
            'market_run_id': 'DAM',
            'anc_type': anc_type,
            'anc_region': anc_region,
            'resultformat': '6'  # CSV in ZIP
        }

        zip_content = self._make_request(params)
        return self._extract_csv_from_zip(zip_content)


def format_datetime_for_caiso(dt: datetime) -> str:
    """
    Format datetime for CAISO API.

    Args:
        dt: Python datetime object

    Returns:
        String in format 'YYYYMMDDTHH:MM-0000'
    """
    return dt.strftime('%Y%m%dT%H:%M-0000')


if __name__ == "__main__":
    # Test the client
    print("CAISO OASIS API Client")
    print("=" * 80)
    print("\nCAISO OASIS API is PUBLIC - no authentication required!")
    print("\nTest usage:")
    print("  client = CAISOAPIClient()")
    print("  df = client.get_day_ahead_lmps('20250101T00:00-0000', '20250102T00:00-0000')")
    print("\nNote: CAISO requires 5-second delays between requests")
