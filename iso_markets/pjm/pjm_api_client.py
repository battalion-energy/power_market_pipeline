#!/usr/bin/env python3
"""
PJM Data Miner 2 API Client
Handles authentication, rate limiting, and data retrieval from PJM APIs.
"""

import os
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RateLimiter:
    """Rate limiter to enforce API request throttling."""

    def __init__(self, max_requests: int = 8, time_window: int = 60):
        """
        Initialize rate limiter.

        Args:
            max_requests: Maximum number of requests allowed
            time_window: Time window in seconds
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []

    def wait_if_needed(self):
        """Wait if necessary to comply with rate limits."""
        now = time.time()

        # Remove requests outside the time window
        self.requests = [req_time for req_time in self.requests
                        if now - req_time < self.time_window]

        if len(self.requests) >= self.max_requests:
            # Calculate how long to wait
            oldest_request = min(self.requests)
            wait_time = self.time_window - (now - oldest_request)
            if wait_time > 0:
                logger.info(f"Rate limit reached. Waiting {wait_time:.2f} seconds...")
                time.sleep(wait_time + 0.1)  # Add small buffer

        # Record this request
        self.requests.append(time.time())


class PJMAPIClient:
    """Client for interacting with PJM Data Miner 2 API."""

    BASE_URL = "https://api.pjm.com/api/v1"

    def __init__(self, api_key: Optional[str] = None,
                 requests_per_minute: int = 6):
        """
        Initialize PJM API client.

        Args:
            api_key: PJM API key (will use PJM_API_KEY env var if not provided)
            requests_per_minute: Maximum requests per minute (default 6 for non-members, 600 for members)
        """
        self.api_key = api_key or os.getenv('PJM_API_KEY')
        if not self.api_key:
            raise ValueError(
                "API key required. Set PJM_API_KEY environment variable or pass api_key parameter.\n"
                "Register for free at: https://apiportal.pjm.com/signup/"
            )

        self.rate_limiter = RateLimiter(max_requests=requests_per_minute, time_window=60)
        self.session = self._create_session()

    def _create_session(self) -> requests.Session:
        """Create a requests session with retry logic."""
        session = requests.Session()

        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests."""
        return {
            'Ocp-Apim-Subscription-Key': self.api_key,
            'Accept': 'application/json',
            'User-Agent': 'PJM-Data-Downloader/1.0'
        }

    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """
        Make a rate-limited API request.

        Args:
            endpoint: API endpoint (relative to BASE_URL)
            params: Query parameters

        Returns:
            JSON response as dictionary
        """
        self.rate_limiter.wait_if_needed()

        url = f"{self.BASE_URL}/{endpoint}"
        logger.info(f"Requesting: {url}")

        try:
            response = self.session.get(
                url,
                headers=self._get_headers(),
                params=params,
                timeout=30
            )

            response.raise_for_status()
            return response.json()

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                raise ValueError(
                    "Authentication failed. Check your API key.\n"
                    "Register at: https://apiportal.pjm.com/signup/"
                )
            elif e.response.status_code == 429:
                logger.warning("Rate limit exceeded. Waiting 60 seconds...")
                time.sleep(60)
                return self._make_request(endpoint, params)
            else:
                logger.error(f"HTTP error: {e}")
                raise

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise

    def get_day_ahead_lmps(self, start_date: str, end_date: str,
                          pnode_id: Optional[str] = None,
                          use_exact_times: bool = False) -> List[Dict]:
        """
        Get day-ahead hourly LMPs (Locational Marginal Prices).

        Args:
            start_date: Start date/time (YYYY-MM-DD or YYYY-MM-DD HH:mm if use_exact_times=True)
            end_date: End date/time (YYYY-MM-DD or YYYY-MM-DD HH:mm if use_exact_times=True)
            pnode_id: Specific pnode ID (optional, for hubs or specific nodes)
            use_exact_times: If True, use start_date/end_date as-is without adding times

        Returns:
            List of price records
        """
        # API requires date range in format: "YYYY-MM-DD HH:mm to YYYY-MM-DD HH:mm"
        if use_exact_times:
            datetime_range = f'{start_date} to {end_date}'
        else:
            datetime_range = f'{start_date} 00:00 to {end_date} 23:59'

        params = {
            'startRow': '1',
            'rowCount': '50000',  # Max allowed per request
            'datetime_beginning_ept': datetime_range
        }

        if pnode_id:
            params['pnode_id'] = pnode_id

        return self._make_request('da_hrl_lmps', params)

    def get_rt_hourly_lmps(self, start_date: str, end_date: str,
                           pnode_id: Optional[str] = None) -> List[Dict]:
        """
        Get real-time hourly LMPs (settlement roll-up of 5-minute runs).

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            pnode_id: Specific pnode ID (optional, for hubs or specific nodes)

        Returns:
            List of price records
        """
        # API requires date range in format: "YYYY-MM-DD HH:mm to YYYY-MM-DD HH:mm"
        params = {
            'startRow': '1',
            'rowCount': '50000',  # Max allowed per request
            'datetime_beginning_ept': f'{start_date} 00:00 to {end_date} 23:59'
        }

        if pnode_id:
            params['pnode_id'] = pnode_id

        return self._make_request('rt_hrl_lmps', params)

    def get_rt_fivemin_lmps(self, start_date: str, end_date: str,
                            pnode_id: Optional[str] = None) -> List[Dict]:
        """
        Get real-time 5-minute LMPs.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            pnode_id: Specific pnode ID (optional)

        Returns:
            List of price records
        """
        params = {
            'startRow': '1',
            'rowCount': '50000',
            'datetime_beginning_ept': f'{start_date} 00:00 to {end_date} 23:59'
        }

        if pnode_id:
            params['pnode_id'] = pnode_id

        return self._make_request('rt_fivemin_lmps', params)

    def get_ancillary_services(self, start_date: str, end_date: str) -> List[Dict]:
        """
        Get day-ahead ancillary services pricing data.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            List of ancillary service price records
        """
        # API requires date range in format: "YYYY-MM-DD HH:mm to YYYY-MM-DD HH:mm"
        params = {
            'startRow': '1',
            'rowCount': '50000',
            'datetime_beginning_ept': f'{start_date} 00:00 to {end_date} 23:59'
        }

        return self._make_request('da_ancillary_services', params)

    def get_pnodes(self) -> List[Dict]:
        """
        Get list of all pnodes.

        Returns:
            List of pnode information
        """
        return self._make_request('pnode', {'rowCount': '50000', 'startRow': '1'})

    def get_hubs(self) -> List[Dict]:
        """
        Get list of pricing hubs.

        Returns:
            List of hub information
        """
        # Hubs are typically identified by pnode_type or specific naming patterns
        pnodes = self.get_pnodes()

        # Filter for hubs - adjust based on actual API response structure
        hubs = [p for p in pnodes if 'HUB' in p.get('pnode_name', '').upper()
                or p.get('pnode_type', '').upper() == 'HUB']

        return hubs


if __name__ == "__main__":
    # Test the client
    print("PJM API Client")
    print("=" * 80)
    print("\nTo use this client, you need a free API key from PJM.")
    print("Register at: https://apiportal.pjm.com/signup/")
    print("\nSet your API key as an environment variable:")
    print("  export PJM_API_KEY='your-api-key-here'")
    print("\nOr pass it directly when creating the client:")
    print("  client = PJMAPIClient(api_key='your-api-key-here')")
