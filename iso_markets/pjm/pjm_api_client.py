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

    def __init__(self, max_requests: int = 8, time_window: int = 60,
                 min_delay_between_requests: float = 2.0):
        """
        Initialize rate limiter.

        Args:
            max_requests: Maximum number of requests allowed
            time_window: Time window in seconds
            min_delay_between_requests: Minimum seconds to wait between requests (default 2.0)
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.min_delay = min_delay_between_requests
        self.requests = []
        self.last_request_time = None
        self.response_times = []  # Track response times for smart backoff
        self.max_response_samples = 50  # Keep last 50 response times

    def wait_if_needed(self):
        """Wait if necessary to comply with rate limits and minimum delay."""
        now = time.time()

        # Enforce minimum delay between requests (conservative approach)
        if self.last_request_time is not None:
            time_since_last = now - self.last_request_time
            if time_since_last < self.min_delay:
                wait_time = self.min_delay - time_since_last
                logger.debug(f"Minimum delay: waiting {wait_time:.2f} seconds...")
                time.sleep(wait_time)
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
                time.sleep(wait_time + 0.5)  # Add buffer
                now = time.time()

        # Record this request
        self.requests.append(now)
        self.last_request_time = now

    def record_response_time(self, duration: float):
        """Record a response time for calculating smart backoff."""
        self.response_times.append(duration)
        if len(self.response_times) > self.max_response_samples:
            self.response_times.pop(0)

    def get_average_response_time(self) -> float:
        """Get average response time, or estimated time if no samples."""
        if not self.response_times:
            return 3.0  # Default estimate
        return sum(self.response_times) / len(self.response_times)

    def get_smart_429_backoff(self) -> float:
        """
        Calculate smart backoff for 429 errors based on actual response times.

        Logic: If we're hitting rate limits with 6 req/min (10s average interval),
        and we already wait ~min_delay between requests, a small additional wait
        should be enough. We use 2x the average response time as initial backoff.
        """
        avg_response = self.get_average_response_time()
        # Start with 2x average response time, minimum 5 seconds
        return max(5.0, avg_response * 2)


class PJMAPIClient:
    """Client for interacting with PJM Data Miner 2 API."""

    BASE_URL = "https://api.pjm.com/api/v1"

    def __init__(self, api_key: Optional[str] = None,
                 requests_per_minute: int = 6,
                 min_delay_between_requests: float = 2.0):
        """
        Initialize PJM API client.

        Args:
            api_key: PJM API key (will use PJM_API_KEY env var if not provided)
            requests_per_minute: Maximum requests per minute (default 5 for safety, 6 is max for non-members)
            min_delay_between_requests: Minimum seconds between requests (default 2.0 for conservative throttling)
        """
        self.api_key = api_key or os.getenv('PJM_API_KEY')
        if not self.api_key:
            raise ValueError(
                "API key required. Set PJM_API_KEY environment variable or pass api_key parameter.\n"
                "Register for free at: https://apiportal.pjm.com/signup/"
            )

        self.rate_limiter = RateLimiter(
            max_requests=requests_per_minute,
            time_window=60,
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
            status_forcelist=[500, 502, 503, 504],  # Removed 429 - handle separately
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

        request_start = time.time()
        try:
            response = self.session.get(
                url,
                headers=self._get_headers(),
                params=params,
                timeout=30
            )

            # Record response time for successful and rate-limited requests
            response_time = time.time() - request_start
            self.rate_limiter.record_response_time(response_time)

            response.raise_for_status()
            return response.json()

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                raise ValueError(
                    "Authentication failed. Check your API key.\n"
                    "Register at: https://apiportal.pjm.com/signup/"
                )
            elif e.response.status_code == 429:
                # Check for Retry-After header
                retry_after = e.response.headers.get('Retry-After')
                if retry_after:
                    logger.warning(f"Rate limit exceeded (429). API says retry after: {retry_after} seconds")
                    e.retry_after = int(retry_after) if retry_after and retry_after.isdigit() else None
                else:
                    logger.warning(f"Rate limit exceeded (429). No Retry-After header provided.")
                    # Use smart backoff based on observed response times
                    smart_backoff = self.rate_limiter.get_smart_429_backoff()
                    e.retry_after = smart_backoff
                    logger.warning(f"Suggesting {smart_backoff:.1f}s backoff (based on avg response time: {self.rate_limiter.get_average_response_time():.1f}s)")
                raise
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
                           pnode_id: Optional[str] = None,
                           use_exact_times: bool = False) -> List[Dict]:
        """
        Get real-time hourly LMPs (settlement roll-up of 5-minute runs).

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

        return self._make_request('rt_hrl_lmps', params)

    def get_rt_fivemin_lmps(self, start_date: str, end_date: str,
                            pnode_id: Optional[str] = None,
                            use_exact_times: bool = False) -> List[Dict]:
        """
        Get real-time 5-minute LMPs.

        IMPORTANT: PJM only retains 5-minute data for ~6 months (186 days).
        Historical data beyond this window is not available via API.

        Args:
            start_date: Start date/time (YYYY-MM-DD or YYYY-MM-DD HH:mm if use_exact_times=True)
            end_date: End date/time (YYYY-MM-DD or YYYY-MM-DD HH:mm if use_exact_times=True)
            pnode_id: Specific pnode ID (optional)
            use_exact_times: If True, use start_date/end_date as-is without adding times

        Returns:
            List of price records

        Note:
            Fixed endpoint: 'rt_fivemin_hrl_lmps' (was 'rt_fivemin_lmps' - caused 404)
        """
        # API requires date range in format: "YYYY-MM-DD HH:mm to YYYY-MM-DD HH:mm"
        if use_exact_times:
            datetime_range = f'{start_date} to {end_date}'
        else:
            datetime_range = f'{start_date} 00:00 to {end_date} 23:59'

        params = {
            'startRow': '1',
            'rowCount': '50000',
            'datetime_beginning_ept': datetime_range
        }

        if pnode_id:
            params['pnode_id'] = pnode_id

        # Fixed endpoint name: was 'rt_fivemin_lmps' (404), now 'rt_fivemin_hrl_lmps' (correct)
        return self._make_request('rt_fivemin_hrl_lmps', params)

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
