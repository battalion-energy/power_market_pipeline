#!/usr/bin/env python3
"""
ENTSO-E Transparency Platform API Client

Handles authentication, rate limiting, and data retrieval from the ENTSO-E
Transparency Platform for all European electricity markets.

Wraps the entsoe-py library with rate limiting and error handling consistent
with the existing power_market_pipeline patterns.
"""

import os
import time
import logging
from datetime import datetime, timedelta
from typing import Optional, List
import pandas as pd

# Import entsoe-py library (will be added to dependencies)
try:
    from entsoe import EntsoePandasClient, EntsoeRawClient
except ImportError:
    raise ImportError(
        "entsoe-py library not installed. Install with: pip install entsoe-py"
    )

from .european_zones import BIDDING_ZONES, get_zone

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Rate limiter to enforce API request throttling.
    Based on PJM pattern but adapted for ENTSO-E API limits.
    """

    def __init__(self, max_requests: int = 10, time_window: int = 60,
                 min_delay_between_requests: float = 1.0):
        """
        Initialize rate limiter.

        Args:
            max_requests: Maximum number of requests allowed (default 10 for ENTSO-E)
            time_window: Time window in seconds (default 60)
            min_delay_between_requests: Minimum seconds to wait between requests (default 1.0)
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.min_delay = min_delay_between_requests
        self.requests = []
        self.last_request_time = None

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


class ENTSOEAPIClient:
    """
    Client for interacting with ENTSO-E Transparency Platform API.

    Wraps entsoe-py library with rate limiting, error handling, and timezone
    management consistent with power_market_pipeline patterns.
    """

    def __init__(self, api_key: Optional[str] = None,
                 requests_per_minute: int = 10,
                 min_delay_between_requests: float = 1.0):
        """
        Initialize ENTSO-E API client.

        Args:
            api_key: ENTSO-E API key (will use ENTSO_E_API_KEY env var if not provided)
            requests_per_minute: Maximum requests per minute (default 10 for conservative throttling)
            min_delay_between_requests: Minimum seconds between requests (default 1.0)

        Raises:
            ValueError: If API key is not provided or found in environment
        """
        self.api_key = api_key or os.getenv('ENTSO_E_API_KEY')
        if not self.api_key:
            raise ValueError(
                "API key required. Set ENTSO_E_API_KEY environment variable or pass api_key parameter.\n"
                "Register at: https://transparency.entsoe.eu/\n"
                "Request API access: Email transparency@entsoe.eu with subject 'Restful API access'"
            )

        self.rate_limiter = RateLimiter(
            max_requests=requests_per_minute,
            time_window=60,
            min_delay_between_requests=min_delay_between_requests
        )

        # Initialize entsoe-py clients
        self.pandas_client = EntsoePandasClient(api_key=self.api_key)
        self.raw_client = EntsoeRawClient(api_key=self.api_key)

        logger.info("ENTSO-E API client initialized")

    def _get_zone_code(self, zone_name: str) -> str:
        """
        Get ENTSO-E zone code from zone name.

        Args:
            zone_name: Short zone name (e.g., 'DE_LU', 'FR', 'NL')

        Returns:
            ENTSO-E zone code (e.g., '10Y1001A1001A82H')

        Raises:
            ValueError: If zone name is not recognized
        """
        zone = get_zone(zone_name)
        if zone is None:
            raise ValueError(
                f"Unknown zone: {zone_name}. "
                f"Available zones: {list(BIDDING_ZONES.keys())}"
            )
        return zone.code

    def _make_request(self, request_func, *args, **kwargs):
        """
        Make an API request with rate limiting and error handling.

        Args:
            request_func: The entsoe-py client method to call
            *args: Positional arguments for the request
            **kwargs: Keyword arguments for the request

        Returns:
            API response (typically pandas DataFrame or Series)

        Raises:
            Exception: If API request fails after retries
        """
        self.rate_limiter.wait_if_needed()

        try:
            result = request_func(*args, **kwargs)
            logger.debug(f"API request successful: {request_func.__name__}")
            return result
        except Exception as e:
            logger.error(f"API request failed: {request_func.__name__} - {str(e)}")
            raise

    def query_day_ahead_prices(self, zone_name: str, start: pd.Timestamp,
                               end: pd.Timestamp) -> pd.DataFrame:
        """
        Query day-ahead market prices for a bidding zone.

        Args:
            zone_name: Bidding zone (e.g., 'DE_LU', 'FR', 'NL')
            start: Start timestamp (timezone-aware)
            end: End timestamp (timezone-aware)

        Returns:
            DataFrame with datetime index (UTC) and 'price_eur_per_mwh' column

        Example:
            >>> client = ENTSOEAPIClient()
            >>> start = pd.Timestamp('2024-01-01', tz='Europe/Berlin')
            >>> end = pd.Timestamp('2024-01-31', tz='Europe/Berlin')
            >>> prices = client.query_day_ahead_prices('DE_LU', start, end)
        """
        zone_code = self._get_zone_code(zone_name)
        logger.info(f"Querying day-ahead prices for {zone_name} from {start} to {end}")

        result = self._make_request(
            self.pandas_client.query_day_ahead_prices,
            zone_code,
            start=start,
            end=end
        )

        # Convert to DataFrame with standard column names
        df = pd.DataFrame({'price_eur_per_mwh': result})

        # Ensure index is in UTC
        if df.index.tz is None:
            logger.warning("API returned timezone-naive data, assuming UTC")
            df.index = df.index.tz_localize('UTC')
        else:
            df.index = df.index.tz_convert('UTC')

        df.index.name = 'datetime_utc'
        logger.info(f"Retrieved {len(df)} day-ahead price records")

        return df

    def query_imbalance_prices(self, zone_name: str, start: pd.Timestamp,
                               end: pd.Timestamp) -> pd.DataFrame:
        """
        Query imbalance prices (closest to "real-time" prices in European markets).

        Args:
            zone_name: Bidding zone (e.g., 'DE_LU', 'FR', 'NL')
            start: Start timestamp (timezone-aware)
            end: End timestamp (timezone-aware)

        Returns:
            DataFrame with datetime index (UTC) and imbalance price columns
            Columns may include: 'short_price' (system short), 'long_price' (system long)

        Note:
            Not all zones provide imbalance prices. Germany typically has 15-minute resolution.
        """
        zone_code = self._get_zone_code(zone_name)
        logger.info(f"Querying imbalance prices for {zone_name} from {start} to {end}")

        try:
            result = self._make_request(
                self.pandas_client.query_imbalance_prices,
                zone_code,
                start=start,
                end=end
            )

            # Convert to DataFrame if Series
            if isinstance(result, pd.Series):
                df = pd.DataFrame({'imbalance_price_eur_per_mwh': result})
            else:
                df = result.copy()
                # Standardize column names
                df.columns = [col.lower().replace(' ', '_') for col in df.columns]

            # Ensure index is in UTC
            if df.index.tz is None:
                logger.warning("API returned timezone-naive data, assuming UTC")
                df.index = df.index.tz_localize('UTC')
            else:
                df.index = df.index.tz_convert('UTC')

            df.index.name = 'datetime_utc'
            logger.info(f"Retrieved {len(df)} imbalance price records")

            return df

        except Exception as e:
            logger.error(f"Failed to retrieve imbalance prices: {str(e)}")
            logger.info("Note: Not all zones provide imbalance price data")
            raise

    def query_imbalance_volumes(self, zone_name: str, start: pd.Timestamp,
                                end: pd.Timestamp) -> pd.DataFrame:
        """
        Query imbalance volumes (amount of balancing energy activated).

        Args:
            zone_name: Bidding zone (e.g., 'DE_LU', 'FR', 'NL')
            start: Start timestamp (timezone-aware)
            end: End timestamp (timezone-aware)

        Returns:
            DataFrame with datetime index (UTC) and imbalance volume columns
        """
        zone_code = self._get_zone_code(zone_name)
        logger.info(f"Querying imbalance volumes for {zone_name} from {start} to {end}")

        try:
            result = self._make_request(
                self.pandas_client.query_imbalance_volumes,
                zone_code,
                start=start,
                end=end
            )

            # Convert to DataFrame
            if isinstance(result, pd.Series):
                df = pd.DataFrame({'imbalance_volume_mwh': result})
            else:
                df = result.copy()

            # Ensure index is in UTC
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            else:
                df.index = df.index.tz_convert('UTC')

            df.index.name = 'datetime_utc'
            logger.info(f"Retrieved {len(df)} imbalance volume records")

            return df

        except Exception as e:
            logger.error(f"Failed to retrieve imbalance volumes: {str(e)}")
            raise

    def query_activated_balancing_energy_prices(self, zone_name: str,
                                                start: pd.Timestamp,
                                                end: pd.Timestamp,
                                                process_type: str = 'A51') -> pd.DataFrame:
        """
        Query activated balancing energy prices.

        Args:
            zone_name: Bidding zone (e.g., 'DE_LU', 'FR', 'NL')
            start: Start timestamp (timezone-aware)
            end: End timestamp (timezone-aware)
            process_type: ENTSO-E process type code
                - 'A51': Automatic Frequency Restoration Reserve (aFRR)
                - 'A47': Manual Frequency Restoration Reserve (mFRR)
                - 'A46': Frequency Containment Reserve (FCR)

        Returns:
            DataFrame with datetime index (UTC) and balancing energy prices

        Note:
            This data may not be available for all zones. Germany provides good coverage.
        """
        zone_code = self._get_zone_code(zone_name)
        logger.info(f"Querying activated balancing energy prices for {zone_name}, "
                   f"process {process_type}, from {start} to {end}")

        try:
            # Note: entsoe-py may return this as XML/ZIP
            # We'll need to handle the raw response
            result = self._make_request(
                self.raw_client.query_activated_balancing_energy_prices,
                zone_code,
                start=start,
                end=end,
                process_type=process_type
            )

            logger.info("Received balancing energy price data (raw XML)")
            logger.warning("Raw XML parsing not yet implemented. Consider using pandas client if available.")

            # TODO: Parse XML response into DataFrame
            # For now, return the raw response
            return result

        except Exception as e:
            logger.error(f"Failed to retrieve activated balancing energy prices: {str(e)}")
            logger.info("Note: This data may not be available for all zones")
            raise

    def query_load(self, zone_name: str, start: pd.Timestamp,
                   end: pd.Timestamp) -> pd.DataFrame:
        """
        Query actual total load (consumption).

        Args:
            zone_name: Bidding zone (e.g., 'DE_LU', 'FR', 'NL')
            start: Start timestamp (timezone-aware)
            end: End timestamp (timezone-aware)

        Returns:
            DataFrame with datetime index (UTC) and 'load_mw' column
        """
        zone_code = self._get_zone_code(zone_name)
        logger.info(f"Querying actual load for {zone_name} from {start} to {end}")

        result = self._make_request(
            self.pandas_client.query_load,
            zone_code,
            start=start,
            end=end
        )

        df = pd.DataFrame({'load_mw': result})

        # Ensure index is in UTC
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        else:
            df.index = df.index.tz_convert('UTC')

        df.index.name = 'datetime_utc'
        logger.info(f"Retrieved {len(df)} load records")

        return df

    def query_generation(self, zone_name: str, start: pd.Timestamp,
                        end: pd.Timestamp) -> pd.DataFrame:
        """
        Query actual generation per production type.

        Args:
            zone_name: Bidding zone (e.g., 'DE_LU', 'FR', 'NL')
            start: Start timestamp (timezone-aware)
            end: End timestamp (timezone-aware)

        Returns:
            DataFrame with datetime index (UTC) and generation by fuel type (MW)
        """
        zone_code = self._get_zone_code(zone_name)
        logger.info(f"Querying generation for {zone_name} from {start} to {end}")

        result = self._make_request(
            self.pandas_client.query_generation,
            zone_code,
            start=start,
            end=end
        )

        df = result.copy()

        # Ensure index is in UTC
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        else:
            df.index = df.index.tz_convert('UTC')

        df.index.name = 'datetime_utc'
        logger.info(f"Retrieved {len(df)} generation records with {len(df.columns)} fuel types")

        return df


def test_api_connection(api_key: Optional[str] = None):
    """
    Test API connection with a small query to Germany.

    Args:
        api_key: ENTSO-E API key (optional, will use env var if not provided)
    """
    try:
        client = ENTSOEAPIClient(api_key=api_key)

        # Test with a small date range
        start = pd.Timestamp('2024-01-01', tz='Europe/Berlin')
        end = pd.Timestamp('2024-01-02', tz='Europe/Berlin')

        logger.info("Testing API connection with day-ahead prices for Germany...")
        df = client.query_day_ahead_prices('DE_LU', start, end)

        print("\n" + "="*80)
        print("API CONNECTION TEST SUCCESSFUL")
        print("="*80)
        print(f"\nRetrieved {len(df)} records")
        print("\nSample data:")
        print(df.head())
        print("\nData types:")
        print(df.dtypes)

        return True

    except Exception as e:
        print("\n" + "="*80)
        print("API CONNECTION TEST FAILED")
        print("="*80)
        print(f"\nError: {str(e)}")
        return False


if __name__ == '__main__':
    import sys

    print("ENTSO-E API Client Test")
    print("="*80)

    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        # Run connection test
        test_api_connection()
    else:
        print("\nUsage:")
        print("  python entso_e_api_client.py --test    # Test API connection")
        print("\nOr import in your scripts:")
        print("  from iso_markets.entso_e.entso_e_api_client import ENTSOEAPIClient")
