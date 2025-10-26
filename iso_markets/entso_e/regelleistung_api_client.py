#!/usr/bin/env python3
"""
Regelleistung.net API Client

Downloads German ancillary services data (FCR, aFRR, mFRR) from Regelleistung.net.
This platform is where all balancing capacity and energy tenders are published
for the German market.

API Access: No authentication required (public data)
Rate Limiting: Conservative 2-second delay between requests
Data Format: Excel (.xlsx) files with tender results

Products:
- FCR (Frequency Containment Reserve): Primary reserve, 30-second activation
- aFRR (automatic Frequency Restoration Reserve): Secondary reserve, 5-min activation
- mFRR (manual Frequency Restoration Reserve): Tertiary reserve, 15-min activation

Market Types:
- CAPACITY: Capacity procurement prices (€/MW per hour or per 4h block)
- ENERGY: Energy activation prices (€/MWh)

Time Structure:
- 6 x 4-hour blocks per day (00-04, 04-08, 08-12, 12-16, 16-20, 20-24)
- aFRR/mFRR have POS (positive/up) and NEG (negative/down) directions
"""

import os
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List
from io import BytesIO
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
    """Simple rate limiter for polite scraping."""

    def __init__(self, min_delay_between_requests: float = 2.0):
        """
        Initialize rate limiter.

        Args:
            min_delay_between_requests: Minimum seconds between requests (default 2.0)
        """
        self.min_delay = min_delay_between_requests
        self.last_request_time = None

    def wait_if_needed(self):
        """Wait if necessary to comply with minimum delay."""
        now = time.time()

        if self.last_request_time is not None:
            time_since_last = now - self.last_request_time
            if time_since_last < self.min_delay:
                wait_time = self.min_delay - time_since_last
                logger.debug(f"Rate limiting: waiting {wait_time:.2f} seconds...")
                time.sleep(wait_time)

        self.last_request_time = time.time()


class RegelleistungAPIClient:
    """
    Client for downloading data from Regelleistung.net.

    The Regelleistung platform provides tender results for balancing services
    in Germany and connected countries. No API key required.
    """

    BASE_URL = "https://www.regelleistung.net/apps/cpp-publisher/api/v1/download/tenders"

    # Product type codes
    PRODUCT_TYPES = {
        'FCR': 'FCR',      # Frequency Containment Reserve (Primary Control)
        'aFRR': 'aFRR',    # Automatic Frequency Restoration Reserve (Secondary Control)
        'mFRR': 'mFRR',    # Manual Frequency Restoration Reserve (Tertiary Control)
    }

    # Market types
    MARKETS = {
        'CAPACITY': 'CAPACITY',  # Capacity market (€/MW)
        'ENERGY': 'ENERGY',      # Energy market (€/MWh)
    }

    def __init__(self, min_delay_between_requests: float = 2.0):
        """
        Initialize Regelleistung API client.

        Args:
            min_delay_between_requests: Minimum seconds between requests (default 2.0 for polite scraping)
        """
        self.rate_limiter = RateLimiter(min_delay_between_requests=min_delay_between_requests)
        self.session = self._create_session()

        logger.info("Regelleistung.net API client initialized (no authentication required)")

    def _create_session(self) -> requests.Session:
        """Create requests session with retry logic."""
        session = requests.Session()

        # Configure retries
        retries = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["GET"]
        )

        adapter = HTTPAdapter(max_retries=retries)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

    def download_tender_results(self, product_type: str, market: str,
                               date: datetime, export_format: str = 'xlsx') -> bytes:
        """
        Download tender results for a specific product, market, and date.

        Args:
            product_type: 'FCR', 'aFRR', or 'mFRR'
            market: 'CAPACITY' or 'ENERGY'
            date: Date for which to download data
            export_format: 'xlsx' or 'csv' (xlsx recommended, csv sometimes unavailable)

        Returns:
            Raw file content (bytes)

        Raises:
            ValueError: If product_type or market is invalid
            requests.HTTPError: If download fails
        """
        # Validate inputs
        if product_type not in self.PRODUCT_TYPES:
            raise ValueError(f"Invalid product_type: {product_type}. "
                           f"Must be one of {list(self.PRODUCT_TYPES.keys())}")

        if market not in self.MARKETS:
            raise ValueError(f"Invalid market: {market}. "
                           f"Must be one of {list(self.MARKETS.keys())}")

        # Build URL
        date_str = date.strftime('%Y-%m-%d')
        url = f"{self.BASE_URL}/resultsoverview"

        params = {
            'productTypes': product_type,
            'market': market,
            'exportFormat': export_format,
            'date': date_str
        }

        # Apply rate limiting
        self.rate_limiter.wait_if_needed()

        # Make request
        logger.info(f"Downloading {product_type} {market} for {date_str}")
        logger.debug(f"URL: {url}")
        logger.debug(f"Params: {params}")

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()

            # Log successful download
            content_length = len(response.content)
            logger.info(f"  Downloaded {content_length:,} bytes")

            return response.content

        except requests.HTTPError as e:
            if e.response.status_code == 404:
                logger.warning(f"  No data available for {product_type} {market} on {date_str}")
                return None
            else:
                logger.error(f"  HTTP error {e.response.status_code}: {e}")
                raise

        except requests.RequestException as e:
            logger.error(f"  Request failed: {e}")
            raise

    def download_and_parse(self, product_type: str, market: str,
                          date: datetime) -> Optional[pd.DataFrame]:
        """
        Download and parse tender results into a DataFrame.

        Args:
            product_type: 'FCR', 'aFRR', or 'mFRR'
            market: 'CAPACITY' or 'ENERGY'
            date: Date for which to download data

        Returns:
            DataFrame with tender results, or None if no data available
        """
        # Download raw file
        content = self.download_tender_results(product_type, market, date, export_format='xlsx')

        if content is None:
            return None

        # Parse Excel file
        try:
            # Wrap bytes in BytesIO to avoid FutureWarning
            excel_buffer = BytesIO(content)

            # Read into DataFrame (assuming first sheet)
            df = pd.read_excel(excel_buffer, sheet_name=0, engine='openpyxl')

            # Add metadata columns
            df['download_date'] = datetime.now()
            df['product_type'] = product_type
            df['market_type'] = market

            logger.info(f"  Parsed {len(df)} records")

            return df

        except Exception as e:
            logger.error(f"  Failed to parse Excel file: {e}")
            raise

    def download_date_range(self, product_type: str, market: str,
                           start_date: datetime, end_date: datetime) -> List[pd.DataFrame]:
        """
        Download tender results for a date range.

        Args:
            product_type: 'FCR', 'aFRR', or 'mFRR'
            market: 'CAPACITY' or 'ENERGY'
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            List of DataFrames, one per day
        """
        logger.info(f"\nDownloading {product_type} {market} from {start_date.date()} to {end_date.date()}")

        all_dfs = []
        current_date = start_date
        failed_dates = []

        while current_date <= end_date:
            try:
                df = self.download_and_parse(product_type, market, current_date)

                if df is not None and not df.empty:
                    all_dfs.append(df)
                else:
                    logger.warning(f"  No data for {current_date.date()}")
                    failed_dates.append(current_date)

            except Exception as e:
                logger.error(f"  Failed to download {current_date.date()}: {e}")
                failed_dates.append(current_date)

            current_date += timedelta(days=1)

        # Summary
        logger.info(f"\nDownload complete:")
        logger.info(f"  Successful: {len(all_dfs)} days")
        logger.info(f"  Failed/No data: {len(failed_dates)} days")

        if failed_dates and len(failed_dates) <= 10:
            logger.info(f"  Failed dates: {[d.date() for d in failed_dates]}")

        return all_dfs


def test_api():
    """Test API with sample downloads."""
    client = RegelleistungAPIClient()

    # Test date
    test_date = datetime(2024, 1, 1)

    print("\n" + "="*80)
    print("REGELLEISTUNG.NET API TEST")
    print("="*80)

    # Test FCR capacity
    print("\n1. Testing FCR CAPACITY...")
    df_fcr = client.download_and_parse('FCR', 'CAPACITY', test_date)
    if df_fcr is not None:
        print(f"   ✓ Success: {len(df_fcr)} records")
        print(f"   Columns: {df_fcr.columns.tolist()[:5]}...")  # First 5 columns
        if 'PRODUCT' in df_fcr.columns or 'PRODUCTNAME' in df_fcr.columns:
            product_col = 'PRODUCT' if 'PRODUCT' in df_fcr.columns else 'PRODUCTNAME'
            print(f"   Products: {df_fcr[product_col].unique()}")

    # Test aFRR capacity
    print("\n2. Testing aFRR CAPACITY...")
    df_afrr_cap = client.download_and_parse('aFRR', 'CAPACITY', test_date)
    if df_afrr_cap is not None:
        print(f"   ✓ Success: {len(df_afrr_cap)} records")
        if 'PRODUCT' in df_afrr_cap.columns:
            print(f"   Products: {df_afrr_cap['PRODUCT'].unique()}")

    # Test aFRR energy
    print("\n3. Testing aFRR ENERGY...")
    df_afrr_energy = client.download_and_parse('aFRR', 'ENERGY', test_date)
    if df_afrr_energy is not None:
        print(f"   ✓ Success: {len(df_afrr_energy)} records")

    # Test mFRR
    print("\n4. Testing mFRR CAPACITY...")
    df_mfrr = client.download_and_parse('mFRR', 'CAPACITY', test_date)
    if df_mfrr is not None:
        print(f"   ✓ Success: {len(df_mfrr)} records")

    print("\n" + "="*80)
    print("TEST COMPLETE - All API endpoints working!")
    print("="*80)


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        test_api()
    else:
        print("Regelleistung.net API Client")
        print("="*80)
        print("\nUsage:")
        print("  python regelleistung_api_client.py --test    # Test API connection")
        print("\nOr import in your scripts:")
        print("  from iso_markets.entso_e.regelleistung_api_client import RegelleistungAPIClient")
