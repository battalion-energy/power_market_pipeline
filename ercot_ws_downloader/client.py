"""
ERCOT Web Service API Client with robust error handling and rate limiting.
"""

import asyncio
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
import logging

load_dotenv()

logger = logging.getLogger(__name__)


class RateLimitException(Exception):
    """Raised when API rate limit is hit (429)."""
    pass


class ERCOTWebServiceClient:
    """
    Client for ERCOT's official REST API with comprehensive error handling.

    Features:
    - Automatic authentication and token refresh
    - Rate limiting to prevent 429 errors
    - Exponential backoff on failures
    - Pagination support
    - Configurable request delays
    """

    BASE_URL = "https://api.ercot.com/api/public-reports"
    AUTH_URL = "https://ercotb2c.b2clogin.com/ercotb2c.onmicrosoft.com/B2C_1_PUBAPI-ROPC-FLOW/oauth2/v2.0/token"
    CLIENT_ID = "fec253ea-0d06-4272-a5e6-b478baeecd70"

    # Rate limiting settings (ERCOT allows 30 requests/minute)
    DEFAULT_RATE_LIMIT_DELAY = 2.5  # seconds between requests (conservative)
    MAX_RETRIES = 5
    TIMEOUT_SECONDS = 60.0

    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        subscription_key: Optional[str] = None,
        rate_limit_delay: Optional[float] = None,
    ):
        """
        Initialize the ERCOT Web Service Client.

        Args:
            username: ERCOT username (defaults to ERCOT_USERNAME env var)
            password: ERCOT password (defaults to ERCOT_PASSWORD env var)
            subscription_key: ERCOT subscription key (defaults to ERCOT_SUBSCRIPTION_KEY env var)
            rate_limit_delay: Delay between requests in seconds (defaults to 2.5s)
        """
        self.username = username or os.getenv("ERCOT_USERNAME")
        self.password = password or os.getenv("ERCOT_PASSWORD")
        self.subscription_key = subscription_key or os.getenv("ERCOT_SUBSCRIPTION_KEY")

        if not all([self.username, self.password, self.subscription_key]):
            raise ValueError(
                "ERCOT API credentials not found. "
                "Set ERCOT_USERNAME, ERCOT_PASSWORD, and ERCOT_SUBSCRIPTION_KEY "
                "environment variables or pass them to the constructor."
            )

        self.rate_limit_delay = rate_limit_delay or self.DEFAULT_RATE_LIMIT_DELAY
        self.last_request_time = 0
        self.token_data = None
        self.token_expiry = None

        logger.info("ERCOT Web Service Client initialized")

    async def _rate_limit(self):
        """Implement rate limiting to avoid API throttling."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
            await asyncio.sleep(sleep_time)

        self.last_request_time = time.time()

    async def _authenticate(self) -> Dict[str, Any]:
        """
        Authenticate with ERCOT and get an access token.

        Returns:
            Dict containing token data
        """
        params = {
            "username": self.username,
            "password": self.password,
            "grant_type": "password",
            "scope": f"openid {self.CLIENT_ID} offline_access",
            "client_id": self.CLIENT_ID,
            "response_type": "token",
        }

        logger.info("Authenticating with ERCOT...")

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.AUTH_URL,
                data=params,
                timeout=self.TIMEOUT_SECONDS
            )

            if response.status_code != 200:
                raise Exception(
                    f"Authentication failed: {response.status_code} - {response.text}"
                )

            self.token_data = response.json()

            # Set token expiry (tokens typically last 1 hour, refresh at 50 minutes)
            if "expires_in" in self.token_data:
                expires_in = int(self.token_data["expires_in"])
                self.token_expiry = time.time() + (expires_in * 0.83)  # 50 minutes for 1-hour token
            else:
                self.token_expiry = time.time() + 3000  # Default 50 minutes

            logger.info("Authentication successful!")
            return self.token_data

    async def _ensure_authenticated(self):
        """Ensure we have a valid token, refreshing if necessary."""
        if (
            not self.token_data
            or not self.token_data.get("access_token")
            or (self.token_expiry and time.time() >= self.token_expiry)
        ):
            await self._authenticate()

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type((httpx.HTTPError, RateLimitException)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def _make_request(
        self, endpoint: str, params: Optional[Dict] = None
    ) -> Dict:
        """
        Make an authenticated request to ERCOT API with retry logic.

        Args:
            endpoint: API endpoint path
            params: Query parameters

        Returns:
            JSON response as dictionary

        Raises:
            RateLimitException: If rate limit (429) is hit
            httpx.HTTPError: For other HTTP errors
        """
        await self._rate_limit()
        await self._ensure_authenticated()

        headers = {
            "Authorization": f"Bearer {self.token_data['access_token']}",
            "Ocp-Apim-Subscription-Key": self.subscription_key,
            "Accept": "application/json",
        }

        url = f"{self.BASE_URL}/{endpoint.lstrip('/')}"

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=self.TIMEOUT_SECONDS,
                )

                # Handle rate limiting
                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 60))
                    logger.warning(
                        f"Rate limit hit! Waiting {retry_after} seconds before retry..."
                    )
                    await asyncio.sleep(retry_after)
                    raise RateLimitException("Rate limit exceeded")

                response.raise_for_status()
                return response.json()

            except httpx.HTTPStatusError as e:
                logger.error(
                    f"HTTP error for {endpoint}: {e.response.status_code} - {e.response.text}"
                )
                raise

    async def get_paginated_data(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        page_size: int = 50000,
        max_pages: Optional[int] = None,
    ) -> List[Dict]:
        """
        Fetch all paginated data from an endpoint.

        Args:
            endpoint: API endpoint path
            params: Base query parameters
            page_size: Number of records per page
            max_pages: Maximum number of pages to fetch (None for all)

        Returns:
            List of all records across all pages
        """
        all_data = []
        page = 1
        params = params or {}

        while True:
            if max_pages and page > max_pages:
                logger.info(f"Reached max_pages limit ({max_pages})")
                break

            page_params = {**params, "page": page, "size": page_size}

            logger.info(
                f"Fetching page {page} from {endpoint} (size={page_size})..."
            )

            response = await self._make_request(endpoint, page_params)

            # Handle different response formats
            if isinstance(response, dict):
                if "data" in response:
                    data = response["data"]
                    meta = response.get("_meta", {})

                    if not data:
                        logger.info("No more data available")
                        break

                    all_data.extend(data)
                    logger.info(
                        f"Retrieved {len(data)} records (total: {len(all_data)})"
                    )

                    # Check if we've reached the last page
                    total_pages = meta.get("totalPages")
                    current_page = meta.get("currentPage", page)

                    if total_pages and current_page >= total_pages:
                        logger.info(f"Reached last page ({total_pages})")
                        break

                    # If we got less than page_size, we're likely done
                    if len(data) < page_size:
                        logger.info("Received partial page, assuming end of data")
                        break

                elif "fields" in response and not response.get("data"):
                    # Empty result with schema only
                    logger.info("No data available for this query")
                    break
                else:
                    # Unknown response format
                    logger.warning(f"Unknown response format: {list(response.keys())}")
                    break

            elif isinstance(response, list):
                # Direct list response
                if not response:
                    break

                all_data.extend(response)
                logger.info(
                    f"Retrieved {len(response)} records (total: {len(all_data)})"
                )

                if len(response) < page_size:
                    break
            else:
                logger.error(f"Unexpected response type: {type(response)}")
                break

            page += 1

        logger.info(f"Completed fetch: {len(all_data)} total records from {endpoint}")
        return all_data

    async def test_connection(self) -> bool:
        """
        Test the API connection and credentials.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            await self._authenticate()
            logger.info("Connection test successful!")
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False
