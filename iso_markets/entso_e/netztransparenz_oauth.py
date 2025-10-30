#!/usr/bin/env python3
"""
OAuth 2.0 Client for Netztransparenz.de API
Implements Client Credentials Grant flow for automated data retrieval
"""

import os
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any
import json


class NetztransparenzOAuthClient:
    """OAuth 2.0 client for Netztransparenz API access"""

    TOKEN_URL = "https://identity.netztransparenz.de/users/connect/token"
    API_BASE = "https://ds.netztransparenz.de/api/v1"

    def __init__(self, client_id: str, client_secret: str):
        """
        Initialize OAuth client

        Args:
            client_id: OAuth Client ID from Netztransparenz client management
            client_secret: OAuth Client Secret from Netztransparenz
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token = None
        self.token_expires_at = None

    def get_access_token(self) -> str:
        """
        Get OAuth 2.0 access token using Client Credentials Grant
        Caches token until expiration

        Returns:
            Access token string
        """
        # Return cached token if still valid
        if self.access_token and self.token_expires_at:
            if datetime.now() < self.token_expires_at - timedelta(minutes=5):
                return self.access_token

        # Request new token
        print("Requesting new OAuth access token...")

        payload = {
            'grant_type': 'client_credentials',
            'client_id': self.client_id,
            'client_secret': self.client_secret
        }

        response = requests.post(self.TOKEN_URL, data=payload)
        response.raise_for_status()

        token_data = response.json()
        self.access_token = token_data['access_token']

        # Calculate expiration (typically 3600 seconds)
        expires_in = token_data.get('expires_in', 3600)
        self.token_expires_at = datetime.now() + timedelta(seconds=expires_in)

        print(f"✅ Token acquired (expires in {expires_in}s)")
        return self.access_token

    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> str:
        """
        Make authenticated API request

        Args:
            endpoint: API endpoint path (e.g., "/data/NrvSaldo/reBAP/Qualitaetsgesichert/...")
            params: Optional query parameters

        Returns:
            CSV text response data
        """
        token = self.get_access_token()

        headers = {
            'Authorization': f'Bearer {token}',
            'Accept': 'text/csv'
        }

        url = f"{self.API_BASE}{endpoint}"
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()

        return response.text

    def get_rebap_data(self, date_from: str, date_to: str) -> str:
        """
        Get reBAP (imbalance price) data

        Args:
            date_from: Start date in ISO format (YYYY-MM-DDTHH:MM:SS)
            date_to: End date in ISO format (YYYY-MM-DDTHH:MM:SS)

        Returns:
            CSV text string with reBAP records
        """
        endpoint = f"/data/NrvSaldo/reBAP/Qualitaetsgesichert/{date_from}/{date_to}"
        return self._make_request(endpoint)

    def test_connection(self) -> bool:
        """
        Test API connection with health endpoint

        Returns:
            True if connection successful
        """
        try:
            token = self.get_access_token()
            headers = {'Authorization': f'Bearer {token}'}
            response = requests.get(f"{self.API_BASE}/health", headers=headers)
            response.raise_for_status()
            print("✅ API connection successful")
            return True
        except Exception as e:
            print(f"❌ API connection failed: {e}")
            return False


def load_credentials_from_env() -> tuple[str, str]:
    """
    Load OAuth credentials from environment variables

    Returns:
        (client_id, client_secret) tuple

    Raises:
        ValueError if credentials not found
    """
    client_id = os.getenv('NETZTRANSPARENZ_CLIENT_ID')
    client_secret = os.getenv('NETZTRANSPARENZ_CLIENT_SECRET')

    if not client_id or not client_secret:
        raise ValueError(
            "Missing OAuth credentials. Please set:\n"
            "  NETZTRANSPARENZ_CLIENT_ID=<your_client_id>\n"
            "  NETZTRANSPARENZ_CLIENT_SECRET=<your_client_secret>\n"
            "\n"
            "Register at: https://extranet.netztransparenz.de/\n"
            "Create client in 'My Clients' section"
        )

    return client_id, client_secret


if __name__ == "__main__":
    # Test the OAuth client
    from dotenv import load_dotenv

    load_dotenv()

    try:
        client_id, client_secret = load_credentials_from_env()
        client = NetztransparenzOAuthClient(client_id, client_secret)

        print("Testing Netztransparenz OAuth Client")
        print("=" * 60)
        print()

        # Test connection
        if client.test_connection():
            print()
            print("Testing reBAP data retrieval...")

            # Get last 2 days of data as test
            date_to = datetime.now()
            date_from = date_to - timedelta(days=2)

            data = client.get_rebap_data(
                date_from.strftime('%Y-%m-%dT00:00:00'),
                date_to.strftime('%Y-%m-%dT23:59:59')
            )

            print(f"✅ Retrieved {len(data)} records")
            print()
            print("Sample data:")
            print(json.dumps(data[:2], indent=2))

    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")
