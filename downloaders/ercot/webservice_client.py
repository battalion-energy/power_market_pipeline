"""ERCOT Web Service API client for recent data (after Dec 11, 2023)."""

import asyncio
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import httpx
import pandas as pd
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

from .constants import TRADING_HUBS, WEBSERVICE_CUTOFF_DATE

load_dotenv()


class ERCOTWebServiceClient:
    """Client for ERCOT's official REST API."""
    
    BASE_URL = "https://api.ercot.com"
    AUTH_URL = "https://ercotb2c.b2clogin.com/ercotb2c.onmicrosoft.com/B2C_1_PUBAPI-ROPC-FLOW/oauth2/v2.0/token"
    CLIENT_ID = "fec253ea-0d06-4272-a5e6-b478baeecd70"
    
    def __init__(self):
        self.username = os.getenv("ERCOT_USERNAME")
        self.password = os.getenv("ERCOT_PASSWORD")
        self.subscription_key = os.getenv("ERCOT_SUBSCRIPTION_KEY")
        
        if not all([self.username, self.password, self.subscription_key]):
            raise ValueError("ERCOT API credentials not found in environment variables")
        
        self.rate_limit_delay = 2  # seconds between requests
        self.last_request_time = 0
        self.token_data = None
    
    async def _rate_limit(self):
        """Implement rate limiting to avoid API throttling."""
        current_time = asyncio.get_event_loop().time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)
        
        self.last_request_time = asyncio.get_event_loop().time()
    
    async def _authenticate(self) -> Dict[str, Any]:
        """Authenticate with ERCOT and get an access token."""
        params = {
            "username": self.username,
            "password": self.password,
            "grant_type": "password",
            "scope": f"openid {self.CLIENT_ID} offline_access",
            "client_id": self.CLIENT_ID,
            "response_type": "token"
        }
        
        print(f"Authenticating with ERCOT...")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(self.AUTH_URL, data=params)
            
            print(f"Auth response status: {response.status_code}")
            print(f"Auth response body: {response.text}")
            
            if response.status_code != 200:
                raise Exception(f"Authentication failed: {response.status_code} - {response.text}")
            
            self.token_data = response.json()
            print(f"Authentication successful! Got token with keys: {list(self.token_data.keys())}")
            return self.token_data
    
    async def _ensure_authenticated(self):
        """Ensure we have a valid token."""
        if not self.token_data or not self.token_data.get("access_token"):
            await self._authenticate()
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def _make_request(
        self, 
        endpoint: str, 
        params: Optional[Dict] = None
    ) -> Dict:
        """Make an authenticated request to ERCOT API."""
        await self._rate_limit()
        await self._ensure_authenticated()
        
        headers = {
            "Authorization": f"Bearer {self.token_data['access_token']}",
            "Ocp-Apim-Subscription-Key": self.subscription_key,
            "Accept": "application/json"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.BASE_URL}{endpoint}",
                params=params,
                headers=headers,
                timeout=30.0
            )
            
            # Log full response details for debugging
            if response.status_code != 200:
                print(f"Error response status: {response.status_code}")
                print(f"Error response headers: {dict(response.headers)}")
                print(f"Error response body: {response.text}")
            
            response.raise_for_status()
            return response.json()
    
    async def get_dam_spp_prices(
        self, 
        start_date: datetime, 
        end_date: datetime,
        settlement_points: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Get Day-Ahead Market Settlement Point Prices."""
        if start_date < WEBSERVICE_CUTOFF_DATE:
            raise ValueError(f"Web service only available after {WEBSERVICE_CUTOFF_DATE}")
        
        # Use the working endpoint
        endpoint = "/api/public-reports/np4-183-cd/dam_hourly_lmp"
        all_data = []
        
        # Process in daily chunks - get all data and filter client-side if needed
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            
            params = {
                "deliveryDateFrom": date_str,
                "deliveryDateTo": date_str
            }
            
            try:
                response = await self._make_request(endpoint, params)
                if isinstance(response, dict) and "data" in response:
                    raw_data = response["data"]
                    
                    # Convert array format to dictionary format
                    for record in raw_data:
                        if len(record) >= 5:  # Ensure we have all fields
                            record_dict = {
                                "delivery_date": record[0],
                                "hour_ending": record[1], 
                                "bus_name": record[2],
                                "lmp": record[3],
                                "dst_flag": record[4]
                            }
                            all_data.append(record_dict)
                            
            except Exception as e:
                print(f"Error fetching DAM LMP for {date_str}: {str(e)}")
            
            current_date += timedelta(days=1)
        
        if all_data:
            df = pd.DataFrame(all_data)
            
            # Filter by settlement points if specified
            if settlement_points:
                # Convert to set for faster lookup
                points_set = set(settlement_points)
                df = df[df['bus_name'].isin(points_set)]
            
            # Rename columns to match standard format
            df.rename(columns={
                "delivery_date": "delivery_date",
                "hour_ending": "hour_ending",
                "bus_name": "settlement_point", 
                "lmp": "spp",
                "dst_flag": "dst_flag"
            }, inplace=True)
            
            return df
        
        return pd.DataFrame()
    
    async def get_rtm_spp_prices(
        self,
        start_date: datetime,
        end_date: datetime,
        settlement_points: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Get Real-Time Market Settlement Point Prices."""
        if start_date < WEBSERVICE_CUTOFF_DATE:
            raise ValueError(f"Web service only available after {WEBSERVICE_CUTOFF_DATE}")
        
        endpoint = "/api/public/rtm-lmp"
        all_data = []
        
        if not settlement_points:
            settlement_points = TRADING_HUBS
        
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            
            for sp in settlement_points:
                # RTM has 15-minute intervals
                for interval in range(1, 97):  # 96 15-minute intervals per day
                    params = {
                        "deliveryDate": date_str,
                        "deliveryInterval": str(interval),
                        "settlementPoint": sp
                    }
                    
                    try:
                        data = await self._make_request(endpoint, params)
                        if "data" in data:
                            all_data.extend(data["data"])
                    except Exception as e:
                        print(f"Error fetching RTM SPP for {sp} on {date_str} interval {interval}: {str(e)}")
            
            current_date += timedelta(days=1)
        
        if all_data:
            df = pd.DataFrame(all_data)
            df.rename(columns={
                "deliveryDate": "delivery_date",
                "deliveryInterval": "interval",
                "settlementPoint": "settlement_point",
                "settlementPointPrice": "spp",
                "dstFlag": "dst_flag"
            }, inplace=True)
            return df
        
        return pd.DataFrame()
    
    async def get_ancillary_prices(
        self,
        service_type: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Get ancillary service prices."""
        if start_date < WEBSERVICE_CUTOFF_DATE:
            raise ValueError(f"Web service only available after {WEBSERVICE_CUTOFF_DATE}")
        
        # Map service types to API endpoints
        service_endpoints = {
            "REGUP": "/api/public/ancillary-service-reg-up",
            "REGDN": "/api/public/ancillary-service-reg-down",
            "SPIN": "/api/public/ancillary-service-spin",
            "NON_SPIN": "/api/public/ancillary-service-non-spin",
            "RRS": "/api/public/ancillary-service-rrs",
            "ECRS": "/api/public/ancillary-service-ecrs"
        }
        
        if service_type not in service_endpoints:
            raise ValueError(f"Unknown service type: {service_type}")
        
        endpoint = service_endpoints[service_type]
        all_data = []
        
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            params = {"deliveryDate": date_str}
            
            try:
                data = await self._make_request(endpoint, params)
                if "data" in data:
                    all_data.extend(data["data"])
            except Exception as e:
                print(f"Error fetching {service_type} prices for {date_str}: {str(e)}")
            
            current_date += timedelta(days=1)
        
        if all_data:
            df = pd.DataFrame(all_data)
            df["service_type"] = service_type
            return df
        
        return pd.DataFrame()
    
    async def test_connection(self) -> bool:
        """Test the API connection and credentials."""
        try:
            # Try different endpoints
            endpoints_to_try = [
                ("/api/public/dam-lmp", {
                    "deliveryDate": datetime.now().strftime("%Y-%m-%d"),
                    "settlementPoint": "HB_BUSAVG"
                }),
                ("/api/public-reports/np4-190-cd/dam_stlmnt_pnt_prices", {
                    "deliveryDateFrom": datetime.now().strftime("%Y-%m-%d"),
                    "deliveryDateTo": datetime.now().strftime("%Y-%m-%d"),
                    "settlementPointName": "HB_BUSAVG"
                }),
                ("/api/public-reports/np4-183-cd/dam_hourly_lmp", {
                    "busName": "HB_BUSAVG",
                    "deliveryDateFrom": datetime.now().strftime("%Y-%m-%d"),
                    "deliveryDateTo": datetime.now().strftime("%Y-%m-%d")
                })
            ]
            
            for endpoint, params in endpoints_to_try:
                try:
                    print(f"Trying endpoint: {endpoint}")
                    await self._make_request(endpoint, params)
                    print(f"Success with endpoint: {endpoint}")
                    return True
                except Exception as e:
                    print(f"Failed {endpoint}: {str(e)}")
                    continue
                    
            return False
        except Exception as e:
            print(f"Connection test failed: {str(e)}")
            return False