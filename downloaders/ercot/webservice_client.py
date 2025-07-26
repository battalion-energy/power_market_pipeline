"""ERCOT Web Service API client for recent data (after Dec 11, 2023)."""

import asyncio
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import httpx
import pandas as pd
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

from .constants import TRADING_HUBS, WEBSERVICE_CUTOFF_DATE

load_dotenv()


class ERCOTWebServiceClient:
    """Client for ERCOT's official REST API."""
    
    BASE_URL = "https://api.ercot.com"
    
    def __init__(self):
        self.username = os.getenv("ERCOT_USERNAME")
        self.password = os.getenv("ERCOT_PASSWORD")
        self.subscription_key = os.getenv("ERCOT_SUBSCRIPTION_KEY")
        
        if not all([self.username, self.password, self.subscription_key]):
            raise ValueError("ERCOT API credentials not found in environment variables")
        
        self.rate_limit_delay = 2  # seconds between requests
        self.last_request_time = 0
    
    async def _rate_limit(self):
        """Implement rate limiting to avoid API throttling."""
        current_time = asyncio.get_event_loop().time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)
        
        self.last_request_time = asyncio.get_event_loop().time()
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def _make_request(
        self, 
        endpoint: str, 
        params: Optional[Dict] = None
    ) -> Dict:
        """Make an authenticated request to ERCOT API."""
        await self._rate_limit()
        
        headers = {
            "Ocp-Apim-Subscription-Key": self.subscription_key,
            "Accept": "application/json"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.BASE_URL}{endpoint}",
                params=params,
                headers=headers,
                auth=(self.username, self.password),
                timeout=30.0
            )
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
        
        endpoint = "/api/public/dam-lmp"
        all_data = []
        
        # If no specific points requested, use trading hubs
        if not settlement_points:
            settlement_points = TRADING_HUBS
        
        # Process in daily chunks
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            
            for sp in settlement_points:
                params = {
                    "deliveryDate": date_str,
                    "settlementPoint": sp
                }
                
                try:
                    data = await self._make_request(endpoint, params)
                    if "data" in data:
                        all_data.extend(data["data"])
                except Exception as e:
                    print(f"Error fetching DAM SPP for {sp} on {date_str}: {str(e)}")
            
            current_date += timedelta(days=1)
        
        if all_data:
            df = pd.DataFrame(all_data)
            # Standardize column names
            df.rename(columns={
                "deliveryDate": "delivery_date",
                "deliveryHour": "hour_ending",
                "settlementPoint": "settlement_point",
                "settlementPointPrice": "spp",
                "dstFlag": "dst_flag"
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
            # Try a simple request
            endpoint = "/api/public/dam-lmp"
            params = {
                "deliveryDate": datetime.now().strftime("%Y-%m-%d"),
                "settlementPoint": "HB_BUSAVG"
            }
            await self._make_request(endpoint, params)
            return True
        except Exception as e:
            print(f"Connection test failed: {str(e)}")
            return False