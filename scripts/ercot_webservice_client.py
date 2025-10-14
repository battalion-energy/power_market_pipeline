from datetime import datetime, date, timedelta
import requests
from typing import List, Optional, Union, Dict, Any 
from dateutil.relativedelta import relativedelta
import time
import json  # For pretty printing JSON responses

from ercot_webservices.ercot_api.client import Client
from ercot_webservices.ercot_api.models.exception import Exception_
from ercot_webservices.ercot_api.models.report import Report
from ercot_webservices.ercot_api.api.np4_190_cd import get_data_dam_stlmnt_pnt_prices
from settings import (
    ERCOT_USERNAME,
    ERCOT_PASSWORD,
    ERCOT_SUBSCRIPTION_KEY,
    ERCOT_BASE_URL
)


class ERCOTWebServiceClient:
    """Client for interacting with ERCOT Web Services API."""
    
    # ERCOT Trading Hubs
    TRADING_HUBS = ["HB_BUSAVG", "HB_HOUSTON", "HB_NORTH", "HB_SOUTH", "HB_WEST"]
    
    # API Constraints
    API_EARLIEST_DATE = date(2023, 12, 11)  # Earliest available date per API limits
    RATE_LIMIT_WAIT = 2  # seconds between requests (30 requests/minute limit)
    MAX_FILES_PER_REQUEST = 1000
    
    def __init__(
        self, 
        username: str = ERCOT_USERNAME,
        password: str = ERCOT_PASSWORD,
        subscription_key: str = ERCOT_SUBSCRIPTION_KEY,
        base_url: str = ERCOT_BASE_URL
    ):
        """Initialize the ERCOT WebService Client."""
        if not all([username, password, subscription_key]):
            missing = []
            if not username:
                missing.append("ERCOT_USERNAME")
            if not password:
                missing.append("ERCOT_PASSWORD")
            if not subscription_key:
                missing.append("ERCOT_SUBSCRIPTION_KEY")
            raise ValueError(
                f"Missing required credentials: {', '.join(missing)}. "
                "Please set them in your .env file."
            )
        
        self.username = username
        self.password = password
        self.subscription_key = subscription_key
        self.base_url = base_url
        self.client = Client(base_url=base_url)
        self.token_data = None
        self.last_request_time = time.time()
        
        # Get initial token
        self.authenticate()
    
    def authenticate(self) -> Dict[str, Any]:
        """Authenticate with ERCOT and get an access token."""
        # Authentication URL
        auth_url = "https://ercotb2c.b2clogin.com/ercotb2c.onmicrosoft.com/B2C_1_PUBAPI-ROPC-FLOW/oauth2/v2.0/token"
        
        # Required parameters
        params = {
            "username": self.username,
            "password": self.password,
            "grant_type": "password",
            "scope": "openid fec253ea-0d06-4272-a5e6-b478baeecd70 offline_access",
            "client_id": "fec253ea-0d06-4272-a5e6-b478baeecd70",
            "response_type": "token"
        }
        
        # Make the request
        response = requests.post(auth_url, data=params)
        
        # Check if request was successful
        if response.status_code != 200:
            raise Exception(f"Authentication failed: {response.status_code} - {response.text}")
        
        # Parse the response
        self.token_data = response.json()
        
        # Print full token details for Postman debugging
        if self.token_data:
            print("Authentication successful. Token details:")
            if 'access_token' in self.token_data:
                print("\n=== FULL ACCESS TOKEN FOR POSTMAN (copy this) ===")
                print(self.token_data['access_token'])
                print("=== END OF ACCESS TOKEN ===\n")
            if 'id_token' in self.token_data:
                print("\n=== FULL ID TOKEN FOR POSTMAN (try this if access token fails) ===")
                print(self.token_data['id_token'])
                print("=== END OF ID TOKEN ===\n")
            
            print("Other token data:")
            for key in self.token_data:
                if key not in ['access_token', 'id_token']:
                    print(f"  {key}: {self.token_data[key]}")
        
        return self.token_data
    
    def get_headers(self) -> Dict[str, str]:
        """Get headers for API requests including authorization token."""
        if not self.token_data or ('access_token' not in self.token_data and 'id_token' not in self.token_data):
            self.authenticate()
        
        # Try access_token first, fall back to id_token if not available
        token = self.token_data.get('access_token', self.token_data.get('id_token', ''))
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Ocp-Apim-Subscription-Key": self.subscription_key,
            "Content-Type": "application/json",  # Add content type header
            "Accept": "application/json"         # Add accept header
        }
        
        return headers
    
    def _respect_rate_limit(self):
        """Ensure we don't exceed API rate limits."""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        if elapsed < self.RATE_LIMIT_WAIT:
            time.sleep(self.RATE_LIMIT_WAIT - elapsed)
        self.last_request_time = time.time()

    def get_dam_settlement_point_prices(
        self,
        start_date: date,
        end_date: Optional[date] = None,
        settlement_points: Optional[List[str]] = None,
        page_size: int = 1000
    ) -> List[dict]:
        """Get DAM Settlement Point Prices for specified trading hubs and date range."""
        # Ensure we don't request data before the API's earliest available date
        start_date = max(start_date, self.API_EARLIEST_DATE)
        
        if end_date is None:
            end_date = date.today()
        
        if settlement_points is None:
            settlement_points = self.TRADING_HUBS
        
        all_results = []
        
        # Force token refresh at the beginning
        print("Refreshing authentication token before starting requests...")
        self.authenticate()
        
        # Print the full subscription key for Postman testing
        print("\n=== SUBSCRIPTION KEY FOR POSTMAN ===")
        print(self.subscription_key)
        print("=== END OF SUBSCRIPTION KEY ===\n")
        
        print(f"Base URL: {self.base_url}")
        
        # Process data in small chunks (3 days at a time)
        chunk_size = 3  # days
        current_start = start_date
        
        # Try both endpoint patterns
        endpoints_to_try = [
            "api/public-reports/np4-183-cd/dam_hourly_lmp",
            "np4-190-cd/dam_stlmnt_pnt_prices"
        ]
        
        # Try different parameter variations
        param_variations = [
            # For dam_hourly_lmp endpoint
            {
                "dam_hourly_lmp": [
                    {"busName": "HB_HOUSTON"},
                    {"busName": "HB_NORTH"},
                    {"busName": "HB_SOUTH"},
                    {"busName": "HB_WEST"},
                    {"busName": "HB_BUSAVG"},
                    {"settlementPoint": "HB_BUSAVG"},
                    {"settlementPoint": "HB_HOUSTON"},
                    # Try other variants
                    {"busName": "BUSAVG_HUB"}, 
                    {"busName": "HOUSTON_HUB"},
                    # Try without bus name
                    {}
                ]
            },
            # For dam_stlmnt_pnt_prices endpoint
            {
                "dam_stlmnt_pnt_prices": [
                    {"busName": "HB_HOUSTON"},
                    {"busName": "HB_NORTH"},
                    {"busName": "HB_SOUTH"},
                    {"busName": "HB_WEST"},
                    {"busName": "HB_BUSAVG"},
                    # Try without settlement point
                    {}
                ]
            }
        ]
        
        for endpoint in endpoints_to_try:
            print(f"\nTrying endpoint: {endpoint}")
            
            # Determine which parameter variations to use based on endpoint
            param_set = next((v[endpoint.split('/')[-1]] for v in param_variations if endpoint.split('/')[-1] in v), [{}])
            
            for param_variation in param_set:
                print(f"\nTrying parameter variation: {param_variation}")
                
                # Reset to beginning of date range
                current_start = start_date
                
                # Try a single day first to test
                test_end = start_date
                
                delivery_date_from = current_start.strftime("%Y-%m-%d")
                delivery_date_to = test_end.strftime("%Y-%m-%d")
                
                print(f"Test query for single day: {delivery_date_from}")
                
                # Construct the URL
                url = f"{self.base_url.rstrip('/')}/{endpoint}"
                
                # Get fresh headers
                headers = self.get_headers()
                
                # Base parameters
                params = {
                    "deliveryDateFrom": delivery_date_from,
                    "deliveryDateTo": delivery_date_to,
                    "page": 1,
                    "size": page_size
                }
                
                # Add parameter variation
                params.update(param_variation)
                
                print(f"Complete test request:")
                print(f"URL: {url}")
                print(f"Headers: {headers}")
                print(f"Params: {params}")
                
                try:
                    print("Making test request...")
                    response = requests.get(url, headers=headers, params=params)
                    
                    print(f"Response status: {response.status_code}")
                    
                    # Always print the full JSON response for debugging
                    try:
                        response_json = response.json()
                        print("\n=== FULL JSON RESPONSE ===")
                        print(json.dumps(response_json, indent=2))
                        print("=== END OF JSON RESPONSE ===\n")
                        
                        # Check for empty data but successful response
                        if response.status_code == 200:
                            if isinstance(response_json, dict):
                                if "data" in response_json and isinstance(response_json["data"], list):
                                    if len(response_json["data"]) == 0:
                                        print("API returned successfully but no records found for these parameters.")
                                        
                                        # Check the metadata for clues
                                        if "_meta" in response_json:
                                            print("Metadata information:")
                                            print(json.dumps(response_json["_meta"], indent=2))
                                        
                                        # If there are fields, print them to understand the API structure
                                        if "fields" in response_json:
                                            print("\nAvailable fields:")
                                            for field in response_json["fields"]:
                                                print(f"  {field['name']} ({field['label']}): {field['dataType']}")
                                    else:
                                        print(f"Success! Found {len(response_json['data'])} records.")
                                        # This is a working combination, proceed with full date range
                                        
                                        # Process the full date range with working parameters
                                        return self._process_full_date_range(
                                            start_date, end_date, chunk_size, url, headers, params
                                        )
                                else:
                                    print("Response does not contain a 'data' array.")
                            elif isinstance(response_json, list) and len(response_json) > 0:
                                print(f"Success! Found {len(response_json)} records.")
                                # This is a working combination, proceed with full date range
                                
                                # Process the full date range with working parameters
                                return self._process_full_date_range(
                                    start_date, end_date, chunk_size, url, headers, params
                                )
                    except Exception as e:
                        print(f"Error parsing JSON response: {str(e)}")
                    
                except Exception as e:
                    print(f"Request error: {str(e)}")
        
        print("All endpoint and parameter combinations failed or returned no data.")
        
        # Suggestions for manual testing
        print("\nSuggestions for troubleshooting:")
        print("1. Check ERCOT documentation for the exact endpoint and parameter names")
        print("2. Verify if data is available for the specified date range")
        print("3. Try different settlement point / busName values")
        print("4. Verify your API credentials and subscription key")
        print("5. Check if the API requires additional parameters")
        print("6. Try with a more recent date range (e.g., last week)")
        
        return all_results

    def _process_full_date_range(
        self, 
        start_date: date, 
        end_date: date, 
        chunk_size: int,
        url: str,
        headers: dict,
        params: dict
    ) -> List[dict]:
        """Process the full date range with working parameters."""
        all_results = []
        current_start = start_date
        
        print(f"\nProcessing full date range from {start_date} to {end_date} with working parameters")
        
        while current_start <= end_date:
            # Calculate end of current chunk
            current_end = min(
                current_start + timedelta(days=chunk_size-1),
                end_date
            )
            
            print(f"Fetching data for period: {current_start} to {current_end}")
            
            # Update date parameters
            params["deliveryDateFrom"] = current_start.strftime("%Y-%m-%d")
            params["deliveryDateTo"] = current_end.strftime("%Y-%m-%d")
            params["page"] = 1  # Reset page for new date range
            
            page = 1
            while True:
                self._respect_rate_limit()
                
                try:
                    print(f"  Making request (page {page})...")
                    response = requests.get(url, headers=headers, params=params)
                    
                    if response.status_code != 200:
                        print(f"  API Error: {response.status_code} - {response.text}")
                        break
                    
                    data = response.json()
                    
                    # Extract data based on response structure
                    if isinstance(data, dict) and "data" in data:
                        current_batch = data["data"]
                        
                        # Check if we're on the last page
                        if "_meta" in data and "totalPages" in data["_meta"]:
                            total_pages = data["_meta"]["totalPages"]
                            current_page = data["_meta"]["currentPage"]
                            print(f"  Page {current_page} of {total_pages}")
                        else:
                            # If no pagination info, use length of batch
                            total_pages = 1 if len(current_batch) < params["size"] else page + 1
                    elif isinstance(data, list):
                        current_batch = data
                        # If response is a list, use length to determine if there are more pages
                        total_pages = 1 if len(current_batch) < params["size"] else page + 1
                    else:
                        print("  Unknown response format")
                        break
                    
                    if not current_batch:
                        print("  No data for this page")
                        break
                    
                    all_results.extend(current_batch)
                    print(f"  Retrieved {len(current_batch)} records")
                    
                    # Check if we need to fetch more pages
                    if page >= total_pages or len(current_batch) < params["size"]:
                        break
                        
                    # Move to next page
                    page += 1
                    params["page"] = page
                    
                except Exception as e:
                    print(f"  Request error: {str(e)}")
                    break
            
            # Move to next chunk
            current_start = current_end + timedelta(days=1)
            print(f"Completed chunk. Cumulative records so far: {len(all_results)}")
            print("Pausing between chunks...")
            time.sleep(3)
        
        print(f"Data retrieval complete. Total records: {len(all_results)}")
        return all_results

    def get_all_trading_hub_prices(self, start_date_str: str = "2022-01-01") -> List[dict]:
        """Convenience method to get prices for all trading hubs since a given date."""
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
        return self.get_dam_settlement_point_prices(
            start_date=start_date,
            settlement_points=self.TRADING_HUBS
        )

def save_dam_data_to_csv(data: List[dict], output_dir: str, year: int):
    """Save DAM data to CSV file by year.
    
    Args:
        data: List of DAM price records
        output_dir: Directory to save CSV files
        year: Year of the data
    """
    import pandas as pd
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Format the delivery date column
    df['deliveryDate'] = pd.to_datetime(df['deliveryDate'])
    
    # Save to CSV
    output_file = os.path.join(output_dir, f'dam_prices_{year}.csv')
    df.to_csv(output_file, index=False)
    print(f"Saved data for {year} to {output_file}")

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    import pandas as pd
    import time
    
    # Load environment variables
    load_dotenv()
    
    # Initialize client
    client = ERCOTWebServiceClient()
    
    # Set date range (respecting API limitations)
    #start_date = date(2023, 12, 11)  # API's earliest available date
    start_date = date(2024, 1, 1)  # API's earliest available date
    end_date = date.today()
    
    try:
        print(f"Fetching DAM prices from {start_date} to {end_date}...")
        data = client.get_dam_settlement_point_prices(
            start_date=start_date,
            end_date=end_date
        )
        
        if data:
            # Create output directory
            output_dir = "dam_price_data"
            os.makedirs(output_dir, exist_ok=True)
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            df['deliveryDate'] = pd.to_datetime(df['deliveryDate'])
            
            # Save monthly file
            output_file = os.path.join(
                output_dir, 
                f"dam_prices_{start_date.strftime('%Y_%m')}_to_{end_date.strftime('%Y_%m')}.csv"
            )
            df.to_csv(output_file, index=False)
            
            print(f"\nSuccessfully saved DAM price data to {output_file}")
            print(f"Total records: {len(data)}")
        else:
            print("No data returned for the specified period")
            
    except Exception as e:
        print(f"Error: {str(e)}")
