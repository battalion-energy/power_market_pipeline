"""ERCOT data processor for standardizing and cleaning data."""

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import pytz


class ERCOTProcessor:
    """Process and standardize ERCOT data files."""
    
    def __init__(self):
        self.tz = pytz.timezone('US/Central')
    
    def process_energy_file(self, file_path: Path, market_type: str) -> pd.DataFrame:
        """Process a raw ERCOT energy price file."""
        # ERCOT files can have different formats
        # Try different read methods
        try:
            df = pd.read_csv(file_path)
        except:
            # Try Excel format
            df = pd.read_excel(file_path)
        
        # Standardize column names based on common ERCOT patterns
        column_mapping = {
            'Delivery Date': 'delivery_date',
            'DeliveryDate': 'delivery_date',
            'Delivery Hour': 'hour_ending',
            'DeliveryHour': 'hour_ending',
            'Hour Ending': 'hour_ending',
            'Delivery Interval': 'interval',
            'DeliveryInterval': 'interval',
            'Settlement Point': 'settlement_point',
            'SettlementPoint': 'settlement_point',
            'Settlement Point Name': 'settlement_point_name',
            'SettlementPointName': 'settlement_point_name',
            'Settlement Point Price': 'spp',
            'SettlementPointPrice': 'spp',
            'SPP': 'spp',
            'LMP': 'lmp',
            'Energy': 'energy_component',
            'Congestion': 'congestion_component',
            'Losses': 'loss_component'
        }
        
        # Rename columns
        df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns}, inplace=True)
        
        # Add market type
        df['market_type'] = market_type
        
        # Convert to timestamp
        df = self._create_timestamp(df, market_type)
        
        # Clean numeric columns
        numeric_columns = ['spp', 'lmp', 'energy_component', 'congestion_component', 'loss_component']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def _create_timestamp(self, df: pd.DataFrame, market_type: str) -> pd.DataFrame:
        """Create proper timestamp from date and hour/interval columns."""
        if 'delivery_date' in df.columns:
            # Convert delivery date to datetime
            df['delivery_date'] = pd.to_datetime(df['delivery_date'])
            
            if market_type == 'DAM' and 'hour_ending' in df.columns:
                # For DAM, hour ending represents the end of the hour
                # Hour 1 = 00:00-01:00, so timestamp is 01:00
                df['timestamp'] = df.apply(
                    lambda row: self.tz.localize(
                        row['delivery_date'].replace(hour=int(row['hour_ending']) % 24)
                    ),
                    axis=1
                )
            elif market_type == 'RTM' and 'interval' in df.columns:
                # For RTM, intervals are 15-minute periods
                # Interval 1 = 00:00-00:15, Interval 2 = 00:15-00:30, etc.
                df['timestamp'] = df.apply(
                    lambda row: self.tz.localize(
                        row['delivery_date'] + pd.Timedelta(minutes=(row['interval'] - 1) * 15)
                    ),
                    axis=1
                )
            else:
                # Fallback to just the date
                df['timestamp'] = df['delivery_date'].apply(lambda x: self.tz.localize(x))
        
        return df
    
    def standardize_energy_data(self, df: pd.DataFrame, market_type: str) -> pd.DataFrame:
        """Standardize energy data from web service format."""
        # Web service data is already fairly standardized
        df['market_type'] = market_type
        
        # Ensure timestamp
        if 'timestamp' not in df.columns:
            df = self._create_timestamp(df, market_type)
        
        # Ensure numeric types
        numeric_columns = ['spp', 'lmp', 'energy_component', 'congestion_component', 'loss_component']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def process_ancillary_file(self, file_path: Path, service_type: str, market_type: str) -> pd.DataFrame:
        """Process a raw ERCOT ancillary service file."""
        df = pd.read_csv(file_path)
        
        # Standardize column names
        column_mapping = {
            'Delivery Date': 'delivery_date',
            'DeliveryDate': 'delivery_date',
            'Hour Ending': 'hour_ending',
            'HourEnding': 'hour_ending',
            'MCPC': 'price',
            'Market Clearing Price': 'price',
            'MarketClearingPrice': 'price',
            'MW': 'quantity_mw',
            'Quantity': 'quantity_mw'
        }
        
        df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns}, inplace=True)
        
        # Add service and market type
        df['service_type'] = service_type
        df['market_type'] = market_type
        
        # Create timestamp
        df = self._create_timestamp(df, market_type)
        
        # Clean numeric columns
        if 'price' in df.columns:
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
        if 'quantity_mw' in df.columns:
            df['quantity_mw'] = pd.to_numeric(df['quantity_mw'], errors='coerce')
        
        return df