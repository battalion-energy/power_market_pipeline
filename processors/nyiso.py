"""NYISO data processor for standardizing and cleaning data."""

from datetime import datetime
from typing import Optional

import pandas as pd
import pytz

from downloaders.nyiso.constants import COLUMN_MAPPINGS


class NYISOProcessor:
    """Process and standardize NYISO data."""
    
    def __init__(self):
        self.tz = pytz.timezone('US/Eastern')
    
    def process_energy_data(
        self, 
        df: pd.DataFrame, 
        market_type: str, 
        node_type: str
    ) -> pd.DataFrame:
        """Process NYISO energy price data."""
        # Apply column mappings
        df.rename(columns=COLUMN_MAPPINGS["LBMP"], inplace=True)
        
        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['timestamp'] = df['timestamp'].dt.tz_localize(self.tz, ambiguous='NaT')
        
        # Add market type
        df['market_type'] = market_type
        
        # Calculate energy component (LMP - congestion - losses)
        df['energy_component'] = (
            df['lmp'] - 
            df.get('congestion_component', 0) - 
            df.get('loss_component', 0)
        )
        
        # Clean numeric columns
        numeric_columns = ['lmp', 'energy_component', 'congestion_component', 'loss_component']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Add location name for readability
        if 'location' in df.columns:
            df['location_name'] = df['location'].apply(self._format_location_name)
        
        # Select final columns
        final_columns = [
            'timestamp', 'location', 'location_name', 'ptid', 
            'market_type', 'lmp', 'energy_component', 
            'congestion_component', 'loss_component'
        ]
        final_columns = [col for col in final_columns if col in df.columns]
        
        return df[final_columns]
    
    def process_regulation_data(
        self, 
        df: pd.DataFrame, 
        market_type: str
    ) -> pd.DataFrame:
        """Process NYISO regulation (ASP) data."""
        # Apply column mappings
        df.rename(columns=COLUMN_MAPPINGS["ASP"], inplace=True)
        
        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['timestamp'] = df['timestamp'].dt.tz_localize(self.tz, ambiguous='NaT')
        
        # Create separate records for each service type
        records = []
        
        for _, row in df.iterrows():
            base_record = {
                'timestamp': row['timestamp'],
                'market_type': market_type,
                'zone': 'NYISO'
            }
            
            # Regulation capacity
            if 'reg_capacity_price' in row and pd.notna(row['reg_capacity_price']):
                records.append({
                    **base_record,
                    'service_type': 'REG_CAPACITY',
                    'price': float(row['reg_capacity_price']),
                    'quantity_mw': None
                })
            
            # Regulation movement
            if 'reg_movement_price' in row and pd.notna(row['reg_movement_price']):
                records.append({
                    **base_record,
                    'service_type': 'REG_MOVEMENT',
                    'price': float(row['reg_movement_price']),
                    'quantity_mw': None
                })
            
            # Regulation service (combined)
            if 'reg_service_price' in row and pd.notna(row['reg_service_price']):
                records.append({
                    **base_record,
                    'service_type': 'REG',
                    'price': float(row['reg_service_price']),
                    'quantity_mw': None
                })
        
        return pd.DataFrame(records)
    
    def process_reserve_data(
        self,
        df: pd.DataFrame,
        service_type: str,
        market_type: str
    ) -> pd.DataFrame:
        """Process NYISO reserve price data."""
        # Apply column mappings
        df.rename(columns=COLUMN_MAPPINGS["RESERVE"], inplace=True)
        
        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['timestamp'] = df['timestamp'].dt.tz_localize(self.tz, ambiguous='NaT')
        
        # Create records for specific service type
        records = []
        
        # Map service types to column names
        service_column_map = {
            'SPIN_10': 'spin_10_price',
            'NON_SYNC_10': 'non_sync_10_price',
            'OPER_30': 'oper_30_price'
        }
        
        if service_type not in service_column_map:
            return pd.DataFrame()
        
        price_column = service_column_map[service_type]
        
        for _, row in df.iterrows():
            if price_column in row and pd.notna(row[price_column]):
                records.append({
                    'timestamp': row['timestamp'],
                    'service_type': service_type,
                    'market_type': market_type,
                    'zone': row.get('zone', 'NYISO'),
                    'price': float(row[price_column]),
                    'quantity_mw': None
                })
        
        return pd.DataFrame(records)
    
    def _format_location_name(self, location: str) -> str:
        """Format location code into readable name."""
        # Clean up common patterns
        if location == "NYISO":
            return "NYISO Reference Bus"
        
        # Title case and remove underscores
        name = location.replace("_", " ").title()
        
        # Fix common abbreviations
        replacements = {
            "N.Y.C.": "New York City",
            "Hud Vl": "Hudson Valley",
            "Mhk Vl": "Mohawk Valley",
            "Capitl": "Capital",
            "Centrl": "Central",
            "Genese": "Genesee",
            "Longil": "Long Island",
            "Millwd": "Millwood",
            "Dunwod": "Dunwoodie"
        }
        
        for old, new in replacements.items():
            if old in name:
                name = name.replace(old, new)
        
        return name