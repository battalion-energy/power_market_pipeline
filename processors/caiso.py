"""CAISO data processor for standardizing and cleaning data."""

from datetime import datetime
from typing import Optional

import pandas as pd
import pytz


class CAISOProcessor:
    """Process and standardize CAISO data files."""
    
    def __init__(self):
        self.tz = pytz.timezone('US/Pacific')
    
    def process_energy_data(self, df: pd.DataFrame, market_type: str) -> pd.DataFrame:
        """Process CAISO energy price data."""
        # CAISO OASIS column mappings
        column_mapping = {
            'INTERVALSTARTTIME_GMT': 'timestamp_gmt',
            'INTERVALENDTIME_GMT': 'timestamp_end_gmt',
            'OPR_DT': 'operating_date',
            'OPR_HR': 'operating_hour',
            'OPR_INTERVAL': 'operating_interval',
            'NODE_ID': 'node',
            'NODE': 'node',
            'NODE_ID_XML': 'node',
            'MARKET_RUN_ID': 'market_run',
            'LMP_TYPE': 'lmp_type',
            'XML_DATA_ITEM': 'data_item',
            'PNODE_RESMRID': 'resource_id',
            'GRPTYPE': 'group_type',
            'POS': 'position',
            'MW': 'mw',  # Energy component
            'MCC': 'mcc',  # Congestion component
            'MLC': 'mlc',  # Loss component
            'VALUE': 'value',
            'LMP': 'lmp'
        }
        
        # Rename columns
        df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns}, inplace=True)
        
        # Filter for LMP data
        if 'lmp_type' in df.columns:
            df = df[df['lmp_type'] == 'LMP'].copy()
        
        # Pivot if data is in long format
        if 'data_item' in df.columns and 'value' in df.columns:
            # Identify the time and node columns
            id_vars = ['timestamp_gmt', 'node', 'operating_date', 'operating_hour']
            id_vars = [col for col in id_vars if col in df.columns]
            
            # Pivot data items to columns
            pivot_df = df.pivot_table(
                index=id_vars,
                columns='data_item',
                values='value',
                aggfunc='first'
            ).reset_index()
            
            # Rename pivoted columns
            pivot_mapping = {
                'LMP_PRC': 'lmp',
                'LMP_CONG_PRC': 'mcc',
                'LMP_LOSS_PRC': 'mlc',
                'LMP_ENE_PRC': 'mw'
            }
            pivot_df.rename(columns=pivot_mapping, inplace=True)
            df = pivot_df
        
        # Add market type
        df['market_type'] = market_type
        
        # Convert timestamp
        if 'timestamp_gmt' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp_gmt']).dt.tz_localize('UTC')
            df['timestamp'] = df['timestamp'].dt.tz_convert(self.tz)
        elif 'operating_date' in df.columns and 'operating_hour' in df.columns:
            # Reconstruct timestamp from date and hour
            df['timestamp'] = pd.to_datetime(df['operating_date'])
            df['timestamp'] = df.apply(
                lambda row: self.tz.localize(
                    row['timestamp'].replace(hour=int(row['operating_hour']) - 1)
                ),
                axis=1
            )
        
        # Clean numeric columns
        numeric_columns = ['lmp', 'mw', 'mcc', 'mlc']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Select final columns
        final_columns = ['timestamp', 'node', 'market_type', 'lmp', 'mw', 'mcc', 'mlc']
        final_columns = [col for col in final_columns if col in df.columns]
        
        return df[final_columns]
    
    def process_ancillary_data(
        self, 
        df: pd.DataFrame, 
        service_type: str, 
        market_type: str
    ) -> pd.DataFrame:
        """Process CAISO ancillary service data."""
        # Column mappings for AS data
        column_mapping = {
            'INTERVALSTARTTIME_GMT': 'timestamp_gmt',
            'OPR_DT': 'operating_date',
            'OPR_HR': 'operating_hour',
            'OPR_INTERVAL': 'operating_interval',
            'MARKET_RUN_ID': 'market_run',
            'AS_TYPE': 'as_type',
            'AS_REGION': 'region',
            'AS_MW': 'quantity_mw',
            'AS_CLEARING_PRC': 'price',
            'MW_AWARDED': 'quantity_mw',
            'ASMP_PRC': 'price',
            'XML_DATA_ITEM': 'data_item',
            'VALUE': 'value'
        }
        
        # Rename columns
        df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns}, inplace=True)
        
        # Filter for specific service type if as_type column exists
        service_type_mapping = {
            'SPIN': 'SP',
            'NON_SPIN': 'NS',
            'REG_UP': 'RU',
            'REG_DN': 'RD',
            'REG_MILEAGE_UP': 'RMU',
            'REG_MILEAGE_DN': 'RMD'
        }
        
        if 'as_type' in df.columns and service_type in service_type_mapping:
            df = df[df['as_type'] == service_type_mapping[service_type]].copy()
        
        # Pivot if needed
        if 'data_item' in df.columns and 'value' in df.columns:
            id_vars = ['timestamp_gmt', 'region', 'operating_date', 'operating_hour']
            id_vars = [col for col in id_vars if col in df.columns]
            
            pivot_df = df.pivot_table(
                index=id_vars,
                columns='data_item',
                values='value',
                aggfunc='first'
            ).reset_index()
            
            # Map pivoted columns
            as_pivot_mapping = {
                'AS_CLEARING_PRC': 'price',
                'AS_MW_CLEARED': 'quantity_mw',
                'AS_COST': 'total_cost'
            }
            pivot_df.rename(columns=as_pivot_mapping, inplace=True)
            df = pivot_df
        
        # Add service and market type
        df['service_type'] = service_type
        df['market_type'] = market_type
        
        # Convert timestamp
        if 'timestamp_gmt' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp_gmt']).dt.tz_localize('UTC')
            df['timestamp'] = df['timestamp'].dt.tz_convert(self.tz)
        elif 'operating_date' in df.columns and 'operating_hour' in df.columns:
            df['timestamp'] = pd.to_datetime(df['operating_date'])
            df['timestamp'] = df.apply(
                lambda row: self.tz.localize(
                    row['timestamp'].replace(hour=int(row['operating_hour']) - 1)
                ),
                axis=1
            )
        
        # Clean numeric columns
        numeric_columns = ['price', 'quantity_mw', 'total_cost']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Rename region to zone for consistency
        if 'region' in df.columns:
            df['zone'] = df['region']
        
        # Select final columns
        final_columns = ['timestamp', 'service_type', 'market_type', 'zone', 'price', 'quantity_mw']
        final_columns = [col for col in final_columns if col in df.columns]
        
        return df[final_columns]