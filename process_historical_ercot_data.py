#!/usr/bin/env python3
"""Process historical ERCOT data from cron job downloads.

This script extracts and processes:
1. DAM Hourly LMPs (June 2024 - March 2025)
2. RT Settlement Point Prices (August 2024 - July 2025)  
3. DAM Clearing Prices for Capacity/Ancillary Services (April 2024 - July 2025)

The data is stored in both the database and exported to annual CSV/Arrow files.
"""

import asyncio
import os
import sys
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv
import structlog
from sqlalchemy import create_engine, text
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc

load_dotenv()

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import init_db, get_db
from database.models_v2 import LMP, AncillaryServices, Location
from sqlalchemy.orm import Session
from sqlalchemy import func

# Configure logging
logger = structlog.get_logger()

# Base directory for ERCOT data
ERCOT_DATA_DIR = "/Users/enrico/data/ERCOT_data"

# Output directories
OUTPUT_DIR = Path("./data/ercot_historical")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Batch size for database inserts
BATCH_SIZE = 10000


class ERCOTHistoricalProcessor:
    """Process historical ERCOT data from downloaded files."""
    
    def __init__(self):
        self.logger = logger.bind(component="ERCOTHistoricalProcessor")
        self.processed_files = set()
        self.error_files = []
        init_db()
        
    def extract_date_from_filename(self, filename: str) -> Optional[datetime]:
        """Extract date from ERCOT filename."""
        # Try different date patterns
        import re
        
        # Pattern 1: .20240605.123251.
        pattern1 = r'\.(\d{8})\.\d{6}\.'
        match = re.search(pattern1, filename)
        if match:
            date_str = match.group(1)
            return datetime.strptime(date_str, '%Y%m%d')
            
        # Pattern 2: _20240823_0000_
        pattern2 = r'_(\d{8})_\d{4}_'
        match = re.search(pattern2, filename)
        if match:
            date_str = match.group(1)
            return datetime.strptime(date_str, '%Y%m%d')
            
        return None
        
    def process_dam_lmp_file(self, zip_path: Path) -> pd.DataFrame:
        """Process a single DAM LMP zip file."""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                # Find CSV file in zip
                csv_files = [f for f in zf.namelist() if f.endswith('.csv')]
                if not csv_files:
                    self.logger.error(f"No CSV found in {zip_path}")
                    return pd.DataFrame()
                    
                # Read CSV
                with zf.open(csv_files[0]) as f:
                    df = pd.read_csv(f)
                    
                # Expected columns: DeliveryDate, HourEnding, BusName, LMP, etc.
                required_cols = ['DeliveryDate', 'HourEnding', 'BusName', 'LMP']
                if not all(col in df.columns for col in required_cols):
                    self.logger.error(f"Missing required columns in {zip_path}")
                    return pd.DataFrame()
                    
                # Parse timestamps
                df['DeliveryDate'] = pd.to_datetime(df['DeliveryDate'])
                
                # Convert HourEnding to hour offset
                df['hour'] = df['HourEnding'].astype(str).str.extract(r'(\d+)').astype(int)
                df['timestamp'] = df['DeliveryDate'] + pd.to_timedelta(df['hour'] - 1, unit='h')
                
                # Rename columns
                df = df.rename(columns={
                    'BusName': 'location',
                    'LMP': 'lmp',
                    'MCC': 'congestion',
                    'MLC': 'loss',
                    'MW': 'energy'
                })
                
                # Add metadata
                df['iso'] = 'ERCOT'
                df['market'] = 'DAM'
                df['interval_start'] = df['timestamp']
                df['interval_end'] = df['timestamp'] + timedelta(hours=1)
                
                # Infer location type
                df['location_type'] = df['location'].apply(self.infer_location_type)
                
                return df[['interval_start', 'interval_end', 'iso', 'location', 
                          'location_type', 'market', 'lmp', 'energy', 'congestion', 'loss']]
                          
        except Exception as e:
            self.logger.error(f"Error processing {zip_path}: {e}")
            self.error_files.append(str(zip_path))
            return pd.DataFrame()
            
    def process_rt_spp_file(self, zip_path: Path) -> pd.DataFrame:
        """Process a single RT SPP zip file."""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                csv_files = [f for f in zf.namelist() if f.endswith('.csv')]
                if not csv_files:
                    return pd.DataFrame()
                    
                with zf.open(csv_files[0]) as f:
                    df = pd.read_csv(f)
                    
                # Expected columns vary, but typically include:
                # SCEDTimestamp, SettlementPoint, LMP, etc.
                if 'SCEDTimestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['SCEDTimestamp'])
                elif 'DeliveryInterval' in df.columns and 'DeliveryDate' in df.columns:
                    # Parse date and interval
                    df['DeliveryDate'] = pd.to_datetime(df['DeliveryDate'])
                    # Interval is 15-minute periods (1-96 per day)
                    df['minutes'] = (df['DeliveryInterval'] - 1) * 15
                    df['timestamp'] = df['DeliveryDate'] + pd.to_timedelta(df['minutes'], unit='m')
                else:
                    self.logger.error(f"Cannot find timestamp column in {zip_path}")
                    return pd.DataFrame()
                    
                # Rename columns
                col_mapping = {
                    'SettlementPoint': 'location',
                    'SettlementPointName': 'location',
                    'SPP': 'lmp',
                    'LMP': 'lmp',
                    'MCC': 'congestion',
                    'MLC': 'loss',
                    'ShadowPriceMW': 'energy'
                }
                
                for old, new in col_mapping.items():
                    if old in df.columns:
                        df = df.rename(columns={old: new})
                        
                # Add metadata
                df['iso'] = 'ERCOT'
                df['market'] = 'RT5M'
                df['interval_start'] = df['timestamp']
                df['interval_end'] = df['timestamp'] + timedelta(minutes=5)
                df['location_type'] = df['location'].apply(self.infer_location_type)
                
                return df[['interval_start', 'interval_end', 'iso', 'location',
                          'location_type', 'market', 'lmp', 'energy', 'congestion', 'loss']]
                          
        except Exception as e:
            self.logger.error(f"Error processing {zip_path}: {e}")
            self.error_files.append(str(zip_path))
            return pd.DataFrame()
            
    def process_ancillary_file(self, zip_path: Path) -> pd.DataFrame:
        """Process a single ancillary services file."""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                csv_files = [f for f in zf.namelist() if f.endswith('.csv')]
                if not csv_files:
                    return pd.DataFrame()
                    
                with zf.open(csv_files[0]) as f:
                    df = pd.read_csv(f)
                    
                # Expected columns: DeliveryDate, HourEnding, AncillaryType, MCPC, etc.
                if 'DeliveryDate' not in df.columns:
                    return pd.DataFrame()
                    
                df['DeliveryDate'] = pd.to_datetime(df['DeliveryDate'])
                
                if 'HourEnding' in df.columns:
                    df['hour'] = df['HourEnding'].astype(str).str.extract(r'(\d+)').astype(int)
                    df['timestamp'] = df['DeliveryDate'] + pd.to_timedelta(df['hour'] - 1, unit='h')
                else:
                    df['timestamp'] = df['DeliveryDate']
                    
                # Process different ancillary service types
                records = []
                
                # Map ERCOT service names to our standard names
                service_mapping = {
                    'REGUP': 'REGUP',
                    'REGDN': 'REGDOWN',
                    'REGDOWN': 'REGDOWN',
                    'RRS': 'RRS',
                    'NSPIN': 'NON_SPIN',
                    'NON-SPIN': 'NON_SPIN',
                    'ECRS': 'ECRS'
                }
                
                for service_col in ['REGUP', 'REGDN', 'RRS', 'NSPIN', 'ECRS']:
                    if service_col in df.columns:
                        service_df = df[['timestamp']].copy()
                        service_df['product'] = service_mapping.get(service_col, service_col)
                        service_df['clearing_price'] = df[service_col]
                        service_df['iso'] = 'ERCOT'
                        service_df['region'] = 'ERCOT'  # System-wide
                        service_df['market'] = 'DAM'
                        service_df['interval_start'] = service_df['timestamp']
                        service_df['interval_end'] = service_df['timestamp'] + timedelta(hours=1)
                        
                        # Add quantity if available
                        qty_col = f'{service_col}_QTY'
                        if qty_col in df.columns:
                            service_df['clearing_quantity'] = df[qty_col]
                            
                        records.append(service_df)
                        
                if records:
                    result_df = pd.concat(records, ignore_index=True)
                    return result_df[['interval_start', 'interval_end', 'iso', 'region',
                                     'market', 'product', 'clearing_price', 'clearing_quantity']]
                else:
                    return pd.DataFrame()
                    
        except Exception as e:
            self.logger.error(f"Error processing {zip_path}: {e}")
            self.error_files.append(str(zip_path))
            return pd.DataFrame()
            
    def infer_location_type(self, location: str) -> str:
        """Infer ERCOT location type from name."""
        if pd.isna(location):
            return 'unknown'
        location = str(location).upper()
        if location.startswith('HB_'):
            return 'hub'
        elif location.startswith('LZ_'):
            return 'zone'
        else:
            return 'node'
            
    def process_directory(self, directory: str, file_pattern: str, 
                         processor_func, data_type: str) -> Tuple[int, int]:
        """Process all files in a directory."""
        dir_path = Path(ERCOT_DATA_DIR) / directory
        if not dir_path.exists():
            self.logger.error(f"Directory not found: {dir_path}")
            return 0, 0
            
        # Get all matching files
        files = list(dir_path.glob(file_pattern))
        self.logger.info(f"Found {len(files)} {data_type} files to process")
        
        # Process files in batches by year
        files_by_year = {}
        for file in files:
            date = self.extract_date_from_filename(file.name)
            if date:
                year = date.year
                if year not in files_by_year:
                    files_by_year[year] = []
                files_by_year[year].append(file)
                
        total_processed = 0
        total_records = 0
        
        for year, year_files in sorted(files_by_year.items()):
            self.logger.info(f"Processing {len(year_files)} files for {year}")
            
            # Process files in parallel
            year_data = []
            with ProcessPoolExecutor(max_workers=4) as executor:
                future_to_file = {
                    executor.submit(processor_func, file): file 
                    for file in year_files[:100]  # Limit for testing
                }
                
                for future in as_completed(future_to_file):
                    file = future_to_file[future]
                    try:
                        df = future.result()
                        if not df.empty:
                            year_data.append(df)
                            total_processed += 1
                    except Exception as e:
                        self.logger.error(f"Error processing {file}: {e}")
                        
            if year_data:
                # Combine all data for the year
                year_df = pd.concat(year_data, ignore_index=True)
                self.logger.info(f"Combined {len(year_df)} records for {year}")
                
                # Save to files
                csv_path = OUTPUT_DIR / f"ercot_{data_type}_{year}.csv"
                arrow_path = OUTPUT_DIR / f"ercot_{data_type}_{year}.parquet"
                
                # Save CSV
                year_df.to_csv(csv_path, index=False)
                self.logger.info(f"Saved {csv_path}")
                
                # Save Arrow/Parquet
                table = pa.Table.from_pandas(year_df)
                pq.write_table(table, arrow_path, compression='snappy')
                self.logger.info(f"Saved {arrow_path}")
                
                # Store in database
                if data_type == 'dam_lmp':
                    records_stored = self.store_lmp_data(year_df)
                elif data_type == 'rt_spp':
                    records_stored = self.store_lmp_data(year_df)
                elif data_type == 'ancillary':
                    records_stored = self.store_ancillary_data(year_df)
                    
                total_records += records_stored
                
                # Clean up memory
                del year_df
                del year_data
                gc.collect()
                
        return total_processed, total_records
        
    def store_lmp_data(self, df: pd.DataFrame) -> int:
        """Store LMP data in database."""
        records_stored = 0
        
        with get_db() as db:
            # Get ERCOT ISO ID
            iso_id = db.execute(text("SELECT id FROM isos WHERE code = 'ERCOT'")).scalar()
            
            # Process in batches
            for i in range(0, len(df), BATCH_SIZE):
                batch_df = df.iloc[i:i + BATCH_SIZE]
                
                # Ensure locations exist
                unique_locations = batch_df[['location', 'location_type']].drop_duplicates()
                for _, loc in unique_locations.iterrows():
                    existing = db.query(Location).filter(
                        Location.iso_id == iso_id,
                        Location.location_id == loc['location']
                    ).first()
                    
                    if not existing:
                        location = Location(
                            iso_id=iso_id,
                            location_id=loc['location'],
                            location_name=loc['location'],
                            location_type=loc['location_type']
                        )
                        db.add(location)
                        
                db.flush()
                
                # Prepare records for bulk insert
                records = []
                for _, row in batch_df.iterrows():
                    # Check if record exists
                    existing = db.query(LMP).filter(
                        LMP.interval_start == row['interval_start'],
                        LMP.iso == row['iso'],
                        LMP.location == row['location'],
                        LMP.market == row['market']
                    ).first()
                    
                    if not existing:
                        record = {
                            'interval_start': row['interval_start'],
                            'interval_end': row['interval_end'],
                            'iso': row['iso'],
                            'location': row['location'],
                            'location_type': row['location_type'],
                            'market': row['market'],
                            'lmp': row.get('lmp'),
                            'energy': row.get('energy'),
                            'congestion': row.get('congestion'),
                            'loss': row.get('loss')
                        }
                        records.append(record)
                        
                if records:
                    db.bulk_insert_mappings(LMP, records)
                    db.commit()
                    records_stored += len(records)
                    
                self.logger.info(f"Stored batch {i//BATCH_SIZE + 1}: {len(records)} records")
                
        return records_stored
        
    def store_ancillary_data(self, df: pd.DataFrame) -> int:
        """Store ancillary services data in database."""
        records_stored = 0
        
        with get_db() as db:
            # Process in batches
            for i in range(0, len(df), BATCH_SIZE):
                batch_df = df.iloc[i:i + BATCH_SIZE]
                
                records = []
                for _, row in batch_df.iterrows():
                    record = {
                        'interval_start': row['interval_start'],
                        'interval_end': row['interval_end'],
                        'iso': row['iso'],
                        'region': row.get('region', 'ERCOT'),
                        'market': row['market'],
                        'product': row['product'],
                        'clearing_price': row.get('clearing_price'),
                        'clearing_quantity': row.get('clearing_quantity')
                    }
                    records.append(record)
                    
                if records:
                    db.bulk_insert_mappings(AncillaryServices, records)
                    db.commit()
                    records_stored += len(records)
                    
        return records_stored
        
    async def run(self):
        """Run the historical data processing."""
        self.logger.info("Starting ERCOT historical data processing")
        
        # Process DAM LMP data
        self.logger.info("\n📊 Processing DAM Hourly LMPs...")
        dam_files, dam_records = self.process_directory(
            "DAM_Hourly_LMPs",
            "*DAMHRLMPNP4183_csv.zip",
            self.process_dam_lmp_file,
            "dam_lmp"
        )
        
        # Process RT SPP data
        self.logger.info("\n📊 Processing RT Settlement Point Prices...")
        rt_files, rt_records = self.process_directory(
            "Settlement_Point_Prices_at_Resource_Nodes,_Hubs_and_Load_Zones",
            "*SPPHLZNP6905*csv.zip",
            self.process_rt_spp_file,
            "rt_spp"
        )
        
        # Process Ancillary Services data
        self.logger.info("\n📊 Processing DAM Ancillary Services...")
        as_files, as_records = self.process_directory(
            "DAM_Clearing_Prices_for_Capacity",
            "*DAMCPCNP4188_csv.zip",
            self.process_ancillary_file,
            "ancillary"
        )
        
        # Summary
        self.logger.info("\n" + "=" * 60)
        self.logger.info("📊 PROCESSING COMPLETE")
        self.logger.info("=" * 60)
        self.logger.info(f"DAM LMP: {dam_files} files processed, {dam_records:,} records stored")
        self.logger.info(f"RT SPP: {rt_files} files processed, {rt_records:,} records stored")
        self.logger.info(f"Ancillary: {as_files} files processed, {as_records:,} records stored")
        self.logger.info(f"Total records: {dam_records + rt_records + as_records:,}")
        
        if self.error_files:
            self.logger.warning(f"\n⚠️  {len(self.error_files)} files had errors:")
            for f in self.error_files[:10]:
                self.logger.warning(f"  - {f}")
                
        # Check final database state
        with get_db() as db:
            lmp_count = db.query(func.count(LMP.iso)).filter(LMP.iso == 'ERCOT').scalar()
            as_count = db.query(func.count(AncillaryServices.iso)).filter(
                AncillaryServices.iso == 'ERCOT'
            ).scalar()
            
            self.logger.info(f"\n📊 Final Database State:")
            self.logger.info(f"LMP records: {lmp_count:,}")
            self.logger.info(f"Ancillary records: {as_count:,}")


async def main():
    """Main entry point."""
    processor = ERCOTHistoricalProcessor()
    await processor.run()


if __name__ == "__main__":
    asyncio.run(main()