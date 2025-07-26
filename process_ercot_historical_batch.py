#!/usr/bin/env python3
"""Process ERCOT historical data in batches with proper column handling."""

import os
import sys
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple, List
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc

load_dotenv()

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import init_db, get_db
from database.models_v2 import LMP, AncillaryServices, Location
from sqlalchemy import text, func

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Base directory for ERCOT data
ERCOT_DATA_DIR = "/Users/enrico/data/ERCOT_data"

# Output directories
OUTPUT_DIR = Path("./data/ercot_historical")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Batch size for database inserts
BATCH_SIZE = 5000
FILE_BATCH_SIZE = 50  # Process files in batches


def extract_date_from_filename(filename: str) -> Optional[datetime]:
    """Extract date from ERCOT filename."""
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


def process_dam_lmp_file(zip_path: str) -> Optional[pd.DataFrame]:
    """Process a single DAM LMP zip file."""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            csv_files = [f for f in zf.namelist() if f.endswith('.csv')]
            if not csv_files:
                return None
                
            with zf.open(csv_files[0]) as f:
                df = pd.read_csv(f)
                
            # Parse date and hour
            df['DeliveryDate'] = pd.to_datetime(df['DeliveryDate'], format='%m/%d/%Y')
            df['hour'] = df['HourEnding'].str.extract(r'(\d+)').astype(int)
            df['interval_start'] = df['DeliveryDate'] + pd.to_timedelta(df['hour'] - 1, unit='h')
            df['interval_end'] = df['interval_start'] + timedelta(hours=1)
            
            # Rename and format
            df = df.rename(columns={
                'BusName': 'location',
                'LMP': 'lmp'
            })
            
            # Add metadata
            df['iso'] = 'ERCOT'
            df['market'] = 'DAM'
            df['location_type'] = df['location'].apply(infer_location_type)
            
            # Note: This file format doesn't have energy, congestion, loss components
            df['energy'] = None
            df['congestion'] = None
            df['loss'] = None
            
            return df[['interval_start', 'interval_end', 'iso', 'location', 
                      'location_type', 'market', 'lmp', 'energy', 'congestion', 'loss']]
                      
    except Exception as e:
        logger.error(f"Error processing {zip_path}: {e}")
        return None


def process_rt_spp_file(zip_path: str) -> Optional[pd.DataFrame]:
    """Process a single RT SPP zip file."""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            csv_files = [f for f in zf.namelist() if f.endswith('.csv')]
            if not csv_files:
                return None
                
            with zf.open(csv_files[0]) as f:
                df = pd.read_csv(f)
                
            # Parse timestamp from date, hour, and interval
            df['DeliveryDate'] = pd.to_datetime(df['DeliveryDate'], format='%m/%d/%Y')
            
            # DeliveryHour is 1-24, DeliveryInterval is 1-4 (15-min intervals within hour)
            # Convert to minutes from start of day
            df['minutes'] = (df['DeliveryHour'] - 1) * 60 + (df['DeliveryInterval'] - 1) * 15
            df['interval_start'] = df['DeliveryDate'] + pd.to_timedelta(df['minutes'], unit='m')
            df['interval_end'] = df['interval_start'] + timedelta(minutes=15)
            
            # Rename columns
            df = df.rename(columns={
                'SettlementPointName': 'location',
                'SettlementPointPrice': 'lmp'
            })
            
            # Add metadata
            df['iso'] = 'ERCOT'
            df['market'] = 'RT15M'  # 15-minute RT market
            df['location_type'] = df.apply(
                lambda row: row.get('SettlementPointType', '').lower() 
                if pd.notna(row.get('SettlementPointType')) 
                else infer_location_type(row['location']), 
                axis=1
            )
            
            # No component prices in this format
            df['energy'] = None
            df['congestion'] = None
            df['loss'] = None
            
            return df[['interval_start', 'interval_end', 'iso', 'location',
                      'location_type', 'market', 'lmp', 'energy', 'congestion', 'loss']]
                      
    except Exception as e:
        logger.error(f"Error processing {zip_path}: {e}")
        return None


def process_ancillary_file(zip_path: str) -> Optional[pd.DataFrame]:
    """Process a single ancillary services file."""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            csv_files = [f for f in zf.namelist() if f.endswith('.csv')]
            if not csv_files:
                return None
                
            with zf.open(csv_files[0]) as f:
                df = pd.read_csv(f)
                
            # Parse date and hour
            df['DeliveryDate'] = pd.to_datetime(df['DeliveryDate'], format='%m/%d/%Y')
            df['hour'] = df['HourEnding'].str.extract(r'(\d+)').astype(int)
            df['interval_start'] = df['DeliveryDate'] + pd.to_timedelta(df['hour'] - 1, unit='h')
            df['interval_end'] = df['interval_start'] + timedelta(hours=1)
            
            # Pivot to get one row per hour with all service types
            df_pivot = df.pivot_table(
                index=['interval_start', 'interval_end'],
                columns='AncillaryType',
                values='MCPC',
                aggfunc='first'
            ).reset_index()
            
            # Melt back to long format with standard product names
            records = []
            service_mapping = {
                'REGUP': 'REGUP',
                'REGDN': 'REGDOWN',
                'RRS': 'RRS',
                'NSPIN': 'NON_SPIN',
                'ECRS': 'ECRS'
            }
            
            for service, standard_name in service_mapping.items():
                if service in df_pivot.columns:
                    service_df = df_pivot[['interval_start', 'interval_end', service]].copy()
                    service_df = service_df.dropna(subset=[service])
                    service_df['product'] = standard_name
                    service_df['clearing_price'] = service_df[service]
                    service_df['iso'] = 'ERCOT'
                    service_df['region'] = 'ERCOT'
                    service_df['market'] = 'DAM'
                    service_df['clearing_quantity'] = None
                    service_df['requirement'] = None
                    records.append(service_df[['interval_start', 'interval_end', 'iso', 
                                              'region', 'market', 'product', 
                                              'clearing_price', 'clearing_quantity', 'requirement']])
                    
            if records:
                return pd.concat(records, ignore_index=True)
            else:
                return None
                
    except Exception as e:
        logger.error(f"Error processing {zip_path}: {e}")
        return None


def infer_location_type(location: str) -> str:
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


def process_file_batch(files: List[Path], processor_func, batch_name: str) -> pd.DataFrame:
    """Process a batch of files."""
    batch_data = []
    
    for i, file in enumerate(files):
        if i % 10 == 0:
            logger.info(f"  Processing {batch_name} file {i+1}/{len(files)}")
            
        df = processor_func(str(file))
        if df is not None and not df.empty:
            batch_data.append(df)
            
    if batch_data:
        return pd.concat(batch_data, ignore_index=True)
    else:
        return pd.DataFrame()


def store_lmp_batch(df: pd.DataFrame) -> int:
    """Store a batch of LMP data."""
    if df.empty:
        return 0
        
    records_stored = 0
    
    with get_db() as db:
        # Get ERCOT ISO ID
        iso_id = db.execute(text("SELECT id FROM isos WHERE code = 'ERCOT'")).scalar()
        
        # Ensure locations exist
        unique_locations = df[['location', 'location_type']].drop_duplicates()
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
        
        # Prepare records for bulk insert (skip duplicates)
        records = []
        for _, row in df.iterrows():
            record = {
                'interval_start': row['interval_start'],
                'interval_end': row['interval_end'],
                'iso': row['iso'],
                'location': row['location'],
                'location_type': row['location_type'],
                'market': row['market'],
                'lmp': row.get('lmp') if pd.notna(row.get('lmp')) else None,
                'energy': row.get('energy') if pd.notna(row.get('energy')) else None,
                'congestion': row.get('congestion') if pd.notna(row.get('congestion')) else None,
                'loss': row.get('loss') if pd.notna(row.get('loss')) else None
            }
            records.append(record)
            
        if records:
            # Use INSERT IGNORE equivalent for PostgreSQL
            from sqlalchemy.dialects.postgresql import insert
            stmt = insert(LMP).values(records)
            stmt = stmt.on_conflict_do_nothing()
            result = db.execute(stmt)
            records_stored = result.rowcount
            db.commit()
            
    return records_stored


def store_ancillary_batch(df: pd.DataFrame) -> int:
    """Store a batch of ancillary services data."""
    if df.empty:
        return 0
        
    records_stored = 0
    
    with get_db() as db:
        records = []
        for _, row in df.iterrows():
            record = {
                'interval_start': row['interval_start'],
                'interval_end': row['interval_end'],
                'iso': row['iso'],
                'region': row.get('region', 'ERCOT'),
                'market': row['market'],
                'product': row['product'],
                'clearing_price': row.get('clearing_price') if pd.notna(row.get('clearing_price')) else None,
                'clearing_quantity': row.get('clearing_quantity') if pd.notna(row.get('clearing_quantity')) else None,
                'requirement': row.get('requirement') if pd.notna(row.get('requirement')) else None
            }
            records.append(record)
            
        if records:
            from sqlalchemy.dialects.postgresql import insert
            stmt = insert(AncillaryServices).values(records)
            stmt = stmt.on_conflict_do_nothing()
            result = db.execute(stmt)
            records_stored = result.rowcount
            db.commit()
            
    return records_stored


def process_directory(directory: str, file_pattern: str, processor_func, 
                     data_type: str, store_func) -> Tuple[int, int]:
    """Process all files in a directory."""
    dir_path = Path(ERCOT_DATA_DIR) / directory
    if not dir_path.exists():
        logger.error(f"Directory not found: {dir_path}")
        return 0, 0
        
    # Get all matching files
    files = sorted(dir_path.glob(file_pattern))
    logger.info(f"Found {len(files)} {data_type} files to process")
    
    if not files:
        return 0, 0
        
    total_processed = 0
    total_records = 0
    
    # Process files in batches
    for i in range(0, len(files), FILE_BATCH_SIZE):
        batch_files = files[i:i + FILE_BATCH_SIZE]
        batch_num = i // FILE_BATCH_SIZE + 1
        total_batches = (len(files) + FILE_BATCH_SIZE - 1) // FILE_BATCH_SIZE
        
        logger.info(f"\nProcessing batch {batch_num}/{total_batches} ({len(batch_files)} files)")
        
        # Process files
        df = process_file_batch(batch_files, processor_func, f"batch_{batch_num}")
        
        if not df.empty:
            logger.info(f"  Processed {len(df):,} records")
            
            # Store in database
            records_stored = store_func(df)
            logger.info(f"  Stored {records_stored:,} new records")
            
            total_processed += len(batch_files)
            total_records += records_stored
            
            # Also save to annual files
            if data_type in ['dam_lmp', 'rt_spp']:
                save_annual_files(df, data_type)
                
        # Clean up memory
        del df
        gc.collect()
        
    return total_processed, total_records


def save_annual_files(df: pd.DataFrame, data_type: str):
    """Save data to annual CSV and Parquet files."""
    if df.empty:
        return
        
    # Group by year
    df['year'] = pd.to_datetime(df['interval_start']).dt.year
    
    for year, year_df in df.groupby('year'):
        # Remove year column before saving
        year_df = year_df.drop(columns=['year'])
        
        # File paths
        csv_path = OUTPUT_DIR / f"ercot_{data_type}_{year}.csv"
        parquet_path = OUTPUT_DIR / f"ercot_{data_type}_{year}.parquet"
        
        # Append to existing files or create new
        if csv_path.exists():
            # Append to CSV
            year_df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            year_df.to_csv(csv_path, index=False)
            
        # For Parquet, we need to read existing and combine
        if parquet_path.exists():
            existing_df = pd.read_parquet(parquet_path)
            combined_df = pd.concat([existing_df, year_df], ignore_index=True)
            table = pa.Table.from_pandas(combined_df)
            pq.write_table(table, parquet_path, compression='snappy')
        else:
            table = pa.Table.from_pandas(year_df)
            pq.write_table(table, parquet_path, compression='snappy')


def main():
    """Main processing function."""
    logger.info("Starting ERCOT historical data processing")
    
    # Initialize database
    init_db()
    
    # Check current database state
    with get_db() as db:
        lmp_count = db.query(func.count(LMP.iso)).filter(LMP.iso == 'ERCOT').scalar()
        as_count = db.query(func.count(AncillaryServices.iso)).filter(
            AncillaryServices.iso == 'ERCOT'
        ).scalar()
        logger.info(f"Starting database state - LMP: {lmp_count:,}, AS: {as_count:,}")
    
    # Process each data type
    results = {}
    
    # DAM LMP
    logger.info("\n📊 Processing DAM Hourly LMPs...")
    dam_files, dam_records = process_directory(
        "DAM_Hourly_LMPs",
        "*DAMHRLMPNP4183_csv.zip",
        process_dam_lmp_file,
        "dam_lmp",
        store_lmp_batch
    )
    results['DAM LMP'] = (dam_files, dam_records)
    
    # RT SPP
    logger.info("\n📊 Processing RT Settlement Point Prices...")
    rt_files, rt_records = process_directory(
        "Settlement_Point_Prices_at_Resource_Nodes,_Hubs_and_Load_Zones",
        "*SPPHLZNP6905*csv.zip",
        process_rt_spp_file,
        "rt_spp",
        store_lmp_batch
    )
    results['RT SPP'] = (rt_files, rt_records)
    
    # Ancillary Services
    logger.info("\n📊 Processing DAM Ancillary Services...")
    as_files, as_records = process_directory(
        "DAM_Clearing_Prices_for_Capacity",
        "*DAMCPCNP4188_csv.zip",
        process_ancillary_file,
        "ancillary",
        store_ancillary_batch
    )
    results['Ancillary'] = (as_files, as_records)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 PROCESSING COMPLETE")
    logger.info("=" * 60)
    
    total_files = 0
    total_records = 0
    for data_type, (files, records) in results.items():
        logger.info(f"{data_type}: {files} files processed, {records:,} records stored")
        total_files += files
        total_records += records
        
    logger.info(f"\nTotal: {total_files} files, {total_records:,} records")
    
    # Final database state
    with get_db() as db:
        lmp_count = db.query(func.count(LMP.iso)).filter(LMP.iso == 'ERCOT').scalar()
        as_count = db.query(func.count(AncillaryServices.iso)).filter(
            AncillaryServices.iso == 'ERCOT'
        ).scalar()
        logger.info(f"\nFinal database state - LMP: {lmp_count:,}, AS: {as_count:,}")


if __name__ == "__main__":
    main()