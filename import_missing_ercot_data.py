#!/usr/bin/env python3
"""Import missing ERCOT data (RT and Ancillary Services only)."""

import os
import sys
from pathlib import Path
from datetime import datetime
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from process_ercot_historical_batch import (
    process_rt_spp_file, process_ancillary_file,
    store_lmp_batch, store_ancillary_batch,
    FILE_BATCH_SIZE
)
from database import init_db, get_db
from sqlalchemy import func, text
from database.models_v2 import LMP, AncillaryServices
import pandas as pd
import gc

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ERCOT_DATA_DIR = "/Users/enrico/data/ERCOT_data"


def process_rt_data():
    """Process RT settlement point prices."""
    logger.info("\n📊 Processing RT Settlement Point Prices...")
    
    rt_dir = Path(ERCOT_DATA_DIR) / "Settlement_Point_Prices_at_Resource_Nodes,_Hubs_and_Load_Zones"
    rt_files = sorted(rt_dir.glob("*SPPHLZNP6905*csv.zip"))
    
    logger.info(f"Found {len(rt_files)} RT files to process")
    
    if not rt_files:
        return 0
    
    total_records = 0
    batch_size = 20  # Smaller batch size for RT data due to high volume
    
    for i in range(0, len(rt_files), batch_size):
        batch_files = rt_files[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(rt_files) + batch_size - 1) // batch_size
        
        logger.info(f"\nProcessing RT batch {batch_num}/{total_batches} ({len(batch_files)} files)")
        
        batch_data = []
        for j, file in enumerate(batch_files):
            if j % 5 == 0:
                logger.info(f"  Processing file {j+1}/{len(batch_files)}")
            
            df = process_rt_spp_file(str(file))
            if df is not None and not df.empty:
                batch_data.append(df)
        
        if batch_data:
            combined_df = pd.concat(batch_data, ignore_index=True)
            logger.info(f"  Combined {len(combined_df):,} records")
            
            # Store in database
            records_stored = store_lmp_batch(combined_df)
            logger.info(f"  Stored {records_stored:,} new records")
            total_records += records_stored
            
            # Clean up
            del combined_df
            del batch_data
            gc.collect()
    
    return total_records


def process_ancillary_data():
    """Process ancillary services data."""
    logger.info("\n📊 Processing DAM Ancillary Services...")
    
    as_dir = Path(ERCOT_DATA_DIR) / "DAM_Clearing_Prices_for_Capacity"
    as_files = sorted(as_dir.glob("*DAMCPCNP4188_csv.zip"))
    
    logger.info(f"Found {len(as_files)} Ancillary Services files to process")
    
    if not as_files:
        return 0
    
    total_records = 0
    
    for i in range(0, len(as_files), FILE_BATCH_SIZE):
        batch_files = as_files[i:i + FILE_BATCH_SIZE]
        batch_num = i // FILE_BATCH_SIZE + 1
        total_batches = (len(as_files) + FILE_BATCH_SIZE - 1) // FILE_BATCH_SIZE
        
        logger.info(f"\nProcessing AS batch {batch_num}/{total_batches} ({len(batch_files)} files)")
        
        batch_data = []
        for j, file in enumerate(batch_files):
            if j % 10 == 0:
                logger.info(f"  Processing file {j+1}/{len(batch_files)}")
            
            df = process_ancillary_file(str(file))
            if df is not None and not df.empty:
                batch_data.append(df)
        
        if batch_data:
            combined_df = pd.concat(batch_data, ignore_index=True)
            logger.info(f"  Combined {len(combined_df):,} records")
            
            # Store in database
            records_stored = store_ancillary_batch(combined_df)
            logger.info(f"  Stored {records_stored:,} new records")
            total_records += records_stored
            
            # Clean up
            del combined_df
            del batch_data
            gc.collect()
    
    return total_records


def main():
    """Main function."""
    logger.info("Starting import of missing ERCOT data")
    
    # Initialize database
    init_db()
    
    # Check current state
    with get_db() as db:
        # Check RT data
        rt_count = db.query(func.count(LMP.iso)).filter(
            LMP.iso == 'ERCOT',
            LMP.market.in_(['RT5M', 'RT15M'])
        ).scalar()
        
        # Check ancillary data
        as_count = db.query(func.count(AncillaryServices.iso)).filter(
            AncillaryServices.iso == 'ERCOT'
        ).scalar()
        
        logger.info(f"Current database state:")
        logger.info(f"  RT records: {rt_count:,}")
        logger.info(f"  Ancillary records: {as_count:,}")
    
    # Process RT data
    rt_records = process_rt_data()
    
    # Process Ancillary Services
    as_records = process_ancillary_data()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 IMPORT COMPLETE")
    logger.info("=" * 60)
    logger.info(f"RT records imported: {rt_records:,}")
    logger.info(f"Ancillary records imported: {as_records:,}")
    
    # Final state
    with get_db() as db:
        # Check RT data
        rt_count = db.query(func.count(LMP.iso)).filter(
            LMP.iso == 'ERCOT',
            LMP.market.in_(['RT5M', 'RT15M'])
        ).scalar()
        
        # Check ancillary data
        as_count = db.query(func.count(AncillaryServices.iso)).filter(
            AncillaryServices.iso == 'ERCOT'
        ).scalar()
        
        # Get date ranges
        rt_range = db.query(
            func.min(LMP.interval_start).label('min_date'),
            func.max(LMP.interval_start).label('max_date')
        ).filter(
            LMP.iso == 'ERCOT',
            LMP.market.in_(['RT5M', 'RT15M'])
        ).first()
        
        as_range = db.query(
            func.min(AncillaryServices.interval_start).label('min_date'),
            func.max(AncillaryServices.interval_start).label('max_date')
        ).filter(
            AncillaryServices.iso == 'ERCOT'
        ).first()
        
        logger.info(f"\nFinal database state:")
        logger.info(f"  RT records: {rt_count:,}")
        if rt_range and rt_range.min_date:
            logger.info(f"    Date range: {rt_range.min_date.date()} to {rt_range.max_date.date()}")
        
        logger.info(f"  Ancillary records: {as_count:,}")
        if as_range and as_range.min_date:
            logger.info(f"    Date range: {as_range.min_date.date()} to {as_range.max_date.date()}")
        
        # Show product breakdown
        if as_count > 0:
            products = db.query(
                AncillaryServices.product,
                func.count(AncillaryServices.product).label('count')
            ).filter(
                AncillaryServices.iso == 'ERCOT'
            ).group_by(AncillaryServices.product).all()
            
            logger.info("\n  Ancillary products:")
            for product, count in products:
                logger.info(f"    {product}: {count:,}")


if __name__ == "__main__":
    main()