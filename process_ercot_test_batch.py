#!/usr/bin/env python3
"""Test processing a small batch of ERCOT historical data."""

import os
import sys
from pathlib import Path

# Add the processing functions
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from process_ercot_historical_batch import (
    process_dam_lmp_file, process_rt_spp_file, process_ancillary_file,
    store_lmp_batch, store_ancillary_batch, logger
)
from database import init_db, get_db
from sqlalchemy import func
from database.models_v2 import LMP, AncillaryServices
import pandas as pd

# Test with just a few files
ERCOT_DATA_DIR = "/Users/enrico/data/ERCOT_data"


def test_small_batch():
    """Test processing a small batch of files."""
    # Initialize database
    init_db()
    
    # Get initial counts
    with get_db() as db:
        initial_lmp = db.query(func.count(LMP.iso)).filter(LMP.iso == 'ERCOT').scalar()
        initial_as = db.query(func.count(AncillaryServices.iso)).filter(
            AncillaryServices.iso == 'ERCOT'
        ).scalar()
        logger.info(f"Initial counts - LMP: {initial_lmp:,}, AS: {initial_as:,}")
    
    # Test DAM LMP
    logger.info("\n📊 Testing DAM LMP processing...")
    dam_dir = Path(ERCOT_DATA_DIR) / "DAM_Hourly_LMPs"
    dam_files = sorted(dam_dir.glob("*DAMHRLMPNP4183_csv.zip"))[:5]
    
    dam_data = []
    for file in dam_files:
        logger.info(f"Processing {file.name}")
        df = process_dam_lmp_file(str(file))
        if df is not None:
            dam_data.append(df)
            logger.info(f"  Got {len(df)} records")
    
    if dam_data:
        dam_df = pd.concat(dam_data, ignore_index=True)
        logger.info(f"Total DAM records to store: {len(dam_df)}")
        dam_stored = store_lmp_batch(dam_df)
        logger.info(f"Stored {dam_stored} DAM records")
    
    # Test RT SPP
    logger.info("\n📊 Testing RT SPP processing...")
    rt_dir = Path(ERCOT_DATA_DIR) / "Settlement_Point_Prices_at_Resource_Nodes,_Hubs_and_Load_Zones"
    rt_files = sorted(rt_dir.glob("*SPPHLZNP6905*csv.zip"))[:5]
    
    rt_data = []
    for file in rt_files:
        logger.info(f"Processing {file.name}")
        df = process_rt_spp_file(str(file))
        if df is not None:
            rt_data.append(df)
            logger.info(f"  Got {len(df)} records")
    
    if rt_data:
        rt_df = pd.concat(rt_data, ignore_index=True)
        logger.info(f"Total RT records to store: {len(rt_df)}")
        rt_stored = store_lmp_batch(rt_df)
        logger.info(f"Stored {rt_stored} RT records")
    
    # Test Ancillary Services
    logger.info("\n📊 Testing Ancillary Services processing...")
    as_dir = Path(ERCOT_DATA_DIR) / "DAM_Clearing_Prices_for_Capacity"
    as_files = sorted(as_dir.glob("*DAMCPCNP4188_csv.zip"))[:5]
    
    as_data = []
    for file in as_files:
        logger.info(f"Processing {file.name}")
        df = process_ancillary_file(str(file))
        if df is not None:
            as_data.append(df)
            logger.info(f"  Got {len(df)} records")
    
    if as_data:
        as_df = pd.concat(as_data, ignore_index=True)
        logger.info(f"Total AS records to store: {len(as_df)}")
        as_stored = store_ancillary_batch(as_df)
        logger.info(f"Stored {as_stored} AS records")
    
    # Final counts
    with get_db() as db:
        final_lmp = db.query(func.count(LMP.iso)).filter(LMP.iso == 'ERCOT').scalar()
        final_as = db.query(func.count(AncillaryServices.iso)).filter(
            AncillaryServices.iso == 'ERCOT'
        ).scalar()
        logger.info(f"\nFinal counts - LMP: {final_lmp:,}, AS: {final_as:,}")
        logger.info(f"New records - LMP: {final_lmp - initial_lmp:,}, AS: {final_as - initial_as:,}")


if __name__ == "__main__":
    test_small_batch()