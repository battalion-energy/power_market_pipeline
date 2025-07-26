#!/usr/bin/env python3
"""Test the incremental processor with just a few files."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from incremental_ercot_processor import IncrementalProcessor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_processor():
    """Test with just a few files."""
    processor = IncrementalProcessor()
    
    # Show current state
    summary = processor.get_processing_summary()
    logger.info(f"Starting state - Files: {summary['total_files_processed']}, Records: {summary['total_records']:,}")
    
    # Test processing a single DAM file
    dam_dir = Path("/Users/enrico/data/ERCOT_data/DAM_Hourly_LMPs")
    dam_files = sorted(dam_dir.glob("*DAMHRLMPNP4183_csv.zip"))[:1]
    
    if dam_files:
        logger.info(f"\nTesting DAM LMP file: {dam_files[0].name}")
        records = processor.process_dam_lmp_file(dam_files[0])
        logger.info(f"Result: {records} records")
    
    # Test processing a single RT file
    rt_dir = Path("/Users/enrico/data/ERCOT_data/Settlement_Point_Prices_at_Resource_Nodes,_Hubs_and_Load_Zones")
    rt_files = sorted(rt_dir.glob("*SPPHLZNP6905*csv.zip"))[:1]
    
    if rt_files:
        logger.info(f"\nTesting RT SPP file: {rt_files[0].name}")
        records = processor.process_rt_spp_file(rt_files[0])
        logger.info(f"Result: {records} records")
    
    # Test processing a single AS file
    as_dir = Path("/Users/enrico/data/ERCOT_data/DAM_Clearing_Prices_for_Capacity")
    as_files = sorted(as_dir.glob("*DAMCPCNP4188_csv.zip"))[:1]
    
    if as_files:
        logger.info(f"\nTesting Ancillary file: {as_files[0].name}")
        records = processor.process_ancillary_file(as_files[0])
        logger.info(f"Result: {records} records")
    
    # Show final state
    final_summary = processor.get_processing_summary()
    logger.info(f"\nFinal state - Files: {final_summary['total_files_processed']}, Records: {final_summary['total_records']:,}")
    
    # Try processing the same files again (should be skipped)
    logger.info("\n--- Testing incremental behavior (should skip) ---")
    
    if dam_files:
        logger.info(f"Re-processing DAM file: {dam_files[0].name}")
        records = processor.process_dam_lmp_file(dam_files[0])
        logger.info(f"Result: {records} records (should be 0)")

if __name__ == "__main__":
    test_processor()