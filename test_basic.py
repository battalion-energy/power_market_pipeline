#!/usr/bin/env python
"""Basic test to check if the pipeline is working."""

import os
import sys
from datetime import datetime, timedelta

# Add the project directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from power_market_pipeline.database import init_db, get_db, ISO

def test_database():
    """Test database connection and initialization."""
    print("Testing database connection...")
    
    try:
        # Initialize database
        init_db()
        print("✓ Database initialized")
        
        # Check ISOs
        with get_db() as db:
            isos = db.query(ISO).all()
            print(f"✓ Found {len(isos)} ISOs: {[iso.code for iso in isos]}")
            
        return True
    except Exception as e:
        print(f"✗ Database error: {e}")
        return False

def test_imports():
    """Test all imports."""
    print("\nTesting imports...")
    
    try:
        from power_market_pipeline.downloaders.ercot.downloader_v2 import ERCOTDownloaderV2
        print("✓ ERCOT downloader imported")
        
        from power_market_pipeline.services.data_fetcher import DataFetcher
        print("✓ Data fetcher imported")
        
        from power_market_pipeline.cli import cli
        print("✓ CLI imported")
        
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Power Market Pipeline - Basic Test")
    print("="*50)
    
    # Test imports first
    if not test_imports():
        sys.exit(1)
    
    # Test database
    if not test_database():
        sys.exit(1)
    
    print("\n✓ All basic tests passed!")