#!/usr/bin/env python
"""Test script to download 3 days of data for each ISO."""

import asyncio
import os
import sys
from datetime import datetime, timedelta

# Add the project directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from power_market_pipeline.downloaders.base_v2 import DownloadConfig
from power_market_pipeline.services.data_fetcher import DataFetcher
from power_market_pipeline.database import init_db, get_db
from sqlalchemy import text


async def test_3day_download():
    """Test downloading 3 days of data."""
    print("Power Market Pipeline - 3 Day Download Test")
    print("="*60)
    
    # Initialize database
    print("\n1. Initializing database...")
    init_db()
    print("✓ Database initialized")
    
    # Set up test parameters
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3)
    
    config = DownloadConfig(
        start_date=start_date,
        end_date=end_date,
        data_types=["lmp"],  # Start with just LMP data
        output_dir="/tmp/power_market_pipeline",
        batch_size=1000,
        retry_attempts=3,
        retry_delay=60
    )
    
    print(f"\n2. Test Parameters:")
    print(f"   Start Date: {start_date.strftime('%Y-%m-%d %H:%M')}")
    print(f"   End Date: {end_date.strftime('%Y-%m-%d %H:%M')}")
    print(f"   Data Types: {config.data_types}")
    
    # Create data fetcher
    fetcher = DataFetcher(config)
    
    # Test each ISO
    isos = ["ERCOT", "CAISO", "ISONE", "NYISO"]
    
    print("\n3. Starting download test for each ISO...")
    print("-"*60)
    
    for iso in isos:
        print(f"\n{iso}:")
        try:
            results = await fetcher.fetch_all_data(
                isos=[iso],
                start_date=start_date,
                end_date=end_date,
                data_types=["lmp"]
            )
            
            if iso in results:
                iso_results = results[iso]
                total_records = sum(iso_results.values())
                print(f"  ✓ Downloaded {total_records} records")
                for data_type, count in iso_results.items():
                    print(f"    - {data_type}: {count} records")
            else:
                print(f"  ✗ No results returned")
                
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Audit database
    print("\n4. Auditing downloaded data...")
    print("-"*60)
    
    with get_db() as db:
        # Check LMP table
        result = db.execute(text("""
            SELECT 
                iso,
                market,
                COUNT(*) as record_count,
                MIN(interval_start) as earliest,
                MAX(interval_start) as latest,
                COUNT(DISTINCT location) as unique_locations
            FROM lmp
            GROUP BY iso, market
            ORDER BY iso, market
        """))
        
        lmp_data = result.fetchall()
        if lmp_data:
            print("\nLMP Data Summary:")
            for row in lmp_data:
                print(f"  {row.iso} {row.market}: {row.record_count} records")
                print(f"    - Date range: {row.earliest} to {row.latest}")
                print(f"    - Locations: {row.unique_locations}")
        else:
            print("\n  No LMP data found in database")
        
        # Check for null values
        result = db.execute(text("""
            SELECT 
                iso,
                COUNT(*) as total,
                SUM(CASE WHEN lmp IS NULL THEN 1 ELSE 0 END) as null_lmp,
                SUM(CASE WHEN energy IS NULL THEN 1 ELSE 0 END) as null_energy,
                SUM(CASE WHEN congestion IS NULL THEN 1 ELSE 0 END) as null_congestion,
                SUM(CASE WHEN loss IS NULL THEN 1 ELSE 0 END) as null_loss
            FROM lmp
            GROUP BY iso
        """))
        
        null_data = result.fetchall()
        if null_data:
            print("\nData Quality Check (NULL values):")
            for row in null_data:
                print(f"  {row.iso}:")
                print(f"    - Total records: {row.total}")
                print(f"    - NULL LMP: {row.null_lmp}")
                print(f"    - NULL energy: {row.null_energy}")
                print(f"    - NULL congestion: {row.null_congestion}")
                print(f"    - NULL loss: {row.null_loss}")


if __name__ == "__main__":
    # Check for required env vars
    required_vars = ["DATABASE_URL"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"Error: Missing required environment variables: {missing_vars}")
        print("\nPlease set the following in your .env file:")
        print("DATABASE_URL=postgresql://user:password@localhost:5432/power_market")
        sys.exit(1)
    
    # Run the test
    asyncio.run(test_3day_download())