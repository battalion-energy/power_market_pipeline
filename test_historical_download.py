#!/usr/bin/env python
"""Test downloading historical data for different time periods."""

import asyncio
import os
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from database import init_db, get_db
from downloaders.base_v2 import DownloadConfig
from services.data_fetcher import DataFetcher
from sqlalchemy import text


class HistoricalDataTester:
    """Test historical data downloads."""
    
    def __init__(self):
        self.test_results = []
        
    async def test_period(self, name: str, start_date: datetime, end_date: datetime, 
                         isos: List[str] = None, data_types: List[str] = None):
        """Test downloading data for a specific period."""
        if isos is None:
            isos = ["ERCOT"]  # Start with ERCOT only
        if data_types is None:
            data_types = ["lmp"]  # Start with LMP only
            
        print(f"\n{'='*80}")
        print(f"Testing: {name}")
        print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"ISOs: {', '.join(isos)}")
        print(f"Data types: {', '.join(data_types)}")
        print(f"{'='*80}")
        
        # Create config
        config = DownloadConfig(
            start_date=start_date,
            end_date=end_date,
            data_types=data_types,
            output_dir="/tmp/power_market_pipeline",
            batch_size=1000,
            retry_attempts=3,
            retry_delay=60
        )
        
        # Create data fetcher
        fetcher = DataFetcher(config)
        
        # Track results
        test_result = {
            "name": name,
            "start_date": start_date,
            "end_date": end_date,
            "isos": isos,
            "data_types": data_types,
            "success": False,
            "records": {},
            "errors": []
        }
        
        try:
            # Fetch data
            print("\nFetching data...")
            results = await fetcher.fetch_all_data(
                isos=isos,
                start_date=start_date,
                end_date=end_date,
                data_types=data_types
            )
            
            # Process results
            total_records = 0
            for iso, iso_results in results.items():
                iso_total = sum(iso_results.values())
                total_records += iso_total
                test_result["records"][iso] = iso_results
                
                print(f"\n{iso}:")
                for data_type, count in iso_results.items():
                    print(f"  {data_type}: {count:,} records")
                print(f"  Total: {iso_total:,} records")
            
            print(f"\nTotal records downloaded: {total_records:,}")
            test_result["success"] = True
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(f"\n{error_msg}")
            test_result["errors"].append(error_msg)
            import traceback
            traceback.print_exc()
        
        # Audit database
        await self.audit_data(name)
        
        self.test_results.append(test_result)
        return test_result
    
    async def audit_data(self, test_name: str):
        """Audit the downloaded data."""
        print(f"\nAuditing data for {test_name}...")
        
        with get_db() as db:
            # Overall statistics
            result = db.execute(text("""
                SELECT 
                    COUNT(*) as total_records,
                    COUNT(DISTINCT iso) as unique_isos,
                    COUNT(DISTINCT market) as unique_markets,
                    COUNT(DISTINCT location) as unique_locations,
                    MIN(interval_start) as earliest_data,
                    MAX(interval_start) as latest_data
                FROM lmp
            """))
            
            stats = result.fetchone()
            print(f"\nDatabase Statistics:")
            print(f"  Total records: {stats.total_records:,}")
            print(f"  ISOs: {stats.unique_isos}")
            print(f"  Markets: {stats.unique_markets}")
            print(f"  Locations: {stats.unique_locations:,}")
            print(f"  Date range: {stats.earliest_data} to {stats.latest_data}")
            
            # Per ISO statistics
            result = db.execute(text("""
                SELECT 
                    iso,
                    market,
                    COUNT(*) as record_count,
                    MIN(interval_start) as earliest,
                    MAX(interval_start) as latest
                FROM lmp
                GROUP BY iso, market
                ORDER BY iso, market
            """))
            
            print(f"\nPer ISO/Market Statistics:")
            for row in result:
                print(f"  {row.iso} {row.market}: {row.record_count:,} records "
                      f"({row.earliest} to {row.latest})")
    
    async def run_all_tests(self):
        """Run all historical data tests."""
        print("Historical Data Download Tests")
        print("="*80)
        
        # Initialize database
        print("Initializing database...")
        init_db()
        print("✓ Database initialized")
        
        # Clear existing data
        print("\nClearing existing data...")
        with get_db() as db:
            db.execute(text("TRUNCATE TABLE lmp CASCADE"))
            db.commit()
        print("✓ Data cleared")
        
        # Test 1: Last 3 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=3)
        await self.test_period(
            "Last 3 Days",
            start_date,
            end_date,
            isos=["ERCOT"],
            data_types=["lmp"]
        )
        
        # Clear data before next test
        print("\nClearing data before next test...")
        with get_db() as db:
            db.execute(text("TRUNCATE TABLE lmp CASCADE"))
            db.commit()
        
        # Test 2: Last month
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        await self.test_period(
            "Last Month (30 days)",
            start_date,
            end_date,
            isos=["ERCOT"],
            data_types=["lmp"]
        )
        
        # Clear data before next test
        print("\nClearing data before next test...")
        with get_db() as db:
            db.execute(text("TRUNCATE TABLE lmp CASCADE"))
            db.commit()
        
        # Test 3: Since Jan 1, 2019 (in chunks)
        # Note: This is a large download, so we'll test with a smaller range first
        print("\nTesting historical download capability...")
        
        # Test with just January 2019 as a proof of concept
        start_date = datetime(2019, 1, 1)
        end_date = datetime(2019, 1, 31)
        await self.test_period(
            "January 2019 (Historical Test)",
            start_date,
            end_date,
            isos=["ERCOT"],
            data_types=["lmp"]
        )
        
        # Summary
        print("\n" + "="*80)
        print("Test Summary")
        print("="*80)
        
        for result in self.test_results:
            status = "✓ PASSED" if result["success"] else "✗ FAILED"
            print(f"\n{result['name']}: {status}")
            if result["success"]:
                for iso, records in result["records"].items():
                    total = sum(records.values())
                    print(f"  {iso}: {total:,} records")
            else:
                for error in result["errors"]:
                    print(f"  Error: {error}")
        
        # Check if we can do full historical
        print("\n" + "="*80)
        print("Full Historical Download Plan")
        print("="*80)
        print("\nTo download all data since Jan 1, 2019:")
        print("1. The system would process in monthly chunks")
        print("2. Estimated data volume:")
        
        days_since_2019 = (datetime.now() - datetime(2019, 1, 1)).days
        
        # Rough estimates
        print(f"   - Days: {days_since_2019:,}")
        print(f"   - Hourly intervals: {days_since_2019 * 24:,}")
        print(f"   - 5-min intervals: {days_since_2019 * 24 * 12:,}")
        print(f"   - With ~5000 nodes: ~{days_since_2019 * 24 * 12 * 5000:,} RT records")
        print("\n3. This would require:")
        print("   - Valid API credentials")
        print("   - Significant storage space")
        print("   - Several hours/days to complete")
        print("   - Rate limiting consideration")


async def main():
    """Run historical data tests."""
    # Check database
    if not os.getenv("DATABASE_URL"):
        print("Error: DATABASE_URL not set")
        sys.exit(1)
    
    # Create tester
    tester = HistoricalDataTester()
    
    # Run tests
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())