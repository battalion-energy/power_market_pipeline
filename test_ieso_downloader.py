"""Test script for IESO downloader.

This script demonstrates how to use the IESODownloaderV2 class to download
Ontario market data.
"""

import asyncio
from datetime import datetime
from pathlib import Path

from downloaders.base_v2 import DownloadConfig
from downloaders.ieso import IESODownloaderV2


async def test_ieso_downloader():
    """Test IESO downloader with sample date range."""

    # Configuration
    config = DownloadConfig(
        start_date=datetime(2025, 5, 1),  # Start of LMP era
        end_date=datetime(2025, 5, 7),     # One week
        data_types=["lmp", "ancillary_services", "load"],
        output_dir="/pool/ssd8tb/data/iso",  # Adjust as needed
        batch_size=1000,
        retry_attempts=3,
        retry_delay=5
    )

    # Create downloader
    downloader = IESODownloaderV2(config)

    print("=" * 80)
    print("IESO Downloader Test")
    print("=" * 80)
    print(f"Start Date: {config.start_date}")
    print(f"End Date: {config.end_date}")
    print(f"Output Dir: {config.output_dir}/IESO_data/csv_files/")
    print("=" * 80)

    # Test 1: Download Day-Ahead LMP
    print("\n1. Testing Day-Ahead LMP download...")
    da_count = await downloader.download_lmp("DAM", config.start_date, config.end_date)
    print(f"   Downloaded {da_count} DA LMP files")

    # Test 2: Download Real-Time LMP
    print("\n2. Testing Real-Time LMP download...")
    rt_count = await downloader.download_lmp("RT5M", config.start_date, config.end_date)
    print(f"   Downloaded {rt_count} RT LMP files")

    # Test 3: Download Ontario Zonal Prices
    print("\n3. Testing Ontario Zonal Prices download...")
    zonal_count = await downloader.download_ontario_zonal_prices(config.start_date, config.end_date)
    print(f"   Downloaded {zonal_count} Zonal Price files")

    # Test 4: Download OEMP
    print("\n4. Testing OEMP download...")
    oemp_count = await downloader.download_oemp(config.start_date, config.end_date)
    print(f"   Downloaded {oemp_count} OEMP files")

    # Test 5: Download Operating Reserves
    print("\n5. Testing Operating Reserve downloads...")
    as_count = await downloader.download_ancillary_services("ALL", "RTM", config.start_date, config.end_date)
    print(f"   Downloaded {as_count} AS files (10S, 10NS, 30OR)")

    # Test 6: Download Load Data
    print("\n6. Testing Load data download...")
    load_count = await downloader.download_load("actual", config.start_date, config.end_date)
    print(f"   Downloaded {load_count} Load files")

    print("\n" + "=" * 80)
    print("Test Summary:")
    print("=" * 80)
    print(f"Total files downloaded: {da_count + rt_count + zonal_count + oemp_count + as_count + load_count}")
    print(f"Output directory: {config.output_dir}/IESO_data/csv_files/")

    # Test 7: Download all markets at once (convenience method)
    print("\n7. Testing download_all_markets() convenience method...")
    results = await downloader.download_all_markets(
        datetime(2025, 5, 1),
        datetime(2025, 5, 2),
        include_legacy=False
    )
    print(f"   Results: {results}")

    print("\n" + "=" * 80)
    print("All tests complete!")
    print("=" * 80)


async def test_legacy_hoep():
    """Test legacy HOEP download (pre-May 2025)."""

    config = DownloadConfig(
        start_date=datetime(2024, 12, 1),
        end_date=datetime(2024, 12, 7),
        data_types=["hoep"],
        output_dir="/pool/ssd8tb/data/iso",
        batch_size=1000,
        retry_attempts=3,
        retry_delay=5
    )

    downloader = IESODownloaderV2(config)

    print("\n" + "=" * 80)
    print("Testing Legacy HOEP Download")
    print("=" * 80)

    hoep_count = await downloader.download_legacy_hoep(config.start_date, config.end_date)
    print(f"Downloaded {hoep_count} HOEP files")
    print("=" * 80)


async def test_transition_period():
    """Test download across the May 2025 transition date."""

    config = DownloadConfig(
        start_date=datetime(2025, 4, 28),  # Before transition
        end_date=datetime(2025, 5, 3),      # After transition
        data_types=["all"],
        output_dir="/pool/ssd8tb/data/iso",
        batch_size=1000,
        retry_attempts=3,
        retry_delay=5
    )

    downloader = IESODownloaderV2(config)

    print("\n" + "=" * 80)
    print("Testing Transition Period (HOEP -> LMP)")
    print("=" * 80)
    print(f"Date range: {config.start_date} to {config.end_date}")
    print(f"Transition date: {downloader.LMP_TRANSITION_DATE}")
    print("=" * 80)

    results = await downloader.download_all_markets(
        config.start_date,
        config.end_date,
        include_legacy=True
    )

    print("\nResults:")
    for dataset, count in results.items():
        print(f"  {dataset}: {count} files")
    print("=" * 80)


async def show_available_locations():
    """Display available IESO locations."""

    config = DownloadConfig(
        start_date=datetime(2025, 5, 1),
        end_date=datetime(2025, 5, 1),
        data_types=["lmp"],
        output_dir="/pool/ssd8tb/data/iso",
    )

    downloader = IESODownloaderV2(config)

    print("\n" + "=" * 80)
    print("IESO Available Locations")
    print("=" * 80)

    locations = await downloader.get_available_locations()

    print(f"\nTotal locations: {len(locations)}")
    print("\nZones:")
    for loc in locations:
        if loc["location_type"] == "zone":
            print(f"  {loc['location_id']:15} - {loc['location_name']}")

    print("\nNote: ~1000 LMP nodes available post-May 2025")
    print("      Node list must be extracted from actual LMP CSV files")
    print("=" * 80)


if __name__ == "__main__":
    print("\nIESO Downloader Test Suite")
    print("==========================\n")

    # Run tests
    asyncio.run(test_ieso_downloader())
    asyncio.run(test_legacy_hoep())
    asyncio.run(test_transition_period())
    asyncio.run(show_available_locations())

    print("\n\nAll tests complete!\n")
