#!/usr/bin/env python3
"""
Automated reBAP data download and update
Uses OAuth 2.0 API to fetch latest data and merge with existing dataset
"""

import sys
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
from io import StringIO

# Import OAuth client
from netztransparenz_oauth import NetztransparenzOAuthClient, load_credentials_from_env


def parse_rebap_api_response(api_data: str) -> pd.DataFrame:
    """
    Convert Netztransparenz API CSV response to standardized DataFrame
    API returns German CSV format (semicolon delimited, comma decimals)

    Args:
        api_data: CSV string from API

    Returns:
        DataFrame with standardized columns
    """
    if not api_data or len(api_data) == 0:
        return pd.DataFrame()

    # Parse CSV from string (same format as manual download)
    df = pd.read_csv(
        StringIO(api_data),
        encoding='utf-8-sig',  # Handle BOM
        sep=';',
        decimal=','
    )

    # Parse timestamps (German format: DD.MM.YYYY HH:MM)
    df['datetime_start'] = pd.to_datetime(
        df['Datum'] + ' ' + df['von'],
        format='%d.%m.%Y %H:%M'
    )

    # Mark as UTC (timestamps are already in UTC from API)
    df['datetime_utc'] = df['datetime_start'].dt.tz_localize('UTC')

    # Create clean dataframe
    df_clean = pd.DataFrame({
        'datetime_utc': df['datetime_utc'],
        'rebap_undersupply_eur_per_mwh': df['reBAP unterdeckt'],
        'rebap_oversupply_eur_per_mwh': df['reBAP ueberdeckt'],
        'download_date': datetime.now()
    })

    return df_clean.sort_values('datetime_utc').reset_index(drop=True)


def get_latest_date_in_dataset(data_dir: Path) -> datetime:
    """
    Find the latest datetime in existing reBAP dataset

    Args:
        data_dir: Directory containing reBAP CSV files

    Returns:
        Latest datetime in dataset, or default start date if no data
    """
    existing_files = list(data_dir.glob("rebap_de_*.csv"))

    if not existing_files:
        # No existing data - start from 2019
        return datetime(2019, 12, 31, 23, 0, 0)

    # Find most recent file
    latest_file = max(existing_files, key=lambda p: p.stat().st_mtime)

    # Read last row to get latest date
    df = pd.read_csv(latest_file)
    df['datetime_utc'] = pd.to_datetime(df['datetime_utc'], utc=True)

    return df['datetime_utc'].max()


def merge_with_existing_data(new_df: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """
    Merge new data with existing dataset, removing duplicates

    Args:
        new_df: New data from API
        data_dir: Directory containing existing data

    Returns:
        Merged and deduplicated DataFrame
    """
    existing_files = list(data_dir.glob("rebap_de_*.csv"))

    if not existing_files:
        print("No existing data found - using new data as-is")
        return new_df

    # Load most recent existing file
    latest_file = max(existing_files, key=lambda p: p.stat().st_mtime)
    print(f"Loading existing data: {latest_file.name}")

    df_existing = pd.read_csv(latest_file)
    df_existing['datetime_utc'] = pd.to_datetime(df_existing['datetime_utc'], utc=True)

    print(f"  Existing records: {len(df_existing):,}")
    print(f"  Date range: {df_existing['datetime_utc'].min()} to {df_existing['datetime_utc'].max()}")
    print()

    # Merge
    print("Merging data...")
    df_combined = pd.concat([df_existing, new_df], ignore_index=True)

    # Remove duplicates (keep latest download_date)
    df_combined = df_combined.sort_values(['datetime_utc', 'download_date'])
    df_combined = df_combined.drop_duplicates(subset=['datetime_utc'], keep='last')
    df_combined = df_combined.sort_values('datetime_utc').reset_index(drop=True)

    print(f"  Combined records: {len(df_combined):,}")
    print(f"  Duplicates removed: {len(df_existing) + len(new_df) - len(df_combined):,}")
    print()

    return df_combined


def verify_continuity(df: pd.DataFrame) -> None:
    """
    Check for gaps in 15-minute time series

    Args:
        df: DataFrame with datetime_utc column
    """
    print("Verifying data continuity...")
    df_check = df.copy()
    df_check['time_diff'] = df_check['datetime_utc'].diff()
    gaps = df_check[df_check['time_diff'] > pd.Timedelta(minutes=15)]

    if len(gaps) > 0:
        print(f"⚠️  Found {len(gaps)} gaps in data:")
        for idx, row in gaps.head(5).iterrows():
            print(f"   Gap at {row['datetime_utc']}: {row['time_diff']}")
    else:
        print("✅ No gaps - continuous 15-minute intervals")
    print()


def save_dataset(df: pd.DataFrame, data_dir: Path) -> Path:
    """
    Save dataset with date range in filename

    Args:
        df: DataFrame to save
        data_dir: Output directory

    Returns:
        Path to saved file
    """
    min_date = df['datetime_utc'].min().strftime('%Y-%m-%d')
    max_date = df['datetime_utc'].max().strftime('%Y-%m-%d')
    output_file = data_dir / f"rebap_de_{min_date}_{max_date}.csv"

    print(f"Saving to: {output_file}")
    df.to_csv(output_file, index=False)
    print(f"  File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
    print()

    return output_file


def main(days_back: int = 7):
    """
    Main update function

    Args:
        days_back: Number of days to download (default: 7 for weekly updates)
    """
    load_dotenv()

    print("=" * 80)
    print("Automated reBAP Data Download")
    print("=" * 80)
    print(f"Date: {datetime.now()}")
    print()

    # Load credentials
    try:
        client_id, client_secret = load_credentials_from_env()
    except ValueError as e:
        print(f"❌ {e}")
        sys.exit(1)

    # Initialize OAuth client
    client = NetztransparenzOAuthClient(client_id, client_secret)

    # Test connection
    if not client.test_connection():
        sys.exit(1)

    print()

    # Determine download range
    data_dir = Path("/pool/ssd8tb/data/iso/ENTSO_E/csv_files/rebap")
    latest_date = get_latest_date_in_dataset(data_dir)

    # Download from latest date to now (with 1-day overlap for safety)
    date_from = latest_date - timedelta(days=1)
    date_to = datetime.now().astimezone()  # Make timezone-aware

    print(f"Download range:")
    print(f"  From: {date_from}")
    print(f"  To: {date_to}")
    print(f"  Days: {(date_to.replace(tzinfo=None) - date_from.replace(tzinfo=None)).days}")
    print()

    # Download data
    print("Downloading reBAP data from API...")
    try:
        api_data = client.get_rebap_data(
            date_from.strftime('%Y-%m-%dT00:00:00'),
            date_to.strftime('%Y-%m-%dT23:59:59')
        )
        print(f"✅ Downloaded {len(api_data)} records from API")
        print()
    except Exception as e:
        print(f"❌ Download failed: {e}")
        sys.exit(1)

    # Parse API response
    print("Converting API data to standardized format...")
    df_new = parse_rebap_api_response(api_data)

    if len(df_new) == 0:
        print("⚠️  No new data received from API")
        sys.exit(0)

    print(f"  New records: {len(df_new):,}")
    print(f"  Date range: {df_new['datetime_utc'].min()} to {df_new['datetime_utc'].max()}")
    print()

    # Merge with existing data
    df_final = merge_with_existing_data(df_new, data_dir)

    # Verify continuity
    verify_continuity(df_final)

    # Save
    output_file = save_dataset(df_final, data_dir)

    # Summary
    print("Summary:")
    print(f"  Total records: {len(df_final):,}")
    print(f"  Date range: {df_final['datetime_utc'].min()} to {df_final['datetime_utc'].max()}")
    print(f"  Price range (undersupply): €{df_final['rebap_undersupply_eur_per_mwh'].min():.2f} to €{df_final['rebap_undersupply_eur_per_mwh'].max():.2f}/MWh")
    print()

    # Sample
    print("Latest 5 records:")
    print(df_final.tail(5).to_string(index=False))
    print()

    print("=" * 80)
    print("✅ reBAP update complete!")
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Download and update reBAP data')
    parser.add_argument('--days', type=int, default=7,
                        help='Number of days to download (default: 7)')
    args = parser.parse_args()

    main(days_back=args.days)
