#!/usr/bin/env python3
"""
Update reBAP data by merging new download with existing data
Handles German CSV format, deduplication, and verification
"""

import sys
import pandas as pd
from datetime import datetime
from pathlib import Path

def convert_german_rebap(filepath):
    """Convert German reBAP CSV to standardized format"""
    df = pd.read_csv(
        filepath,
        encoding='utf-8-sig',  # Handle BOM
        sep=';',
        decimal=','
    )

    # Parse timestamps
    df['datetime_start'] = pd.to_datetime(
        df['Datum'] + ' ' + df['von'],
        format='%d.%m.%Y %H:%M'
    )

    # Mark as UTC
    df['datetime_utc'] = df['datetime_start'].dt.tz_localize('UTC')

    # Create clean dataframe
    df_clean = pd.DataFrame({
        'datetime_utc': df['datetime_utc'],
        'rebap_undersupply_eur_per_mwh': df['reBAP unterdeckt'],
        'rebap_oversupply_eur_per_mwh': df['reBAP ueberdeckt'],
        'download_date': datetime.now()
    })

    return df_clean.sort_values('datetime_utc').reset_index(drop=True)

def main():
    if len(sys.argv) < 2:
        print("Usage: convert_rebap_update.py <new_file> [existing_file]")
        sys.exit(1)

    new_file = Path(sys.argv[1])
    existing_file = Path(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] else None

    print(f"New data: {new_file}")
    print()

    # Convert new data
    print("Converting new data...")
    df_new = convert_german_rebap(new_file)
    print(f"  New records: {len(df_new):,}")
    print(f"  Date range: {df_new['datetime_utc'].min()} to {df_new['datetime_utc'].max()}")
    print()

    # Load existing data if present
    if existing_file and existing_file.exists():
        print(f"Loading existing data: {existing_file}")
        df_existing = pd.read_csv(existing_file)
        df_existing['datetime_utc'] = pd.to_datetime(df_existing['datetime_utc'], utc=True)
        print(f"  Existing records: {len(df_existing):,}")
        print(f"  Date range: {df_existing['datetime_utc'].min()} to {df_existing['datetime_utc'].max()}")
        print()

        # Merge and deduplicate
        print("Merging data...")
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)

        # Remove duplicates (keep latest download_date)
        df_combined = df_combined.sort_values(['datetime_utc', 'download_date'])
        df_combined = df_combined.drop_duplicates(subset=['datetime_utc'], keep='last')
        df_combined = df_combined.sort_values('datetime_utc').reset_index(drop=True)

        print(f"  Combined records: {len(df_combined):,}")
        print(f"  Duplicates removed: {len(df_existing) + len(df_new) - len(df_combined):,}")
        print()

        df_final = df_combined
    else:
        print("No existing data - using new data as is")
        df_final = df_new

    # Verify continuity
    print("Verifying data continuity...")
    df_final['time_diff'] = df_final['datetime_utc'].diff()
    gaps = df_final[df_final['time_diff'] > pd.Timedelta(minutes=15)]

    if len(gaps) > 0:
        print(f"⚠️  Found {len(gaps)} gaps in data:")
        for idx, row in gaps.head(5).iterrows():
            print(f"   Gap at {row['datetime_utc']}: {row['time_diff']}")
    else:
        print("✅ No gaps - continuous 15-minute intervals")
    print()

    # Drop time_diff column
    df_final = df_final.drop(columns=['time_diff'])

    # Generate output filename
    min_date = df_final['datetime_utc'].min().strftime('%Y-%m-%d')
    max_date = df_final['datetime_utc'].max().strftime('%Y-%m-%d')
    output_dir = Path("/pool/ssd8tb/data/iso/ENTSO_E/csv_files/rebap")
    output_file = output_dir / f"rebap_de_{min_date}_{max_date}.csv"

    # Save
    print(f"Saving to: {output_file}")
    df_final.to_csv(output_file, index=False)
    print(f"  File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
    print()

    # Summary
    print("Summary:")
    print(f"  Total records: {len(df_final):,}")
    print(f"  Date range: {df_final['datetime_utc'].min()} to {df_final['datetime_utc'].max()}")
    print(f"  Price range (undersupply): €{df_final['rebap_undersupply_eur_per_mwh'].min():.2f} to €{df_final['rebap_undersupply_eur_per_mwh'].max():.2f}/MWh")
    print()

    # Sample
    print("Latest 5 records:")
    print(df_final.tail(5).to_string(index=False))

    # Archive new file
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(exist_ok=True)
    archive_name = f"rebap_raw_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    archive_path = raw_dir / archive_name
    new_file.rename(archive_path)
    print()
    print(f"✅ Archived raw file: {archive_path}")

if __name__ == "__main__":
    main()
