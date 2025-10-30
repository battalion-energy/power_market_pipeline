#!/usr/bin/env python3
"""
Incremental download of grid data (redispatch, curtailment)
Downloads last N days and merges with existing data
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv
from io import StringIO

# Import OAuth client
from netztransparenz_oauth import NetztransparenzOAuthClient, load_credentials_from_env


def parse_csv_response(csv_text: str) -> pd.DataFrame:
    """Parse German CSV format from API"""
    if not csv_text or len(csv_text) == 0:
        return pd.DataFrame()

    df = pd.read_csv(
        StringIO(csv_text),
        encoding='utf-8-sig',
        sep=';',
        decimal=','
    )
    return df


def merge_with_existing(new_df: pd.DataFrame, existing_file: Path, key_columns: list = None) -> pd.DataFrame:
    """Merge new data with existing file, removing duplicates"""
    if not existing_file.exists():
        print(f"  No existing file, using new data")
        return new_df

    # Load existing
    existing = pd.read_csv(existing_file)
    print(f"  Existing: {len(existing):,} records")

    # Merge
    combined = pd.concat([existing, new_df], ignore_index=True)

    # Remove duplicates based on key columns
    if key_columns and len(key_columns) > 0:
        # Verify all key columns exist
        valid_keys = [k for k in key_columns if k in combined.columns]
        if valid_keys:
            combined = combined.drop_duplicates(subset=valid_keys, keep='last')
    else:
        # No key columns - remove exact duplicates
        combined = combined.drop_duplicates(keep='last')

    combined = combined.reset_index(drop=True)
    print(f"  Combined: {len(combined):,} records")
    print(f"  New records added: {len(combined) - len(existing):,}")

    return combined


def download_redispatch_incremental(client: NetztransparenzOAuthClient, days: int, output_dir: Path):
    """Download incremental redispatch data"""
    print("=" * 80)
    print("REDISPATCH (Incremental Update)")
    print("=" * 80)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    start_str = start_date.strftime('%Y-%m-%dT00:00:00')
    end_str = end_date.strftime('%Y-%m-%dT23:59:59')

    print(f"  Downloading: {start_date.date()} to {end_date.date()}")

    try:
        endpoint = f"/data/redispatch/{start_str}/{end_str}"
        csv_data = client._make_request(endpoint)
        df_new = parse_csv_response(csv_data)

        if len(df_new) == 0:
            print("  No new data")
            print()
            return

        print(f"  Downloaded: {len(df_new):,} records")

        # Find existing file
        existing_files = list(output_dir.glob("redispatch_*.csv"))
        if existing_files:
            existing_file = sorted(existing_files, key=lambda x: x.stat().st_mtime)[-1]

            # Merge - check what columns exist
            key_cols = []
            if 'Datum' in df_new.columns:
                key_cols.append('Datum')
            if 'von' in df_new.columns:
                key_cols.append('von')

            df_combined = merge_with_existing(df_new, existing_file, key_columns=key_cols if key_cols else None)

            # Generate new filename with extended date range
            if 'Datum' in df_combined.columns:
                df_combined['date_parsed'] = pd.to_datetime(df_combined['Datum'], format='%d.%m.%Y', errors='coerce')
                min_date = df_combined['date_parsed'].min().strftime('%Y-%m-%d')
                max_date = df_combined['date_parsed'].max().strftime('%Y-%m-%d')
                df_combined = df_combined.drop(columns=['date_parsed'])

                output_file = output_dir / f"redispatch_{min_date}_{max_date}.csv"
            else:
                output_file = output_dir / f"redispatch_{start_str}_{end_str}.csv"

            df_combined.to_csv(output_file, index=False)
            size_mb = output_file.stat().st_size / 1024 / 1024
            print(f"  Saved: {output_file.name} ({size_mb:.2f} MB)")

        else:
            # First download
            output_file = output_dir / f"redispatch_{start_str}_{end_str}.csv"
            df_new.to_csv(output_file, index=False)
            size_mb = output_file.stat().st_size / 1024 / 1024
            print(f"  Saved: {output_file.name} ({size_mb:.2f} MB)")

        print()

    except Exception as e:
        print(f"  ❌ Error: {e}")
        print()


def download_curtailment_incremental(client: NetztransparenzOAuthClient, days: int, output_dir: Path):
    """Download incremental curtailment data"""
    print("=" * 80)
    print("CURTAILMENT (Incremental Update)")
    print("=" * 80)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    start_str = start_date.strftime('%Y-%m-%dT00:00:00')
    end_str = end_date.strftime('%Y-%m-%dT23:59:59')

    print(f"  Downloading: {start_date.date()} to {end_date.date()}")
    print()

    # Designated curtailment
    print("1. Designated Curtailment")
    try:
        endpoint = f"/data/AusgewieseneABSM/{start_str}/{end_str}"
        csv_data = client._make_request(endpoint)
        df_new = parse_csv_response(csv_data)

        if len(df_new) > 0:
            print(f"  Downloaded: {len(df_new):,} records")

            # Find existing file
            existing_files = list(output_dir.glob("curtailment_designated_*.csv"))
            if existing_files:
                existing_file = sorted(existing_files, key=lambda x: x.stat().st_mtime)[-1]

                # Determine key columns
                key_cols = []
                if 'Datum' in df_new.columns:
                    key_cols.append('Datum')
                if 'von' in df_new.columns:
                    key_cols.append('von')

                df_combined = merge_with_existing(df_new, existing_file, key_columns=key_cols if key_cols else None)

                # Generate filename
                if 'Datum' in df_combined.columns:
                    df_combined['date_parsed'] = pd.to_datetime(df_combined['Datum'], format='%d.%m.%Y', errors='coerce')
                    min_date = df_combined['date_parsed'].min().strftime('%Y-%m-%d')
                    max_date = df_combined['date_parsed'].max().strftime('%Y-%m-%d')
                    df_combined = df_combined.drop(columns=['date_parsed'])
                    output_file = output_dir / f"curtailment_designated_{min_date}_{max_date}.csv"
                else:
                    output_file = output_dir / f"curtailment_designated_{start_str}_{end_str}.csv"

                df_combined.to_csv(output_file, index=False)
                size_mb = output_file.stat().st_size / 1024 / 1024
                print(f"  Saved: {output_file.name} ({size_mb:.2f} MB)")
            else:
                output_file = output_dir / f"curtailment_designated_{start_str}_{end_str}.csv"
                df_new.to_csv(output_file, index=False)
                size_mb = output_file.stat().st_size / 1024 / 1024
                print(f"  Saved: {output_file.name} ({size_mb:.2f} MB)")
        else:
            print("  No new data")

        print()

    except Exception as e:
        print(f"  ❌ Error: {e}")
        print()

    # Allocated curtailment
    print("2. Allocated Curtailment")
    try:
        endpoint = f"/data/ZugeteilteABSM/{start_str}/{end_str}"
        csv_data = client._make_request(endpoint)
        df_new = parse_csv_response(csv_data)

        if len(df_new) > 0:
            print(f"  Downloaded: {len(df_new):,} records")

            # Find existing file
            existing_files = list(output_dir.glob("curtailment_allocated_*.csv"))
            if existing_files:
                existing_file = sorted(existing_files, key=lambda x: x.stat().st_mtime)[-1]

                # Determine key columns
                key_cols = []
                if 'Datum' in df_new.columns:
                    key_cols.append('Datum')
                if 'von' in df_new.columns:
                    key_cols.append('von')

                df_combined = merge_with_existing(df_new, existing_file, key_columns=key_cols if key_cols else None)

                # Generate filename
                if 'Datum' in df_combined.columns:
                    df_combined['date_parsed'] = pd.to_datetime(df_combined['Datum'], format='%d.%m.%Y', errors='coerce')
                    min_date = df_combined['date_parsed'].min().strftime('%Y-%m-%d')
                    max_date = df_combined['date_parsed'].max().strftime('%Y-%m-%d')
                    df_combined = df_combined.drop(columns=['date_parsed'])
                    output_file = output_dir / f"curtailment_allocated_{min_date}_{max_date}.csv"
                else:
                    output_file = output_dir / f"curtailment_allocated_{start_str}_{end_str}.csv"

                df_combined.to_csv(output_file, index=False)
                size_mb = output_file.stat().st_size / 1024 / 1024
                print(f"  Saved: {output_file.name} ({size_mb:.2f} MB)")
            else:
                output_file = output_dir / f"curtailment_allocated_{start_str}_{end_str}.csv"
                df_new.to_csv(output_file, index=False)
                size_mb = output_file.stat().st_size / 1024 / 1024
                print(f"  Saved: {output_file.name} ({size_mb:.2f} MB)")
        else:
            print("  No new data")

        print()

    except Exception as e:
        print(f"  ❌ Error: {e}")
        print()


def main():
    parser = argparse.ArgumentParser(description='Download incremental grid data')
    parser.add_argument('--days', type=int, default=30,
                        help='Number of days to download (default: 30)')
    args = parser.parse_args()

    load_dotenv()

    print("=" * 80)
    print("GRID DATA INCREMENTAL UPDATE")
    print("=" * 80)
    print(f"Date: {datetime.now()}")
    print(f"Downloading last {args.days} days")
    print()

    # Load OAuth credentials
    try:
        client_id, client_secret = load_credentials_from_env()
    except ValueError as e:
        print(f"❌ {e}")
        sys.exit(1)

    client = NetztransparenzOAuthClient(client_id, client_secret)

    # Test connection
    if not client.test_connection():
        sys.exit(1)

    print()

    # Output directories
    base_dir = Path("/pool/ssd8tb/data/iso/ENTSO_E/csv_files")
    redispatch_dir = base_dir / "redispatch"
    curtailment_dir = base_dir / "curtailment"

    # Download incremental data
    download_redispatch_incremental(client, args.days, redispatch_dir)
    download_curtailment_incremental(client, args.days, curtailment_dir)

    print("=" * 80)
    print("✅ GRID DATA INCREMENTAL UPDATE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
