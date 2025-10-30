#!/usr/bin/env python3
"""
Download Spain electricity market data via ENTSO-E API
- Day-ahead prices
- Ancillary services (FCR, aFRR, mFRR)
- Imbalance prices (if available)
"""

import sys
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from entsoe import EntsoePandasClient
from dotenv import load_dotenv
import os


def download_spain_da_prices(client: EntsoePandasClient, start_date: pd.Timestamp, end_date: pd.Timestamp, output_dir: Path):
    """Download Spain day-ahead prices (chunked by year)"""
    print("=" * 80)
    print("SPAIN - Day-Ahead Prices")
    print("=" * 80)
    print(f"  Period: {start_date.date()} to {end_date.date()}")
    print("  Note: Downloading year-by-year to avoid API errors")
    print()

    all_data = []
    current_year = start_date.year
    end_year = end_date.year

    for year in range(current_year, end_year + 1):
        year_start = pd.Timestamp(f'{year}-01-01', tz='UTC')
        year_end = pd.Timestamp(f'{year}-12-31 23:59:59', tz='UTC')

        # Adjust first and last year
        if year == current_year:
            year_start = start_date
        if year == end_year:
            year_end = end_date

        print(f"  Year {year}: {year_start.date()} to {year_end.date()}")

        try:
            # Spain country code
            df = client.query_day_ahead_prices('ES', start=year_start, end=year_end)

            # Convert to DataFrame format
            if isinstance(df, pd.Series):
                df = df.reset_index()
                df.columns = ['datetime_utc', 'price_eur_per_mwh']

            all_data.append(df)
            print(f"    Retrieved: {len(df):,} records")

        except Exception as e:
            print(f"    ❌ Error: {e}")

    if not all_data:
        print()
        print("  No data retrieved")
        print()
        return False

    # Combine all years
    df_combined = pd.concat(all_data, ignore_index=True)
    df_combined = df_combined.drop_duplicates(subset=['datetime_utc'])
    df_combined = df_combined.sort_values('datetime_utc')

    print()
    print(f"  Total records: {len(df_combined):,}")
    print(f"  Date range: {df_combined['datetime_utc'].min()} to {df_combined['datetime_utc'].max()}")

    # Save
    output_file = output_dir / f"da_prices_ES_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}.csv"
    df_combined.to_csv(output_file, index=False)
    size_mb = output_file.stat().st_size / 1024 / 1024
    print(f"  Saved: {output_file.name} ({size_mb:.2f} MB)")
    print()

    return True


def download_spain_ancillary_capacity(client: EntsoePandasClient, start_date: pd.Timestamp, end_date: pd.Timestamp, output_dir: Path):
    """Download Spain ancillary services capacity prices"""
    print("=" * 80)
    print("SPAIN - Ancillary Services (Capacity)")
    print("=" * 80)

    services = {
        'FCR': 'Frequency Containment Reserve',
        'aFRR': 'Automatic Frequency Restoration Reserve',
        'mFRR': 'Manual Frequency Restoration Reserve'
    }

    all_data = []

    for service_code, service_name in services.items():
        print(f"\n{service_code} - {service_name}")
        print("-" * 40)

        try:
            # Query procured balancing capacity
            df = client.query_procured_balancing_capacity(
                'ES',
                start=start_date,
                end=end_date,
                process_type=service_code
            )

            if df is not None and len(df) > 0:
                # Reset index to get datetime as column
                df = df.reset_index()
                df['service_type'] = service_code
                all_data.append(df)

                print(f"  Records: {len(df):,}")
                if 'index' in df.columns:
                    print(f"  Date range: {df['index'].min()} to {df['index'].max()}")
            else:
                print(f"  No data available")

        except Exception as e:
            print(f"  ❌ Error: {e}")

    if all_data:
        # Combine all services
        df_combined = pd.concat(all_data, ignore_index=True)

        output_file = output_dir / f"ancillary_capacity_ES_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}.csv"
        df_combined.to_csv(output_file, index=False)
        size_mb = output_file.stat().st_size / 1024 / 1024
        print()
        print(f"  Combined saved: {output_file.name} ({size_mb:.2f} MB)")
        print()
        return True
    else:
        print()
        print("  No ancillary capacity data available")
        print()
        return False


def download_spain_imbalance_prices(client: EntsoePandasClient, start_date: pd.Timestamp, end_date: pd.Timestamp, output_dir: Path):
    """Download Spain imbalance prices (if available)"""
    print("=" * 80)
    print("SPAIN - Imbalance Prices")
    print("=" * 80)

    try:
        df = client.query_imbalance_prices('ES', start=start_date, end=end_date)

        if df is not None and len(df) > 0:
            df = df.reset_index()

            print(f"  Records: {len(df):,}")

            output_file = output_dir / f"imbalance_prices_ES_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}.csv"
            df.to_csv(output_file, index=False)
            size_mb = output_file.stat().st_size / 1024 / 1024
            print(f"  Saved: {output_file.name} ({size_mb:.2f} MB)")
            print()
            return True
        else:
            print("  No imbalance price data available")
            print()
            return False

    except Exception as e:
        print(f"  ❌ Error: {e}")
        print("  Note: Spain may not publish imbalance prices via ENTSO-E")
        print()
        return False


def download_spain_generation(client: EntsoePandasClient, start_date: pd.Timestamp, end_date: pd.Timestamp, output_dir: Path):
    """Download Spain generation by type (useful for renewable penetration analysis)"""
    print("=" * 80)
    print("SPAIN - Generation by Type")
    print("=" * 80)

    try:
        df = client.query_generation('ES', start=start_date, end=end_date)

        if df is not None and len(df) > 0:
            df = df.reset_index()

            print(f"  Records: {len(df):,}")
            print(f"  Generation types: {df.columns.tolist()}")

            output_file = output_dir / f"generation_by_type_ES_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}.csv"
            df.to_csv(output_file, index=False)
            size_mb = output_file.stat().st_size / 1024 / 1024
            print(f"  Saved: {output_file.name} ({size_mb:.2f} MB)")
            print()
            return True
        else:
            print("  No generation data available")
            print()
            return False

    except Exception as e:
        print(f"  ❌ Error: {e}")
        print()
        return False


def main():
    load_dotenv()

    print("=" * 80)
    print("SPAIN ELECTRICITY MARKET DATA DOWNLOAD")
    print("=" * 80)
    print(f"Date: {datetime.now()}")
    print()

    # Get API key
    api_key = os.getenv('ENTSO_E_API_KEY')
    if not api_key:
        print("❌ ENTSO_E_API_KEY not found in environment")
        sys.exit(1)

    client = EntsoePandasClient(api_key=api_key)

    # Date range: 2019 to now
    start_date = pd.Timestamp('2019-01-01', tz='UTC')
    end_date = pd.Timestamp(datetime.now(), tz='UTC')

    print(f"Downloading data from {start_date.date()} to {end_date.date()}")
    print()

    # Output directory
    base_dir = Path("/pool/ssd8tb/data/iso/ENTSO_E/csv_files")
    spain_dir = base_dir / "spain"
    spain_dir.mkdir(exist_ok=True)

    # Download all datasets
    success_count = 0

    # Day-ahead prices (most important)
    if download_spain_da_prices(client, start_date, end_date, spain_dir):
        success_count += 1

    # Ancillary services
    if download_spain_ancillary_capacity(client, start_date, end_date, spain_dir):
        success_count += 1

    # Imbalance prices (may not be available)
    if download_spain_imbalance_prices(client, start_date, end_date, spain_dir):
        success_count += 1

    # Generation by type (useful context)
    if download_spain_generation(client, start_date, end_date, spain_dir):
        success_count += 1

    print("=" * 80)
    print(f"✅ SPAIN DATA DOWNLOAD COMPLETE")
    print(f"   Successfully downloaded {success_count} dataset(s)")
    print("=" * 80)


if __name__ == "__main__":
    main()
