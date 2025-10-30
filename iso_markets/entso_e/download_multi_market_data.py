#!/usr/bin/env python3
"""
Download electricity market data for multiple European markets
Markets: Italy, France, Netherlands, Belgium, Austria, Denmark
"""

import sys
import pandas as pd
from pathlib import Path
from datetime import datetime
from entsoe import EntsoePandasClient
from dotenv import load_dotenv
import os


# Market configurations
MARKETS = {
    'IT': {
        'name': 'Italy',
        'zones': ['IT_CNOR', 'IT_CSUD', 'IT_NORD', 'IT_SARD', 'IT_SICI', 'IT_SUD'],  # Italy has zones
        'single_zone': False
    },
    'FR': {
        'name': 'France',
        'zones': ['FR'],
        'single_zone': True
    },
    'NL': {
        'name': 'Netherlands',
        'zones': ['NL'],
        'single_zone': True
    },
    'BE': {
        'name': 'Belgium',
        'zones': ['BE'],
        'single_zone': True
    },
    'AT': {
        'name': 'Austria',
        'zones': ['AT'],
        'single_zone': True
    },
    'DK_1': {
        'name': 'Denmark West',
        'zones': ['DK_1'],
        'single_zone': True
    },
    'DK_2': {
        'name': 'Denmark East',
        'zones': ['DK_2'],
        'single_zone': True
    }
}


def download_da_prices_chunked(client: EntsoePandasClient, country_code: str, start_date: pd.Timestamp,
                                end_date: pd.Timestamp, output_dir: Path):
    """Download day-ahead prices with yearly chunking"""
    market_name = MARKETS.get(country_code, {}).get('name', country_code)

    print("=" * 80)
    print(f"{market_name} - Day-Ahead Prices")
    print("=" * 80)

    all_data = []
    current_year = start_date.year
    end_year = end_date.year

    for year in range(current_year, end_year + 1):
        year_start = pd.Timestamp(f'{year}-01-01', tz='UTC')
        year_end = pd.Timestamp(f'{year}-12-31 23:59:59', tz='UTC')

        if year == current_year:
            year_start = start_date
        if year == end_year:
            year_end = end_date

        print(f"  Year {year}: {year_start.date()} to {year_end.date()}")

        try:
            df = client.query_day_ahead_prices(country_code, start=year_start, end=year_end)

            if isinstance(df, pd.Series):
                df = df.reset_index()
                df.columns = ['datetime_utc', 'price_eur_per_mwh']

            all_data.append(df)
            print(f"    Retrieved: {len(df):,} records")

        except Exception as e:
            print(f"    ❌ Error: {str(e)[:100]}")

    if not all_data:
        print("  No data available via ENTSO-E")
        print()
        return False

    df_combined = pd.concat(all_data, ignore_index=True)
    df_combined = df_combined.drop_duplicates()
    df_combined = df_combined.sort_values('datetime_utc')

    print()
    print(f"  Total records: {len(df_combined):,}")
    print(f"  Date range: {df_combined['datetime_utc'].min()} to {df_combined['datetime_utc'].max()}")

    output_file = output_dir / f"da_prices_{country_code}_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}.csv"
    df_combined.to_csv(output_file, index=False)
    size_mb = output_file.stat().st_size / 1024 / 1024
    print(f"  Saved: {output_file.name} ({size_mb:.2f} MB)")
    print()

    return True


def download_imbalance_prices(client: EntsoePandasClient, country_code: str, start_date: pd.Timestamp,
                               end_date: pd.Timestamp, output_dir: Path):
    """Download imbalance prices"""
    market_name = MARKETS.get(country_code, {}).get('name', country_code)

    print(f"{market_name} - Imbalance Prices")
    print("-" * 40)

    try:
        df = client.query_imbalance_prices(country_code, start=start_date, end=end_date)

        if df is not None and len(df) > 0:
            df = df.reset_index()
            print(f"  Records: {len(df):,}")

            output_file = output_dir / f"imbalance_prices_{country_code}_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}.csv"
            df.to_csv(output_file, index=False)
            size_mb = output_file.stat().st_size / 1024 / 1024
            print(f"  Saved: {output_file.name} ({size_mb:.2f} MB)")
            return True
        else:
            print("  No data available")
            return False

    except Exception as e:
        print(f"  ❌ Error: {str(e)[:80]}")
        return False


def download_generation(client: EntsoePandasClient, country_code: str, start_date: pd.Timestamp,
                        end_date: pd.Timestamp, output_dir: Path):
    """Download generation by type"""
    market_name = MARKETS.get(country_code, {}).get('name', country_code)

    print(f"{market_name} - Generation by Type")
    print("-" * 40)

    try:
        df = client.query_generation(country_code, start=start_date, end=end_date)

        if df is not None and len(df) > 0:
            df = df.reset_index()
            print(f"  Records: {len(df):,}")

            output_file = output_dir / f"generation_by_type_{country_code}_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}.csv"
            df.to_csv(output_file, index=False)
            size_mb = output_file.stat().st_size / 1024 / 1024
            print(f"  Saved: {output_file.name} ({size_mb:.2f} MB)")
            return True
        else:
            print("  No data available")
            return False

    except Exception as e:
        print(f"  ❌ Error: {str(e)[:80]}")
        return False


def download_market_data(client: EntsoePandasClient, country_code: str, start_date: pd.Timestamp,
                         end_date: pd.Timestamp, base_dir: Path):
    """Download all available data for a market"""
    market_name = MARKETS.get(country_code, {}).get('name', country_code)

    print()
    print("=" * 80)
    print(f"DOWNLOADING: {market_name} ({country_code})")
    print("=" * 80)
    print()

    # Create market directory
    market_dir = base_dir / country_code.lower().replace('_', '')
    market_dir.mkdir(exist_ok=True, parents=True)

    success_count = 0

    # Day-ahead prices
    if download_da_prices_chunked(client, country_code, start_date, end_date, market_dir):
        success_count += 1

    # Imbalance prices
    print()
    if download_imbalance_prices(client, country_code, start_date, end_date, market_dir):
        success_count += 1

    # Generation by type
    print()
    if download_generation(client, country_code, start_date, end_date, market_dir):
        success_count += 1

    print()
    print(f"  ✅ {market_name}: {success_count} dataset(s) downloaded")
    print()

    return success_count


def main():
    load_dotenv()

    print("=" * 80)
    print("MULTI-MARKET EUROPEAN ELECTRICITY DATA DOWNLOAD")
    print("=" * 80)
    print(f"Date: {datetime.now()}")
    print()

    # Get API key
    api_key = os.getenv('ENTSO_E_API_KEY')
    if not api_key:
        print("❌ ENTSO_E_API_KEY not found in environment")
        sys.exit(1)

    client = EntsoePandasClient(api_key=api_key)

    # Date range
    start_date = pd.Timestamp('2019-01-01', tz='UTC')
    end_date = pd.Timestamp(datetime.now(), tz='UTC')

    print(f"Period: {start_date.date()} to {end_date.date()}")
    print()

    # Base output directory
    base_dir = Path("/pool/ssd8tb/data/iso/ENTSO_E/csv_files")

    # Download priority markets
    priority_markets = ['IT', 'FR', 'NL', 'BE', 'AT', 'DK_1', 'DK_2']

    total_success = 0
    total_markets = len(priority_markets)

    for country_code in priority_markets:
        try:
            count = download_market_data(client, country_code, start_date, end_date, base_dir)
            total_success += (1 if count > 0 else 0)
        except Exception as e:
            print(f"❌ Error downloading {country_code}: {e}")
            print()

    print("=" * 80)
    print(f"✅ MULTI-MARKET DOWNLOAD COMPLETE")
    print(f"   Successfully downloaded data for {total_success}/{total_markets} markets")
    print("=" * 80)


if __name__ == "__main__":
    main()
