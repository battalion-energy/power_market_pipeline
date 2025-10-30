#!/usr/bin/env python3
"""
Download additional high-value German datasets
- Spot Market Prices (all exchanges)
- Negative Price Events (1h, 3h, 4h, 6h)
- ID-AEP (Intraday index)
"""

import sys
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


def chunk_date_range(start_date: datetime, end_date: datetime, chunk_months: int = 3):
    """Split date range into chunks to avoid API errors"""
    chunks = []
    current = start_date

    while current < end_date:
        chunk_end = min(current + timedelta(days=chunk_months * 30), end_date)
        chunks.append((current, chunk_end))
        current = chunk_end + timedelta(days=1)

    return chunks


def download_spot_prices_chunked(client: NetztransparenzOAuthClient, start_date: datetime, end_date: datetime, output_dir: Path):
    """Download spot market prices with chunking"""
    print("=" * 80)
    print("SPOT MARKET PRICES (All Exchanges)")
    print("=" * 80)
    print("Note: Using 3-month chunks to avoid API errors")
    print()

    all_data = []
    chunks = chunk_date_range(start_date, end_date, chunk_months=3)

    for i, (chunk_start, chunk_end) in enumerate(chunks, 1):
        start_str = chunk_start.strftime('%Y-%m-%dT00:00:00')
        end_str = chunk_end.strftime('%Y-%m-%dT23:59:59')

        print(f"  Chunk {i}/{len(chunks)}: {chunk_start.date()} to {chunk_end.date()}")

        try:
            endpoint = f"/data/Spotmarktpreise/{start_str}/{end_str}"
            csv_data = client._make_request(endpoint)
            df = parse_csv_response(csv_data)

            if len(df) > 0:
                all_data.append(df)
                print(f"    Retrieved: {len(df):,} records")
            else:
                print(f"    No data")
        except Exception as e:
            print(f"    ❌ Error: {e}")

    if not all_data:
        print("  No data retrieved")
        return

    # Combine all chunks
    df_combined = pd.concat(all_data, ignore_index=True)
    df_combined = df_combined.drop_duplicates()

    print()
    print(f"  Total records: {len(df_combined):,}")

    # Parse timestamps
    if 'Datum' in df_combined.columns and 'von' in df_combined.columns:
        df_combined['datetime_start'] = pd.to_datetime(
            df_combined['Datum'] + ' ' + df_combined['von'],
            format='%d.%m.%Y %H:%M',
            errors='coerce'
        )
        df_combined['datetime_utc'] = df_combined['datetime_start'].dt.tz_localize('UTC')
        print(f"  Date range: {df_combined['datetime_utc'].min()} to {df_combined['datetime_utc'].max()}")

    # Save
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    output_file = output_dir / f"spot_prices_{start_str}_{end_str}.csv"
    df_combined.to_csv(output_file, index=False)
    size_mb = output_file.stat().st_size / 1024 / 1024
    print(f"  Saved: {output_file.name} ({size_mb:.2f} MB)")
    print()


def download_negative_price_events(client: NetztransparenzOAuthClient, start_date: datetime, end_date: datetime, output_dir: Path):
    """Download negative price events for all logic types"""
    print("=" * 80)
    print("NEGATIVE PRICE EVENTS")
    print("=" * 80)

    start_str = start_date.strftime('%Y-%m-%dT00:00:00')
    end_str = end_date.strftime('%Y-%m-%dT23:59:59')

    logic_types = [1, 3, 4, 6]  # 1h, 3h, 4h, 6h consecutive negative prices

    for logic in logic_types:
        print(f"\n{logic}-Hour Consecutive Negative Prices")
        print("-" * 40)

        try:
            endpoint = f"/data/NegativePreise/{logic}/{start_str}/{end_str}"
            csv_data = client._make_request(endpoint)
            df = parse_csv_response(csv_data)

            if len(df) == 0:
                print(f"  No events found")
                continue

            print(f"  Events: {len(df):,}")

            # Parse dates if available
            if 'Datum' in df.columns:
                df['date'] = pd.to_datetime(df['Datum'], format='%d.%m.%Y', errors='coerce')
                print(f"  Date range: {df['date'].min()} to {df['date'].max()}")

            output_file = output_dir / f"negative_prices_{logic}h_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}.csv"
            df.to_csv(output_file, index=False)
            size_mb = output_file.stat().st_size / 1024 / 1024
            print(f"  Saved: {output_file.name} ({size_mb:.2f} MB)")

        except Exception as e:
            print(f"  ❌ Error: {e}")

    print()


def download_id_aep(client: NetztransparenzOAuthClient, start_date: datetime, end_date: datetime, output_dir: Path):
    """Download ID-AEP (Intraday Auction Energy Price)"""
    print("=" * 80)
    print("ID-AEP (Intraday Auction Energy Price)")
    print("=" * 80)

    start_str = start_date.strftime('%Y-%m-%dT00:00:00')
    end_str = end_date.strftime('%Y-%m-%dT23:59:59')

    print(f"  Downloading: {start_date.date()} to {end_date.date()}")

    try:
        endpoint = f"/data/IdAep/{start_str}/{end_str}"
        csv_data = client._make_request(endpoint)
        df = parse_csv_response(csv_data)

        if len(df) == 0:
            print("  No data available")
            return

        print(f"  Records: {len(df):,}")

        # Parse timestamps
        if 'Datum' in df.columns and 'von' in df.columns:
            df['datetime_start'] = pd.to_datetime(
                df['Datum'] + ' ' + df['von'],
                format='%d.%m.%Y %H:%M',
                errors='coerce'
            )
            df['datetime_utc'] = df['datetime_start'].dt.tz_localize('UTC')
            print(f"  Date range: {df['datetime_utc'].min()} to {df['datetime_utc'].max()}")

        output_file = output_dir / f"id_aep_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}.csv"
        df.to_csv(output_file, index=False)
        size_mb = output_file.stat().st_size / 1024 / 1024
        print(f"  Saved: {output_file.name} ({size_mb:.2f} MB)")

    except Exception as e:
        print(f"  ❌ Error: {e}")

    print()


def main():
    load_dotenv()

    print("=" * 80)
    print("ADDITIONAL GERMAN GRID DATA DOWNLOAD")
    print("Spot Prices, Negative Price Events, ID-AEP")
    print("=" * 80)
    print(f"Date: {datetime.now()}")
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

    # Date range: 2019-01-01 to now
    start_date = datetime(2019, 1, 1)
    end_date = datetime.now()

    # Output directories
    base_dir = Path("/pool/ssd8tb/data/iso/ENTSO_E/csv_files")

    spot_dir = base_dir / "spot_prices"
    spot_dir.mkdir(exist_ok=True)

    negative_dir = base_dir / "negative_prices"
    negative_dir.mkdir(exist_ok=True)

    id_aep_dir = base_dir / "id_aep"
    id_aep_dir.mkdir(exist_ok=True)

    # Download all datasets
    try:
        download_spot_prices_chunked(client, start_date, end_date, spot_dir)
    except Exception as e:
        print(f"❌ Spot Prices Error: {e}")
        print()

    try:
        download_negative_price_events(client, start_date, end_date, negative_dir)
    except Exception as e:
        print(f"❌ Negative Price Events Error: {e}")
        print()

    try:
        download_id_aep(client, start_date, end_date, id_aep_dir)
    except Exception as e:
        print(f"❌ ID-AEP Error: {e}")
        print()

    print("=" * 80)
    print("✅ ADDITIONAL GERMAN DATA DOWNLOAD COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
