#!/usr/bin/env python3
"""
Automated download of high-value German grid datasets
- Redispatch (congestion management)
- Curtailment (renewable shutdowns)
- VoAA (Value of Avoided Activation)
- Spot Market Prices
- Capacity Reserve
- curative Redispatch
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


def download_redispatch(client: NetztransparenzOAuthClient, start_date: str, end_date: str, output_dir: Path):
    """Download redispatch (congestion management) data"""
    print("=" * 80)
    print("REDISPATCH (Congestion Management)")
    print("=" * 80)

    endpoint = f"/data/redispatch/{start_date}/{end_date}"
    csv_data = client._make_request(endpoint)

    df = parse_csv_response(csv_data)
    if len(df) == 0:
        print("  No data available")
        return

    print(f"  Records: {len(df):,}")
    if 'Datum' in df.columns:
        df['date'] = pd.to_datetime(df['Datum'], format='%d.%m.%Y', errors='coerce')
        print(f"  Date range: {df['date'].min()} to {df['date'].max()}")

    output_file = output_dir / f"redispatch_{start_date}_{end_date}.csv"
    df.to_csv(output_file, index=False)
    size_mb = output_file.stat().st_size / 1024 / 1024
    print(f"  Saved: {output_file.name} ({size_mb:.2f} MB)")
    print()


def download_curtailment(client: NetztransparenzOAuthClient, start_date: str, end_date: str, output_dir: Path):
    """Download renewable curtailment data"""
    print("=" * 80)
    print("CURTAILMENT (Renewable Abregelung)")
    print("=" * 80)

    # Designated curtailment
    print("1. Designated Curtailment (Ausgewiesene ABSM)")
    endpoint1 = f"/data/AusgewieseneABSM/{start_date}/{end_date}"
    csv_data1 = client._make_request(endpoint1)
    df1 = parse_csv_response(csv_data1)

    if len(df1) > 0:
        print(f"  Records: {len(df1):,}")
        output_file1 = output_dir / f"curtailment_designated_{start_date}_{end_date}.csv"
        df1.to_csv(output_file1, index=False)
        size_mb = output_file1.stat().st_size / 1024 / 1024
        print(f"  Saved: {output_file1.name} ({size_mb:.2f} MB)")
    else:
        print("  No data available")

    print()

    # Allocated curtailment
    print("2. Allocated Curtailment (Zugeteilte ABSM)")
    endpoint2 = f"/data/ZugeteilteABSM/{start_date}/{end_date}"
    csv_data2 = client._make_request(endpoint2)
    df2 = parse_csv_response(csv_data2)

    if len(df2) > 0:
        print(f"  Records: {len(df2):,}")
        output_file2 = output_dir / f"curtailment_allocated_{start_date}_{end_date}.csv"
        df2.to_csv(output_file2, index=False)
        size_mb = output_file2.stat().st_size / 1024 / 1024
        print(f"  Saved: {output_file2.name} ({size_mb:.2f} MB)")
    else:
        print("  No data available")

    print()


def download_voaa(client: NetztransparenzOAuthClient, start_date: str, end_date: str, output_dir: Path):
    """Download VoAA (Value of Avoided Activation) data"""
    print("=" * 80)
    print("VoAA (Value of Avoided Activation)")
    print("=" * 80)

    endpoint = f"/data/NrvSaldo/VoAA/Qualitaetsgesichert/{start_date}/{end_date}"
    csv_data = client._make_request(endpoint)

    df = parse_csv_response(csv_data)
    if len(df) == 0:
        print("  No data available")
        return

    print(f"  Records: {len(df):,}")

    # Parse timestamps if available
    if 'Datum' in df.columns and 'von' in df.columns:
        df['datetime_start'] = pd.to_datetime(
            df['Datum'] + ' ' + df['von'],
            format='%d.%m.%Y %H:%M',
            errors='coerce'
        )
        df['datetime_utc'] = df['datetime_start'].dt.tz_localize('UTC')
        print(f"  Date range: {df['datetime_utc'].min()} to {df['datetime_utc'].max()}")

    output_file = output_dir / f"voaa_{start_date}_{end_date}.csv"
    df.to_csv(output_file, index=False)
    size_mb = output_file.stat().st_size / 1024 / 1024
    print(f"  Saved: {output_file.name} ({size_mb:.2f} MB)")
    print()


def download_spot_prices(client: NetztransparenzOAuthClient, start_date: str, end_date: str, output_dir: Path):
    """Download spot market prices from all exchanges"""
    print("=" * 80)
    print("SPOT MARKET PRICES (All Exchanges)")
    print("=" * 80)

    endpoint = f"/data/Spotmarktpreise/{start_date}/{end_date}"
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

    output_file = output_dir / f"spot_prices_{start_date}_{end_date}.csv"
    df.to_csv(output_file, index=False)
    size_mb = output_file.stat().st_size / 1024 / 1024
    print(f"  Saved: {output_file.name} ({size_mb:.2f} MB)")
    print()


def download_curative_redispatch(client: NetztransparenzOAuthClient, start_date: str, end_date: str, output_dir: Path):
    """Download curative redispatch (kRD) data"""
    print("=" * 80)
    print("CURATIVE REDISPATCH (kRD)")
    print("=" * 80)

    endpoint = f"/data/VorhaltungkRD/{start_date}/{end_date}"
    csv_data = client._make_request(endpoint)

    df = parse_csv_response(csv_data)
    if len(df) == 0:
        print("  No data available")
        return

    print(f"  Records: {len(df):,}")
    if 'Datum' in df.columns:
        df['date'] = pd.to_datetime(df['Datum'], format='%d.%m.%Y', errors='coerce')
        print(f"  Date range: {df['date'].min()} to {df['date'].max()}")

    output_file = output_dir / f"curative_redispatch_{start_date}_{end_date}.csv"
    df.to_csv(output_file, index=False)
    size_mb = output_file.stat().st_size / 1024 / 1024
    print(f"  Saved: {output_file.name} ({size_mb:.2f} MB)")
    print()


def download_capacity_reserve(client: NetztransparenzOAuthClient, start_date: str, end_date: str, output_dir: Path):
    """Download capacity reserve data"""
    print("=" * 80)
    print("CAPACITY RESERVE (Strategic Reserve)")
    print("=" * 80)

    endpoint = f"/data/Kapazitaetsreserve/{start_date}/{end_date}"
    csv_data = client._make_request(endpoint)

    df = parse_csv_response(csv_data)
    if len(df) == 0:
        print("  No data available")
        return

    print(f"  Records: {len(df):,}")
    if 'Datum' in df.columns:
        df['date'] = pd.to_datetime(df['Datum'], format='%d.%m.%Y', errors='coerce')
        print(f"  Date range: {df['date'].min()} to {df['date'].max()}")

    output_file = output_dir / f"capacity_reserve_{start_date}_{end_date}.csv"
    df.to_csv(output_file, index=False)
    size_mb = output_file.stat().st_size / 1024 / 1024
    print(f"  Saved: {output_file.name} ({size_mb:.2f} MB)")
    print()


def main():
    load_dotenv()

    print("=" * 80)
    print("GERMAN GRID DATA DOWNLOAD")
    print("High-Value Datasets for BESS Operations")
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
    start_date = "2019-01-01T00:00:00"
    end_date = datetime.now().strftime('%Y-%m-%dT23:59:59')

    # Output directories
    base_dir = Path("/pool/ssd8tb/data/iso/ENTSO_E/csv_files")

    redispatch_dir = base_dir / "redispatch"
    redispatch_dir.mkdir(exist_ok=True)

    curtailment_dir = base_dir / "curtailment"
    curtailment_dir.mkdir(exist_ok=True)

    voaa_dir = base_dir / "voaa"
    voaa_dir.mkdir(exist_ok=True)

    spot_dir = base_dir / "spot_prices"
    spot_dir.mkdir(exist_ok=True)

    curative_rd_dir = base_dir / "curative_redispatch"
    curative_rd_dir.mkdir(exist_ok=True)

    capacity_dir = base_dir / "capacity_reserve"
    capacity_dir.mkdir(exist_ok=True)

    # Download all datasets
    try:
        download_redispatch(client, start_date, end_date, redispatch_dir)
    except Exception as e:
        print(f"  ❌ Error: {e}")
        print()

    try:
        download_curtailment(client, start_date, end_date, curtailment_dir)
    except Exception as e:
        print(f"  ❌ Error: {e}")
        print()

    try:
        download_voaa(client, start_date, end_date, voaa_dir)
    except Exception as e:
        print(f"  ❌ Error: {e}")
        print()

    try:
        download_spot_prices(client, start_date, end_date, spot_dir)
    except Exception as e:
        print(f"  ❌ Error: {e}")
        print()

    try:
        download_curative_redispatch(client, start_date, end_date, curative_rd_dir)
    except Exception as e:
        print(f"  ❌ Error: {e}")
        print()

    try:
        download_capacity_reserve(client, start_date, end_date, capacity_dir)
    except Exception as e:
        print(f"  ❌ Error: {e}")
        print()

    print("=" * 80)
    print("✅ GRID DATA DOWNLOAD COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
