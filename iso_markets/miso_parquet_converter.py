#!/usr/bin/env python3
"""
MISO Parquet Converter

MEMORY OPTIMIZED: Uses chunked processing (BATCH_SIZE=50, CHUNK_SIZE=100k)

Converts MISO market data from CSV to unified parquet format.

Data Types:
- Day-Ahead Ex-Post LMP (hourly)
- Day-Ahead Ex-Ante LMP (hourly)
- Real-Time Final Settlement (hourly)
- Real-Time 5-Minute LMP
- Ancillary Services (Ramp MCPs)

Usage:
    python miso_parquet_converter.py --year 2024
    python miso_parquet_converter.py --all
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List
from zoneinfo import ZoneInfo
import glob
import gc
import zipfile
import io

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from unified_iso_parquet_converter import UnifiedISOParquetConverter


class MISOParquetConverter(UnifiedISOParquetConverter):
    """MISO-specific parquet converter."""

    # MEMORY OPTIMIZATION
    BATCH_SIZE = 50
    CHUNK_SIZE = 100000

    def __init__(
        self,
        csv_data_dir: str = "/pool/ssd8tb/data/iso/MISO/csv_files",
        parquet_output_dir: str = "/pool/ssd8tb/data/iso/unified_iso_data/parquet/miso",
        metadata_dir: str = "/pool/ssd8tb/data/iso/unified_iso_data/metadata"
    ):
        super().__init__(
            iso_name="MISO",
            csv_data_dir=csv_data_dir,
            parquet_output_dir=parquet_output_dir,
            metadata_dir=metadata_dir,
            iso_timezone="America/Chicago"  # MISO uses Central Time
        )

        # MISO AS product mapping
        self.as_product_mapping = {
            'RegMCP': 'REG',
            'SpinMCP': 'SPIN',
            'SupplementalMCP': 'SUPPLEMENTAL',
            'RampCapabilityUp': 'RAMP_UP',
            'RampCapabilityDown': 'RAMP_DOWN'
        }


    def _process_csv_files_in_batches(self, csv_files, year=None):
        """Generator that yields DataFrames from CSV files in batches (MEMORY SAFE)."""
        total_files = len(csv_files)
        self.logger.info(f"Processing {total_files} files in batches of {self.BATCH_SIZE}")

        for batch_start in range(0, total_files, self.BATCH_SIZE):
            batch_end = min(batch_start + self.BATCH_SIZE, total_files)
            batch_files = csv_files[batch_start:batch_end]

            self.logger.info(f"Processing batch {batch_start//self.BATCH_SIZE + 1}: files {batch_start+1}-{batch_end} of {total_files}")

            dfs = []
            for csv_file in batch_files:
                try:
                    chunks = []
                    for chunk in pd.read_csv(csv_file, chunksize=self.CHUNK_SIZE):
                        chunks.append(chunk)

                    if chunks:
                        df = pd.concat(chunks, ignore_index=True)
                        dfs.append(df)

                    del chunks
                    gc.collect()

                except Exception as e:
                    self.logger.error(f"Error reading {csv_file}: {e}")
                    continue

            if dfs:
                batch_df = pd.concat(dfs, ignore_index=True)
                del dfs
                gc.collect()

                yield batch_df

                del batch_df
                gc.collect()

    def _read_miso_csv_files(
        self,
        csv_dir: Path,
        year: Optional[int] = None
    ) -> pd.DataFrame:
        """Read MISO CSV files."""
        if not csv_dir.exists():
            self.logger.warning(f"CSV directory not found: {csv_dir}")
            return pd.DataFrame()

        csv_files = list(csv_dir.glob("*.csv"))

        if year:
            csv_files = [f for f in csv_files if str(year) in f.name]

        if not csv_files:
            self.logger.warning(f"No CSV files found in {csv_dir}" + (f" for year {year}" if year else ""))
            return pd.DataFrame()

        self.logger.info(f"Reading {len(csv_files)} CSV files from {csv_dir}")

        dfs = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                dfs.append(df)
            except Exception as e:
                self.logger.error(f"Error reading {csv_file}: {e}")
                continue

        if not dfs:
            return pd.DataFrame()

        combined_df = pd.concat(dfs, ignore_index=True)
        self.logger.info(f"Combined {len(combined_df):,} rows from {len(dfs)} files")

        return combined_df

    def _parse_miso_datetime(self, df: pd.DataFrame, datetime_col: str = 'MarketDateTime') -> pd.DataFrame:
        """
        Parse MISO datetime format.

        MISO typically uses MarketDateTime in format: YYYY-MM-DD HH:MM:SS
        Times are in Central Time (CT).
        """
        # Parse as Central Time, then convert to UTC
        df['datetime_local'] = pd.to_datetime(df[datetime_col])
        df['datetime_utc'] = self.normalize_datetime_to_utc(df['datetime_local'])

        return df

    def _read_miso_pivoted_csv(self, csv_file: Path) -> pd.DataFrame:
        """
        Read MISO pivoted CSV format.

        Two formats supported:
        1. Regular files with 4-line header:
           Line 1: Title
           Line 2: Date (MM/DD/YYYY)
           Line 3: Blank
           Line 4: Note about timezone (EST)
           Line 5+: Data with Node,Type,Value,HE 1,HE 2,...,HE 24

        2. Hub-only files (no header):
           Line 1: Node,Type,Value,HE 1,HE 2,...,HE 24
           Line 2+: Data rows
        """
        try:
            # Read first line to detect format
            with open(csv_file, 'r') as f:
                first_line = f.readline().strip()

            # Check if it's a hub-only file (starts with "Node,Type,Value")
            if first_line.startswith('Node,Type,Value'):
                # Hub-only format: no header, extract date from filename
                # Filename format: 20240728_da_expost_lmp_hubs_only.csv
                date_str = csv_file.name.split('_')[0]  # Get "20240728"
                file_date = pd.to_datetime(date_str, format='%Y%m%d')

                # Read CSV with headers
                df = pd.read_csv(csv_file)
            else:
                # Regular format with 4-line header
                with open(csv_file, 'r') as f:
                    f.readline()  # Skip title
                    date_line = f.readline().strip()

                # Parse the date
                file_date = pd.to_datetime(date_line)

                # Read the actual data, skipping first 4 rows
                df = pd.read_csv(csv_file, skiprows=4)

            # Unpivot the hour-ending columns
            he_columns = [col for col in df.columns if col.startswith('HE ')]

            if not he_columns:
                self.logger.warning(f"No HE columns found in {csv_file}")
                return pd.DataFrame()

            # Melt the dataframe
            df_melted = df.melt(
                id_vars=['Node', 'Type', 'Value'],
                value_vars=he_columns,
                var_name='hour_ending',
                value_name='price'
            )

            # Extract hour from 'HE 1' -> 1
            df_melted['hour'] = df_melted['hour_ending'].str.extract(r'HE (\d+)').astype(int)

            # Create datetime: hour ending 1 means hour 0 (00:00-01:00)
            # MISO uses EST, not CT!
            df_melted['datetime_local'] = file_date + pd.to_timedelta(df_melted['hour'] - 1, unit='h')

            # Filter for LMP values only
            df_lmp = df_melted[df_melted['Value'] == 'LMP'].copy()

            return df_lmp

        except Exception as e:
            self.logger.error(f"Error reading {csv_file}: {e}")
            return pd.DataFrame()

    def convert_da_energy(self, year: Optional[int] = None) -> None:
        """Convert Day-Ahead energy prices to parquet."""

        # MISO has both ex-post and ex-ante DA prices
        # Ex-post is the actual settlement price, ex-ante is the forecast

        # DA Ex-Post (primary)
        self.logger.info("Converting DA Ex-Post Energy prices with pivoted format...")
        csv_dir = self.csv_data_dir / "da_expost"

        if not csv_dir.exists():
            self.logger.warning(f"DA expost directory not found: {csv_dir}")
            return

        csv_files = list(csv_dir.glob("*.csv"))
        if year:
            csv_files = [f for f in csv_files if str(year) in f.name]

        if not csv_files:
            self.logger.warning("No DA expost CSV files found")
            return

        self.logger.info(f"Reading {len(csv_files)} DA expost CSV files with pivoted format")

        # Process files in batches to avoid memory exhaustion
        all_dfs = []
        for batch_start in range(0, len(csv_files), self.BATCH_SIZE):
            batch_end = min(batch_start + self.BATCH_SIZE, len(csv_files))
            batch_files = csv_files[batch_start:batch_end]

            self.logger.info(f"Processing MISO batch {batch_start//self.BATCH_SIZE + 1}: files {batch_start+1}-{batch_end} of {len(csv_files)}")

            dfs = []
            for csv_file in batch_files:
                df_file = self._read_miso_pivoted_csv(csv_file)
                if not df_file.empty:
                    dfs.append(df_file)

            if dfs:
                batch_df = pd.concat(dfs, ignore_index=True)
                all_dfs.append(batch_df)
                del dfs, batch_df
                gc.collect()

        if not all_dfs:
            self.logger.warning("No valid MISO DA data loaded")
            return

        df = pd.concat(all_dfs, ignore_index=True)
        del all_dfs
        gc.collect()

        market_label = "ex-post"

        # Convert datetime to UTC (MISO data is in EST, not CT!)
        # Deduplicate timestamps to avoid DST errors with many duplicate datetimes
        unique_dt = df['datetime_local'].drop_duplicates()
        unique_dt_utc = self.normalize_datetime_to_utc(unique_dt, source_tz=ZoneInfo('America/New_York'))
        dt_mapping = pd.Series(unique_dt_utc.values, index=unique_dt.values)
        df['datetime_utc'] = df['datetime_local'].map(dt_mapping)

        # For pivoted format, columns are: Node, Type, Value, hour_ending, hour, price, datetime_local, datetime_utc
        # Determine node type from Type column
        def get_location_type(type_val):
            if type_val == 'Hub':
                return 'HUB'
            elif type_val == 'Loadzone':
                return 'LOAD_ZONE'
            elif type_val == 'Interface':
                return 'INTERFACE'
            else:
                return 'NODE'

        df['settlement_location_type'] = df['Type'].apply(get_location_type)

        df_unified = pd.DataFrame({
            'datetime_utc': df['datetime_utc'],
            'datetime_local': df['datetime_local'],
            'interval_start_utc': df['datetime_utc'],
            'interval_end_utc': df['datetime_utc'] + pd.Timedelta(hours=1),
            'delivery_date': df['datetime_local'].dt.date,
            'delivery_hour': df['datetime_local'].dt.hour + 1,
            'delivery_interval': np.uint8(0),
            'interval_minutes': np.uint8(60),
            'iso': 'MISO',
            'market_type': 'DA',
            'settlement_location': df['Node'],
            'settlement_location_type': df['settlement_location_type'],
            'settlement_location_id': df['Node'].astype(str),
            'zone': None,  # No zone info in pivoted format
            'voltage_kv': None,
            'lmp_total': pd.to_numeric(df['price'], errors='coerce').astype('float64'),
            'lmp_energy': None,  # Components not available in pivoted format
            'lmp_congestion': None,
            'lmp_loss': None,
            'system_lambda': None,
            'dst_flag': None,
            'data_source': f'MISO {market_label}',
            'version': 1,
            'is_current': True
        })

        df_unified = self.enforce_price_types(df_unified)
        df_unified = df_unified.sort_values('datetime_utc')
        df_unified = df_unified.drop_duplicates(subset=['datetime_utc', 'settlement_location'], keep='last')

        is_valid, issues = self.validate_data(df_unified)
        if not is_valid:
            self.logger.warning(f"Validation issues: {issues}")

        years = df_unified['delivery_date'].apply(lambda x: x.year).unique()

        for yr in years:
            if year and yr != year:
                continue

            df_year = df_unified[df_unified['delivery_date'].apply(lambda x: x.year) == yr]

            output_dir = self.parquet_output_dir / "da_energy_hourly_nodal"
            output_file = output_dir / f"da_energy_hourly_nodal_{yr}.parquet"

            self.write_parquet_atomic(
                df_year,
                output_file,
                self.ENERGY_SCHEMA,
                compression='snappy'
            )

        # Extract node metadata
        nodes = self.extract_unique_locations(
            df_unified,
            location_col='settlement_location',
            location_type_col='settlement_location_type',
            location_id_col='settlement_location_id'
        )

        self.save_metadata_json('nodes', {
            'iso': 'MISO',
            'last_updated': datetime.now().isoformat(),
            'total_nodes': len(nodes),
            'nodes': nodes
        })

    def convert_rt_energy(self, year: Optional[int] = None) -> None:
        """Convert Real-Time energy prices to parquet."""

        # RT Hourly
        self.logger.info("Converting RT Energy (Hourly) prices...")
        csv_dir = self.csv_data_dir / "rt_final"
        df = self._read_miso_csv_files(csv_dir, year)

        if not df.empty:
            df = self._parse_miso_datetime(df)

            # Similar column mapping as DA
            node_col = next((col for col in df.columns if col in ['Node', 'PNODE', 'Location']), df.columns[0])
            lmp_col = next((col for col in df.columns if 'lmp' in col.lower() or 'price' in col.lower()), 'Value')

            df_unified = pd.DataFrame({
                'datetime_utc': df['datetime_utc'],
                'datetime_local': df['datetime_local'],
                'interval_start_utc': df['datetime_utc'],
                'interval_end_utc': df['datetime_utc'] + pd.Timedelta(hours=1),
                'delivery_date': df['datetime_local'].dt.date,
                'delivery_hour': df['datetime_local'].dt.hour + 1,
                'delivery_interval': np.uint8(0),
                'interval_minutes': np.uint8(60),
                'iso': 'MISO',
                'market_type': 'RT',
                'settlement_location': df[node_col],
                'settlement_location_type': 'NODE',
                'settlement_location_id': df[node_col].astype(str),
                'zone': df.get('Zone', None),
                'voltage_kv': None,
                'lmp_total': df[lmp_col].astype('float64'),
                'lmp_energy': None,
                'lmp_congestion': None,
                'lmp_loss': None,
                'system_lambda': None,
                'dst_flag': None,
                'data_source': 'MISO RT Final',
                'version': 1,
                'is_current': True
            })

            df_unified = self.enforce_price_types(df_unified)
            df_unified = df_unified.sort_values('datetime_utc')

            years = df_unified['delivery_date'].apply(lambda x: x.year).unique()

            for yr in years:
                if year and yr != year:
                    continue

                df_year = df_unified[df_unified['delivery_date'].apply(lambda x: x.year) == yr]

                output_dir = self.parquet_output_dir / "rt_energy_hourly_nodal"
                output_file = output_dir / f"rt_energy_hourly_nodal_{yr}.parquet"

                self.write_parquet_atomic(
                    df_year,
                    output_file,
                    self.ENERGY_SCHEMA,
                    compression='snappy'
                )

        # RT 5-Minute
        self.logger.info("Converting RT Energy (5-Minute) prices...")
        csv_dir = self.csv_data_dir / "rt_5min_lmp"
        df_5min = self._read_miso_csv_files(csv_dir, year)

        if not df_5min.empty:
            df_5min = self._parse_miso_datetime(df_5min)

            node_col = next((col for col in df_5min.columns if col in ['Node', 'PNODE', 'Location']), df_5min.columns[0])
            lmp_col = next((col for col in df_5min.columns if 'lmp' in col.lower() or 'price' in col.lower()), 'Value')

            df_5min_unified = pd.DataFrame({
                'datetime_utc': df_5min['datetime_utc'],
                'datetime_local': df_5min['datetime_local'],
                'interval_start_utc': df_5min['datetime_utc'],
                'interval_end_utc': df_5min['datetime_utc'] + pd.Timedelta(minutes=5),
                'delivery_date': df_5min['datetime_local'].dt.date,
                'delivery_hour': df_5min['datetime_local'].dt.hour + 1,
                'delivery_interval': (df_5min['datetime_local'].dt.minute // 5 + 1).astype('uint8'),
                'interval_minutes': np.uint8(5),
                'iso': 'MISO',
                'market_type': 'RT',
                'settlement_location': df_5min[node_col],
                'settlement_location_type': 'NODE',
                'settlement_location_id': df_5min[node_col].astype(str),
                'zone': df_5min.get('Zone', None),
                'voltage_kv': None,
                'lmp_total': df_5min[lmp_col].astype('float64'),
                'lmp_energy': None,
                'lmp_congestion': None,
                'lmp_loss': None,
                'system_lambda': None,
                'dst_flag': None,
                'data_source': 'MISO RT 5-min',
                'version': 1,
                'is_current': True
            })

            df_5min_unified = self.enforce_price_types(df_5min_unified)
            df_5min_unified = df_5min_unified.sort_values('datetime_utc')

            years = df_5min_unified['delivery_date'].apply(lambda x: x.year).unique()

            for yr in years:
                if year and yr != year:
                    continue

                df_year = df_5min_unified[df_5min_unified['delivery_date'].apply(lambda x: x.year) == yr]

                output_dir = self.parquet_output_dir / "rt_energy_5min_nodal"
                output_file = output_dir / f"rt_energy_5min_nodal_{yr}.parquet"

                self.write_parquet_atomic(
                    df_year,
                    output_file,
                    self.ENERGY_SCHEMA,
                    compression='snappy'
                )

    def convert_ancillary_services(self, year: Optional[int] = None) -> None:
        """Convert ancillary services to parquet."""
        self.logger.info("Converting Ancillary Services...")

        csv_dir = self.csv_data_dir / "ancillary_services"
        df = self._read_miso_csv_files(csv_dir, year)

        if df.empty:
            self.logger.warning("No AS data to convert")
            return

        df = self._parse_miso_datetime(df)

        # MISO AS column mapping
        as_product_col = next((col for col in df.columns if 'product' in col.lower() or 'type' in col.lower()), 'Product')
        price_col = next((col for col in df.columns if 'mcp' in col.lower() or 'price' in col.lower()), 'MCP')

        df_unified = pd.DataFrame({
            'datetime_utc': df['datetime_utc'],
            'datetime_local': df['datetime_local'],
            'interval_start_utc': df['datetime_utc'],
            'interval_end_utc': df['datetime_utc'] + pd.Timedelta(hours=1),
            'delivery_date': df['datetime_local'].dt.date,
            'delivery_hour': df['datetime_local'].dt.hour + 1,
            'interval_minutes': np.uint8(60),
            'iso': 'MISO',
            'market_type': 'DA',
            'as_product': df[as_product_col],
            'as_product_standard': df[as_product_col].map(self.as_product_mapping).fillna('OTHER'),
            'as_region': df.get('Region', None),
            'market_clearing_price': df[price_col].astype('float64'),
            'cleared_quantity_mw': df.get('ClearedMW', None),
            'unit': '$/MW',
            'data_source': 'MISO',
            'version': 1,
            'is_current': True
        })

        df_unified = df_unified.sort_values('datetime_utc')

        years = df_unified['delivery_date'].apply(lambda x: x.year).unique()

        for yr in years:
            if year and yr != year:
                continue

            df_year = df_unified[df_unified['delivery_date'].apply(lambda x: x.year) == yr]

            output_dir = self.parquet_output_dir / "as_hourly"
            output_file = output_dir / f"as_hourly_{yr}.parquet"

            self.write_parquet_atomic(
                df_year,
                output_file,
                self.AS_SCHEMA,
                compression='snappy'
            )

        # Extract AS product metadata
        as_products = []
        for product in df_unified['as_product'].unique():
            if pd.notna(product):
                as_products.append({
                    'product_name': product,
                    'product_standard': self.as_product_mapping.get(product, 'OTHER'),
                    'unit': '$/MW',
                    'active': True
                })

        self.save_metadata_json('ancillary_services', {
            'iso': 'MISO',
            'last_updated': datetime.now().isoformat(),
            'products': as_products
        })


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Convert MISO data to unified parquet format')
    parser.add_argument('--year', type=int, help='Specific year to process')
    parser.add_argument('--all', action='store_true', help='Process all years')
    parser.add_argument('--da-only', action='store_true', help='Only convert DA prices')
    parser.add_argument('--rt-only', action='store_true', help='Only convert RT prices')
    parser.add_argument('--as-only', action='store_true', help='Only convert AS prices')
    parser.add_argument('--csv-dir', help='Override CSV data directory')
    parser.add_argument('--output-dir', help='Override output directory')

    args = parser.parse_args()

    kwargs = {}
    if args.csv_dir:
        kwargs['csv_data_dir'] = args.csv_dir
    if args.output_dir:
        kwargs['parquet_output_dir'] = args.output_dir

    converter = MISOParquetConverter(**kwargs)

    convert_da = not (args.rt_only or args.as_only)
    convert_rt = not (args.da_only or args.as_only)
    convert_as = not (args.da_only or args.rt_only)

    converter.run_full_conversion(
        year=args.year,
        convert_da=convert_da,
        convert_rt=convert_rt,
        convert_as=convert_as
    )


if __name__ == "__main__":
    main()
