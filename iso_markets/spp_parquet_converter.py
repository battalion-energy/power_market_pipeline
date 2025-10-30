#!/usr/bin/env python3
"""
SPP Parquet Converter

MEMORY OPTIMIZED: Uses chunked processing (BATCH_SIZE=50, CHUNK_SIZE=100k)

Converts Southwest Power Pool market data from CSV to unified parquet format.

Usage:
    python spp_parquet_converter.py --year 2024
    python spp_parquet_converter.py --all
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import gc
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from unified_iso_parquet_converter import UnifiedISOParquetConverter


class SPPParquetConverter(UnifiedISOParquetConverter):
    """SPP-specific parquet converter."""

    # MEMORY OPTIMIZATION
    BATCH_SIZE = 50
    CHUNK_SIZE = 100000

    def __init__(
        self,
        csv_data_dir: str = "/pool/ssd8tb/data/iso/SPP/csv_files",
        parquet_output_dir: str = "/pool/ssd8tb/data/iso/unified_iso_data/parquet/spp",
        metadata_dir: str = "/pool/ssd8tb/data/iso/unified_iso_data/metadata"
    ):
        super().__init__(
            iso_name="SPP",
            csv_data_dir=csv_data_dir,
            parquet_output_dir=parquet_output_dir,
            metadata_dir=metadata_dir,
            iso_timezone="America/Chicago"  # SPP uses Central Time
        )

    def _read_spp_csv_files(self, csv_dir: Path, year=None) -> pd.DataFrame:
        """Read SPP CSV files."""
        if not csv_dir.exists():
            self.logger.warning(f"CSV directory not found: {csv_dir}")
            return pd.DataFrame()

        csv_files = list(csv_dir.glob("*.csv"))
        if year:
            csv_files = [f for f in csv_files if str(year) in f.name]

        if not csv_files:
            return pd.DataFrame()

        self.logger.info(f"Reading {len(csv_files)} CSV files from {csv_dir} in batches")

        # Process files in batches to avoid memory exhaustion
        all_dfs = []
        for batch_start in range(0, len(csv_files), self.BATCH_SIZE):
            batch_end = min(batch_start + self.BATCH_SIZE, len(csv_files))
            batch_files = csv_files[batch_start:batch_end]

            self.logger.info(f"Processing SPP batch {batch_start//self.BATCH_SIZE + 1}: files {batch_start+1}-{batch_end} of {len(csv_files)}")

            dfs = []
            for csv_file in batch_files:
                try:
                    df = pd.read_csv(csv_file)
                    dfs.append(df)
                except Exception as e:
                    self.logger.error(f"Error reading {csv_file}: {e}")

            if dfs:
                batch_df = pd.concat(dfs, ignore_index=True)
                all_dfs.append(batch_df)
                del dfs, batch_df
                gc.collect()

        if not all_dfs:
            return pd.DataFrame()

        combined_df = pd.concat(all_dfs, ignore_index=True)
        del all_dfs
        gc.collect()

        return combined_df

    def convert_da_energy(self, year=None) -> None:
        """Convert Day-Ahead energy prices."""
        csv_dir = self.csv_data_dir / "da_lmp"
        df = self._read_spp_csv_files(csv_dir, year)

        if df.empty:
            self.logger.warning("No DA data to convert")
            return

        # Parse datetime (SPP data is timezone-aware, convert directly to UTC)
        # Schema v2.0.0: Keep datetime_local timezone-aware!
        datetime_col = next((col for col in df.columns if 'date' in col.lower() or 'time' in col.lower() or 'interval' in col.lower()), df.columns[0])
        df['datetime_utc'] = pd.to_datetime(df[datetime_col], utc=True)
        df['datetime_local'] = df['datetime_utc'].dt.tz_convert('America/Chicago')

        location_col = next((col for col in df.columns if 'settlement' in col.lower() or 'pnode' in col.lower() or 'location' in col.lower()), 'Settlement Location')
        lmp_col = next((col for col in df.columns if 'lmp' in col.lower() or 'price' in col.lower()), 'LMP')

        df_unified = pd.DataFrame({
            'datetime_utc': df['datetime_utc'],
            'datetime_local': df['datetime_local'],
            'interval_start_utc': df['datetime_utc'],
            'interval_end_utc': df['datetime_utc'] + pd.Timedelta(hours=1),
            'delivery_date': df['datetime_local'].dt.date,
            'delivery_hour': df['datetime_local'].dt.hour + 1,
            'delivery_interval': np.uint8(0),
            'interval_minutes': np.uint8(60),
            'iso': 'SPP',
            'market_type': 'DA',
            'settlement_location': df[location_col],
            'settlement_location_type': 'SETTLEMENT_LOCATION',
            'settlement_location_id': df[location_col].astype(str),
            'zone': None,
            'voltage_kv': None,
            'lmp_total': df[lmp_col].astype('float64'),
            'lmp_energy': None,
            'lmp_congestion': None,
            'lmp_loss': None,
            'system_lambda': None,
            'dst_flag': None,
            'data_source': 'SPP',
            'version': 1,
            'is_current': True
        })

        df_unified = self.enforce_price_types(df_unified)
        df_unified = df_unified.sort_values('datetime_utc')
        df_unified = df_unified.drop_duplicates(subset=['datetime_utc', 'settlement_location'], keep='last')

        years = df_unified['delivery_date'].apply(lambda x: x.year).unique()
        for yr in years:
            if year and yr != year:
                continue

            df_year = df_unified[df_unified['delivery_date'].apply(lambda x: x.year) == yr]
            output_dir = self.parquet_output_dir / "da_energy_hourly"
            output_file = output_dir / f"da_energy_hourly_{yr}.parquet"

            self.write_parquet_atomic(df_year, output_file, self.ENERGY_SCHEMA, compression='snappy')


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

    def convert_rt_energy(self, year=None) -> None:
        """Convert Real-Time energy prices."""
        csv_dir = self.csv_data_dir / "rt_lmp"
        df = self._read_spp_csv_files(csv_dir, year)

        if df.empty:
            self.logger.warning("No RT data to convert")
            return

        # Similar structure to DA (Schema v2.0.0: keep timezone-aware)
        datetime_col = next((col for col in df.columns if 'date' in col.lower() or 'time' in col.lower() or 'interval' in col.lower()), df.columns[0])
        df['datetime_local'] = pd.to_datetime(df[datetime_col]).dt.tz_localize('America/Chicago')
        df['datetime_utc'] = df['datetime_local'].dt.tz_convert('UTC')

        location_col = next((col for col in df.columns if 'settlement' in col.lower() or 'pnode' in col.lower() or 'location' in col.lower()), 'Settlement Location')
        lmp_col = next((col for col in df.columns if 'lmp' in col.lower() or 'price' in col.lower()), 'LMP')

        df_unified = pd.DataFrame({
            'datetime_utc': df['datetime_utc'],
            'datetime_local': df['datetime_local'],
            'interval_start_utc': df['datetime_utc'],
            'interval_end_utc': df['datetime_utc'] + pd.Timedelta(hours=1),
            'delivery_date': df['datetime_local'].dt.date,
            'delivery_hour': df['datetime_local'].dt.hour + 1,
            'delivery_interval': np.uint8(0),
            'interval_minutes': np.uint8(60),
            'iso': 'SPP',
            'market_type': 'RT',
            'settlement_location': df[location_col],
            'settlement_location_type': 'SETTLEMENT_LOCATION',
            'settlement_location_id': df[location_col].astype(str),
            'zone': None,
            'voltage_kv': None,
            'lmp_total': df[lmp_col].astype('float64'),
            'lmp_energy': None,
            'lmp_congestion': None,
            'lmp_loss': None,
            'system_lambda': None,
            'dst_flag': None,
            'data_source': 'SPP',
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
            output_dir = self.parquet_output_dir / "rt_energy_hourly"
            output_file = output_dir / f"rt_energy_hourly_{yr}.parquet"

            self.write_parquet_atomic(df_year, output_file, self.ENERGY_SCHEMA, compression='snappy')

    def convert_ancillary_services(self, year=None) -> None:
        """Convert ancillary services."""
        self.logger.info("SPP AS conversion - not yet implemented")


def main():
    parser = argparse.ArgumentParser(description='Convert SPP data to unified parquet format')
    parser.add_argument('--year', type=int, help='Specific year to process')
    parser.add_argument('--all', action='store_true', help='Process all years')
    parser.add_argument('--da-only', action='store_true', help='Only convert DA prices')
    parser.add_argument('--rt-only', action='store_true', help='Only convert RT prices')
    parser.add_argument('--csv-dir', help='Override CSV data directory')
    parser.add_argument('--output-dir', help='Override output directory')

    args = parser.parse_args()

    kwargs = {}
    if args.csv_dir:
        kwargs['csv_data_dir'] = args.csv_dir
    if args.output_dir:
        kwargs['parquet_output_dir'] = args.output_dir

    converter = SPPParquetConverter(**kwargs)
    convert_da = not args.rt_only
    convert_rt = not args.da_only

    converter.run_full_conversion(year=args.year, convert_da=convert_da, convert_rt=convert_rt, convert_as=False)


if __name__ == "__main__":
    main()
