#!/usr/bin/env python3
"""
NYISO Parquet Converter

MEMORY OPTIMIZED: Uses chunked processing (BATCH_SIZE=50, CHUNK_SIZE=100k)

Converts NYISO market data from CSV to unified parquet format.

Data Types:
- Day-Ahead Zonal Prices (11 zones, hourly)
- Real-Time Zonal Prices (hourly and 5-minute)
- Ancillary Services

Usage:
    python nyiso_parquet_converter.py --year 2024
    python nyiso_parquet_converter.py --all
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional
import zipfile
import gc

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from unified_iso_parquet_converter import UnifiedISOParquetConverter


class NYISOParquetConverter(UnifiedISOParquetConverter):
    """NYISO-specific parquet converter."""

    # MEMORY OPTIMIZATION
    BATCH_SIZE = 50
    CHUNK_SIZE = 100000

    def __init__(
        self,
        csv_data_dir: str = "/pool/ssd8tb/data/iso/NYISO_data/csv_files",
        parquet_output_dir: str = "/pool/ssd8tb/data/iso/unified_iso_data/parquet/nyiso",
        metadata_dir: str = "/pool/ssd8tb/data/iso/unified_iso_data/metadata"
    ):
        super().__init__(
            iso_name="NYISO",
            csv_data_dir=csv_data_dir,
            parquet_output_dir=parquet_output_dir,
            metadata_dir=metadata_dir,
            iso_timezone="America/New_York"
        )

        # NYISO zones
        self.zones = [
            'CAPITL', 'CENTRL', 'DUNWOD', 'GENESE', 'HUD VL',
            'LONGIL', 'MHK VL', 'MILLWD', 'N.Y.C.', 'NORTH', 'WEST'
        ]

    def _read_nyiso_csv_files(self, csv_dir: Path, year: Optional[int] = None) -> pd.DataFrame:
        """Read NYISO CSV files, handling ZIP archives if needed."""
        if not csv_dir.exists():
            self.logger.warning(f"CSV directory not found: {csv_dir}")
            return pd.DataFrame()

        # Look for both CSV and ZIP files
        csv_files = list(csv_dir.glob("*.csv"))
        zip_files = list(csv_dir.glob("*.zip"))

        if year:
            csv_files = [f for f in csv_files if str(year) in f.name]
            zip_files = [f for f in zip_files if str(year) in f.name]

        self.logger.info(f"Found {len(csv_files)} CSV and {len(zip_files)} ZIP files")

        # Process CSV files in batches
        all_dfs = []
        total_files = len(csv_files) + len(zip_files)

        if csv_files:
            for batch_start in range(0, len(csv_files), self.BATCH_SIZE):
                batch_end = min(batch_start + self.BATCH_SIZE, len(csv_files))
                batch_files = csv_files[batch_start:batch_end]

                self.logger.info(f"Processing NYISO CSV batch {batch_start//self.BATCH_SIZE + 1}: files {batch_start+1}-{batch_end} of {len(csv_files)}")

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

        # Extract and read ZIP files in batches
        if zip_files:
            for batch_start in range(0, len(zip_files), self.BATCH_SIZE):
                batch_end = min(batch_start + self.BATCH_SIZE, len(zip_files))
                batch_zip_files = zip_files[batch_start:batch_end]

                self.logger.info(f"Processing NYISO ZIP batch {batch_start//self.BATCH_SIZE + 1}: files {batch_start+1}-{batch_end} of {len(zip_files)}")

                dfs = []
                for zip_file in batch_zip_files:
                    try:
                        with zipfile.ZipFile(zip_file, 'r') as zf:
                            for name in zf.namelist():
                                if name.endswith('.csv'):
                                    with zf.open(name) as f:
                                        df = pd.read_csv(f)
                                        dfs.append(df)
                    except Exception as e:
                        self.logger.error(f"Error reading {zip_file}: {e}")

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

        self.logger.info(f"Combined {len(combined_df):,} rows from {total_files} files")

        return combined_df

    def convert_da_energy(self, year: Optional[int] = None) -> None:
        """Convert Day-Ahead energy prices to parquet."""
        self.logger.info("Converting DA Energy (Zonal) prices...")

        csv_dir = self.csv_data_dir / "lmp_day_ahead_hourly"
        if not csv_dir.exists():
            csv_dir = self.csv_data_dir / "day_ahead" / "csv"
        if not csv_dir.exists():
            csv_dir = self.csv_data_dir / "dam"

        df = self._read_nyiso_csv_files(csv_dir, year)

        if df.empty:
            self.logger.warning("No DA data to convert")
            return

        # NYISO datetime parsing (data is timezone-aware, convert directly to UTC)
        # Schema v2.0.0: Keep datetime_local timezone-aware!
        if 'Time' in df.columns:
            df['datetime_utc'] = pd.to_datetime(df['Time'], utc=True)
            df['datetime_local'] = df['datetime_utc'].dt.tz_convert('America/New_York')
        elif 'Time Stamp' in df.columns:
            df['datetime_utc'] = pd.to_datetime(df['Time Stamp'], utc=True)
            df['datetime_local'] = df['datetime_utc'].dt.tz_convert('America/New_York')
        elif 'Timestamp' in df.columns:
            df['datetime_utc'] = pd.to_datetime(df['Timestamp'], utc=True)
            df['datetime_local'] = df['datetime_utc'].dt.tz_convert('America/New_York')
        elif 'Date' in df.columns:
            df['datetime_local'] = pd.to_datetime(df['Date']).dt.tz_localize('America/New_York')
            df['datetime_utc'] = df['datetime_local'].dt.tz_convert('UTC')
        else:
            # Try first column
            df['datetime_utc'] = pd.to_datetime(df.iloc[:, 0], utc=True)
            df['datetime_local'] = df['datetime_utc'].dt.tz_convert('America/New_York')

        # Zone and price columns
        zone_col = 'Location' if 'Location' in df.columns else next((col for col in df.columns if 'zone' in col.lower() or 'name' in col.lower()), 'Zone')
        lmp_col = 'LMP' if 'LMP' in df.columns else next((col for col in df.columns if 'lbmp' in col.lower() or 'lmp' in col.lower()), 'LBMP')

        df_unified = pd.DataFrame({
            'datetime_utc': df['datetime_utc'],
            'datetime_local': df['datetime_local'],
            'interval_start_utc': df['datetime_utc'],
            'interval_end_utc': df['datetime_utc'] + pd.Timedelta(hours=1),
            'delivery_date': df['datetime_local'].dt.date,
            'delivery_hour': df['datetime_local'].dt.hour + 1,
            'delivery_interval': np.uint8(0),
            'interval_minutes': np.uint8(60),
            'iso': 'NYISO',
            'market_type': 'DA',
            'settlement_location': df[zone_col],
            'settlement_location_type': 'ZONE',
            'settlement_location_id': df[zone_col].astype(str),
            'zone': df[zone_col],
            'voltage_kv': None,
            'lmp_total': df[lmp_col].astype('float64'),
            'lmp_energy': df.get('Energy', None),
            'lmp_congestion': df.get('Congestion', None),
            'lmp_loss': df.get('Losses', None),
            'system_lambda': None,
            'dst_flag': None,
            'data_source': 'NYISO',
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

        # Zone metadata
        zones = self.extract_unique_locations(df_unified, 'settlement_location', 'settlement_location_type', 'settlement_location_id')

        self.save_metadata_json('zones', {
            'iso': 'NYISO',
            'last_updated': datetime.now().isoformat(),
            'total_zones': len(zones),
            'zones': zones
        })


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

    def convert_rt_energy(self, year: Optional[int] = None) -> None:
        """Convert Real-Time energy prices to parquet."""
        self.logger.info("Converting RT Energy prices...")

        csv_dir = self.csv_data_dir / "real-time" / "csv"
        if not csv_dir.exists():
            csv_dir = self.csv_data_dir / "rt"

        df = self._read_nyiso_csv_files(csv_dir, year)

        if df.empty:
            self.logger.warning("No RT data to convert")
            return

        # Parse datetime (Schema v2.0.0: keep timezone-aware)
        if 'Time Stamp' in df.columns:
            df['datetime_local'] = pd.to_datetime(df['Time Stamp']).dt.tz_localize('America/New_York')
        elif 'Timestamp' in df.columns:
            df['datetime_local'] = pd.to_datetime(df['Timestamp']).dt.tz_localize('America/New_York')
        else:
            df['datetime_local'] = pd.to_datetime(df.iloc[:, 0]).dt.tz_localize('America/New_York')

        df['datetime_utc'] = df['datetime_local'].dt.tz_convert('UTC')

        # Detect interval
        time_diff = (df['datetime_local'].diff().dt.total_seconds() / 60).mode()
        interval_minutes = int(time_diff[0]) if len(time_diff) > 0 else 60

        zone_col = next((col for col in df.columns if 'zone' in col.lower() or 'name' in col.lower()), 'Zone')
        lmp_col = next((col for col in df.columns if 'lbmp' in col.lower() or 'lmp' in col.lower()), 'LBMP')

        df_unified = pd.DataFrame({
            'datetime_utc': df['datetime_utc'],
            'datetime_local': df['datetime_local'],
            'interval_start_utc': df['datetime_utc'],
            'interval_end_utc': df['datetime_utc'] + pd.Timedelta(minutes=interval_minutes),
            'delivery_date': df['datetime_local'].dt.date,
            'delivery_hour': df['datetime_local'].dt.hour + 1,
            'delivery_interval': (df['datetime_local'].dt.minute // interval_minutes + 1).astype('uint8') if interval_minutes < 60 else np.uint8(0),
            'interval_minutes': np.uint8(interval_minutes),
            'iso': 'NYISO',
            'market_type': 'RT',
            'settlement_location': df[zone_col],
            'settlement_location_type': 'ZONE',
            'settlement_location_id': df[zone_col].astype(str),
            'zone': df[zone_col],
            'voltage_kv': None,
            'lmp_total': df[lmp_col].astype('float64'),
            'lmp_energy': df.get('Energy', None),
            'lmp_congestion': df.get('Congestion', None),
            'lmp_loss': df.get('Losses', None),
            'system_lambda': None,
            'dst_flag': None,
            'data_source': 'NYISO',
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

            if interval_minutes == 5:
                output_dir = self.parquet_output_dir / "rt_energy_5min"
                output_file = output_dir / f"rt_energy_5min_{yr}.parquet"
            else:
                output_dir = self.parquet_output_dir / "rt_energy_hourly"
                output_file = output_dir / f"rt_energy_hourly_{yr}.parquet"

            self.write_parquet_atomic(df_year, output_file, self.ENERGY_SCHEMA, compression='snappy')

    def convert_ancillary_services(self, year: Optional[int] = None) -> None:
        """Convert ancillary services to parquet."""
        self.logger.info("Converting Ancillary Services...")

        csv_dir = self.csv_data_dir / "ancillary_services"
        df = self._read_nyiso_csv_files(csv_dir, year)

        if df.empty:
            self.logger.warning("No AS data to convert")
            return

        # Similar structure to energy prices
        # Implementation depends on actual NYISO AS format
        self.logger.info("NYISO AS conversion - format detection needed")


def main():
    parser = argparse.ArgumentParser(description='Convert NYISO data to unified parquet format')
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

    converter = NYISOParquetConverter(**kwargs)

    convert_da = not (args.rt_only or args.as_only)
    convert_rt = not (args.da_only or args.as_only)
    convert_as = not (args.da_only or args.rt_only)

    converter.run_full_conversion(year=args.year, convert_da=convert_da, convert_rt=convert_rt, convert_as=convert_as)


if __name__ == "__main__":
    main()
