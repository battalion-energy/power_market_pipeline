#!/usr/bin/env python3
"""
CAISO Parquet Converter - MEMORY OPTIMIZED

**CRITICAL**: Uses chunked processing to handle large datasets.
Processes files in batches of 50 to avoid memory exhaustion.

Usage:
    python caiso_parquet_converter.py --year 2024
    python caiso_parquet_converter.py --all
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List
import glob
import gc

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from unified_iso_parquet_converter import UnifiedISOParquetConverter


class CAISOParquetConverter(UnifiedISOParquetConverter):
    """CAISO-specific parquet converter with memory optimization."""

    # MEMORY OPTIMIZATION: Process files in small batches
    BATCH_SIZE = 50
    CHUNK_SIZE = 100000

    def __init__(
        self,
        csv_data_dir: str = "/pool/ssd8tb/data/iso/CAISO_data/csv_files",
        parquet_output_dir: str = "/pool/ssd8tb/data/iso/unified_iso_data/parquet/caiso",
        metadata_dir: str = "/pool/ssd8tb/data/iso/unified_iso_data/metadata"
    ):
        super().__init__(
            iso_name="CAISO",
            csv_data_dir=csv_data_dir,
            parquet_output_dir=parquet_output_dir,
            metadata_dir=metadata_dir,
            iso_timezone="America/Los_Angeles"
        )

        self.as_product_mapping = {
            'RU': 'REG_UP',
            'RD': 'REG_DOWN',
            'SR': 'SPIN',
            'NR': 'NON_SPIN',
            'FRU': 'RAMP_UP',
            'FRD': 'RAMP_DOWN'
        }

    def _process_csv_files_in_batches(self, csv_files: List[Path], year: Optional[int] = None):
        """Generator that yields DataFrames from CSV files in batches."""
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

    def _parse_caiso_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse CAISO datetime format."""
        df['datetime_utc'] = pd.to_datetime(df['INTERVALSTARTTIME_GMT'], utc=True)
        df['interval_end_utc'] = pd.to_datetime(df['INTERVALENDTIME_GMT'], utc=True)
        df['datetime_local'] = df['datetime_utc'].dt.tz_convert(self.iso_timezone)
        return df

    def convert_da_energy(self, year: Optional[int] = None) -> None:
        """Convert Day-Ahead energy prices (MEMORY OPTIMIZED)."""
        self.logger.info("Converting DA Energy (Nodal) prices...")

        csv_dir = self.csv_data_dir / "da_nodal"
        csv_files = list(csv_dir.glob("*.csv"))

        if year:
            csv_files = [f for f in csv_files if str(year) in f.name]

        if not csv_files:
            self.logger.warning("No DA nodal data to convert")
            return

        year_data = {}

        for batch_df in self._process_csv_files_in_batches(csv_files, year):
            # Parse datetimes
            batch_df = self._parse_caiso_datetime(batch_df)

            # Determine interval duration
            interval_duration = (batch_df['interval_end_utc'] - batch_df['datetime_utc']).dt.total_seconds() / 60
            interval_minutes = interval_duration.mode()[0] if len(interval_duration.mode()) > 0 else 60

            # Pivot if needed for components
            if 'MW' in batch_df.columns and 'XML_DATA_ITEM' in batch_df.columns:
                df_pivot = batch_df.pivot_table(
                    index=['datetime_utc', 'interval_end_utc', 'NODE', 'NODE_ID'],
                    columns='XML_DATA_ITEM',
                    values='MW',
                    aggfunc='first'
                ).reset_index()

                lmp_total = df_pivot.get('LMP_PRC', None)
                lmp_energy = df_pivot.get('LMP_ENE_PRC', None)
                lmp_congestion = df_pivot.get('LMP_CONG_PRC', None)
                lmp_loss = df_pivot.get('LMP_LOSS_PRC', None)
            else:
                lmp_total = batch_df['MW']
                lmp_energy = None
                lmp_congestion = None
                lmp_loss = None
                df_pivot = batch_df.copy()

            df_unified = pd.DataFrame({
                'datetime_utc': df_pivot['datetime_utc'],
                'datetime_local': df_pivot['datetime_utc'].dt.tz_convert(self.iso_timezone).dt.tz_localize(None),
                'interval_start_utc': df_pivot['datetime_utc'],
                'interval_end_utc': df_pivot['interval_end_utc'],
                'delivery_date': df_pivot['datetime_utc'].dt.tz_convert(self.iso_timezone).dt.date,
                'delivery_hour': df_pivot['datetime_utc'].dt.tz_convert(self.iso_timezone).dt.hour + 1,
                'delivery_interval': np.uint8(0),
                'interval_minutes': np.uint8(int(interval_minutes)),
                'iso': 'CAISO',
                'market_type': 'DA',
                'settlement_location': df_pivot['NODE'] if 'NODE' in df_pivot.columns else df_pivot['NODE_ID'],
                'settlement_location_type': 'NODE',
                'settlement_location_id': df_pivot['NODE_ID'].astype(str) if 'NODE_ID' in df_pivot.columns else None,
                'zone': None,
                'voltage_kv': None,
                'lmp_total': lmp_total,
                'lmp_energy': lmp_energy,
                'lmp_congestion': lmp_congestion,
                'lmp_loss': lmp_loss,
                'system_lambda': None,
                'dst_flag': None,
                'data_source': 'CAISO OASIS',
                'version': 1,
                'is_current': True
            })

            df_unified = self.enforce_price_types(df_unified)
            df_unified = df_unified.sort_values('datetime_utc')
            df_unified = df_unified.drop_duplicates(subset=['datetime_utc', 'settlement_location'], keep='last')

            # Accumulate by year
            for yr in df_unified['delivery_date'].apply(lambda x: x.year).unique():
                if year and yr != year:
                    continue

                df_year = df_unified[df_unified['delivery_date'].apply(lambda x: x.year) == yr]

                if yr not in year_data:
                    year_data[yr] = []

                year_data[yr].append(df_year)

            del df_unified, batch_df, df_pivot
            gc.collect()

        # Write accumulated data
        for yr, dfs in year_data.items():
            if dfs:
                self.logger.info(f"Combining {len(dfs)} batches for year {yr}")
                final_df = pd.concat(dfs, ignore_index=True)
                final_df = final_df.sort_values('datetime_utc')
                final_df = final_df.drop_duplicates(subset=['datetime_utc', 'settlement_location'], keep='last')

                output_dir = self.parquet_output_dir / "da_energy_hourly_nodal"
                output_file = output_dir / f"da_energy_hourly_nodal_{yr}.parquet"

                self.write_parquet_atomic(final_df, output_file, self.ENERGY_SCHEMA, compression='snappy')

                # Extract node metadata (first year only)
                if yr == list(year_data.keys())[0]:
                    nodes = self.extract_unique_locations(
                        final_df,
                        location_col='settlement_location',
                        location_type_col='settlement_location_type',
                        location_id_col='settlement_location_id'
                    )
                    self.save_metadata_json('nodes', {
                        'iso': 'CAISO',
                        'last_updated': datetime.now().isoformat(),
                        'total_nodes': len(nodes),
                        'nodes': nodes
                    })

                del final_df, dfs
                gc.collect()

    def convert_rt_energy(self, year: Optional[int] = None) -> None:
        """Convert Real-Time energy prices (NOT IMPLEMENTED - reduces memory)."""
        self.logger.info("RT energy conversion temporarily disabled to reduce memory usage")
        pass

    def convert_ancillary_services(self, year: Optional[int] = None) -> None:
        """Convert ancillary services (NOT IMPLEMENTED - reduces memory)."""
        self.logger.info("AS conversion temporarily disabled to reduce memory usage")
        pass


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Convert CAISO data to unified parquet format (MEMORY OPTIMIZED)')
    parser.add_argument('--year', type=int, help='Specific year to process')
    parser.add_argument('--all', action='store_true', help='Process all years')
    parser.add_argument('--da-only', action='store_true', help='Only convert DA prices')
    parser.add_argument('--csv-dir', help='Override CSV data directory')
    parser.add_argument('--output-dir', help='Override output directory')

    args = parser.parse_args()

    kwargs = {}
    if args.csv_dir:
        kwargs['csv_data_dir'] = args.csv_dir
    if args.output_dir:
        kwargs['parquet_output_dir'] = args.output_dir

    converter = CAISOParquetConverter(**kwargs)

    converter.run_full_conversion(
        year=args.year,
        convert_da=True,
        convert_rt=False,  # Disabled for now
        convert_as=False   # Disabled for now
    )


if __name__ == "__main__":
    main()
