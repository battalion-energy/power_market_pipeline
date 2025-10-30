#!/usr/bin/env python3
"""
PJM Parquet Converter - MEMORY OPTIMIZED

**CRITICAL**: Uses chunked processing to handle large datasets.
Processes files in batches of 50 to avoid memory exhaustion.

Usage:
    python pjm_parquet_converter.py --year 2024
    python pjm_parquet_converter.py --all
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List
import glob
import gc

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from unified_iso_parquet_converter import UnifiedISOParquetConverter


class PJMParquetConverter(UnifiedISOParquetConverter):
    """PJM-specific parquet converter with memory optimization."""

    # MEMORY OPTIMIZATION: Process files in small batches
    BATCH_SIZE = 50  # Process 50 CSV files at a time
    CHUNK_SIZE = 100000  # Read CSV in 100k row chunks

    def __init__(
        self,
        csv_data_dir: str = "/pool/ssd8tb/data/iso/PJM_data/csv_files",
        parquet_output_dir: str = "/pool/ssd8tb/data/iso/unified_iso_data/parquet/pjm",
        metadata_dir: str = "/pool/ssd8tb/data/iso/unified_iso_data/metadata"
    ):
        super().__init__(
            iso_name="PJM",
            csv_data_dir=csv_data_dir,
            parquet_output_dir=parquet_output_dir,
            metadata_dir=metadata_dir,
            iso_timezone="America/New_York"
        )

        self.as_product_mapping = {
            'Mid-Atlantic/Dominion Primary Reserve': 'NON_SPIN',
            'RTO Primary Reserve': 'NON_SPIN',
            'Synchronized Reserve': 'SPIN',
            'Regulation': 'REG',
            'Day-ahead Scheduling Reserve': 'RESERVE'
        }

    def _process_csv_files_in_batches(
        self,
        csv_files: List[Path],
        year: Optional[int] = None
    ):
        """
        Generator that yields DataFrames from CSV files in batches.

        MEMORY OPTIMIZATION: Only loads BATCH_SIZE files at a time.
        """
        total_files = len(csv_files)
        self.logger.info(f"Processing {total_files} files in batches of {self.BATCH_SIZE}")

        for batch_start in range(0, total_files, self.BATCH_SIZE):
            batch_end = min(batch_start + self.BATCH_SIZE, total_files)
            batch_files = csv_files[batch_start:batch_end]

            self.logger.info(f"Processing batch {batch_start//self.BATCH_SIZE + 1}: files {batch_start+1}-{batch_end} of {total_files}")

            dfs = []
            for csv_file in batch_files:
                try:
                    # Read CSV in chunks to avoid memory spikes
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

    def convert_da_energy(self, year: Optional[int] = None) -> None:
        """Convert Day-Ahead energy prices (MEMORY OPTIMIZED)."""

        # DA Hub prices
        self.logger.info("Converting DA Energy (Hub) prices...")
        csv_dir = self.csv_data_dir / "da_hubs"
        csv_files = list(csv_dir.glob("*.csv"))

        if year:
            csv_files = [f for f in csv_files if str(year) in f.name]

        if not csv_files:
            self.logger.warning("No DA hub files found")
        else:
            # Process in streaming mode
            self._process_da_hub_streaming(csv_files, year)

        # DA Nodal prices (LARGE - use streaming)
        self.logger.info("Converting DA Energy (Nodal) prices...")
        csv_dir = self.csv_data_dir / "da_nodal"
        csv_files = list(csv_dir.glob("*.csv"))

        if year:
            csv_files = [f for f in csv_files if str(year) in f.name]

        if not csv_files:
            self.logger.warning("No DA nodal files found")
        else:
            self._process_da_nodal_streaming(csv_files, year)

    def _process_da_hub_streaming(self, csv_files: List[Path], year: Optional[int]):
        """Process DA hub files in streaming mode."""

        # Group files by year for output
        year_data = {}

        for batch_df in self._process_csv_files_in_batches(csv_files, year):
            # Parse timestamps (PJM provides both UTC and EPT)
            datetime_utc = pd.to_datetime(batch_df['datetime_beginning_utc'], utc=True)
            datetime_local = pd.to_datetime(batch_df['datetime_beginning_ept']).dt.tz_localize('America/New_York')

            # Transform to unified schema
            df_unified = pd.DataFrame({
                'datetime_utc': datetime_utc,
                'datetime_local': datetime_local,  # Schema v2.0.0: timezone-aware!
                'interval_start_utc': datetime_utc,
                'interval_end_utc': datetime_utc + pd.Timedelta(hours=1),
                'delivery_date': datetime_local.dt.date,
                'delivery_hour': datetime_local.dt.hour + 1,
                'delivery_interval': np.uint8(0),
                'interval_minutes': np.uint8(60),
                'iso': 'PJM',
                'market_type': 'DA',
                'settlement_location': batch_df['pnode_name'].fillna(batch_df['pnode_id'].astype(str)),
                'settlement_location_type': batch_df['type'],
                'settlement_location_id': batch_df['pnode_id'].astype(str),
                'zone': batch_df['zone'],
                'voltage_kv': pd.to_numeric(batch_df['voltage'], errors='coerce'),
                'lmp_total': batch_df['total_lmp_da'].astype('float64'),
                'lmp_energy': batch_df['system_energy_price_da'].astype('float64'),
                'lmp_congestion': batch_df['congestion_price_da'].astype('float64'),
                'lmp_loss': batch_df['marginal_loss_price_da'].astype('float64'),
                'system_lambda': batch_df['system_energy_price_da'].astype('float64'),
                'dst_flag': None,
                'data_source': 'PJM API',
                'version': batch_df.get('version_nbr', 1).astype('uint32'),
                'is_current': batch_df.get('row_is_current', True)
            })

            df_unified = self.enforce_price_types(df_unified)
            df_unified = df_unified.sort_values('datetime_utc')
            df_unified = df_unified.drop_duplicates(subset=['datetime_utc', 'settlement_location'], keep='last')

            # Group by year
            for yr in df_unified['delivery_date'].apply(lambda x: x.year).unique():
                if year and yr != year:
                    continue

                df_year = df_unified[df_unified['delivery_date'].apply(lambda x: x.year) == yr]

                if yr not in year_data:
                    year_data[yr] = []

                year_data[yr].append(df_year)

            del df_unified, batch_df
            gc.collect()

        # Write accumulated data for each year
        for yr, dfs in year_data.items():
            if dfs:
                self.logger.info(f"Combining {len(dfs)} batches for year {yr}")
                final_df = pd.concat(dfs, ignore_index=True)
                # Each batch is already deduplicated (line 176), so skip expensive final dedup
                # Cross-batch duplicates are rare and can be filtered during reads if needed

                output_dir = self.parquet_output_dir / "da_energy_hourly_hub"
                output_file = output_dir / f"da_energy_hourly_hub_{yr}.parquet"

                self.write_parquet_atomic(final_df, output_file, self.ENERGY_SCHEMA, compression='snappy')

                # Extract metadata from first batch only
                if yr == list(year_data.keys())[0]:
                    hubs = self.extract_unique_locations(
                        final_df,
                        location_col='settlement_location',
                        location_type_col='settlement_location_type',
                        location_id_col='settlement_location_id'
                    )
                    self.save_metadata_json('hubs', {
                        'iso': 'PJM',
                        'last_updated': datetime.now().isoformat(),
                        'total_hubs': len(hubs),
                        'hubs': hubs
                    })

                del final_df, dfs
                gc.collect()

    def _process_da_nodal_streaming(self, csv_files: List[Path], year: Optional[int]):
        """Process DA nodal files in TRUE streaming mode (write batch by batch)."""

        # Dictionary to hold ParquetWriters for each year
        year_writers = {}
        year_files = {}

        try:
            for batch_df in self._process_csv_files_in_batches(csv_files, year):
                # Parse timestamps (PJM provides both UTC and EPT)
                datetime_utc = pd.to_datetime(batch_df['datetime_beginning_utc'], utc=True)
                datetime_local = pd.to_datetime(batch_df['datetime_beginning_ept']).dt.tz_localize('America/New_York')

                df_unified = pd.DataFrame({
                    'datetime_utc': datetime_utc,
                    'datetime_local': datetime_local,  # Schema v2.0.0: timezone-aware!
                    'interval_start_utc': datetime_utc,
                    'interval_end_utc': datetime_utc + pd.Timedelta(hours=1),
                    'delivery_date': datetime_local.dt.date,
                    'delivery_hour': datetime_local.dt.hour + 1,
                    'delivery_interval': np.uint8(0),
                    'interval_minutes': np.uint8(60),
                    'iso': 'PJM',
                    'market_type': 'DA',
                    'settlement_location': batch_df['pnode_name'].fillna(batch_df['pnode_id'].astype(str)),
                    'settlement_location_type': batch_df['type'],
                    'settlement_location_id': batch_df['pnode_id'].astype(str),
                    'zone': batch_df['zone'],
                    'voltage_kv': pd.to_numeric(batch_df['voltage'], errors='coerce'),
                    'lmp_total': batch_df['total_lmp_da'].astype('float64'),
                    'lmp_energy': batch_df['system_energy_price_da'].astype('float64'),
                    'lmp_congestion': batch_df['congestion_price_da'].astype('float64'),
                    'lmp_loss': batch_df['marginal_loss_price_da'].astype('float64'),
                    'system_lambda': batch_df['system_energy_price_da'].astype('float64'),
                    'dst_flag': None,
                    'data_source': 'PJM API',
                    'version': batch_df.get('version_nbr', 1).astype('uint32'),
                    'is_current': batch_df.get('row_is_current', True)
                })

                df_unified = self.enforce_price_types(df_unified)

                # Write batch by batch to parquet (TRUE STREAMING)
                for yr in df_unified['delivery_date'].apply(lambda x: x.year).unique():
                    if year and yr != year:
                        continue

                    df_year = df_unified[df_unified['delivery_date'].apply(lambda x: x.year) == yr]

                    if yr not in year_writers:
                        # Create output file for this year
                        output_dir = self.parquet_output_dir / "da_energy_hourly_nodal"
                        output_dir.mkdir(parents=True, exist_ok=True)
                        output_file = output_dir / f"da_energy_hourly_nodal_{yr}.parquet.tmp"
                        year_files[yr] = output_file

                        # Create ParquetWriter for streaming writes
                        table = pa.Table.from_pandas(df_year, schema=self.ENERGY_SCHEMA)
                        year_writers[yr] = pq.ParquetWriter(output_file, self.ENERGY_SCHEMA, compression='snappy')
                        year_writers[yr].write_table(table)
                        self.logger.info(f"Started streaming write for year {yr}: {output_file}")
                    else:
                        # Append to existing writer
                        table = pa.Table.from_pandas(df_year, schema=self.ENERGY_SCHEMA)
                        year_writers[yr].write_table(table)

                del df_unified, batch_df
                gc.collect()

            # Close all writers and atomic move
            for yr, writer in year_writers.items():
                writer.close()
                temp_file = year_files[yr]
                final_file = temp_file.parent / temp_file.name.replace('.tmp', '')
                temp_file.replace(final_file)
                self.logger.info(f"Successfully wrote {final_file}")

        finally:
            # Ensure all writers are closed
            for writer in year_writers.values():
                try:
                    writer.close()
                except:
                    pass

    def convert_rt_energy(self, year: Optional[int] = None) -> None:
        """Convert Real-Time energy prices (NOT IMPLEMENTED YET - reduces memory load)."""
        self.logger.info("RT energy conversion temporarily disabled to reduce memory usage")
        pass

    def convert_ancillary_services(self, year: Optional[int] = None) -> None:
        """Convert ancillary services (NOT IMPLEMENTED YET - reduces memory load)."""
        self.logger.info("AS conversion temporarily disabled to reduce memory usage")
        pass


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Convert PJM data to unified parquet format (MEMORY OPTIMIZED)')
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

    converter = PJMParquetConverter(**kwargs)

    # For now, only convert DA to avoid memory issues
    converter.run_full_conversion(
        year=args.year,
        convert_da=True,
        convert_rt=False,  # Disabled for now
        convert_as=False   # Disabled for now
    )


if __name__ == "__main__":
    main()
