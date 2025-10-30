#!/usr/bin/env python3
"""
ERCOT Parquet Converter - UNIFIED FORMAT

MEMORY OPTIMIZED: Uses chunked processing (BATCH_SIZE=50, CHUNK_SIZE=100k)

**IMPORTANT**: This creates NEW parquet files in the unified format.
DO NOT modify or touch the existing ERCOT legacy parquet files in:
    /home/enrico/data/ERCOT_data/rollup_files/

This converter reads from ERCOT CSV sources and creates unified-format
parquet files in:
    /pool/ssd8tb/data/iso/unified_iso_data/parquet/ercot/

Usage:
    python ercot_parquet_converter.py --year 2024
    python ercot_parquet_converter.py --all
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import glob
import gc

sys.path.insert(0, str(Path(__file__).parent))
from unified_iso_parquet_converter import UnifiedISOParquetConverter


class ERCOTParquetConverter(UnifiedISOParquetConverter):
    """
    ERCOT-specific parquet converter for UNIFIED format.

    This does NOT touch the legacy ERCOT parquet files.
    Creates new files in unified format alongside PJM, CAISO, etc.
    """

    # MEMORY OPTIMIZATION
    BATCH_SIZE = 50
    CHUNK_SIZE = 100000

    def __init__(
        self,
        csv_data_dir: str = "/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data",
        parquet_output_dir: str = "/pool/ssd8tb/data/iso/unified_iso_data/parquet/ercot",
        metadata_dir: str = "/pool/ssd8tb/data/iso/unified_iso_data/metadata"
    ):
        super().__init__(
            iso_name="ERCOT",
            csv_data_dir=csv_data_dir,
            parquet_output_dir=parquet_output_dir,
            metadata_dir=metadata_dir,
            iso_timezone="America/Chicago"  # Central Time
        )

        # ERCOT settlement point types
        self.settlement_types = {
            'HB': 'HUB',
            'LZ': 'LOAD_ZONE',
            'RN': 'RESOURCE_NODE'
        }

        # ERCOT AS products
        self.as_product_mapping = {
            'REGUP': 'REG_UP',
            'REGDN': 'REG_DOWN',
            'RRS': 'SPIN',
            'ECRS': 'NON_SPIN',
            'NSPIN': 'NON_SPIN'
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

    def _read_ercot_csv_files(self, pattern: str, year=None) -> pd.DataFrame:
        """Read ERCOT CSV files matching pattern."""
        csv_files = glob.glob(str(self.csv_data_dir / pattern))

        if year:
            csv_files = [f for f in csv_files if str(year) in f]

        if not csv_files:
            self.logger.warning(f"No files found for pattern: {pattern}")
            return pd.DataFrame()

        self.logger.info(f"Reading {len(csv_files)} CSV files")

        dfs = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                dfs.append(df)
            except Exception as e:
                self.logger.error(f"Error reading {csv_file}: {e}")

        if not dfs:
            return pd.DataFrame()

        return pd.concat(dfs, ignore_index=True)

    def convert_da_energy(self, year=None) -> None:
        """Convert Day-Ahead energy prices (hourly)."""
        self.logger.info("Converting DA Energy prices...")

        # ERCOT DA files: dam_prices_*.csv in DAM_Settlement_Point_Prices directory
        dam_dir = self.csv_data_dir / "DAM_Settlement_Point_Prices"
        if not dam_dir.exists():
            self.logger.warning(f"DAM directory not found: {dam_dir}")
            return

        csv_files = list(dam_dir.glob("dam_prices_*.csv"))
        if year:
            csv_files = [f for f in csv_files if str(year) in f.name]

        if not csv_files:
            self.logger.warning("No ERCOT DA data found")
            return

        self.logger.info(f"Reading {len(csv_files)} ERCOT DAM CSV files in batches")

        # Process files in batches to avoid memory exhaustion
        all_dfs = []
        for batch_start in range(0, len(csv_files), self.BATCH_SIZE):
            batch_end = min(batch_start + self.BATCH_SIZE, len(csv_files))
            batch_files = csv_files[batch_start:batch_end]

            self.logger.info(f"Processing ERCOT batch {batch_start//self.BATCH_SIZE + 1}: files {batch_start+1}-{batch_end} of {len(csv_files)}")

            dfs = []
            for csv_file in batch_files:
                try:
                    df_temp = pd.read_csv(csv_file, header=None, names=['DeliveryDate', 'HourEnding', 'SettlementPoint', 'SettlementPointPrice', 'DSTFlag'], skiprows=1)
                    dfs.append(df_temp)
                except Exception as e:
                    self.logger.error(f"Error reading {csv_file}: {e}")

            if dfs:
                batch_df = pd.concat(dfs, ignore_index=True)
                all_dfs.append(batch_df)
                del dfs, batch_df
                gc.collect()

        if not all_dfs:
            self.logger.warning("No valid ERCOT DA data loaded")
            return

        df = pd.concat(all_dfs, ignore_index=True)
        del all_dfs
        gc.collect()

        if df.empty:
            self.logger.warning("No ERCOT DA data found")
            return

        # Parse ERCOT datetime
        # DeliveryDate is in format MM/DD/YYYY, HourEnding is 01:00 - 24:00
        df['DeliveryDate'] = pd.to_datetime(df['DeliveryDate'])

        # Hour ending: convert "01:00" - "24:00" to hour number
        df['hour_num'] = df['HourEnding'].str.split(':').str[0].astype(int)

        # Create datetime (hour ending 1 = hour 0 in datetime, hour ending 24 = hour 23)
        df['datetime_local'] = df['DeliveryDate'] + pd.to_timedelta(df['hour_num'] - 1, unit='h')

        # Handle DST using ERCOT's DSTFlag column
        # CRITICAL: DSTFlag is NOT a general DST indicator!
        # DSTFlag='Y' ONLY appears on the SECOND occurrence of the repeated hour during fall-back
        # DSTFlag='N' is used for ALL other hours (including summer DST hours!)
        #
        # Strategy: Use America/Chicago timezone with ambiguous parameter
        # - For non-ambiguous times: pandas handles DST automatically
        # - For ambiguous times (Nov fall-back): DSTFlag tells us which occurrence
        #   - DSTFlag='N' = first occurrence (still in CDT before 2am fall-back)
        #   - DSTFlag='Y' = second occurrence (now in CST after fall-back)

        # Convert using America/Chicago with DST handling
        # Schema v2.0.0: Keep both datetime_utc and datetime_local timezone-aware!
        # Vectorized approach: most times are unambiguous, handle separately
        try:
            # Try to localize all at once with ambiguous='infer' (works for 99% of rows)
            df['datetime_local'] = df['datetime_local'].dt.tz_localize('America/Chicago', ambiguous='infer')
            df['datetime_utc'] = df['datetime_local'].dt.tz_convert('UTC')
        except:
            # If inference fails (fall-back day), use DSTFlag to distinguish
            # DSTFlag='N' = first occurrence (CDT), 'Y' = second occurrence (CST)
            df['datetime_local'] = df['datetime_local'].dt.tz_localize(
                'America/Chicago',
                ambiguous=(df['DSTFlag'] == 'N')  # True=first (CDT), False=second (CST)
            )
            df['datetime_utc'] = df['datetime_local'].dt.tz_convert('UTC')

        df_unified = pd.DataFrame({
            'datetime_utc': df['datetime_utc'],
            'datetime_local': df['datetime_local'],  # Now timezone-aware!
            'interval_start_utc': df['datetime_utc'],
            'interval_end_utc': df['datetime_utc'] + pd.Timedelta(hours=1),
            'delivery_date': df['DeliveryDate'].dt.date,
            'delivery_hour': df['hour_num'].astype('uint8'),
            'delivery_interval': np.uint8(0),
            'interval_minutes': np.uint8(60),
            'iso': 'ERCOT',
            'market_type': 'DA',
            'settlement_location': df['SettlementPoint'],
            'settlement_location_type': 'NODE',  # or map from SettlementPointType if available
            'settlement_location_id': df['SettlementPoint'].astype(str),
            'zone': None,
            'voltage_kv': None,
            'lmp_total': df['SettlementPointPrice'].astype('float64'),
            'lmp_energy': None,
            'lmp_congestion': None,
            'lmp_loss': None,
            'system_lambda': None,
            'dst_flag': df['DSTFlag'].astype(str) if 'DSTFlag' in df.columns else None,
            'data_source': 'ERCOT DAM',
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

    def convert_rt_energy(self, year=None) -> None:
        """Convert Real-Time energy prices (15-minute)."""
        self.logger.info("Converting RT Energy prices (15-min)...")

        # ERCOT RT files: cdr.00012301.*.SPPHLZNP6905_*.csv
        df = self._read_ercot_csv_files("cdr.00012301.*.SPPHLZNP6905_*.csv", year)

        if df.empty:
            self.logger.warning("No ERCOT RT data found")
            return

        # Parse datetime
        df['DeliveryDate'] = pd.to_datetime(df['DeliveryDate'])
        df['DeliveryHour'] = df['DeliveryHour'].astype(int)
        df['DeliveryInterval'] = df['DeliveryInterval'].astype(int)

        # Calculate datetime: hour (0-23) + interval (1-4)
        df['hour_offset'] = df['DeliveryHour'] - 1
        df['interval_offset'] = (df['DeliveryInterval'] - 1) * 15  # 0, 15, 30, 45 minutes

        # Schema v2.0.0: Keep timezone-aware timestamps
        df['datetime_local'] = df['DeliveryDate'] + pd.to_timedelta(df['hour_offset'], unit='h') + pd.to_timedelta(df['interval_offset'], unit='m')
        df['datetime_local'] = df['datetime_local'].dt.tz_localize('America/Chicago', ambiguous='infer')
        df['datetime_utc'] = df['datetime_local'].dt.tz_convert('UTC')

        # Get settlement point type if available
        settlement_type = df.get('SettlementPointType', 'NODE')

        df_unified = pd.DataFrame({
            'datetime_utc': df['datetime_utc'],
            'datetime_local': df['datetime_local'],
            'interval_start_utc': df['datetime_utc'],
            'interval_end_utc': df['datetime_utc'] + pd.Timedelta(minutes=15),
            'delivery_date': df['DeliveryDate'].dt.date,
            'delivery_hour': df['DeliveryHour'].astype('uint8'),
            'delivery_interval': df['DeliveryInterval'].astype('uint8'),
            'interval_minutes': np.uint8(15),
            'iso': 'ERCOT',
            'market_type': 'RT',
            'settlement_location': df['SettlementPointName'],
            'settlement_location_type': settlement_type.map(self.settlement_types) if isinstance(settlement_type, pd.Series) else 'NODE',
            'settlement_location_id': df['SettlementPointName'].astype(str),
            'zone': None,
            'voltage_kv': None,
            'lmp_total': df['SettlementPointPrice'].astype('float64'),
            'lmp_energy': None,
            'lmp_congestion': None,
            'lmp_loss': None,
            'system_lambda': None,
            'dst_flag': df['DSTFlag'].astype(str) if 'DSTFlag' in df.columns else None,
            'data_source': 'ERCOT SCED',
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
            output_dir = self.parquet_output_dir / "rt_energy_15min"
            output_file = output_dir / f"rt_energy_15min_{yr}.parquet"

            self.write_parquet_atomic(df_year, output_file, self.ENERGY_SCHEMA, compression='snappy')

    def convert_ancillary_services(self, year=None) -> None:
        """Convert ancillary services (hourly)."""
        self.logger.info("Converting Ancillary Services...")

        # ERCOT AS files: cdr.00012329.*.DAMCPCNP4188.csv
        df = self._read_ercot_csv_files("cdr.00012329.*.DAMCPCNP4188.csv", year)

        if df.empty:
            self.logger.warning("No ERCOT AS data found")
            return

        # Parse datetime (Schema v2.0.0: keep timezone-aware)
        df['DeliveryDate'] = pd.to_datetime(df['DeliveryDate'])
        df['hour_num'] = df['HourEnding'].str.split(':').str[0].astype(int)
        df['datetime_local'] = df['DeliveryDate'] + pd.to_timedelta(df['hour_num'] - 1, unit='h')
        df['datetime_local'] = df['datetime_local'].dt.tz_localize('America/Chicago', ambiguous='infer')
        df['datetime_utc'] = df['datetime_local'].dt.tz_convert('UTC')

        df_unified = pd.DataFrame({
            'datetime_utc': df['datetime_utc'],
            'datetime_local': df['datetime_local'],
            'interval_start_utc': df['datetime_utc'],
            'interval_end_utc': df['datetime_utc'] + pd.Timedelta(hours=1),
            'delivery_date': df['DeliveryDate'].dt.date,
            'delivery_hour': df['hour_num'].astype('uint8'),
            'interval_minutes': np.uint8(60),
            'iso': 'ERCOT',
            'market_type': 'DA',
            'as_product': df['AncillaryType'],
            'as_product_standard': df['AncillaryType'].map(self.as_product_mapping).fillna('OTHER'),
            'as_region': None,
            'market_clearing_price': df['MCPC'].astype('float64'),
            'cleared_quantity_mw': None,
            'unit': '$/MW',
            'data_source': 'ERCOT DAM',
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

            self.write_parquet_atomic(df_year, output_file, self.AS_SCHEMA, compression='snappy')


def main():
    parser = argparse.ArgumentParser(description='Convert ERCOT data to UNIFIED parquet format (separate from legacy)')
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

    print("=" * 80)
    print("ERCOT UNIFIED FORMAT CONVERTER")
    print("=" * 80)
    print("Creating NEW parquet files in unified format")
    print("Output: /home/enrico/data/unified_iso_data/parquet/ercot/")
    print()
    print("LEGACY ERCOT files NOT touched:")
    print("  /home/enrico/data/ERCOT_data/rollup_files/")
    print("=" * 80)
    print()

    converter = ERCOTParquetConverter(**kwargs)

    convert_da = not (args.rt_only or args.as_only)
    convert_rt = not (args.da_only or args.as_only)
    convert_as = not (args.da_only or args.rt_only)

    converter.run_full_conversion(year=args.year, convert_da=convert_da, convert_rt=convert_rt, convert_as=convert_as)


if __name__ == "__main__":
    main()
