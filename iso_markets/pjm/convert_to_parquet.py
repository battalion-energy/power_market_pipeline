#!/usr/bin/env python3
"""
Convert PJM CSV files to Parquet format (ERCOT-style pipeline)

Following the 4-stage pipeline from ISO_DATA_PIPELINE_STRATEGY.md:
- Stage 2: Raw Parquet Conversion (year-partitioned, type-enforced)
- Stage 3: Flattened Parquet (wide format, nodes as columns)
- Stage 4: Combined datasets (multi-market joins)

Usage:
    python convert_to_parquet.py --market da_hubs --year 2024
    python convert_to_parquet.py --market rt_hourly --all-years
    python convert_to_parquet.py --all-markets --year 2024
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PJMParquetConverter:
    """Convert PJM CSV files to Parquet following ERCOT-style pipeline."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.csv_dir = data_dir / 'csv_files'
        self.parquet_dir = data_dir / 'parquet_files'
        self.flattened_dir = data_dir / 'flattened'
        self.combined_dir = data_dir / 'combined'

        # Create output directories
        self.parquet_dir.mkdir(parents=True, exist_ok=True)
        self.flattened_dir.mkdir(parents=True, exist_ok=True)
        self.combined_dir.mkdir(parents=True, exist_ok=True)

    def stage_2_raw_parquet(self, market_type: str, year: int = None):
        """
        Stage 2: Convert CSV to raw Parquet with schema enforcement.

        Args:
            market_type: Type of market data (da_hubs, rt_hourly, da_ancillary_services, etc.)
            year: Specific year to process (None = all years)
        """
        logger.info(f"Stage 2: Converting {market_type} CSVs to raw Parquet")

        csv_market_dir = self.csv_dir / market_type
        if not csv_market_dir.exists():
            logger.error(f"CSV directory not found: {csv_market_dir}")
            return

        # Get all CSV files
        csv_files = sorted(csv_market_dir.glob('*.csv'))
        if not csv_files:
            logger.warning(f"No CSV files found in {csv_market_dir}")
            return

        logger.info(f"Found {len(csv_files)} CSV files")

        # Group files by year (from filename or data)
        files_by_year = {}
        for csv_file in csv_files:
            # Try to extract year from filename
            # Format: HUB_YEAR-MM-DD_YEAR-MM-DD.csv or nodal_da_lmp_YEAR-MM-DD_YEAR-MM-DD.csv
            try:
                parts = csv_file.stem.split('_')
                # Find date part (YYYY-MM-DD format)
                for part in parts:
                    if len(part) == 10 and part[4] == '-' and part[7] == '-':
                        file_year = int(part[:4])
                        if year is None or file_year == year:
                            if file_year not in files_by_year:
                                files_by_year[file_year] = []
                            files_by_year[file_year].append(csv_file)
                        break
            except (ValueError, IndexError):
                logger.warning(f"Could not extract year from {csv_file.name}")

        # Process each year
        for file_year in sorted(files_by_year.keys()):
            year_files = files_by_year[file_year]
            logger.info(f"Processing {file_year}: {len(year_files)} files")

            # Read and concatenate all CSVs for this year
            dfs = []
            for csv_file in year_files:
                try:
                    df = pd.read_csv(csv_file)
                    dfs.append(df)
                except Exception as e:
                    logger.error(f"Error reading {csv_file.name}: {e}")

            if not dfs:
                logger.warning(f"No data loaded for {file_year}")
                continue

            # Concatenate all dataframes
            df_combined = pd.concat(dfs, ignore_index=True)
            logger.info(f"  Combined {len(df_combined):,} rows for {file_year}")

            # Enforce data types (all price columns → Float64)
            price_columns = [col for col in df_combined.columns if any(
                keyword in col.lower() for keyword in
                ['price', 'lmp', 'energy', 'congestion', 'loss', 'marginal']
            )]

            for col in price_columns:
                try:
                    df_combined[col] = pd.to_numeric(df_combined[col], errors='coerce').astype('float64')
                except Exception as e:
                    logger.warning(f"Could not convert {col} to float64: {e}")

            # Sort by datetime if available
            datetime_cols = [col for col in df_combined.columns if 'datetime' in col.lower()]
            if datetime_cols:
                df_combined = df_combined.sort_values(datetime_cols[0])

            # Save as Parquet
            output_dir = self.parquet_dir / market_type
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"{market_type}_{file_year}.parquet"

            df_combined.to_parquet(
                output_file,
                engine='pyarrow',
                compression='snappy',
                index=False
            )

            file_size_mb = output_file.stat().st_size / 1024 / 1024
            logger.info(f"  ✓ Saved {output_file.name} ({file_size_mb:.1f} MB)")

    def stage_3_flattened_parquet(self, market_type: str, year: int = None):
        """
        Stage 3: Transform to flattened wide format (nodes as columns).

        Args:
            market_type: Type of market data
            year: Specific year to process (None = all years)
        """
        logger.info(f"Stage 3: Creating flattened Parquet for {market_type}")

        parquet_market_dir = self.parquet_dir / market_type
        if not parquet_market_dir.exists():
            logger.error(f"Parquet directory not found: {parquet_market_dir}")
            logger.info("Run Stage 2 first to create raw Parquet files")
            return

        # Get all parquet files
        parquet_files = sorted(parquet_market_dir.glob('*.parquet'))
        if not parquet_files:
            logger.warning(f"No Parquet files found in {parquet_market_dir}")
            return

        for parquet_file in parquet_files:
            # Check year filter
            if year is not None:
                try:
                    file_year = int(parquet_file.stem.split('_')[-1])
                    if file_year != year:
                        continue
                except (ValueError, IndexError):
                    pass

            logger.info(f"Processing {parquet_file.name}")

            try:
                df = pd.read_parquet(parquet_file)

                # Identify key columns
                datetime_col = next((col for col in df.columns if 'datetime_beginning_ept' in col.lower()), None)
                node_col = next((col for col in df.columns if 'pnode_name' in col.lower()), None)
                price_col = next((col for col in df.columns if 'total_lmp' in col.lower()), None)

                if not all([datetime_col, node_col, price_col]):
                    logger.warning(f"Missing required columns in {parquet_file.name}")
                    logger.warning(f"  datetime: {datetime_col}, node: {node_col}, price: {price_col}")
                    continue

                # Remove duplicates before pivoting (keep last value for each datetime/node pair)
                df = df.drop_duplicates(subset=[datetime_col, node_col], keep='last')

                # Pivot to wide format (nodes as columns)
                df_flat = df.pivot(
                    index=datetime_col,
                    columns=node_col,
                    values=price_col
                )

                # Reset index to make datetime a column
                df_flat = df_flat.reset_index()

                # Save flattened version
                output_dir = self.flattened_dir / market_type
                output_dir.mkdir(parents=True, exist_ok=True)
                output_file = output_dir / f"{market_type}_flattened_{parquet_file.stem.split('_')[-1]}.parquet"

                df_flat.to_parquet(
                    output_file,
                    engine='pyarrow',
                    compression='snappy',
                    index=False
                )

                rows, cols = df_flat.shape
                file_size_mb = output_file.stat().st_size / 1024 / 1024
                logger.info(f"  ✓ Saved {output_file.name}")
                logger.info(f"    {rows:,} timestamps × {cols-1} nodes ({file_size_mb:.1f} MB)")

            except Exception as e:
                logger.error(f"Error processing {parquet_file.name}: {e}")

    def stage_4_combined_datasets(self, year: int):
        """
        Stage 4: Create combined multi-market datasets.

        Joins DA LMP, RT hourly, and ancillary services by timestamp.

        Args:
            year: Year to process
        """
        logger.info(f"Stage 4: Creating combined datasets for {year}")

        # Load flattened datasets
        da_file = self.flattened_dir / 'da_hubs' / f'da_hubs_flattened_{year}.parquet'
        rt_file = self.flattened_dir / 'rt_hourly' / f'rt_hourly_flattened_{year}.parquet'
        as_file = self.parquet_dir / 'da_ancillary_services' / f'da_ancillary_services_{year}.parquet'

        dfs = {}

        if da_file.exists():
            logger.info(f"Loading DA LMP data...")
            dfs['da'] = pd.read_parquet(da_file)

        if rt_file.exists():
            logger.info(f"Loading RT hourly data...")
            dfs['rt'] = pd.read_parquet(rt_file)

        if as_file.exists():
            logger.info(f"Loading ancillary services data...")
            # AS data is not flattened, needs special handling
            df_as = pd.read_parquet(as_file)
            # Pivot AS data by service type
            if 'ancillary_service' in df_as.columns:
                # Remove duplicates before pivoting
                df_as = df_as.drop_duplicates(subset=['datetime_beginning_ept', 'ancillary_service'], keep='last')
                df_as_flat = df_as.pivot(
                    index='datetime_beginning_ept',
                    columns='ancillary_service',
                    values='value'
                )
                df_as_flat = df_as_flat.reset_index()
                dfs['as'] = df_as_flat

        if len(dfs) < 2:
            logger.warning(f"Need at least 2 market types to combine. Found: {list(dfs.keys())}")
            return

        # Combine datasets
        logger.info(f"Combining {len(dfs)} market types...")

        # Start with first dataset
        combined = list(dfs.values())[0]
        datetime_col = [col for col in combined.columns if 'datetime' in col.lower()][0]

        # Join additional datasets
        for i, (market_name, df) in enumerate(list(dfs.items())[1:], 1):
            df_datetime_col = [col for col in df.columns if 'datetime' in col.lower()][0]

            # Rename columns to add market prefix
            df_renamed = df.rename(columns={
                col: f"{market_name}_{col}" for col in df.columns if col != df_datetime_col
            })

            # Merge on datetime
            combined = combined.merge(
                df_renamed,
                left_on=datetime_col,
                right_on=df_datetime_col,
                how='outer'
            )

        # Save combined dataset
        output_dir = self.combined_dir / 'yearly'
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"combined_all_markets_{year}.parquet"

        combined.to_parquet(
            output_file,
            engine='pyarrow',
            compression='snappy',
            index=False
        )

        rows, cols = combined.shape
        file_size_mb = output_file.stat().st_size / 1024 / 1024
        logger.info(f"✓ Saved combined dataset: {output_file.name}")
        logger.info(f"  {rows:,} rows × {cols} columns ({file_size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description='Convert PJM CSVs to Parquet (ERCOT-style 4-stage pipeline)'
    )
    parser.add_argument('--market', type=str, help='Market type (da_hubs, rt_hourly, etc.)')
    parser.add_argument('--all-markets', action='store_true', help='Process all market types')
    parser.add_argument('--year', type=int, help='Specific year to process')
    parser.add_argument('--all-years', action='store_true', help='Process all years')
    parser.add_argument('--stage', type=int, choices=[2, 3, 4], help='Specific stage to run')
    parser.add_argument('--all-stages', action='store_true', help='Run all stages (2, 3, 4)')

    args = parser.parse_args()

    # Get data directory
    data_dir = Path(os.getenv('PJM_DATA_DIR', '/home/enrico/data/PJM_data'))

    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return

    converter = PJMParquetConverter(data_dir)

    # Determine markets to process
    if args.all_markets:
        markets = ['da_hubs', 'rt_hourly', 'da_ancillary_services']
    elif args.market:
        markets = [args.market]
    else:
        logger.error("Must specify --market or --all-markets")
        return

    # Determine stages to run
    if args.all_stages:
        stages = [2, 3, 4]
    elif args.stage:
        stages = [args.stage]
    else:
        stages = [2, 3, 4]  # Default to all stages

    # Determine year
    year = args.year if not args.all_years else None

    # Run pipeline
    for stage in stages:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running Stage {stage}")
        logger.info(f"{'='*60}\n")

        if stage == 2:
            for market in markets:
                converter.stage_2_raw_parquet(market, year)

        elif stage == 3:
            for market in markets:
                if market != 'da_ancillary_services':  # AS handled differently
                    converter.stage_3_flattened_parquet(market, year)

        elif stage == 4:
            if year:
                converter.stage_4_combined_datasets(year)
            else:
                # Find all years and process each
                years = set()
                for market in markets:
                    market_dir = data_dir / 'parquet_files' / market
                    if market_dir.exists():
                        for f in market_dir.glob('*.parquet'):
                            try:
                                y = int(f.stem.split('_')[-1])
                                years.add(y)
                            except:
                                pass
                for y in sorted(years):
                    converter.stage_4_combined_datasets(y)

    logger.info("\n✓ Parquet conversion complete!")


if __name__ == "__main__":
    main()
