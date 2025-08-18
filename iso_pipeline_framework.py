#!/usr/bin/env python3
"""
ISO Pipeline Framework
Generic implementation of the 4-stage data processing pipeline for any ISO
"""

import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import Dict, List, Optional, Union
import json
import os
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import logging
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ISOPipelineBase(ABC):
    """Base class for ISO-specific data pipelines."""
    
    # Standard column mappings all ISOs should implement
    STANDARD_COLUMNS = {
        'datetime': pl.Datetime,
        'settlement_point': pl.Utf8,
        'settlement_type': pl.Utf8,
        'iso': pl.Utf8,
        'da_lmp': pl.Float64,
        'rt_lmp': pl.Float64,
        'rt_energy': pl.Float64,
        'rt_congestion': pl.Float64,
        'rt_loss': pl.Float64,
        'as_reg_up': pl.Float64,
        'as_reg_down': pl.Float64,
        'as_spin': pl.Float64,
        'as_nonspin': pl.Float64,
    }
    
    def __init__(self, iso_name: str, base_dir: Path, config: Optional[Dict] = None):
        self.iso_name = iso_name
        self.base_dir = Path(base_dir)
        self.config = config or {}
        
        # Setup directory structure
        self.setup_directories()
        
        # Load or create schema registry
        self.schema_registry = self.load_schema_registry()
        
        # Configure parallelism
        self.setup_parallelism()
    
    def setup_directories(self):
        """Create standard directory structure."""
        self.dirs = {
            'raw_downloads': self.base_dir / 'raw_downloads',
            'csv_files': self.base_dir / 'csv_files',
            'rollup_files': self.base_dir / 'rollup_files',
            'flattened': self.base_dir / 'rollup_files' / 'flattened',
            'combined': self.base_dir / 'rollup_files' / 'combined',
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def setup_parallelism(self):
        """Configure parallel processing based on system resources."""
        import multiprocessing
        
        cpu_count = multiprocessing.cpu_count()
        mem_gb = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024**3)
        
        # Adaptive configuration based on resources
        if mem_gb > 128:
            self.batch_size = 2000
            self.num_threads = min(cpu_count, 32)
            self.io_threads = 16
        elif mem_gb > 64:
            self.batch_size = 1000
            self.num_threads = min(cpu_count, 24)
            self.io_threads = 12
        elif mem_gb > 32:
            self.batch_size = 500
            self.num_threads = min(cpu_count, 16)
            self.io_threads = 8
        else:
            self.batch_size = 100
            self.num_threads = min(cpu_count, 8)
            self.io_threads = 4
        
        logger.info(f"Parallelism config: {self.num_threads} threads, "
                   f"batch_size={self.batch_size}, io_threads={self.io_threads}")
    
    @abstractmethod
    def get_column_mappings(self) -> Dict[str, str]:
        """Return ISO-specific column mappings to standard names."""
        pass
    
    @abstractmethod
    def get_schema_overrides(self) -> Dict[str, pl.DataType]:
        """Return ISO-specific type overrides."""
        pass
    
    # ============= Stage 1: CSV Extraction =============
    
    def stage1_extract_csv(self, source_dir: Optional[Path] = None):
        """Extract CSV files from archives."""
        logger.info(f"Stage 1: Extracting CSV files for {self.iso_name}")
        
        source_dir = source_dir or self.dirs['raw_downloads']
        
        # Find all archive files
        archives = list(source_dir.glob("**/*.zip")) + list(source_dir.glob("**/*.tar.gz"))
        
        logger.info(f"Found {len(archives)} archives to process")
        
        with ThreadPoolExecutor(max_workers=self.io_threads) as executor:
            futures = [executor.submit(self.extract_archive, arc) for arc in archives]
            for future in futures:
                future.result()
        
        logger.info("Stage 1 complete: CSV extraction finished")
    
    def extract_archive(self, archive_path: Path):
        """Extract a single archive."""
        import zipfile
        import tarfile
        
        output_dir = self.dirs['csv_files'] / archive_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            if archive_path.suffix == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zf:
                    zf.extractall(output_dir)
            elif archive_path.suffix in ['.gz', '.tar']:
                with tarfile.open(archive_path, 'r:*') as tf:
                    tf.extractall(output_dir)
            
            logger.debug(f"Extracted {archive_path.name}")
        except Exception as e:
            logger.error(f"Failed to extract {archive_path}: {e}")
    
    # ============= Stage 2: Raw Parquet Conversion =============
    
    def stage2_csv_to_parquet(self, dataset_types: Optional[List[str]] = None):
        """Convert CSV files to raw parquet format."""
        logger.info(f"Stage 2: Converting CSV to Parquet for {self.iso_name}")
        
        if dataset_types is None:
            dataset_types = ['DA_prices', 'RT_prices', 'AS_prices']
        
        for dataset_type in dataset_types:
            self.process_dataset_to_parquet(dataset_type)
        
        logger.info("Stage 2 complete: Raw parquet conversion finished")
    
    def process_dataset_to_parquet(self, dataset_type: str):
        """Process a specific dataset type to parquet."""
        csv_dir = self.dirs['csv_files'] / dataset_type
        if not csv_dir.exists():
            logger.warning(f"No CSV files found for {dataset_type}")
            return
        
        csv_files = list(csv_dir.glob("**/*.csv"))
        logger.info(f"Processing {len(csv_files)} CSV files for {dataset_type}")
        
        # Group files by year
        files_by_year = {}
        for csv_file in csv_files:
            # Extract year from filename or content
            year = self.extract_year_from_file(csv_file)
            if year not in files_by_year:
                files_by_year[year] = []
            files_by_year[year].append(csv_file)
        
        # Process each year in parallel
        output_dir = self.dirs['rollup_files'] / dataset_type
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with ProcessPoolExecutor(max_workers=self.num_threads) as executor:
            futures = []
            for year, files in files_by_year.items():
                output_path = output_dir / f"{dataset_type}_{year}.parquet"
                future = executor.submit(
                    self.convert_year_to_parquet,
                    files, output_path, dataset_type
                )
                futures.append(future)
            
            for future in futures:
                future.result()
    
    def convert_year_to_parquet(self, csv_files: List[Path], output_path: Path, dataset_type: str):
        """Convert a year's worth of CSV files to a single parquet file."""
        schema_overrides = self.get_schema_overrides()
        column_mappings = self.get_column_mappings()
        
        # Process files in batches
        all_dfs = []
        for i in range(0, len(csv_files), self.batch_size):
            batch = csv_files[i:i + self.batch_size]
            
            batch_dfs = []
            for csv_file in batch:
                try:
                    # Read with schema overrides
                    df = pl.read_csv(
                        csv_file,
                        dtypes=schema_overrides,
                        try_parse_dates=True,
                        ignore_errors=True
                    )
                    
                    # Apply column mappings
                    for old_name, new_name in column_mappings.items():
                        if old_name in df.columns:
                            df = df.rename({old_name: new_name})
                    
                    # Add ISO identifier
                    df = df.with_columns(pl.lit(self.iso_name).alias("iso"))
                    
                    batch_dfs.append(df)
                except Exception as e:
                    logger.error(f"Failed to process {csv_file}: {e}")
            
            if batch_dfs:
                combined = pl.concat(batch_dfs, how="diagonal")
                all_dfs.append(combined)
        
        # Combine all batches and save
        if all_dfs:
            final_df = pl.concat(all_dfs, how="diagonal")
            
            # Sort by datetime if available
            if "datetime" in final_df.columns:
                final_df = final_df.sort("datetime")
            
            # Write to parquet
            final_df.write_parquet(
                output_path,
                compression="zstd",
                statistics=True,
                row_group_size=100_000
            )
            
            logger.info(f"Created {output_path.name} with {len(final_df):,} rows")
    
    def extract_year_from_file(self, csv_file: Path) -> int:
        """Extract year from filename or file content."""
        # Try filename first
        import re
        year_match = re.search(r'20\d{2}', csv_file.name)
        if year_match:
            return int(year_match.group())
        
        # Fall back to reading first few rows
        try:
            df = pl.read_csv(csv_file, n_rows=10)
            if 'datetime' in df.columns or 'date' in df.columns:
                date_col = 'datetime' if 'datetime' in df.columns else 'date'
                first_date = df[date_col][0]
                if isinstance(first_date, str):
                    return pd.to_datetime(first_date).year
                return first_date.year
        except:
            pass
        
        # Default to current year
        return datetime.now().year
    
    # ============= Stage 3: Flattened Parquet =============
    
    def stage3_flatten_parquet(self):
        """Flatten parquet files from long to wide format."""
        logger.info(f"Stage 3: Flattening parquet files for {self.iso_name}")
        
        # Process each market type
        self.flatten_da_prices()
        self.flatten_rt_prices()
        self.flatten_as_prices()
        
        logger.info("Stage 3 complete: Flattening finished")
    
    def flatten_da_prices(self):
        """Flatten day-ahead prices to wide format (hourly)."""
        input_dir = self.dirs['rollup_files'] / 'DA_prices'
        output_dir = self.dirs['flattened']
        
        for parquet_file in input_dir.glob("*.parquet"):
            logger.info(f"Flattening {parquet_file.name}")
            
            df = pl.read_parquet(parquet_file)
            
            # Pivot settlement points to columns
            if 'settlement_point' in df.columns and 'da_lmp' in df.columns:
                flattened = df.pivot(
                    values="da_lmp",
                    index="datetime",
                    columns="settlement_point"
                )
                
                # Save flattened version
                output_path = output_dir / f"DA_prices_flat_{parquet_file.stem.split('_')[-1]}.parquet"
                flattened.write_parquet(output_path, compression="zstd")
                
                logger.info(f"Created flattened DA prices: {output_path.name}")
    
    def flatten_rt_prices(self):
        """Flatten real-time prices preserving native granularity."""
        input_dir = self.dirs['rollup_files'] / 'RT_prices'
        output_dir = self.dirs['flattened']
        
        for parquet_file in input_dir.glob("*.parquet"):
            logger.info(f"Flattening {parquet_file.name}")
            
            df = pl.read_parquet(parquet_file)
            
            # Pivot settlement points to columns
            if 'settlement_point' in df.columns and 'rt_lmp' in df.columns:
                flattened = df.pivot(
                    values="rt_lmp",
                    index="datetime",
                    columns="settlement_point"
                )
                
                # Save flattened version
                output_path = output_dir / f"RT_prices_flat_{parquet_file.stem.split('_')[-1]}.parquet"
                flattened.write_parquet(output_path, compression="zstd")
                
                # Also create hourly aggregated version
                hourly = self.aggregate_to_hourly(df)
                hourly_path = output_dir / f"RT_prices_hourly_{parquet_file.stem.split('_')[-1]}.parquet"
                hourly.write_parquet(hourly_path, compression="zstd")
                
                logger.info(f"Created flattened RT prices: {output_path.name}")
    
    def flatten_as_prices(self):
        """Flatten ancillary service prices to wide format."""
        input_dir = self.dirs['rollup_files'] / 'AS_prices'
        output_dir = self.dirs['flattened']
        
        for parquet_file in input_dir.glob("*.parquet"):
            logger.info(f"Flattening {parquet_file.name}")
            
            df = pl.read_parquet(parquet_file)
            
            # Create wide format with all AS products as columns
            # This will vary by ISO
            output_path = output_dir / f"AS_prices_flat_{parquet_file.stem.split('_')[-1]}.parquet"
            df.write_parquet(output_path, compression="zstd")
            
            logger.info(f"Created flattened AS prices: {output_path.name}")
    
    def aggregate_to_hourly(self, df: pl.DataFrame) -> pl.DataFrame:
        """Aggregate sub-hourly data to hourly."""
        if 'datetime' not in df.columns:
            return df
        
        # Add hour column
        df = df.with_columns(
            pl.col("datetime").dt.truncate("1h").alias("hour")
        )
        
        # Aggregate numeric columns
        numeric_cols = [col for col in df.columns 
                       if df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]]
        
        agg_exprs = []
        for col in numeric_cols:
            if col not in ['hour', 'datetime']:
                agg_exprs.extend([
                    pl.col(col).mean().alias(f"{col}_avg"),
                    pl.col(col).max().alias(f"{col}_max"),
                    pl.col(col).min().alias(f"{col}_min"),
                    pl.col(col).std().alias(f"{col}_std")
                ])
        
        if agg_exprs:
            return df.group_by("hour").agg(agg_exprs).sort("hour")
        return df
    
    # ============= Stage 4: Combined Parquet =============
    
    def stage4_create_combined(self):
        """Create combined datasets from flattened parquet files."""
        logger.info(f"Stage 4: Creating combined datasets for {self.iso_name}")
        
        # Create different combinations
        self.create_da_as_combined()
        self.create_da_as_rt_hourly()
        self.create_da_as_rt_native()
        self.create_monthly_splits()
        
        logger.info("Stage 4 complete: Combined datasets created")
    
    def create_da_as_combined(self):
        """Combine DA and AS prices (both hourly)."""
        output_dir = self.dirs['combined'] / 'yearly'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get years available
        da_files = list(self.dirs['flattened'].glob("DA_prices_flat_*.parquet"))
        
        for da_file in da_files:
            year = da_file.stem.split('_')[-1]
            as_file = self.dirs['flattened'] / f"AS_prices_flat_{year}.parquet"
            
            if as_file.exists():
                da_df = pl.read_parquet(da_file)
                as_df = pl.read_parquet(as_file)
                
                # Join on datetime
                combined = da_df.join(as_df, on="datetime", how="outer")
                
                output_path = output_dir / f"DA_AS_combined_{year}.parquet"
                combined.write_parquet(output_path, compression="zstd")
                
                logger.info(f"Created DA+AS combined for {year}")
    
    def create_da_as_rt_hourly(self):
        """Combine DA, AS, and hourly aggregated RT."""
        output_dir = self.dirs['combined'] / 'yearly'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Implementation similar to above but with RT hourly data
        pass
    
    def create_da_as_rt_native(self):
        """Combine DA, AS, and native granularity RT."""
        output_dir = self.dirs['combined'] / 'yearly'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Implementation with RT at native granularity
        # DA/AS values repeated for each RT interval
        pass
    
    def create_monthly_splits(self):
        """Split combined datasets into monthly files."""
        yearly_dir = self.dirs['combined'] / 'yearly'
        monthly_dir = self.dirs['combined'] / 'monthly'
        
        for yearly_file in yearly_dir.glob("*.parquet"):
            df = pl.read_parquet(yearly_file)
            
            if 'datetime' in df.columns:
                # Add month column
                df = df.with_columns(
                    pl.col("datetime").dt.strftime("%Y-%m").alias("month")
                )
                
                # Split by month
                for month in df['month'].unique():
                    month_df = df.filter(pl.col("month") == month)
                    
                    month_dir = monthly_dir / month
                    month_dir.mkdir(parents=True, exist_ok=True)
                    
                    output_name = yearly_file.stem.replace(
                        yearly_file.stem.split('_')[-1], month
                    )
                    output_path = month_dir / f"{output_name}.parquet"
                    
                    month_df.drop("month").write_parquet(
                        output_path, compression="zstd"
                    )
                
                logger.info(f"Created monthly splits for {yearly_file.name}")
    
    # ============= Verification =============
    
    def verify_pipeline(self):
        """Verify the entire pipeline."""
        logger.info(f"Verifying pipeline for {self.iso_name}")
        
        report = {
            'iso': self.iso_name,
            'timestamp': datetime.now().isoformat(),
            'stages': {}
        }
        
        # Verify each stage
        report['stages']['csv'] = self.verify_csv_files()
        report['stages']['raw_parquet'] = self.verify_raw_parquet()
        report['stages']['flattened'] = self.verify_flattened()
        report['stages']['combined'] = self.verify_combined()
        
        # Save report
        report_path = self.base_dir / 'pipeline_verification_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Verification report saved to {report_path}")
        return report
    
    def verify_csv_files(self) -> Dict:
        """Verify CSV files."""
        csv_count = len(list(self.dirs['csv_files'].glob("**/*.csv")))
        return {
            'file_count': csv_count,
            'status': 'OK' if csv_count > 0 else 'NO_FILES'
        }
    
    def verify_raw_parquet(self) -> Dict:
        """Verify raw parquet files."""
        results = {}
        for dataset_dir in self.dirs['rollup_files'].iterdir():
            if dataset_dir.is_dir():
                parquet_files = list(dataset_dir.glob("*.parquet"))
                if parquet_files:
                    total_rows = 0
                    for pf in parquet_files:
                        df = pl.read_parquet(pf)
                        total_rows += len(df)
                    
                    results[dataset_dir.name] = {
                        'files': len(parquet_files),
                        'rows': total_rows,
                        'status': 'OK'
                    }
        return results
    
    def verify_flattened(self) -> Dict:
        """Verify flattened parquet files."""
        flat_files = list(self.dirs['flattened'].glob("*.parquet"))
        return {
            'file_count': len(flat_files),
            'status': 'OK' if flat_files else 'NO_FILES'
        }
    
    def verify_combined(self) -> Dict:
        """Verify combined datasets."""
        combined_files = list(self.dirs['combined'].glob("**/*.parquet"))
        return {
            'file_count': len(combined_files),
            'status': 'OK' if combined_files else 'NO_FILES'
        }
    
    # ============= Schema Registry =============
    
    def load_schema_registry(self) -> Dict:
        """Load or create schema registry."""
        registry_path = self.base_dir / 'schema_registry.json'
        
        if registry_path.exists():
            with open(registry_path, 'r') as f:
                return json.load(f)
        
        return {}
    
    def save_schema_registry(self):
        """Save schema registry."""
        registry_path = self.base_dir / 'schema_registry.json'
        with open(registry_path, 'w') as f:
            json.dump(self.schema_registry, f, indent=2)


# ============= ISO-Specific Implementations =============

class ERCOTPipeline(ISOPipelineBase):
    """ERCOT-specific pipeline implementation."""
    
    def get_column_mappings(self) -> Dict[str, str]:
        return {
            'SettlementPoint': 'settlement_point',
            'SettlementPointName': 'settlement_point',
            'DeliveryDate': 'datetime',
            'DeliveryHour': 'hour_ending',
            'DeliveryInterval': 'interval',
            'LMP': 'rt_lmp',
            'SPP': 'da_lmp',
            'REGUP': 'as_reg_up',
            'REGDN': 'as_reg_down',
            'RRS': 'as_spin',
            'NSRS': 'as_nonspin',
        }
    
    def get_schema_overrides(self) -> Dict[str, pl.DataType]:
        return {
            'LMP': pl.Float64,
            'SPP': pl.Float64,
            'REGUP': pl.Float64,
            'REGDN': pl.Float64,
            'RRS': pl.Float64,
            'NSRS': pl.Float64,
            'HASL': pl.Float64,
            'LASL': pl.Float64,
            'HDL': pl.Float64,
            'LDL': pl.Float64,
        }


class CAISOPipeline(ISOPipelineBase):
    """CAISO-specific pipeline implementation."""
    
    def get_column_mappings(self) -> Dict[str, str]:
        return {
            'NODE': 'settlement_point',
            'NODE_ID': 'settlement_point',
            'OPR_DATE': 'datetime',
            'OPR_HR': 'hour_ending',
            'OPR_INTERVAL': 'interval',
            'LMP_PRC': 'rt_lmp',
            'DA_LMP': 'da_lmp',
            'MW': 'load_mw',
            'ENERGY_PRC': 'rt_energy',
            'CONGESTION_PRC': 'rt_congestion',
            'LOSS_PRC': 'rt_loss',
        }
    
    def get_schema_overrides(self) -> Dict[str, pl.DataType]:
        return {
            'LMP_PRC': pl.Float64,
            'DA_LMP': pl.Float64,
            'MW': pl.Float64,
            'ENERGY_PRC': pl.Float64,
            'CONGESTION_PRC': pl.Float64,
            'LOSS_PRC': pl.Float64,
        }


class PJMPipeline(ISOPipelineBase):
    """PJM-specific pipeline implementation."""
    
    def get_column_mappings(self) -> Dict[str, str]:
        return {
            'pnode_name': 'settlement_point',
            'datetime_beginning_ept': 'datetime',
            'total_lmp_da': 'da_lmp',
            'total_lmp_rt': 'rt_lmp',
            'energy_lmp_da': 'da_energy',
            'energy_lmp_rt': 'rt_energy',
            'congestion_lmp_da': 'da_congestion',
            'congestion_lmp_rt': 'rt_congestion',
            'marginal_loss_lmp_da': 'da_loss',
            'marginal_loss_lmp_rt': 'rt_loss',
        }
    
    def get_schema_overrides(self) -> Dict[str, pl.DataType]:
        return {
            'total_lmp_da': pl.Float64,
            'total_lmp_rt': pl.Float64,
            'energy_lmp_da': pl.Float64,
            'energy_lmp_rt': pl.Float64,
            'congestion_lmp_da': pl.Float64,
            'congestion_lmp_rt': pl.Float64,
            'marginal_loss_lmp_da': pl.Float64,
            'marginal_loss_lmp_rt': pl.Float64,
        }


# ============= Pipeline Runner =============

def run_pipeline(iso_name: str, base_dir: str, stages: Optional[List[int]] = None):
    """Run the pipeline for a specific ISO."""
    
    # Select appropriate pipeline class
    pipeline_classes = {
        'ERCOT': ERCOTPipeline,
        'CAISO': CAISOPipeline,
        'PJM': PJMPipeline,
    }
    
    if iso_name not in pipeline_classes:
        raise ValueError(f"Unknown ISO: {iso_name}")
    
    # Create pipeline instance
    pipeline_class = pipeline_classes[iso_name]
    pipeline = pipeline_class(iso_name, Path(base_dir))
    
    # Run requested stages (default: all)
    if stages is None:
        stages = [1, 2, 3, 4]
    
    if 1 in stages:
        pipeline.stage1_extract_csv()
    
    if 2 in stages:
        pipeline.stage2_csv_to_parquet()
    
    if 3 in stages:
        pipeline.stage3_flatten_parquet()
    
    if 4 in stages:
        pipeline.stage4_create_combined()
    
    # Always verify at the end
    report = pipeline.verify_pipeline()
    
    print(f"\nPipeline complete for {iso_name}")
    print(f"Verification report: {json.dumps(report, indent=2)}")
    
    return report


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ISO Data Pipeline Framework")
    parser.add_argument("iso", choices=["ERCOT", "CAISO", "PJM"], help="ISO to process")
    parser.add_argument("base_dir", help="Base directory for ISO data")
    parser.add_argument("--stages", nargs="+", type=int, choices=[1, 2, 3, 4],
                       help="Specific stages to run (default: all)")
    
    args = parser.parse_args()
    
    run_pipeline(args.iso, args.base_dir, args.stages)