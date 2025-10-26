#!/usr/bin/env python3
"""
Unified ISO Parquet Converter - Base Class

This module provides the base class for converting ISO market data from CSV to
the unified parquet format. Each ISO will have a specific implementation that
inherits from this base class.

Design Principles:
1. Consistent schema across all ISOs
2. Year-based partitioning for scalability
3. Atomic file updates to prevent corruption
4. Data validation and quality checks
5. Metadata generation for hubs, nodes, and AS products
"""

import os
import sys
import json
import logging
import tempfile
import resource
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from zoneinfo import ZoneInfo


class UnifiedISOParquetConverter(ABC):
    """Base class for converting ISO data to unified parquet format."""

    # Define schema version
    SCHEMA_VERSION = "1.0.0"

    # Unified energy prices schema
    ENERGY_SCHEMA = pa.schema([
        ('datetime_utc', pa.timestamp('ns', tz='UTC')),
        ('datetime_local', pa.timestamp('ns')),
        ('interval_start_utc', pa.timestamp('ns', tz='UTC')),
        ('interval_end_utc', pa.timestamp('ns', tz='UTC')),
        ('delivery_date', pa.date32()),
        ('delivery_hour', pa.uint8()),
        ('delivery_interval', pa.uint8()),
        ('interval_minutes', pa.uint8()),
        ('iso', pa.string()),
        ('market_type', pa.string()),
        ('settlement_location', pa.string()),
        ('settlement_location_type', pa.string()),
        ('settlement_location_id', pa.string()),
        ('zone', pa.string()),
        ('voltage_kv', pa.float64()),
        ('lmp_total', pa.float64()),
        ('lmp_energy', pa.float64()),
        ('lmp_congestion', pa.float64()),
        ('lmp_loss', pa.float64()),
        ('system_lambda', pa.float64()),
        ('dst_flag', pa.string()),
        ('data_source', pa.string()),
        ('version', pa.uint32()),
        ('is_current', pa.bool_()),
    ])

    # Unified ancillary services schema
    AS_SCHEMA = pa.schema([
        ('datetime_utc', pa.timestamp('ns', tz='UTC')),
        ('datetime_local', pa.timestamp('ns')),
        ('interval_start_utc', pa.timestamp('ns', tz='UTC')),
        ('interval_end_utc', pa.timestamp('ns', tz='UTC')),
        ('delivery_date', pa.date32()),
        ('delivery_hour', pa.uint8()),
        ('interval_minutes', pa.uint8()),
        ('iso', pa.string()),
        ('market_type', pa.string()),
        ('as_product', pa.string()),
        ('as_product_standard', pa.string()),
        ('as_region', pa.string()),
        ('market_clearing_price', pa.float64()),
        ('cleared_quantity_mw', pa.float64()),
        ('unit', pa.string()),
        ('data_source', pa.string()),
        ('version', pa.uint32()),
        ('is_current', pa.bool_()),
    ])

    def __init__(
        self,
        iso_name: str,
        csv_data_dir: str,
        parquet_output_dir: str,
        metadata_dir: str,
        iso_timezone: str,
        logger: Optional[logging.Logger] = None,
        memory_limit_gb: Optional[int] = None
    ):
        """
        Initialize the converter.

        Args:
            iso_name: ISO identifier (e.g., 'PJM', 'CAISO')
            csv_data_dir: Root directory containing CSV files
            parquet_output_dir: Output directory for parquet files
            metadata_dir: Directory for metadata JSON files
            iso_timezone: Timezone string (e.g., 'America/New_York')
            logger: Optional logger instance
            memory_limit_gb: Optional memory limit in GB (default: 50GB). Set to prevent system crashes.
        """
        # Set memory limit FIRST to protect against runaway memory usage
        if memory_limit_gb is None:
            # Default: 50GB limit (safe for 256GB system with other processes running)
            memory_limit_gb = int(os.getenv('ISO_CONVERTER_MEMORY_LIMIT_GB', '50'))

        if memory_limit_gb > 0:
            memory_limit_bytes = memory_limit_gb * 1024**3
            try:
                # Set virtual memory limit (RLIMIT_AS)
                # If exceeded, Python will raise MemoryError instead of crashing the system
                resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes))
                print(f"✅ Memory limit set to {memory_limit_gb}GB for safety", file=sys.stderr)
            except (ValueError, OSError) as e:
                print(f"⚠️  Warning: Could not set memory limit: {e}", file=sys.stderr)

        self.iso_name = iso_name.upper()
        self.csv_data_dir = Path(csv_data_dir)
        self.parquet_output_dir = Path(parquet_output_dir)
        self.metadata_dir = Path(metadata_dir)
        self.iso_timezone = ZoneInfo(iso_timezone)
        self.memory_limit_gb = memory_limit_gb

        # Setup logger
        self.logger = logger or self._setup_logger()

        # Create output directories
        self.parquet_output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        # Initialize metadata tracking
        self.hubs_metadata = []
        self.nodes_metadata = []
        self.zones_metadata = []
        self.as_products_metadata = []

    def _setup_logger(self) -> logging.Logger:
        """Setup default logger."""
        logger = logging.getLogger(f"{self.iso_name}_ParquetConverter")
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    @abstractmethod
    def convert_da_energy(self, year: Optional[int] = None) -> None:
        """Convert Day-Ahead energy prices to parquet."""
        pass

    @abstractmethod
    def convert_rt_energy(self, year: Optional[int] = None) -> None:
        """Convert Real-Time energy prices to parquet."""
        pass

    @abstractmethod
    def convert_ancillary_services(self, year: Optional[int] = None) -> None:
        """Convert ancillary services to parquet."""
        pass

    def normalize_datetime_to_utc(
        self,
        dt_series: pd.Series,
        source_tz: Optional[ZoneInfo] = None
    ) -> pd.Series:
        """
        Normalize datetime to UTC.

        Args:
            dt_series: Pandas datetime series
            source_tz: Source timezone (defaults to ISO timezone)

        Returns:
            UTC-normalized datetime series
        """
        tz = source_tz or self.iso_timezone

        # If already timezone-aware, convert to UTC
        if dt_series.dt.tz is not None:
            return dt_series.dt.tz_convert(timezone.utc)

        # Localize to source timezone, then convert to UTC
        # Handle DST transitions: ambiguous times during fall-back, nonexistent during spring-forward
        return dt_series.dt.tz_localize(
            tz,
            ambiguous='infer',  # Infer DST for ambiguous times
            nonexistent='shift_forward'  # Shift forward for nonexistent times
        ).dt.tz_convert(timezone.utc)

    def enforce_price_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enforce that all price columns are Float64.

        CRITICAL: This prevents type mismatch errors when combining years.

        Args:
            df: DataFrame with price columns

        Returns:
            DataFrame with enforced types
        """
        price_keywords = [
            'price', 'lmp', 'energy', 'congestion', 'loss',
            'marginal', 'lambda', 'clearing'
        ]

        for col in df.columns:
            if any(keyword in col.lower() for keyword in price_keywords):
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')

        return df

    def validate_data(
        self,
        df: pd.DataFrame,
        check_duplicates: bool = True,
        check_sorted: bool = True,
        check_gaps: bool = False
    ) -> Tuple[bool, List[str]]:
        """
        Validate data quality.

        Args:
            df: DataFrame to validate
            check_duplicates: Check for duplicate rows
            check_sorted: Check if sorted by datetime
            check_gaps: Check for time series gaps

        Returns:
            (is_valid, list_of_issues)
        """
        issues = []

        # Check for duplicates
        if check_duplicates and 'datetime_utc' in df.columns and 'settlement_location' in df.columns:
            duplicates = df.duplicated(subset=['datetime_utc', 'settlement_location'], keep=False)
            if duplicates.any():
                n_dupes = duplicates.sum()
                issues.append(f"Found {n_dupes} duplicate (datetime, location) pairs")

        # Check if sorted
        if check_sorted and 'datetime_utc' in df.columns:
            if not df['datetime_utc'].is_monotonic_increasing:
                issues.append("Data is not sorted by datetime_utc")

        # Check for gaps (basic check)
        if check_gaps and 'datetime_utc' in df.columns and 'interval_minutes' in df.columns:
            # This is a simplified check - full implementation would need to be more sophisticated
            expected_intervals = len(df) // df['settlement_location'].nunique()
            actual_intervals = df['datetime_utc'].nunique()
            if actual_intervals < expected_intervals * 0.95:  # Allow 5% missing
                issues.append(f"Potential gaps detected: expected ~{expected_intervals} intervals, found {actual_intervals}")

        # Check price ranges
        price_cols = [col for col in df.columns if 'lmp' in col.lower() or 'price' in col.lower()]
        for col in price_cols:
            if df[col].min() < -1000 or df[col].max() > 10000:
                issues.append(f"Price in {col} outside expected range: [{df[col].min():.2f}, {df[col].max():.2f}]")

        is_valid = len(issues) == 0
        return is_valid, issues

    def write_parquet_atomic(
        self,
        df: pd.DataFrame,
        output_path: Path,
        schema: pa.Schema,
        compression: str = 'snappy'
    ) -> None:
        """
        Write parquet file atomically.

        Creates a temporary file, validates it, then atomically replaces the
        target file. This prevents corruption from interrupted writes.

        Args:
            df: DataFrame to write
            output_path: Target output path
            schema: PyArrow schema
            compression: Compression algorithm
        """
        self.logger.info(f"Writing {len(df):,} rows to {output_path}")

        # Create temporary file in same directory (for atomic move)
        temp_dir = output_path.parent
        temp_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(
            mode='wb',
            suffix='.parquet',
            dir=temp_dir,
            delete=False
        ) as tmp_file:
            temp_path = Path(tmp_file.name)

        try:
            # Write to temporary file
            table = pa.Table.from_pandas(df, schema=schema)
            pq.write_table(
                table,
                temp_path,
                compression=compression,
                row_group_size=1000000,  # 1M rows per group
                use_dictionary=True,
                write_statistics=True
            )

            # Verify temp file
            test_df = pd.read_parquet(temp_path)
            if len(test_df) != len(df):
                raise ValueError(f"Verification failed: written {len(test_df)} rows, expected {len(df)}")

            # Atomic move
            temp_path.replace(output_path)
            self.logger.info(f"Successfully wrote {output_path}")

        except Exception as e:
            self.logger.error(f"Error writing parquet: {e}")
            if temp_path.exists():
                temp_path.unlink()
            raise

    def save_metadata_json(self, metadata_type: str, data: Dict[str, Any]) -> None:
        """
        Save metadata to JSON file.

        Args:
            metadata_type: Type of metadata (e.g., 'hubs', 'nodes')
            data: Metadata dictionary
        """
        filename = f"{self.iso_name.lower()}_{metadata_type}.json"
        output_path = self.metadata_dir / metadata_type / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        self.logger.info(f"Saved {metadata_type} metadata to {output_path}")

    def extract_unique_locations(
        self,
        df: pd.DataFrame,
        location_col: str = 'settlement_location',
        location_type_col: str = 'settlement_location_type',
        location_id_col: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract unique settlement locations for metadata.

        Args:
            df: DataFrame with location data
            location_col: Column name for location
            location_type_col: Column name for location type
            location_id_col: Optional column for location ID

        Returns:
            List of location metadata dictionaries
        """
        cols = [location_col, location_type_col]
        if location_id_col and location_id_col in df.columns:
            cols.append(location_id_col)
        if 'zone' in df.columns:
            cols.append('zone')
        if 'voltage_kv' in df.columns:
            cols.append('voltage_kv')

        unique_locs = df[cols].drop_duplicates()

        locations = []
        for _, row in unique_locs.iterrows():
            loc_dict = {
                'name': row[location_col],
                'type': row[location_type_col],
                'active': True
            }

            if location_id_col and location_id_col in row:
                loc_dict['id'] = str(row[location_id_col])

            if 'zone' in row and pd.notna(row['zone']):
                loc_dict['zone'] = row['zone']

            if 'voltage_kv' in row and pd.notna(row['voltage_kv']):
                loc_dict['voltage_kv'] = float(row['voltage_kv'])

            locations.append(loc_dict)

        return locations

    def run_full_conversion(
        self,
        year: Optional[int] = None,
        convert_da: bool = True,
        convert_rt: bool = True,
        convert_as: bool = True
    ) -> None:
        """
        Run full conversion process for all market types.

        Args:
            year: Specific year to process (None = all years)
            convert_da: Convert day-ahead prices
            convert_rt: Convert real-time prices
            convert_as: Convert ancillary services
        """
        self.logger.info(f"Starting full conversion for {self.iso_name}")

        if year:
            self.logger.info(f"Processing year: {year}")
        else:
            self.logger.info("Processing all available years")

        try:
            if convert_da:
                self.logger.info("Converting Day-Ahead energy prices...")
                self.convert_da_energy(year)

            if convert_rt:
                self.logger.info("Converting Real-Time energy prices...")
                self.convert_rt_energy(year)

            if convert_as:
                self.logger.info("Converting Ancillary Services...")
                self.convert_ancillary_services(year)

            self.logger.info(f"Conversion complete for {self.iso_name}")

        except Exception as e:
            self.logger.error(f"Conversion failed: {e}", exc_info=True)
            raise


if __name__ == "__main__":
    print("This is a base class. Use specific ISO implementations.")
    print("Available implementations:")
    print("  - pjm_parquet_converter.py")
    print("  - caiso_parquet_converter.py")
    print("  - miso_parquet_converter.py")
    print("  - nyiso_parquet_converter.py")
    print("  - isone_parquet_converter.py")
    print("  - spp_parquet_converter.py")
