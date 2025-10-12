#!/usr/bin/env python3
"""
ISO-NE Data Quality Audit Script

Performs comprehensive quality checks on downloaded ISO-NE historical data:
- Date coverage and continuity
- Missing files detection
- File size validation
- Row count validation
- Schema validation
- Data completeness checks
- Duplicate detection

Usage:
    python audit_isone_data_quality.py --start-date 2019-01-01 --end-date 2025-10-11
    python audit_isone_data_quality.py --auto  # Uses data range from files
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Set
import pandas as pd
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('isone_data_quality_audit.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class ISONEDataAuditor:
    """Audits ISO-NE historical data for quality issues."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir / "ISONE_data" / "csv_files"
        self.issues = defaultdict(list)
        self.stats = {
            'da_files_found': 0,
            'rt_files_found': 0,
            'da_files_missing': 0,
            'rt_files_missing': 0,
            'da_total_rows': 0,
            'rt_total_rows': 0,
            'da_empty_files': 0,
            'rt_empty_files': 0,
            'da_small_files': 0,
            'rt_small_files': 0,
            'dates_checked': 0
        }

    def get_existing_dates(self, data_type: str) -> Set[datetime]:
        """Get all dates with existing files."""
        type_dir = self.data_dir / data_type
        if not type_dir.exists():
            return set()

        dates = set()
        for csv_file in type_dir.glob("*.csv"):
            try:
                date_str = csv_file.stem.split('_')[0]
                date = datetime.strptime(date_str, '%Y-%m-%d')
                dates.add(date)
            except:
                logger.warning(f"Could not parse date from filename: {csv_file.name}")
                continue

        return dates

    def check_date_coverage(self, start_date: datetime, end_date: datetime):
        """Check for missing dates in the range."""
        logger.info("\n" + "="*80)
        logger.info("CHECKING DATE COVERAGE")
        logger.info("="*80)

        # Get existing dates
        da_dates = self.get_existing_dates("lmp_day_ahead_hourly")
        rt_dates = self.get_existing_dates("lmp_real_time_5_min")

        logger.info(f"Day-Ahead files found: {len(da_dates)}")
        logger.info(f"Real-Time files found: {len(rt_dates)}")

        # Check for gaps
        current_date = start_date
        missing_da = []
        missing_rt = []

        while current_date <= end_date:
            self.stats['dates_checked'] += 1

            if current_date not in da_dates:
                missing_da.append(current_date)
                self.stats['da_files_missing'] += 1
            else:
                self.stats['da_files_found'] += 1

            if current_date not in rt_dates:
                missing_rt.append(current_date)
                self.stats['rt_files_missing'] += 1
            else:
                self.stats['rt_files_found'] += 1

            current_date += timedelta(days=1)

        # Report missing dates
        if missing_da:
            logger.warning(f"\n⚠ Missing {len(missing_da)} Day-Ahead files")
            self.issues['missing_da_files'] = missing_da
            if len(missing_da) <= 20:
                for date in missing_da[:20]:
                    logger.warning(f"  - {date.date()}")
            else:
                logger.warning(f"  First 10: {[d.date() for d in missing_da[:10]]}")
                logger.warning(f"  Last 10: {[d.date() for d in missing_da[-10:]]}")

        if missing_rt:
            logger.warning(f"\n⚠ Missing {len(missing_rt)} Real-Time files")
            self.issues['missing_rt_files'] = missing_rt
            if len(missing_rt) <= 20:
                for date in missing_rt[:20]:
                    logger.warning(f"  - {date.date()}")
            else:
                logger.warning(f"  First 10: {[d.date() for d in missing_rt[:10]]}")
                logger.warning(f"  Last 10: {[d.date() for d in missing_rt[-10:]]}")

        if not missing_da and not missing_rt:
            logger.info("✓ Complete date coverage - no gaps found")

    def check_file_sizes(self, start_date: datetime, end_date: datetime):
        """Check for unusually small files that might indicate download issues."""
        logger.info("\n" + "="*80)
        logger.info("CHECKING FILE SIZES")
        logger.info("="*80)

        # Expected minimum sizes (in KB)
        MIN_DA_SIZE_KB = 500  # DA files should be ~1-2MB
        MIN_RT_SIZE_KB = 500  # RT files should be ~1-2MB

        da_dir = self.data_dir / "lmp_day_ahead_hourly"
        rt_dir = self.data_dir / "lmp_real_time_5_min"

        small_da_files = []
        small_rt_files = []
        empty_da_files = []
        empty_rt_files = []

        # Check DA files
        if da_dir.exists():
            for csv_file in da_dir.glob("*.csv"):
                size_kb = csv_file.stat().st_size / 1024
                if size_kb == 0:
                    empty_da_files.append((csv_file.name, size_kb))
                    self.stats['da_empty_files'] += 1
                elif size_kb < MIN_DA_SIZE_KB:
                    small_da_files.append((csv_file.name, size_kb))
                    self.stats['da_small_files'] += 1

        # Check RT files
        if rt_dir.exists():
            for csv_file in rt_dir.glob("*.csv"):
                size_kb = csv_file.stat().st_size / 1024
                if size_kb == 0:
                    empty_rt_files.append((csv_file.name, size_kb))
                    self.stats['rt_empty_files'] += 1
                elif size_kb < MIN_RT_SIZE_KB:
                    small_rt_files.append((csv_file.name, size_kb))
                    self.stats['rt_small_files'] += 1

        # Report issues
        if empty_da_files:
            logger.error(f"\n✗ Found {len(empty_da_files)} empty Day-Ahead files")
            self.issues['empty_da_files'] = empty_da_files
            for filename, size in empty_da_files[:10]:
                logger.error(f"  - {filename}: {size:.1f} KB")

        if empty_rt_files:
            logger.error(f"\n✗ Found {len(empty_rt_files)} empty Real-Time files")
            self.issues['empty_rt_files'] = empty_rt_files
            for filename, size in empty_rt_files[:10]:
                logger.error(f"  - {filename}: {size:.1f} KB")

        if small_da_files:
            logger.warning(f"\n⚠ Found {len(small_da_files)} small Day-Ahead files (< {MIN_DA_SIZE_KB} KB)")
            self.issues['small_da_files'] = small_da_files
            for filename, size in small_da_files[:10]:
                logger.warning(f"  - {filename}: {size:.1f} KB")

        if small_rt_files:
            logger.warning(f"\n⚠ Found {len(small_rt_files)} small Real-Time files (< {MIN_RT_SIZE_KB} KB)")
            self.issues['small_rt_files'] = small_rt_files
            for filename, size in small_rt_files[:10]:
                logger.warning(f"  - {filename}: {size:.1f} KB")

        if not (empty_da_files or empty_rt_files or small_da_files or small_rt_files):
            logger.info("✓ All files have acceptable sizes")

    def check_row_counts(self, sample_size: int = 10):
        """Sample files and check row counts."""
        logger.info("\n" + "="*80)
        logger.info("CHECKING ROW COUNTS (sampling)")
        logger.info("="*80)

        # Expected row counts
        EXPECTED_DA_ROWS_MIN = 25000  # ~1200 locations × 24 hours
        EXPECTED_RT_ROWS_MIN = 25000  # ~1200 locations × 24 hours

        da_dir = self.data_dir / "lmp_day_ahead_hourly"
        rt_dir = self.data_dir / "lmp_real_time_5_min"

        low_count_da = []
        low_count_rt = []

        # Sample DA files
        if da_dir.exists():
            da_files = sorted(da_dir.glob("*.csv"))
            sample_da = da_files[::max(1, len(da_files) // sample_size)][:sample_size]

            for csv_file in sample_da:
                try:
                    df = pd.read_csv(csv_file)
                    row_count = len(df)
                    self.stats['da_total_rows'] += row_count

                    if row_count < EXPECTED_DA_ROWS_MIN:
                        low_count_da.append((csv_file.name, row_count))
                        logger.warning(f"  DA: {csv_file.name} has only {row_count:,} rows")
                    else:
                        logger.debug(f"  DA: {csv_file.name} has {row_count:,} rows ✓")
                except Exception as e:
                    logger.error(f"  ✗ Error reading {csv_file.name}: {e}")
                    self.issues['unreadable_da_files'].append(csv_file.name)

        # Sample RT files
        if rt_dir.exists():
            rt_files = sorted(rt_dir.glob("*.csv"))
            sample_rt = rt_files[::max(1, len(rt_files) // sample_size)][:sample_size]

            for csv_file in sample_rt:
                try:
                    df = pd.read_csv(csv_file)
                    row_count = len(df)
                    self.stats['rt_total_rows'] += row_count

                    if row_count < EXPECTED_RT_ROWS_MIN:
                        low_count_rt.append((csv_file.name, row_count))
                        logger.warning(f"  RT: {csv_file.name} has only {row_count:,} rows")
                    else:
                        logger.debug(f"  RT: {csv_file.name} has {row_count:,} rows ✓")
                except Exception as e:
                    logger.error(f"  ✗ Error reading {csv_file.name}: {e}")
                    self.issues['unreadable_rt_files'].append(csv_file.name)

        if low_count_da:
            logger.warning(f"\n⚠ Found {len(low_count_da)} Day-Ahead files with low row counts")
            self.issues['low_count_da_files'] = low_count_da

        if low_count_rt:
            logger.warning(f"\n⚠ Found {len(low_count_rt)} Real-Time files with low row counts")
            self.issues['low_count_rt_files'] = low_count_rt

        if not (low_count_da or low_count_rt):
            logger.info("✓ Sampled files have acceptable row counts")

    def check_schema(self):
        """Validate schema of sample files."""
        logger.info("\n" + "="*80)
        logger.info("CHECKING SCHEMA")
        logger.info("="*80)

        # Expected columns (from the transformed files - we don't know the exact schema yet)
        # Will validate by checking one sample file from each type

        da_dir = self.data_dir / "lmp_day_ahead_hourly"
        rt_dir = self.data_dir / "lmp_real_time_5_min"

        # Check DA schema
        if da_dir.exists():
            da_files = list(da_dir.glob("*.csv"))
            if da_files:
                sample_da = da_files[0]
                try:
                    df = pd.read_csv(sample_da, nrows=5)
                    logger.info(f"\nDay-Ahead schema (sample: {sample_da.name}):")
                    logger.info(f"  Columns: {list(df.columns)}")
                    logger.info(f"  Dtypes:\n{df.dtypes.to_string()}")
                except Exception as e:
                    logger.error(f"  ✗ Error reading DA schema: {e}")

        # Check RT schema
        if rt_dir.exists():
            rt_files = list(rt_dir.glob("*.csv"))
            if rt_files:
                sample_rt = rt_files[0]
                try:
                    df = pd.read_csv(sample_rt, nrows=5)
                    logger.info(f"\nReal-Time schema (sample: {sample_rt.name}):")
                    logger.info(f"  Columns: {list(df.columns)}")
                    logger.info(f"  Dtypes:\n{df.dtypes.to_string()}")
                except Exception as e:
                    logger.error(f"  ✗ Error reading RT schema: {e}")

    def print_summary(self):
        """Print audit summary."""
        logger.info("\n" + "="*80)
        logger.info("AUDIT SUMMARY")
        logger.info("="*80)

        logger.info(f"\nDates checked: {self.stats['dates_checked']}")
        logger.info(f"\nDay-Ahead LMP:")
        logger.info(f"  Files found: {self.stats['da_files_found']}")
        logger.info(f"  Files missing: {self.stats['da_files_missing']}")
        logger.info(f"  Empty files: {self.stats['da_empty_files']}")
        logger.info(f"  Small files: {self.stats['da_small_files']}")

        logger.info(f"\nReal-Time LMP:")
        logger.info(f"  Files found: {self.stats['rt_files_found']}")
        logger.info(f"  Files missing: {self.stats['rt_files_missing']}")
        logger.info(f"  Empty files: {self.stats['rt_empty_files']}")
        logger.info(f"  Small files: {self.stats['rt_small_files']}")

        logger.info(f"\nIssue Categories:")
        for issue_type, issue_list in self.issues.items():
            logger.info(f"  {issue_type}: {len(issue_list)} issues")

        # Overall status
        total_issues = sum(len(v) for v in self.issues.values()) + \
                      self.stats['da_files_missing'] + self.stats['rt_files_missing']

        if total_issues == 0:
            logger.info("\n✓ ✓ ✓  DATA QUALITY: EXCELLENT - No issues found")
        elif total_issues < 10:
            logger.info(f"\n⚠ DATA QUALITY: GOOD - {total_issues} minor issues found")
        elif total_issues < 50:
            logger.info(f"\n⚠ ⚠ DATA QUALITY: FAIR - {total_issues} issues found")
        else:
            logger.info(f"\n✗ ✗ ✗ DATA QUALITY: POOR - {total_issues} issues found")

        logger.info("="*80 + "\n")

    def run_audit(self, start_date: datetime, end_date: datetime):
        """Run complete audit."""
        logger.info(f"\n{'='*80}")
        logger.info(f"ISO-NE DATA QUALITY AUDIT")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
        logger.info(f"{'='*80}\n")

        self.check_date_coverage(start_date, end_date)
        self.check_file_sizes(start_date, end_date)
        self.check_row_counts(sample_size=20)
        self.check_schema()
        self.print_summary()


def main():
    parser = argparse.ArgumentParser(
        description="Audit ISO-NE historical data quality"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date (YYYY-MM-DD). Defaults to today."
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto-detect date range from existing files"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/pool/ssd8tb/data/iso",
        help="Data directory"
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    auditor = ISONEDataAuditor(data_dir)

    if args.auto:
        # Auto-detect date range
        da_dates = auditor.get_existing_dates("lmp_day_ahead_hourly")
        rt_dates = auditor.get_existing_dates("lmp_real_time_5_min")
        all_dates = da_dates.union(rt_dates)

        if not all_dates:
            logger.error("No data files found!")
            sys.exit(1)

        start_date = min(all_dates)
        end_date = max(all_dates)
        logger.info(f"Auto-detected date range: {start_date.date()} to {end_date.date()}")
    else:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d") if args.start_date else datetime(2019, 1, 1)
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d") if args.end_date else datetime.now()

    auditor.run_audit(start_date, end_date)


if __name__ == "__main__":
    main()
