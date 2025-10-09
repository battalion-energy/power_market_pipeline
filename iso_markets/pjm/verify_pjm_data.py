#!/usr/bin/env python3
"""
Verify PJM Data Completeness

Checks for gaps in downloaded PJM data:
- Missing dates
- Missing hours within dates
- Unexpected row counts
- Data quality issues

Usage:
    python verify_pjm_data.py
    python verify_pjm_data.py --market da_hubs
    python verify_pjm_data.py --market da_nodal --verbose
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def verify_hub_data(data_dir: Path, market_type: str, start_date: str, end_date: str, verbose: bool = False):
    """Verify hub-level data for completeness."""

    logger.info(f"\n{'='*70}")
    logger.info(f"Verifying {market_type}")
    logger.info(f"{'='*70}")

    csv_dir = data_dir / 'csv_files' / market_type
    if not csv_dir.exists():
        logger.error(f"Directory not found: {csv_dir}")
        return

    # Get all CSV files
    csv_files = sorted(csv_dir.glob('*.csv'))
    if not csv_files:
        logger.warning(f"No CSV files found in {csv_dir}")
        return

    logger.info(f"Found {len(csv_files)} CSV files")

    # Parse dates from filenames and load data
    date_coverage = defaultdict(lambda: {'files': [], 'hours': set(), 'rows': 0})

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)

            # Find datetime column
            datetime_col = None
            for col in df.columns:
                if 'datetime_beginning' in col.lower():
                    datetime_col = col
                    break

            if datetime_col:
                df[datetime_col] = pd.to_datetime(df[datetime_col])

                for dt in df[datetime_col].unique():
                    date_str = dt.strftime('%Y-%m-%d')
                    hour = dt.hour
                    date_coverage[date_str]['files'].append(csv_file.name)
                    date_coverage[date_str]['hours'].add(hour)
                    date_coverage[date_str]['rows'] += len(df[df[datetime_col] == dt])

        except Exception as e:
            logger.error(f"Error processing {csv_file.name}: {e}")

    # Check for missing dates
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    current = start

    missing_dates = []
    incomplete_dates = []

    while current <= end:
        date_str = current.strftime('%Y-%m-%d')

        if date_str not in date_coverage:
            missing_dates.append(date_str)
        else:
            hours = date_coverage[date_str]['hours']
            expected_hours = 24

            if len(hours) < expected_hours:
                missing_hours = sorted(set(range(24)) - hours)
                incomplete_dates.append({
                    'date': date_str,
                    'hours_found': len(hours),
                    'missing_hours': missing_hours,
                    'files': date_coverage[date_str]['files']
                })

        current += timedelta(days=1)

    # Report results
    total_days = (end - start).days + 1
    complete_days = total_days - len(missing_dates) - len(incomplete_dates)

    logger.info(f"\n📊 Summary:")
    logger.info(f"  Total expected days: {total_days}")
    logger.info(f"  Complete days: {complete_days}")
    logger.info(f"  Incomplete days: {len(incomplete_dates)}")
    logger.info(f"  Missing days: {len(missing_dates)}")
    logger.info(f"  Coverage: {complete_days/total_days*100:.1f}%")

    if missing_dates:
        logger.warning(f"\n⚠️  Missing dates ({len(missing_dates)} days):")
        for i, date in enumerate(missing_dates[:10]):
            logger.warning(f"  {date}")
        if len(missing_dates) > 10:
            logger.warning(f"  ... and {len(missing_dates) - 10} more")

    if incomplete_dates:
        logger.warning(f"\n⚠️  Incomplete dates ({len(incomplete_dates)} days):")
        for i, info in enumerate(incomplete_dates[:10]):
            logger.warning(f"  {info['date']}: {info['hours_found']}/24 hours (missing hours: {info['missing_hours'][:5]}{'...' if len(info['missing_hours']) > 5 else ''})")
            if verbose:
                logger.warning(f"    Files: {', '.join(info['files'])}")
        if len(incomplete_dates) > 10:
            logger.warning(f"  ... and {len(incomplete_dates) - 10} more")

    if not missing_dates and not incomplete_dates:
        logger.info("✅ No gaps found - data is complete!")

    return {
        'total_days': total_days,
        'complete_days': complete_days,
        'incomplete_days': len(incomplete_dates),
        'missing_days': len(missing_dates),
        'coverage_pct': complete_days/total_days*100
    }


def verify_nodal_data(data_dir: Path, start_date: str, end_date: str, verbose: bool = False):
    """Verify nodal data (day-by-day files) for completeness."""

    logger.info(f"\n{'='*70}")
    logger.info(f"Verifying da_nodal (Day-by-day files)")
    logger.info(f"{'='*70}")

    csv_dir = data_dir / 'csv_files' / 'da_nodal'
    if not csv_dir.exists():
        logger.error(f"Directory not found: {csv_dir}")
        return

    # Get all daily files
    csv_files = sorted(csv_dir.glob('nodal_da_lmp_*.csv'))

    logger.info(f"Found {len(csv_files)} daily files")

    # Parse dates from filenames
    date_coverage = {}

    for csv_file in csv_files:
        try:
            # Extract date from filename: nodal_da_lmp_YYYY-MM-DD.csv
            date_str = csv_file.stem.split('_')[-1]

            # Load file to check hours and rows
            df = pd.read_csv(csv_file)

            # Find datetime column
            datetime_col = None
            for col in df.columns:
                if 'datetime_beginning' in col.lower():
                    datetime_col = col
                    break

            if datetime_col:
                df[datetime_col] = pd.to_datetime(df[datetime_col])
                hours = sorted(df[datetime_col].dt.hour.unique())
                unique_nodes = df['pnode_id'].nunique() if 'pnode_id' in df.columns else 0

                date_coverage[date_str] = {
                    'file': csv_file.name,
                    'rows': len(df),
                    'hours': hours,
                    'hours_count': len(hours),
                    'nodes': unique_nodes
                }

        except Exception as e:
            logger.error(f"Error processing {csv_file.name}: {e}")

    # Check for missing dates
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    current = start

    missing_dates = []
    incomplete_dates = []

    while current <= end:
        date_str = current.strftime('%Y-%m-%d')

        if date_str not in date_coverage:
            missing_dates.append(date_str)
        else:
            info = date_coverage[date_str]
            expected_hours = 24

            if info['hours_count'] < expected_hours:
                missing_hours = sorted(set(range(24)) - set(info['hours']))
                incomplete_dates.append({
                    'date': date_str,
                    'hours_found': info['hours_count'],
                    'missing_hours': missing_hours,
                    'rows': info['rows'],
                    'nodes': info['nodes'],
                    'file': info['file']
                })

        current += timedelta(days=1)

    # Report results
    total_days = (end - start).days + 1
    complete_days = total_days - len(missing_dates) - len(incomplete_dates)

    logger.info(f"\n📊 Summary:")
    logger.info(f"  Total expected days: {total_days}")
    logger.info(f"  Complete days: {complete_days}")
    logger.info(f"  Incomplete days: {len(incomplete_dates)}")
    logger.info(f"  Missing days: {len(missing_dates)}")
    logger.info(f"  Coverage: {complete_days/total_days*100:.1f}%")

    # Sample stats
    if date_coverage:
        sample_date = list(date_coverage.keys())[0]
        sample = date_coverage[sample_date]
        logger.info(f"\n📈 Sample data (from {sample_date}):")
        logger.info(f"  Rows: {sample['rows']:,}")
        logger.info(f"  Unique nodes: {sample['nodes']:,}")
        logger.info(f"  Hours covered: {sample['hours_count']}/24")

    if missing_dates:
        logger.warning(f"\n⚠️  Missing dates ({len(missing_dates)} days):")
        for i, date in enumerate(missing_dates[:10]):
            logger.warning(f"  {date}")
        if len(missing_dates) > 10:
            logger.warning(f"  ... and {len(missing_dates) - 10} more")

    if incomplete_dates:
        logger.warning(f"\n⚠️  Incomplete dates ({len(incomplete_dates)} days):")
        for i, info in enumerate(incomplete_dates[:10]):
            logger.warning(f"  {info['date']}: {info['hours_found']}/24 hours, {info['rows']:,} rows, {info['nodes']:,} nodes")
            logger.warning(f"    Missing hours: {info['missing_hours']}")
            if verbose:
                logger.warning(f"    File: {info['file']}")
        if len(incomplete_dates) > 10:
            logger.warning(f"  ... and {len(incomplete_dates) - 10} more")

    if not missing_dates and not incomplete_dates:
        logger.info("✅ No gaps found - nodal data is complete!")

    return {
        'total_days': total_days,
        'complete_days': complete_days,
        'incomplete_days': len(incomplete_dates),
        'missing_days': len(missing_dates),
        'coverage_pct': complete_days/total_days*100
    }


def main():
    parser = argparse.ArgumentParser(
        description='Verify PJM data completeness'
    )
    parser.add_argument('--market', type=str,
                       choices=['da_hubs', 'rt_hourly', 'da_ancillary_services', 'da_nodal', 'all'],
                       default='all',
                       help='Market type to verify')
    parser.add_argument('--start-date', type=str, default='2023-10-07',
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2025-10-06',
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')

    args = parser.parse_args()

    data_dir = Path(os.getenv('PJM_DATA_DIR', '/home/enrico/data/PJM_data'))

    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return

    logger.info(f"PJM Data Verification")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Date range: {args.start_date} to {args.end_date}")

    results = {}

    # Verify requested markets
    if args.market == 'all' or args.market == 'da_hubs':
        results['da_hubs'] = verify_hub_data(data_dir, 'da_hubs', args.start_date, args.end_date, args.verbose)

    if args.market == 'all' or args.market == 'rt_hourly':
        results['rt_hourly'] = verify_hub_data(data_dir, 'rt_hourly', args.start_date, args.end_date, args.verbose)

    if args.market == 'all' or args.market == 'da_ancillary_services':
        results['da_ancillary_services'] = verify_hub_data(data_dir, 'da_ancillary_services', args.start_date, args.end_date, args.verbose)

    if args.market == 'all' or args.market == 'da_nodal':
        results['da_nodal'] = verify_nodal_data(data_dir, args.start_date, args.end_date, args.verbose)

    # Overall summary
    logger.info(f"\n{'='*70}")
    logger.info("Overall Summary")
    logger.info(f"{'='*70}")

    for market, stats in results.items():
        if stats:
            logger.info(f"{market:30s}: {stats['complete_days']:4d}/{stats['total_days']:4d} days ({stats['coverage_pct']:5.1f}%)")

    logger.info(f"\n✓ Verification complete!")


if __name__ == "__main__":
    main()
