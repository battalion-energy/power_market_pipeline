#!/usr/bin/env python3
"""
Quality Control Script for Parquet Files
- Check file integrity
- Verify schemas across years
- Check for missing columns
- Validate data completeness
- Report statistics
"""

import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from collections import defaultdict
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

PARQUET_DIR = Path("/Users/enrico/data/ERCOT_data/forecast_parquet_full_files")

def analyze_parquet_file(parquet_path):
    """Analyze a single parquet file and return metadata."""
    try:
        # Read parquet metadata
        parquet_file = pq.ParquetFile(parquet_path)
        metadata = parquet_file.metadata
        schema = parquet_file.schema

        # Read actual data
        df = pd.read_parquet(parquet_path)

        info = {
            'file': parquet_path.name,
            'num_rows': len(df),
            'num_columns': len(df.columns),
            'columns': list(df.columns),
            'file_size_mb': parquet_path.stat().st_size / (1024 * 1024),
            'null_counts': df.isnull().sum().to_dict(),
            'dtypes': df.dtypes.to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
            'sample_rows': df.head(3).to_dict('records') if len(df) > 0 else []
        }

        return info, None

    except Exception as e:
        return None, str(e)


def group_files_by_source(parquet_files):
    """Group parquet files by data source."""
    sources = defaultdict(list)

    for f in parquet_files:
        # Extract source name (everything before the year)
        parts = f.stem.rsplit('_', 1)
        if len(parts) == 2:
            source_name = parts[0]
            year = parts[1]
            sources[source_name].append((year, f))

    return sources


def check_schema_consistency(files_info):
    """Check if schemas are consistent across years for same source."""
    issues = []

    # Group by source
    all_columns = defaultdict(set)

    for info in files_info:
        source_name = info['file'].rsplit('_', 1)[0]
        all_columns[source_name].update(info['columns'])

    # Check if all files for same source have same columns
    for source, files in files_info.items():
        if isinstance(files, list):
            column_sets = [set(f['columns']) for f in files]
            if len(column_sets) > 1:
                # Check if all sets are the same
                first_set = column_sets[0]
                for i, col_set in enumerate(column_sets[1:], 1):
                    missing = first_set - col_set
                    extra = col_set - first_set
                    if missing or extra:
                        issues.append({
                            'source': source,
                            'file': files[i]['file'],
                            'missing_columns': list(missing),
                            'extra_columns': list(extra)
                        })

    return issues


def main():
    """Main QC function."""
    logging.info("=" * 80)
    logging.info("PARQUET FILES QUALITY CONTROL")
    logging.info("=" * 80)

    # Find all parquet files
    parquet_files = sorted(PARQUET_DIR.glob("*.parquet"))
    logging.info(f"Found {len(parquet_files)} parquet files\n")

    # Analyze each file
    all_info = []
    errors = []

    logging.info("Analyzing files...")
    for pf in parquet_files:
        info, error = analyze_parquet_file(pf)
        if error:
            errors.append({'file': pf.name, 'error': error})
            logging.error(f"✗ {pf.name}: {error}")
        else:
            all_info.append(info)
            logging.info(f"✓ {pf.name}: {info['num_rows']:,} rows, {info['num_columns']} cols, {info['file_size_mb']:.1f} MB")

    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    # Group by source
    sources = group_files_by_source(parquet_files)

    for source_name, year_files in sorted(sources.items()):
        print(f"\n{source_name}:")
        print(f"  Years: {len(year_files)}")

        # Get info for this source
        source_info = [info for info in all_info if info['file'].startswith(source_name)]

        if source_info:
            total_rows = sum(info['num_rows'] for info in source_info)
            total_size = sum(info['file_size_mb'] for info in source_info)

            print(f"  Total rows: {total_rows:,}")
            print(f"  Total size: {total_size:.1f} MB")
            print(f"  Columns: {source_info[0]['num_columns']}")

            # Check schema consistency within this source
            column_sets = [set(info['columns']) for info in source_info]
            if len(set(frozenset(cs) for cs in column_sets)) > 1:
                print(f"  ⚠️  WARNING: Inconsistent schemas across years!")
                # Show differences
                all_cols = set()
                for cs in column_sets:
                    all_cols.update(cs)

                for info in source_info:
                    missing = all_cols - set(info['columns'])
                    if missing:
                        year = info['file'].rsplit('_', 1)[1].replace('.parquet', '')
                        print(f"    Year {year} missing: {missing}")
            else:
                print(f"  ✓ Schema consistent across all years")
                print(f"  Column names: {', '.join(source_info[0]['columns'][:5])}...")

    # Check for null values
    print("\n" + "=" * 80)
    print("NULL VALUE CHECK")
    print("=" * 80)

    for info in all_info:
        null_cols = {col: count for col, count in info['null_counts'].items() if count > 0}
        if null_cols:
            print(f"\n{info['file']}:")
            for col, count in sorted(null_cols.items(), key=lambda x: x[1], reverse=True)[:5]:
                pct = (count / info['num_rows'] * 100) if info['num_rows'] > 0 else 0
                print(f"  {col}: {count:,} nulls ({pct:.1f}%)")

    # Sample data from a few key files
    print("\n" + "=" * 80)
    print("SAMPLE DATA (First 3 rows from key files)")
    print("=" * 80)

    key_sources = [
        'Settlement_Point_Prices_at_Resource_Nodes_Hubs_and_Load_Zones_2025',
        'DAM_Settlement_Point_Prices_2025',
        'Wind_Power_Production_-_Actual_5-Minute_Averaged_Values_2025'
    ]

    for source in key_sources:
        matching = [info for info in all_info if source in info['file']]
        if matching:
            info = matching[0]
            print(f"\n{info['file']}:")
            print(f"  Columns: {info['columns']}")
            if info['sample_rows']:
                print(f"  Sample row 1: {info['sample_rows'][0]}")

    # Overall summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)

    total_rows = sum(info['num_rows'] for info in all_info)
    total_size = sum(info['file_size_mb'] for info in all_info)
    total_memory = sum(info['memory_usage_mb'] for info in all_info)

    print(f"Total files: {len(all_info)}")
    print(f"Total rows: {total_rows:,}")
    print(f"Total disk size: {total_size:.1f} MB")
    print(f"Total memory if loaded: {total_memory:.1f} MB")
    print(f"Unique data sources: {len(sources)}")

    if errors:
        print(f"\n⚠️  Errors: {len(errors)}")
        for err in errors:
            print(f"  - {err['file']}: {err['error']}")
    else:
        print("\n✓ No errors found!")

    print("\n" + "=" * 80)
    print("QUALITY CONTROL COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
