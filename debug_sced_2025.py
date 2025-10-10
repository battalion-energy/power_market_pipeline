#!/usr/bin/env python3
"""
Debug SCED 2025 processing to identify problematic files.
Tests each file individually and checks for schema issues.
"""

import pandas as pd
from pathlib import Path
import sys

# Configuration
csv_dir = Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/60-Day_SCED_Disclosure_Reports/csv")
pattern = "60d_SCED_Gen_Resource_Data-*-25.csv"

# Find all 2025 files
files = sorted(csv_dir.glob(pattern))
print(f"Found {len(files)} SCED 2025 CSV files\n")

failed_files = []
successful_files = []
all_columns = set()
file_schemas = {}

for idx, file in enumerate(files, 1):
    try:
        print(f"[{idx}/{len(files)}] Reading: {file.name}...", end=" ", flush=True)

        # Try to read the file
        df = pd.read_csv(file, low_memory=False)

        rows = len(df)
        cols = list(df.columns)
        col_count = len(cols)

        # Track all unique columns
        all_columns.update(cols)
        file_schemas[file.name] = cols

        print(f"✅ {rows:,} rows, {col_count} cols")
        successful_files.append(file.name)

    except Exception as e:
        print(f"❌ ERROR: {e}")
        failed_files.append((file.name, str(e)))

print("\n" + "="*80)
print(f"Summary:")
print(f"  ✅ Successful: {len(successful_files)}")
print(f"  ❌ Failed: {len(failed_files)}")
print(f"  Total unique columns across all files: {len(all_columns)}")

if failed_files:
    print("\nFailed files:")
    for filename, error in failed_files:
        print(f"  - {filename}: {error}")
    sys.exit(1)

# Check for schema variations
print("\nChecking for schema variations...")
column_counts = {}
for filename, cols in file_schemas.items():
    col_count = len(cols)
    if col_count not in column_counts:
        column_counts[col_count] = []
    column_counts[col_count].append(filename)

print(f"Files by column count:")
for col_count in sorted(column_counts.keys()):
    files_with_count = column_counts[col_count]
    print(f"  {col_count} columns: {len(files_with_count)} files")
    if len(files_with_count) <= 3:
        for f in files_with_count:
            print(f"    - {f}")

# Find column differences
if len(column_counts) > 1:
    print("\n⚠️  WARNING: Files have different schemas!")
    # Compare first file of each schema type
    schema_samples = {}
    for col_count, files in column_counts.items():
        schema_samples[col_count] = set(file_schemas[files[0]])

    all_schema_cols = set()
    for cols in schema_samples.values():
        all_schema_cols.update(cols)

    print(f"\nColumns that don't appear in all schemas:")
    for col in sorted(all_schema_cols):
        appears_in = [count for count, cols in schema_samples.items() if col in cols]
        if len(appears_in) != len(schema_samples):
            print(f"  - {col}: in schemas with {appears_in} columns")

print("\n" + "="*80)
print("✅ All files readable - schema analysis complete")
