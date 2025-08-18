#!/usr/bin/env python3
"""
Test how Polars handles dates in Parquet files.
"""

import polars as pl
from datetime import date

# Create a test DataFrame with dates
df = pl.DataFrame({
    'date_string': ['01/15/2023', '06/30/2023', '12/31/2023'],
    'value': [100.0, 200.0, 300.0]
})

print("Original DataFrame:")
print(df)
print(f"Schema: {df.schema}")

# Parse the date string to Date type
df = df.with_columns(
    pl.col('date_string').str.strptime(pl.Date, "%m/%d/%Y").alias('parsed_date')
)

print("\nAfter parsing dates:")
print(df)
print(f"Schema: {df.schema}")

# Save to Parquet
df.write_parquet('test_polars_dates.parquet')

# Read back
df_read = pl.read_parquet('test_polars_dates.parquet')
print("\n" + "="*80)
print("After reading from Parquet:")
print(df_read)
print(f"Schema: {df_read.schema}")

# Check what pandas sees
import pandas as pd
df_pandas = pd.read_parquet('test_polars_dates.parquet')
print("\n" + "="*80)
print("Pandas view of the Parquet file:")
print(df_pandas)
print(f"Types: {df_pandas.dtypes.to_dict()}")

# Create dates manually using days since epoch
print("\n" + "="*80)
print("Manual date creation in Polars:")

# Calculate days since epoch for our dates
days_since_epoch = [
    (date(2023, 1, 15) - date(1970, 1, 1)).days,
    (date(2023, 6, 30) - date(1970, 1, 1)).days,
    (date(2023, 12, 31) - date(1970, 1, 1)).days
]

print(f"Days since epoch: {days_since_epoch}")

# Create DataFrame with manual dates
df_manual = pl.DataFrame({
    'days': days_since_epoch,
    'value': [100.0, 200.0, 300.0]
})

# Cast to Date type
df_manual = df_manual.with_columns(
    pl.col('days').cast(pl.Date).alias('date_from_days')
)

print("\nDataFrame with manual dates:")
print(df_manual)
print(f"Schema: {df_manual.schema}")

# Save and verify
df_manual.write_parquet('test_manual_dates.parquet')
df_manual_read = pd.read_parquet('test_manual_dates.parquet')
print("\nPandas view of manual dates:")
print(df_manual_read)

# Clean up
import os
os.remove('test_polars_dates.parquet')
os.remove('test_manual_dates.parquet')