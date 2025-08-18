#!/usr/bin/env python3
"""
Test how dates should be encoded in Parquet files.
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from datetime import datetime, date

# Create a test DataFrame with different date representations
test_data = {
    'date_as_string': ['01/15/2023', '06/30/2023', '12/31/2023'],
    'date_as_datetime': [
        pd.to_datetime('2023-01-15'),
        pd.to_datetime('2023-06-30'), 
        pd.to_datetime('2023-12-31')
    ],
    'date_as_date': [
        date(2023, 1, 15),
        date(2023, 6, 30),
        date(2023, 12, 31)
    ],
    'value': [100.0, 200.0, 300.0]
}

df = pd.DataFrame(test_data)

# Convert string to proper date
df['parsed_date'] = pd.to_datetime(df['date_as_string'], format='%m/%d/%Y').dt.date

print("DataFrame types:")
print(df.dtypes)
print("\nDataFrame:")
print(df)

# Write to Parquet
df.to_parquet('test_dates.parquet')

# Read back and check
df_read = pd.read_parquet('test_dates.parquet')
print("\n" + "="*80)
print("After reading from Parquet:")
print("\nDataFrame types:")
print(df_read.dtypes)
print("\nDataFrame:")
print(df_read)

# Check the Parquet schema
parquet_file = pq.ParquetFile('test_dates.parquet')
print("\n" + "="*80)
print("Parquet Schema:")
print(parquet_file.schema)

# Check how PyArrow handles dates
print("\n" + "="*80)
print("PyArrow Table Schema:")
table = pa.Table.from_pandas(df)
print(table.schema)

# Check the actual values stored
print("\n" + "="*80)
print("Raw values in Arrow Table:")
for field in table.schema:
    col = table[field.name]
    print(f"\n{field.name} ({field.type}):")
    print(f"  Values: {col.to_pylist()}")

# Now test Date32 type specifically
print("\n" + "="*80)
print("Testing Date32 type:")

# Create a table with explicit Date32 type
dates_as_days = [
    (date(2023, 1, 15) - date(1970, 1, 1)).days,
    (date(2023, 6, 30) - date(1970, 1, 1)).days,
    (date(2023, 12, 31) - date(1970, 1, 1)).days
]
print(f"Days since epoch: {dates_as_days}")

# Create Arrow array with Date32 type
date32_array = pa.array(dates_as_days, type=pa.date32())
print(f"Date32 array: {date32_array}")
print(f"As Python dates: {date32_array.to_pylist()}")

# Create table with Date32
schema = pa.schema([
    ('date_field', pa.date32()),
    ('value', pa.float64())
])
table = pa.table({
    'date_field': date32_array,
    'value': [100.0, 200.0, 300.0]
})

# Write and read back
pq.write_table(table, 'test_date32.parquet')
df_date32 = pd.read_parquet('test_date32.parquet')

print("\n" + "="*80)
print("Date32 Parquet file:")
print(f"DataFrame types:\n{df_date32.dtypes}")
print(f"\nDataFrame:\n{df_date32}")

# Clean up
import os
os.remove('test_dates.parquet')
os.remove('test_date32.parquet')