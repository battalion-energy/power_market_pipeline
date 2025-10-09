import pandas as pd
import pyarrow.parquet as pq

file_path = "/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/rollup_files/SCED_Gen_Resources/2025.parquet"

table = pq.read_table(file_path, columns=['SCEDTimeStamp'])
df = table.to_pandas()
df['SCEDTimeStamp'] = pd.to_datetime(df['SCEDTimeStamp'])

print(f"2025.parquet:")
print(f"  Rows: {len(df):,}")
print(f"  Min date: {df['SCEDTimeStamp'].min()}")
print(f"  Max date: {df['SCEDTimeStamp'].max()}")
print(f"  Has NaT: {df['SCEDTimeStamp'].isna().sum()} rows")
