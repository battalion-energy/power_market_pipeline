import pandas as pd
import pyarrow.parquet as pq

file_path = "/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/rollup_files/DAM_Gen_Resources/2024.parquet"

table = pq.read_table(file_path, columns=['DeliveryDate'])
df = table.to_pandas()
df['DeliveryDate'] = pd.to_datetime(df['DeliveryDate'])

print(f"2024.parquet:")
print(f"  Rows: {len(df):,}")
print(f"  Min date: {df['DeliveryDate'].min()}")
print(f"  Max date: {df['DeliveryDate'].max()}")
print(f"  Has NaT: {df['DeliveryDate'].isna().sum()} rows")
