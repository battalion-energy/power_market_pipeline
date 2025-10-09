import pyarrow.parquet as pq

file_path = "/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/rollup_files/DAM_Gen_Resources/2025.parquet"

# Read schema
schema = pq.read_schema(file_path)
print(f"Schema for DAM Gen Resources 2025.parquet:\n")
for field in schema:
    print(f"  {field.name}: {field.type}")

# Check a sample of data
table = pq.read_table(file_path)
df = table.to_pandas()
print(f"\n{len(df):,} total rows")
print(f"\nFirst row sample:")
print(df.iloc[0])
