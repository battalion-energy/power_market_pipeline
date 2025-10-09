import pandas as pd
import pyarrow.parquet as pq

for year in [2023, 2024, 2025]:
    file_path = f"/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/rollup_files/SCED_Gen_Resources/{year}.parquet"
    
    try:
        # Read just the timestamp column (note capital S in SCEDTimeStamp)
        table = pq.read_table(file_path, columns=['SCEDTimeStamp'])
        df = table.to_pandas()
        
        # Convert to datetime if not already
        df['SCEDTimeStamp'] = pd.to_datetime(df['SCEDTimeStamp'])
        
        print(f"\n{year}.parquet:")
        print(f"  Rows: {len(df):,}")
        print(f"  Min date: {df['SCEDTimeStamp'].min()}")
        print(f"  Max date: {df['SCEDTimeStamp'].max()}")
        
    except Exception as e:
        print(f"\n{year}.parquet: ERROR - {e}")
