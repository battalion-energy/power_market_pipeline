# BESS Revenue Analysis: Real Data Sources

## Executive Summary
**WE HAVE ALL THE DATA WE NEED.** No need for any mock data or hacks.

## Primary Data Sources (All Available Now)

### 1. BESS Resource Identification
**Source:** `/Users/enrico/data/ERCOT_data/60-Day_DAM_Disclosure_Reports/csv/60d_DAM_Gen_Resource_Data-*.csv`

**Query to Find All BESS:**
```sql
SELECT DISTINCT 
    "Resource Name",
    "QSE",
    "Resource Type"
FROM dam_gen_resource_data
WHERE "Resource Type" = 'PWRSTR'
ORDER BY "Resource Name";
```

**Sample Real BESS Resources Found:**
- ADL_BESS1
- ALP_BESS_BESS1
- ALVIN_UNIT1
- ANCHOR_BESS1
- ANCHOR_BESS2
- ANGLETON_ESU1
- ANSON_BESS1
- APOLLO_BESS1
- ... (hundreds more)

### 2. DAM Energy Awards
**Source:** Same DAM disclosure files

**Fields We Need:**
- `Awarded Quantity` - MW awarded in DAM
- `Energy Settlement Point Price` - $/MWh
- `Hour Ending` - Which hour
- `Delivery Date` - Which day

**Revenue Calculation:**
```
DAM Energy Revenue = Σ(Awarded_MW[h] × DAM_Price[h])
```

### 3. Ancillary Service Awards
**Source:** Same DAM disclosure files

**AS Products & Fields:**
- `RegUp Awarded` - Regulation Up (MW)
- `RegDown Awarded` - Regulation Down (MW)
- `RRS Awarded` or `RRSFFR/RRSPFR/RRSUFR Awarded` - Responsive Reserve
- `ECRS Awarded` - ERCOT Contingency Reserve
- `NonSpin Awarded` - Non-Spinning Reserve

**AS Clearing Prices:**
**Source:** `/Users/enrico/data/ERCOT_data/DAM_Clearing_Prices_for_Capacity/`
- RegUp MCPC - $/MW for RegUp capacity
- RegDown MCPC - $/MW for RegDown capacity
- RRS MCPC - $/MW for RRS
- ECRS MCPC - $/MW for ECRS
- NonSpin MCPC - $/MW for NonSpin

### 4. Real-Time Dispatch
**Source:** `/Users/enrico/data/ERCOT_data/60-Day_SCED_Disclosure_Reports/csv/60d_SCED_Gen_Resource_Data-*.csv`

**Critical Fields:**
- `Base Point` - MW dispatch instruction
- `Telemetered Net Output` - Actual MW output
- `SCED Time Stamp` - 5-minute interval timestamp
- `Resource Name` - Which BESS

**RT Revenue Calculation:**
```
RT Revenue = Σ((Actual_MW - DAM_MW) × RT_Price) for all 5-min intervals
```

### 5. State of Charge
**Source:** `/Users/enrico/data/ERCOT_data/60-Day_COP_Adjustment_Period_Snapshot/`

**Fields:**
- `Hour Beginning Planned SOC` - MWh stored at hour start
- `Minimum SOC` - Min allowed MWh
- `Maximum SOC` - Max allowed MWh (this is capacity!)

### 6. Settlement Point Prices
**Source:** Already processed into Parquet files

**DAM Prices:**
- `/Users/enrico/data/ERCOT_data/rollup_files/DA_prices/*.parquet`
- Hourly prices by settlement point

**RT Prices:**
- `/Users/enrico/data/ERCOT_data/rollup_files/RT_prices/*.parquet`
- 15-minute prices by settlement point (pre-2024)
- 15-minute prices real-time (post-Aug 2024)

## Key Insights About the Data

### Resource Type Codes
- `PWRSTR` = Power Storage (BESS)
- `WIND` = Wind Generation
- `SOLAR` = Solar Generation
- `CC` = Combined Cycle
- `CT` = Combustion Turbine

### Settlement Point Mapping
BESS resources map to settlement points via:
1. `Settlement Point Name` field in Gen Resource Data
2. QSE registration data
3. Network model files

### Capacity Determination
**Max Discharge Capacity:** HSL field (High Sustained Limit)
**Max Charge Capacity:** |LSL| field (Low Sustained Limit, negative for charging)
**Storage Capacity:** From COP snapshots `Maximum SOC` field

### Efficiency Calculation
Can be derived from actual operations:
```
Efficiency = Energy_Discharged / Energy_Charged
```
Over a complete cycle where SOC returns to initial state.

## Example Data Extraction

### Get BESS List for Specific Date
```python
import pandas as pd

# Read DAM disclosure for specific date
df = pd.read_csv('60d_DAM_Gen_Resource_Data-31-JAN-25.csv')

# Filter for BESS only
bess_df = df[df['Resource Type'] == 'PWRSTR']

# Get unique BESS resources
bess_list = bess_df['Resource Name'].unique()
print(f"Found {len(bess_list)} BESS resources")

# Get capacity info
bess_capacity = bess_df.groupby('Resource Name').agg({
    'HSL': 'max',  # Max discharge
    'LSL': 'min'   # Max charge (negative)
}).reset_index()
```

### Calculate Daily Revenue
```python
def calculate_daily_revenue(resource_name, date):
    # Load awards
    dam_df = pd.read_csv(f'60d_DAM_Gen_Resource_Data-{date}.csv')
    resource_dam = dam_df[dam_df['Resource Name'] == resource_name]
    
    # Load prices
    prices_df = pd.read_parquet(f'DA_prices/{date.year}.parquet')
    
    # Calculate revenue
    revenue = 0
    for _, row in resource_dam.iterrows():
        hour = row['Hour Ending']
        mw = row['Awarded Quantity']
        price = prices_df[
            (prices_df['SettlementPoint'] == row['Settlement Point Name']) &
            (prices_df['HourEnding'] == hour)
        ]['Price'].values[0]
        
        revenue += mw * price
    
    return revenue
```

## What We DON'T Need Mock Data For

### Things We Can Calculate From Real Data:
1. ✅ Actual BESS fleet (from PWRSTR resources)
2. ✅ Real capacities (from HSL/LSL)
3. ✅ Actual awards (from disclosure files)
4. ✅ Real prices (from price files)
5. ✅ True dispatch (from SCED files)
6. ✅ AS performance (from deployment data)
7. ✅ State of charge (from COP snapshots)

### Things We Should NOT Make Up:
1. ❌ Arbitrary price thresholds
2. ❌ Fake AS awards
3. ❌ Assumed efficiencies without data
4. ❌ Made-up BESS resources
5. ❌ Fictional revenue guarantees

## Next Steps

1. **Extract Complete BESS Fleet List**
```bash
grep "PWRSTR" 60d_DAM_Gen_Resource_Data-*.csv | \
  cut -d',' -f5 | sort -u > all_bess_resources.txt
```

2. **Build Resource Registry Database**
```sql
CREATE TABLE bess_registry (
    resource_name VARCHAR PRIMARY KEY,
    qse VARCHAR,
    settlement_point VARCHAR,
    max_discharge_mw FLOAT,
    max_charge_mw FLOAT,
    storage_capacity_mwh FLOAT,
    first_seen_date DATE,
    last_seen_date DATE
);
```

3. **Start Processing Real Awards**
- Parse DAM awards by date and resource
- Match with settlement point prices
- Calculate actual revenues

## The Truth

**We have 28,775 disclosure CSV files with real BESS operations.**

There is absolutely no excuse for using mock data. Every number we need is in these files. The only challenge is parsing them correctly and implementing proper algorithms.

Stop making things up. Start using real data.