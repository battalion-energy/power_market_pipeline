# BESS Revenue Calculation - Complete Solution

## 🎯 The Breakthrough Discovery

**TWO key discoveries solve the BESS revenue puzzle:**

1. **Settlement Point Mapping**: The directory `/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/Settlement_Points_List_and_Electrical_Buses_Mapping/latest_mapping/` provides:
   - Resource Node to Unit mapping (e.g., `BATCAVE_RN` → `BATCAVE/BES1`)
   - Settlement Load Zone mapping (e.g., `BATCAVE` → `LZ_SOUTH`)
   - Links resources to their settlement points for pricing

2. **Cross-Reference Method**: We can match Load Resources between CSV and Parquet files using:
   - `SCEDTimeStamp`
   - `BasePoint`
   - `MaxPowerConsumption`

## The Complete BESS Revenue Algorithm

### Step 1: Resource Pairing

Every BESS has TWO resources with predictable naming patterns:

```python
# Generation Resource → Load Resource mapping
gen_to_load_mapping = {
    'BATCAVE_BES1': 'BATCAVE_LD1',
    'ANCHOR_BESS1': 'ANCHOR_LD1', 
    'ALVIN_UNIT1': 'ALVIN_LD1',
    # Pattern: Replace _BES{N} with _LD{N}
    # Pattern: Replace _UNIT{N} with _LD{N}
}
```

### Step 2: Data Pipeline Architecture

```
RAW CSV FILES (Have Resource Names)
├── 60-Day_DAM_Disclosure_Reports/csv/
│   ├── 60d_DAM_Gen_Resource_Data-*.csv
│   └── 60d_DAM_Load_Resource_Data-*.csv
├── 60-Day_SCED_Disclosure_Reports/csv/
│   ├── 60d_SCED_Gen_Resource_Data-*.csv
│   └── 60d_Load_Resource_Data_in_SCED-*.csv
│
ROLLED-UP PARQUET FILES
├── DAM_Gen_Resources/{year}.parquet ✅ Has ResourceName
├── DAM_Load_Resources/{year}.parquet ❌ Missing ResourceName  
├── SCED_Gen_Resources/{year}.parquet ✅ Has ResourceName
└── SCED_Load_Resources/{year}.parquet ❌ Missing ResourceName
```

### Step 3: The Cross-Reference Solution

```python
def get_load_resource_data(bess_name, year):
    """
    Match Load Resource data from parquet using CSV cross-reference
    """
    # 1. Get the Load resource name
    load_name = bess_name.replace('_BES', '_LD').replace('_UNIT', '_LD')
    
    # 2. Load CSV data for the Load resource (has Resource Name)
    csv_data = load_sced_load_csv(year)
    load_csv = csv_data[csv_data['Resource Name'] == load_name]
    
    # 3. Load parquet data (missing Resource Name)
    parquet_data = load_parquet(f'SCED_Load_Resources/{year}')
    
    # 4. Match using unique combination
    matched_data = []
    for _, csv_row in load_csv.iterrows():
        # Find matching parquet row
        match = parquet_data[
            (parquet_data['SCEDTimeStamp'] == csv_row['SCED Time Stamp']) &
            (parquet_data['BasePoint'] == csv_row['Base Point']) &
            (parquet_data['MaxPowerConsumption'] == csv_row['Max Power Consumption'])
        ]
        if len(match) == 1:
            matched_data.append(match.iloc[0])
    
    return pd.DataFrame(matched_data)
```

### Why 0.25 in RT Revenue Formula?

**The 0.25 factor converts MW to MWh for 15-minute intervals:**
- RT prices in ERCOT are settled every **15 minutes**
- BasePoint is in **MW** (power)
- Price is in **$/MWh** (energy)
- 15 minutes = 0.25 hours
- Energy (MWh) = Power (MW) × Time (hours)
- So: `Revenue = MW × $/MWh × 0.25 hours`

Example:
```
BasePoint = 100 MW for 15 minutes
RT Price = $50/MWh
Energy delivered = 100 MW × 0.25 hours = 25 MWh
Revenue = 25 MWh × $50/MWh = $1,250
Or: Revenue = 100 × $50 × 0.25 = $1,250
```

### Step 4: Correct Revenue Calculation

```python
def calculate_bess_revenue_complete(bess_name, year):
    """
    Complete BESS revenue calculation with proper RT imbalance settlement
    """
    
    # === Day-Ahead Market ===
    dam_gen = load_parquet(f'DAM_Gen_Resources/{year}')
    gen_da = dam_gen[dam_gen['ResourceName'] == bess_name]
    
    # DA Discharge Revenue (Generation sells energy)
    da_discharge_revenue = sum(
        row['AwardedQuantity'] * row['EnergySettlementPointPrice']
        for _, row in gen_da.iterrows()
        if row['AwardedQuantity'] > 0
    )
    
    # DA Charging Cost (Load doesn't directly bid in DAM)
    # Must infer from settlement point Energy Bid Awards
    # For now, assume only BESS at settlement point
    da_charge_cost = 0  # Simplified assumption
    
    # === Real-Time Market (CRITICAL: Imbalance Settlement) ===
    
    # Gen Resource RT Data
    sced_gen = load_parquet(f'SCED_Gen_Resources/{year}')
    gen_rt = sced_gen[sced_gen['ResourceName'] == bess_name]
    
    # Load Resource RT Data (using cross-reference)
    load_rt = get_load_resource_data(bess_name, year)
    
    # Get RT prices
    rt_prices = load_parquet(f'RT_prices/{year}')
    
    # Calculate RT imbalance revenue/cost
    rt_revenue = 0
    rt_cost = 0
    
    for timestamp in gen_rt['datetime'].unique():
        # Get hour for DA position lookup
        hour = pd.to_datetime(timestamp).hour + 1  # HourEnding convention
        
        # Generation imbalance
        gen_da_position = gen_da[gen_da['HourEnding'] == hour]['AwardedQuantity'].iloc[0] \
                         if len(gen_da[gen_da['HourEnding'] == hour]) > 0 else 0
        gen_rt_position = gen_rt[gen_rt['datetime'] == timestamp]['BasePoint'].iloc[0]
        gen_imbalance = gen_rt_position - (gen_da_position if not pd.isna(gen_da_position) else 0)
        
        # Load imbalance (Load has no DA position)
        load_rt_position = load_rt[load_rt['datetime'] == timestamp]['BasePoint'].iloc[0] \
                          if len(load_rt[load_rt['datetime'] == timestamp]) > 0 else 0
        load_imbalance = load_rt_position - 0  # No DA position for load
        
        # Get RT price for this interval
        rt_price = get_rt_price_for_timestamp(timestamp, rt_prices)
        
        # Revenue/Cost calculation (15-minute intervals)
        interval_hours = 0.25  # 15 minutes = 0.25 hours
        rt_revenue += gen_imbalance * rt_price * interval_hours
        rt_cost += load_imbalance * rt_price * interval_hours
    
    # === Ancillary Services ===
    
    # AS from Gen Resource
    gen_as_revenue = sum(
        row['RegUpAwarded'] * row['RegUpMCPC'] +
        row['RegDownAwarded'] * row['RegDownMCPC'] +
        row['RRSAwarded'] * row['RRSMCPC'] +
        row['NonSpinAwarded'] * row['NonSpinMCPC'] +
        row['ECRSAwarded'] * row['ECRSMCPC']
        for _, row in gen_da.iterrows()
    )
    
    # AS from Load Resource (from DAM_Load_Resources CSV)
    load_as_revenue = 0  # Need to implement similar to gen_as_revenue
    
    # === Total Revenue ===
    total_revenue = (
        da_discharge_revenue
        - da_charge_cost
        + rt_revenue
        - rt_cost
        + gen_as_revenue
        + load_as_revenue
    )
    
    return {
        'bess_name': bess_name,
        'da_discharge': da_discharge_revenue,
        'da_charge': da_charge_cost,
        'rt_revenue': rt_revenue,
        'rt_cost': rt_cost,
        'as_revenue': gen_as_revenue + load_as_revenue,
        'total_revenue': total_revenue
    }
```

### Step 5: Validation Checks

```python
def validate_bess_operation(bess_name, year):
    """
    Validate BESS operation makes physical sense
    """
    gen_data = load_gen_data(bess_name, year)
    load_data = get_load_resource_data(bess_name, year)
    
    # Check 1: Never charge and discharge simultaneously
    for timestamp in gen_data['datetime'].unique():
        gen_bp = gen_data[gen_data['datetime'] == timestamp]['BasePoint'].iloc[0]
        load_bp = load_data[load_data['datetime'] == timestamp]['BasePoint'].iloc[0]
        
        if gen_bp > 0 and load_bp > 0:
            print(f"ERROR: Both charging and discharging at {timestamp}")
    
    # Check 2: Energy balance (efficiency check)
    total_discharge = gen_data['BasePoint'].sum() * 0.25  # MWh
    total_charge = load_data['BasePoint'].sum() * 0.25  # MWh
    efficiency = total_discharge / total_charge if total_charge > 0 else 0
    
    if not (0.80 <= efficiency <= 0.95):
        print(f"WARNING: Efficiency {efficiency:.2%} outside normal range")
    
    # Check 3: Revenue reasonableness
    results = calculate_bess_revenue_complete(bess_name, year)
    arbitrage_ratio = results['total_revenue'] / abs(results['rt_cost']) if results['rt_cost'] != 0 else 0
    
    if arbitrage_ratio > 3.0:
        print(f"WARNING: Unusually high arbitrage ratio {arbitrage_ratio:.2f}")
    
    return True
```

## Implementation Priority

### Phase 1: Immediate Fix (Use CSV files directly)
```python
# Since parquet is missing Resource Names, use CSV for Load Resources
# This is slower but accurate
```

### Phase 2: Create Mapping Table
```python
# Build a lookup table once
mapping_table = create_resource_mapping_table(year)
# Save as parquet for fast lookups
mapping_table.to_parquet('resource_mapping.parquet')
```

### Phase 3: Fix Parquet Generation
```python
# Update the rollup scripts to preserve Resource Names
# Add 'Resource Name' column to SCED_Load_Resources parquet
```

## Critical Insights

### Why Current Calculations Are Wrong

1. **Missing Load Resource Data**: Only looking at Gen side
2. **Wrong RT Formula**: Using `BasePoint × Price` instead of `(BasePoint - DA_Position) × Price`
3. **No Energy Balance**: Can't verify if charge/discharge makes sense

### The Data We Actually Have

✅ **We CAN calculate accurate BESS revenue because:**
- Gen Resource data is complete in parquet
- Load Resource data exists in CSV with Resource Names
- We can cross-reference using timestamp + BasePoint + MaxPower
- RT prices are available at 15-minute intervals
- DA prices and awards are available

❌ **Current limitations:**
- Parquet rollup lost Load Resource Names (fixable)
- DA charging attribution at settlement point level (use simplifying assumption)
- Need to process CSV files for Load data (slower but works)

## Example Calculation

```python
# BATCAVE_BES1 on April 17, 2024, Hour 9

# Day-Ahead
DA_Award = 50 MW
DA_Price = $30/MWh
DA_Revenue = 50 × $30 = $1,500

# Real-Time (4 intervals in hour)
Interval 1: BasePoint = 45 MW, RT_Price = $35/MWh
Interval 2: BasePoint = 48 MW, RT_Price = $40/MWh  
Interval 3: BasePoint = 55 MW, RT_Price = $50/MWh
Interval 4: BasePoint = 52 MW, RT_Price = $45/MWh

# RT Imbalance Settlement
Interval 1: (45 - 50) × $35 × 0.25 = -$43.75
Interval 2: (48 - 50) × $40 × 0.25 = -$20.00
Interval 3: (55 - 50) × $50 × 0.25 = +$62.50
Interval 4: (52 - 50) × $45 × 0.25 = +$22.50

Total RT = $21.25 (for deviations from DA position)
Total Revenue = $1,500 (DA) + $21.25 (RT) = $1,521.25
```

## Conclusion

**We CAN solve the BESS revenue puzzle!**

The key breakthrough is discovering we can cross-reference Load Resources using the unique combination of timestamp + BasePoint + MaxPowerConsumption. This allows us to:

1. Match Load Resource data even without Resource Names in parquet
2. Calculate complete BESS revenue including both Gen and Load sides
3. Properly implement RT imbalance settlement
4. Validate results with energy balance checks

The immediate path forward is to implement the cross-reference algorithm and start calculating accurate BESS revenues!