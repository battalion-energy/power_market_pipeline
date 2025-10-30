# TBX Comprehensive Calculator - Usage Guide

## Overview

The comprehensive TBX calculator generates theoretical battery arbitrage revenue for 1-12 hour duration batteries across all ERCOT settlement points, using both Day Ahead and Real-Time prices.

## Output Files

**Location:** `/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/tbx_comprehensive/`

**Files:** `tbx_all_YYYY.parquet` (one file per year, 2010-2025)

**Total Size:** 388 MB (all years)

## Schema

### Core Columns

- `settlement_point` (string): ERCOT settlement point name (e.g., "LZ_WEST", "RUSSEKST_RN")
- `settlement_point_type` (string): Type of settlement point (LZ, HB, RN, DC, etc.)
- `delivery_date` (date): Delivery date
- `year` (int): Year
- `period_type` (string): 'daily', 'monthly', 'quarterly', 'annual', or 'ytd'
- `days_in_period` (int): Number of days in the aggregation period

### Revenue Columns (24 total)

**Day Ahead:** `tb1_da`, `tb2_da`, ..., `tb12_da` ($/MW for period)
**Real-Time:** `tb1_rt`, `tb2_rt`, ..., `tb12_rt` ($/MW for period)

**Interpretation:**
- TB1 = 1-hour battery (1 MW / 1 MWh)
- TB2 = 2-hour battery (1 MW / 2 MWh)
- TB4 = 4-hour battery (1 MW / 4 MWh)
- ... up to TB12 = 12-hour battery (1 MW / 12 MWh)

Values represent revenue per MW of battery capacity for the specified period.

### Timestamp Columns

- `period_start` (datetime): Start of period
- `period_end` (datetime): End of period
- `period_start_local` (datetime with tz): Start in US/Central time
- `period_start_utc` (datetime with tz): Start in UTC
- `period_end_local` (datetime with tz): End in US/Central time
- `period_end_utc` (datetime with tz): End in UTC

### Aggregation Columns

- `month` (float): Month number (1-12) for monthly records
- `quarter` (float): Quarter number (1-4) for quarterly records

## Usage Examples

### Example 1: Get Annual TB2 Revenue for LZ_WEST in 2024

```python
import pyarrow.parquet as pq

df = pq.read_table('tbx_all_2024.parquet').to_pandas()

# Filter for annual records
annual = df[df['period_type'] == 'annual']

# Get LZ_WEST
lz_west = annual[annual['settlement_point'] == 'LZ_WEST'].iloc[0]

print(f"LZ_WEST TB2 Day Ahead: ${lz_west['tb2_da']:,.2f}/MW-year")
print(f"LZ_WEST TB2 Real-Time: ${lz_west['tb2_rt']:,.2f}/MW-year")
print(f"LZ_WEST TB2 Total: ${lz_west['tb2_da'] + lz_west['tb2_rt']:,.2f}/MW-year")

# For a 100 MW / 200 MWh battery
power_mw = 100
annual_revenue = (lz_west['tb2_da'] + lz_west['tb2_rt']) * power_mw
print(f"\n100 MW Battery Annual Revenue: ${annual_revenue:,.2f}")
```

**Output:**
```
LZ_WEST TB2 Day Ahead: $63,967.71/MW-year
LZ_WEST TB2 Real-Time: $80,581.34/MW-year
LZ_WEST TB2 Total: $144,549.05/MW-year

100 MW Battery Annual Revenue: $14,454,905.00
```

### Example 2: Compare TB2 vs TB4 Premium

```python
annual = df[df['period_type'] == 'annual']
annual['tb2_total'] = annual['tb2_da'] + annual['tb2_rt']
annual['tb4_total'] = annual['tb4_da'] + annual['tb4_rt']
annual['tb4_premium'] = annual['tb4_total'] / annual['tb2_total']

# Top 10 by TB4 premium
top10 = annual.nlargest(10, 'tb4_premium')[
    ['settlement_point', 'tb2_total', 'tb4_total', 'tb4_premium']
]
print(top10)
```

### Example 3: Monthly Trend Analysis

```python
# Get monthly data for a specific settlement point
monthly = df[
    (df['period_type'] == 'monthly') &
    (df['settlement_point'] == 'LZ_WEST')
].sort_values('month')

# Calculate combined revenue
monthly['tb2_combined'] = monthly['tb2_da'] + monthly['tb2_rt']

# Plot
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(monthly['month'], monthly['tb2_da'], label='TB2 Day Ahead', marker='o')
plt.plot(monthly['month'], monthly['tb2_rt'], label='TB2 Real-Time', marker='s')
plt.plot(monthly['month'], monthly['tb2_combined'], label='TB2 Combined', marker='^', linewidth=2)
plt.xlabel('Month')
plt.ylabel('Revenue ($/MW-month)')
plt.title('LZ_WEST TB2 Monthly Revenue - 2024')
plt.legend()
plt.grid(True)
plt.savefig('lz_west_monthly_trend.png')
```

### Example 4: Find Best Locations for New Battery Projects

```python
# Get annual data
annual = df[df['period_type'] == 'annual']

# Calculate total TB4 revenue (DA + RT)
annual['tb4_total'] = annual['tb4_da'] + annual['tb4_rt']

# Filter for resource nodes (RN) and load zones (LZ)
viable_locations = annual[
    annual['settlement_point_type'].isin(['RN', 'LZ'])
].nlargest(20, 'tb4_total')

print("Top 20 Locations for 4-Hour Battery:")
for i, row in enumerate(viable_locations.itertuples(), 1):
    print(f"{i}. {row.settlement_point} ({row.settlement_point_type})")
    print(f"   Annual TB4 Revenue: ${row.tb4_total:,.2f}/MW-year")
    print(f"   Days of data: {row.days_in_period}")
```

### Example 5: Compare All Battery Durations

```python
# Get one settlement point's annual data
point = annual[annual['settlement_point'] == 'RUSSEKST_RN'].iloc[0]

# Extract all TBX values
durations = range(1, 13)
da_revenues = [point[f'tb{i}_da'] for i in durations]
rt_revenues = [point[f'tb{i}_rt'] for i in durations]
total_revenues = [da + rt for da, rt in zip(da_revenues, rt_revenues)]

# Plot
plt.figure(figsize=(12, 6))
plt.plot(durations, da_revenues, label='Day Ahead', marker='o')
plt.plot(durations, rt_revenues, label='Real-Time', marker='s')
plt.plot(durations, total_revenues, label='Combined', marker='^', linewidth=2)
plt.xlabel('Battery Duration (hours)')
plt.ylabel('Annual Revenue ($/MW-year)')
plt.title('Battery Duration vs Revenue - RUSSEKST_RN 2024')
plt.legend()
plt.grid(True)
plt.xticks(durations)
plt.savefig('battery_duration_comparison.png')
```

### Example 6: Multi-Year Trend Analysis

```python
import pandas as pd

# Load multiple years
years = range(2020, 2025)
all_data = []

for year in years:
    df = pd.read_parquet(f'tbx_all_{year}.parquet')
    annual = df[
        (df['period_type'] == 'annual') &
        (df['settlement_point'] == 'LZ_WEST')
    ]
    all_data.append(annual)

combined = pd.concat(all_data, ignore_index=True)
combined['tb2_total'] = combined['tb2_da'] + combined['tb2_rt']
combined['tb4_total'] = combined['tb4_da'] + combined['tb4_rt']

# Plot trend
plt.figure(figsize=(12, 6))
plt.plot(combined['year'], combined['tb2_total'], label='TB2', marker='o', linewidth=2)
plt.plot(combined['year'], combined['tb4_total'], label='TB4', marker='s', linewidth=2)
plt.xlabel('Year')
plt.ylabel('Annual Revenue ($/MW-year)')
plt.title('LZ_WEST Multi-Year Battery Revenue Trend')
plt.legend()
plt.grid(True)
plt.savefig('multi_year_trend.png')
```

### Example 7: Quarterly Business Planning

```python
# Get quarterly data
quarterly = df[
    (df['period_type'] == 'quarterly') &
    (df['settlement_point'] == 'LZ_HOUSTON')
]

quarterly['tb4_combined'] = quarterly['tb4_da'] + quarterly['tb4_rt']
quarterly['avg_daily'] = quarterly['tb4_combined'] / quarterly['days_in_period']

print("LZ_HOUSTON Quarterly TB4 Revenue:")
for _, row in quarterly.iterrows():
    print(f"Q{int(row['quarter'])}: ${row['tb4_combined']:,.2f} "
          f"({row['days_in_period']} days, ${row['avg_daily']:.2f}/MW-day)")
```

## Key Metrics

### 2024 Performance Highlights

**Top Performer:** RUSSEKST_RN
- TB2: $255,223/MW-year ($103,450 DA + $151,773 RT)
- TB4: $395,155/MW-year ($171,094 DA + $224,061 RT)

**Major Hubs:**
- LZ_WEST: TB2 $144,549/MW-year, TB4 $214,042/MW-year
- LZ_SOUTH: TB2 $115,766/MW-year, TB4 $166,812/MW-year
- LZ_NORTH: TB2 $109,006/MW-year, TB4 $150,598/MW-year

### Typical TB4/TB2 Premium
The TB4 premium (TB4 revenue ÷ TB2 revenue) typically ranges from 1.4x to 1.6x, meaning 4-hour batteries earn 40-60% more than 2-hour batteries.

## Notes

1. **Perfect Foresight:** These calculations assume perfect knowledge of future prices (theoretical upper bound)

2. **Round-Trip Efficiency:** Applied at 90% (configurable in script)

3. **No Operational Constraints:** Does not account for:
   - State of charge limits
   - Ramp rates
   - Market participation rules
   - Degradation
   - Operating costs

4. **Settlement Points:** 969 unique locations including:
   - Load Zones (LZ_*)
   - Hubs (HB_*)
   - Resource Nodes (RN suffix)
   - DC Ties (DC_*)

5. **Time Zones:** All timestamps include both UTC and US/Central (local) versions

## Running the Calculator

```bash
# Run for specific years
uv run python tools/tbx_comprehensive.py --years 2024 2025

# Run for all years
uv run python tools/tbx_comprehensive.py --years 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025

# Custom efficiency
uv run python tools/tbx_comprehensive.py --years 2024 --efficiency 0.85

# Skip DA or RT
uv run python tools/tbx_comprehensive.py --years 2024 --skip-rt
```

## Script Location

`/home/enrico/projects/power_market_pipeline/tools/tbx_comprehensive.py`

## Related Documentation

- `/home/enrico/projects/power_market_pipeline/specs/TBX_CALCULATOR.md` - Original TBX specification
- `/home/enrico/projects/power_market_pipeline/specs/TBX_UNITS_DOCUMENTATION.md` - Units and calculation details
- `/home/enrico/projects/power_market_pipeline/docs/TBX_IMPLEMENTATION_SUMMARY.md` - Implementation notes
