# TBX Calculator Documentation

## Overview
The TBX (TB2/TB4) Calculator analyzes battery energy storage arbitrage opportunities across all ERCOT pricing nodes. It calculates theoretical revenues for 2-hour (TB2) and 4-hour (TB4) battery systems using historical day-ahead and real-time price data.

## Key Concepts

### TB2 (2-Hour Battery)
- Charges during the 2 lowest-price hours of each day
- Discharges during the 2 highest-price hours of each day
- Represents a battery with 2-hour duration at full power

### TB4 (4-Hour Battery)
- Charges during the 4 lowest-price hours of each day
- Discharges during the 4 highest-price hours of each day
- Represents a battery with 4-hour duration at full power

### Efficiency
- Default: 90% round-trip efficiency (10% losses)
- Applied to discharge revenue (you only get 90% of what you charged)
- Configurable via `--efficiency` parameter

## Data Sources

### Input Files
Located in `/home/enrico/data/ERCOT_data/rollup_files/flattened/`:
- `DA_prices_YYYY.parquet` - Day-ahead hourly prices
- `RT_prices_hourly_YYYY.parquet` - Real-time hourly prices (aggregated from 15-min)
- `RT_prices_15min_YYYY.parquet` - Real-time 15-minute prices (not currently used)

### Price Nodes
The calculator processes all ERCOT pricing nodes including:
- **Load Zones (LZ_*)**: LZ_WEST, LZ_HOUSTON, LZ_NORTH, LZ_SOUTH, LZ_AEN, LZ_CPS, LZ_LCRA, LZ_RAYBN
- **Hubs (HB_*)**: HB_WEST, HB_HOUSTON, HB_NORTH, HB_SOUTH, HB_PAN, HB_BUSAVG, HB_HUBAVG
- **DC Ties (DC_*)**: DC_E, DC_L, DC_N, DC_R, DC_S

## Output Files

### Location
All results are stored in `/home/enrico/data/ERCOT_data/tbx_results/`

### File Structure
For each year (2021-2024), three aggregation levels are created:

#### Daily Results
- **Files**: `tbx_daily_YYYY.parquet`
- **Contents**: Daily arbitrage revenues for each node
- **Columns**:
  - `date`: Trading date
  - `node`: Settlement point name
  - `tb2_da_revenue`: TB2 day-ahead arbitrage revenue ($)
  - `tb2_rt_revenue`: TB2 real-time arbitrage revenue ($)
  - `tb4_da_revenue`: TB4 day-ahead arbitrage revenue ($)
  - `tb4_rt_revenue`: TB4 real-time arbitrage revenue ($)

#### Monthly Results
- **Files**: `tbx_monthly_YYYY.parquet`
- **Contents**: Monthly aggregated revenues
- **Columns**:
  - `node`: Settlement point name
  - `month`: Month number (1-12)
  - `tb2_da_revenue`: Monthly TB2 DA revenue sum ($)
  - `tb2_rt_revenue`: Monthly TB2 RT revenue sum ($)
  - `tb4_da_revenue`: Monthly TB4 DA revenue sum ($)
  - `tb4_rt_revenue`: Monthly TB4 RT revenue sum ($)
  - `days_count`: Number of days in aggregation

#### Annual Results
- **Files**: `tbx_annual_YYYY.parquet`
- **Contents**: Annual aggregated revenues
- **Columns**:
  - `node`: Settlement point name
  - `tb2_da_revenue`: Annual TB2 DA revenue sum ($)
  - `tb2_rt_revenue`: Annual TB2 RT revenue sum ($)
  - `tb4_da_revenue`: Annual TB4 DA revenue sum ($)
  - `tb4_rt_revenue`: Annual TB4 RT revenue sum ($)
  - `days_count`: Number of days in aggregation

## Usage

### Basic Command
```bash
# Calculate TBX for default years (2021-2024)
make tbx

# Or directly:
uv run python calculate_tbx_v2.py
```

### Custom Parameters
```bash
# Specify custom efficiency and years
uv run python calculate_tbx_v2.py \
  --efficiency 0.85 \
  --years 2022 2023 2024 \
  --data-dir /custom/data/path \
  --output-dir /custom/output/path

# Using make with custom parameters
make tbx-custom EFFICIENCY=0.85 YEARS="2022 2023" OUTPUT_DIR=/custom/output
```

## Algorithm

### Daily Arbitrage Calculation
For each day and each node:

1. **Identify Charge Hours**: Sort 24 hourly prices, select N lowest hours (N=2 for TB2, N=4 for TB4)
2. **Identify Discharge Hours**: Select N highest price hours
3. **Calculate Costs**: Sum prices during charge hours
4. **Calculate Revenue**: Sum prices during discharge hours × efficiency
5. **Net Revenue**: (Discharge Revenue × Efficiency) - Charge Cost

### Example Calculation
```
Day's hourly prices: [10, 20, 15, 30, 25, 40, 35, 50, ...] $/MWh
TB2 Calculation:
- Charge hours: Hours 0,2 (prices: 10, 15)
- Discharge hours: Hours 7,5 (prices: 50, 40)
- Charge cost: 10 + 15 = $25
- Discharge revenue: (50 + 40) × 0.9 = $81
- Net revenue: 81 - 25 = $56
```

## Key Findings (2021-2024)

### Best Nodes for Battery Storage
1. **LZ_WEST**: Consistently highest arbitrage values
   - 2023: $142,305 (TB2), $235,312 (TB4)
   - 2024: $63,968 (TB2), $97,212 (TB4)

2. **LZ_LCRA**: Strong performance across years
3. **DC_S**: High values during extreme events (2021-2022)

### Revenue Trends
- **2023**: Peak arbitrage year (2x higher than 2024)
- **2021**: Winter Storm Uri created extreme arbitrage opportunities
- **2024**: Lower volatility = reduced arbitrage revenues
- **TB4 vs TB2**: TB4 generates ~65-70% more revenue

## Implementation Details

### Python Version
- **File**: `calculate_tbx_v2.py`
- **Dependencies**: pandas, numpy, pathlib
- **Processing**: Sequential by year, parallel by node
- **Performance**: Processes 20 nodes × 365 days in ~5 seconds per year

### Rust Version (Planned)
- **File**: `src/tbx_calculator_polars.rs`
- **Status**: Implemented but not compiled due to Rust version constraints
- **Features**: Full parallelization with Rayon, Polars DataFrames

## Future Enhancements

### Planned Features
1. **Real-Time 15-minute Analysis**: Use native 15-min RT prices
2. **Cycling Constraints**: Add daily cycle limits
3. **State of Charge Tracking**: Enforce energy balance
4. **Degradation Modeling**: Account for battery capacity fade
5. **Ancillary Services**: Include regulation/reserve revenues

### Visualization
1. **Heatmaps**: Monthly revenue by node
2. **Time Series**: Daily revenue trends
3. **Geographic**: Map overlay of arbitrage values
4. **Comparison**: TB2 vs TB4 efficiency curves

## Data Quality Notes

### Current Limitations
- RT revenues show as 0.0 (RT hourly data may not be available for all years)
- DA prices are primary signal for arbitrage analysis
- 366 days processed for leap years, 365 for others

### Validation
- All nodes show consistent day counts
- Revenue values align with known price volatility events
- West Texas consistently shows highest values (renewable curtailment)

## Related Documentation
- [ISO Data Pipeline Strategy](./ISO_DATA_PIPELINE_STRATEGY.md)
- [ERCOT Data Processing](../CLAUDE.md)
- [BESS Revenue Analysis](./BESS_REVENUE_TRACKING.md)