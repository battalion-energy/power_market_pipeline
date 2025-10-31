# Hourly Resource Outage Capacity Processing - Summary

## ✓ COMPLETED TASKS

### 1. Extracted ZIP files to CSV
- **Location:** `/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_clean_batch_dataset/Hourly Resource Outage Capacity/csv`
- **Status:** 241 zip files extracted → 59,822 CSV files
- **Coverage:** 2019-2025 (7 years)

### 2. Converted CSV to Yearly Parquet
- **Location:** `/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_clean_batch_dataset/parquet/Hourly Resource Outage Capacity/`
- **Files Created:** 7 yearly parquet files (2019-2025)
- **Total Size:** 31 MB
- **Records:** 10,967,712 rows

### 3. Aggregated and Processed Outage Data
- **Script:** `process_outage_data_FIXED_v2.py`
- **Output:** `/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/processed/generator_outages_2019_2025.parquet`
- **Records:** 59,999 hourly observations
- **Date Range:** 2019-01-01 to 2025-11-04 (most recent!)
- **Size:** 0.5 MB

### Features Generated:
- `outage_total_MW` - Total generator outages
- `outage_thermal_MW` - Thermal plant outages
- `outage_renewable_MW` - Renewable outages
- `outage_new_equip_MW` - New equipment outages
- `outage_change_1h` - Hour-over-hour changes
- `outage_thermal_change_1h` - Thermal outage changes
- `outage_roll_3h_mean` - 3-hour rolling average
- `outage_roll_3h_max` - 3-hour rolling max
- `outage_roll_24h_mean` - 24-hour rolling average
- `outage_roll_24h_max` - 24-hour rolling max
- `high_outage_flag` - >20 GW outages
- `critical_outage_flag` - >30 GW outages
- `large_thermal_outage_flag` - >15 GW thermal outages
- `sudden_outage_flag` - >5 GW/h change

## 📊 KEY FINDINGS

### Data Gap Identified:
- **Master Dataset End Date:** 2025-05-08 01:00:00
- **New Outage Data End Date:** 2025-11-04 22:00:00
- **GAP:** 4,341 additional hours of new data available!

### Statistics (2019-2025):
- Average total outages: 15.4 GW
- Average thermal outages: 12.4 GW
- Average renewable outages: 3.0 GW
- High outage events (>20 GW): 32% of hours
- Critical outages (>30 GW): 3.5% of hours

## 🎯 NEXT STEPS

### 1. Update Master Dataset with New Outage Data (May-Nov 2025)
The current master dataset at:
`/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/master_CORRECTED_with_fixed_solar_wind_outages.parquet`

Already has outage features but needs to be extended with:
- 4,341 new hourly observations from May-Nov 2025
- Requires merging with other data sources for those dates (load, prices, solar, wind, etc.)

### 2. Verify Other Data Sources Have Coverage Through Nov 2025
Check if these datasets go through Nov 2025:
- Actual System Load
- Solar Production
- Wind Production  
- Settlement Point Prices
- ORDC Pricing

### 3. Create Full Master Dataset Update Script
Script should:
- Load current master dataset (ends May 2025)
- Load all processed feature datasets
- Merge features for May-Nov 2025 period
- Append to master dataset
- Validate completeness

## 📁 FILE LOCATIONS

### Source Data (Parquet by Year):
```
/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_clean_batch_dataset/parquet/Hourly Resource Outage Capacity/
├── Hourly Resource Outage Capacity_2019.parquet
├── Hourly Resource Outage Capacity_2020.parquet
├── Hourly Resource Outage Capacity_2021.parquet
├── Hourly Resource Outage Capacity_2022.parquet
├── Hourly Resource Outage Capacity_2023.parquet
├── Hourly Resource Outage Capacity_2024.parquet
└── Hourly Resource Outage Capacity_2025.parquet
```

### Processed Aggregated Data:
```
/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/processed/generator_outages_2019_2025.parquet
```

### Master Dataset (Current):
```
/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/master_CORRECTED_with_fixed_solar_wind_outages.parquet
```

### Processing Scripts:
```
iso_markets/ercot/process_hourly_outage_clean_batch.py (extraction + parquet conversion)
ai_forecasting/process_outage_data_FIXED_v2.py (aggregation + feature engineering)
```
