# BESS Revenue Calculation Methodology

## Executive Summary

Battery Energy Storage Systems (BESS) in ERCOT participate in multiple revenue streams:
1. **Energy Arbitrage** - Buying (charging) when prices are low, selling (discharging) when prices are high
2. **Ancillary Services** - Providing grid stability services (RegUp, RegDown, RRS, ECRS, NonSpin, FFR)

**CRITICAL INSIGHT**: Until December 2025, BESS resources are split into TWO components in ERCOT:
- **Generation Resource** (e.g., `MADERO_UNIT1`) - handles DISCHARGING
- **Load Resource** (e.g., `MADERO_LD1`) - handles CHARGING

Both must be tracked together to calculate complete BESS revenues and state of charge.

## Data Sources

### 1. Energy Operations Data

#### Day-Ahead Market (DAM)
- **Generation (Discharge) Data**: `/60-Day_DAM_Disclosure_Reports/csv/60d_DAM_Gen_Resource_Data-*.csv`
- **Load (Charge) Data**: `/60-Day_DAM_Disclosure_Reports/csv/60d_DAM_Load_Resource_Data-*.csv`

#### Real-Time Market (SCED)
- **Generation (Discharge) Data**: `/60-Day_SCED_Disclosure_Reports/csv/60d_SCED_Gen_Resource_Data-*.csv`
- **Load (Charge) Data**: `/60-Day_SCED_Disclosure_Reports/csv/60d_Load_Resource_Data_in_SCED-*.csv`

### 2. Price Data

#### Energy Prices
- **DAM Prices**: `/DAM_Settlement_Point_Prices/*.csv`
  - Columns: `DeliveryDate`, `HourEnding`, `SettlementPoint`, `SettlementPointPrice`
  
- **RT Prices**: `/Settlement_Point_Prices_at_Resource_Nodes,_Hubs_and_Load_Zones/*.csv`
  - Columns: `DeliveryDate`, `DeliveryHour`, `DeliveryInterval`, `SettlementPointName`, `SettlementPointPrice`

#### Ancillary Service Prices
- **DAM AS Prices**: `/DAM_Clearing_Prices_for_Capacity/*.csv`
  - Columns: `DeliveryDate`, `HourEnding`, `AncillaryType`, `MCPC`
  - Types: `REGUP`, `REGDN`, `RRS`, `ECRS`, `NONSPIN`, `RRSFFR`

## Resource Matching Logic

### Naming Convention
For a battery named "MADERO":
- **Generation Resources**: `MADERO_UNIT1`, `MADERO_UNIT2` (can have multiple units)
- **Load Resources**: `MADERO_LD1`, `MADERO_LD2` (typically _LD1, sometimes _LD2)
- **Settlement Point**: `MADERO_RN` or `MADERO_ALL`

### Matching Algorithm
```python
def match_battery_resources(resource_name):
    # Extract base name
    if '_UNIT' in resource_name:
        base_name = resource_name.split('_UNIT')[0]
        resource_type = 'generation'
    elif '_LD' in resource_name:
        base_name = resource_name.split('_LD')[0]
        resource_type = 'load'
    else:
        # Handle other patterns
        base_name = resource_name
        resource_type = 'unknown'
    
    return {
        'base_name': base_name,
        'type': resource_type,
        'gen_resources': [f"{base_name}_UNIT{i}" for i in range(1, 5)],
        'load_resources': [f"{base_name}_LD{i}" for i in range(1, 3)],
        'settlement_points': [f"{base_name}_RN", f"{base_name}_ALL"]
    }
```

## Revenue Calculation Components

### 1. Energy Revenue

#### Day-Ahead Energy Revenue
```
DAM_Energy_Revenue = DAM_Discharge_Revenue - DAM_Charge_Cost

Where:
- DAM_Discharge_Revenue = Σ(DAM_Gen_Awards[h] × DAM_Price[h])
- DAM_Charge_Cost = Σ(DAM_Load_Awards[h] × DAM_Price[h])
```

**Data Sources**:
- DAM_Gen_Awards: `60d_DAM_Gen_Resource_Data` → `Base Point` column
  - This is the MW amount the battery was awarded to discharge in the day-ahead market
  - Represents the scheduled/committed discharge for each hour
- DAM_Load_Awards: `60d_DAM_Load_Resource_Data` → (need to identify specific column - likely energy bid awards)
- DAM_Price: `DAM_Settlement_Point_Prices` → `SettlementPointPrice`

#### Real-Time Energy Revenue
```
RT_Energy_Revenue = RT_Discharge_Revenue - RT_Charge_Cost

Where:
- RT_Discharge_Revenue = Σ(RT_Gen_Output[i] × RT_Price[i])
- RT_Charge_Cost = Σ(RT_Load_Consumption[i] × RT_Price[i])
```

**Data Sources**:
- RT_Gen_Output: `60d_SCED_Gen_Resource_Data` → `Telemetered Net Output` column
  - This is the ACTUAL measured discharge output in MW (what the battery really discharged)
  - Measured via telemetry every 5 minutes
  - Note: `Output Schedule` column shows what SCED instructed, `Base Point` shows dispatch instruction
  - Use `Telemetered Net Output` for actual revenue calculations
- RT_Load_Consumption: `60d_Load_Resource_Data_in_SCED` → `Real Power Consumption` column
  - This is the ACTUAL measured charging input in MW
  - Measured via telemetry every 5 minutes
- RT_Price: `Settlement_Point_Prices_at_Resource_Nodes` → `SettlementPointPrice`

### 2. Ancillary Services Revenue

#### Service Types and Award Columns

**Generation Resources** (`60d_DAM_Gen_Resource_Data`):
- **RegUp**: `REGUP Awarded` × `REGUP MCPC`
- **RegDown**: `REGDN Awarded` × `REGDN MCPC`
- **RRS**: `RRSPFR Awarded` + `RRSFFR Awarded` + `RRSUFR Awarded` × `RRS MCPC`
- **ECRS**: `ECRS Awarded` × `ECRS MCPC`
- **NonSpin**: `NonSpin Awarded` × `NonSpin MCPC`

**Load Resources** (`60d_DAM_Load_Resource_Data`):
- **RegUp**: `RegUp Awarded` × `RegUp MCPC`
- **RegDown**: `RegDown Awarded` × `RegDown MCPC`
- **RRS**: `RRSPFR Awarded` + `RRSFFR Awarded` + `RRSUFR Awarded` × `RRS MCPC`
- **ECRS**: `ECRSSD Awarded` + `ECRSMD Awarded` × `ECRS MCPC`
- **NonSpin**: `NonSpin Awarded` × `NonSpin MCPC`

#### Total AS Revenue Calculation
```
AS_Revenue = Σ(service_types) [
    (Gen_AS_Awards[service] + Load_AS_Awards[service]) × AS_Price[service]
]
```

### 3. State of Charge Tracking

```python
def calculate_soc(initial_soc, efficiency=0.85):
    soc = initial_soc
    
    for interval in time_series:
        # Get charge/discharge for this interval
        charge_mw = get_load_consumption(interval)  # From Load Resource
        discharge_mw = get_gen_output(interval)     # From Gen Resource
        
        # Update SOC (5-minute intervals = 1/12 hour)
        soc += (charge_mw * efficiency * (1/12))  # Charging
        soc -= (discharge_mw * (1/12))             # Discharging
        
        # Apply constraints
        soc = max(0, min(soc, battery_capacity))
        
    return soc
```

## Complete Revenue Formula

```
Total_BESS_Revenue = DAM_Energy_Revenue 
                   + RT_Energy_Revenue 
                   + AS_Revenue_Gen 
                   + AS_Revenue_Load

Where:
- DAM_Energy_Revenue = Σ(DAM_Discharge × DAM_Price) - Σ(DAM_Charge × DAM_Price)
- RT_Energy_Revenue = Σ(RT_Discharge × RT_Price) - Σ(RT_Load × RT_Price)
- AS_Revenue_Gen = Σ(AS_Awards_Gen × AS_Prices)
- AS_Revenue_Load = Σ(AS_Awards_Load × AS_Prices)
```

## Implementation Checklist

### Phase 1: Data Collection
- [ ] Add `60d_Load_Resource_Data_in_SCED-*.csv` to rollup process
- [ ] Add `60d_DAM_Load_Resource_Data-*.csv` to rollup process
- [ ] Create Parquet files for Load Resource data
- [ ] Verify all price files are being processed

### Phase 2: Resource Matching
- [ ] Implement battery name matching logic (base_name extraction)
- [ ] Create lookup table linking Gen and Load resources
- [ ] Map resources to settlement points

### Phase 3: Revenue Calculation
- [ ] Calculate DAM energy revenues (charge + discharge)
- [ ] Calculate RT energy revenues (charge + discharge)
- [ ] Calculate AS revenues for Gen resources
- [ ] Calculate AS revenues for Load resources
- [ ] Implement SOC tracking with charging data

### Phase 4: Validation
- [ ] Verify charge/discharge cycles are physically possible
- [ ] Check SOC stays within battery capacity limits
- [ ] Validate total revenues against known benchmarks
- [ ] Cross-check with ERCOT settlement statements

## Critical Columns Summary

### Generation Resources
- **Power Output**: `Telemetered Net Output`, `Base Point`
- **AS Awards**: `REGUP Awarded`, `REGDN Awarded`, `RRSPFR Awarded`, `ECRS Awarded`, `NonSpin Awarded`
- **AS Prices**: `REGUP MCPC`, `REGDN MCPC`, `RRS MCPC`, `ECRS MCPC`, `NonSpin MCPC`

### Load Resources
- **Power Consumption**: `Real Power Consumption` (SCED), TBD for DAM
- **AS Awards**: `RegUp Awarded`, `RegDown Awarded`, `RRSPFR Awarded`, `ECRSSD Awarded`, `NonSpin Awarded`
- **AS Prices**: `RegUp MCPC`, `RegDown MCPC`, `RRS MCPC`, `ECRS MCPC`, `NonSpin MCPC`

### Price Files
- **DAM Energy**: `SettlementPointPrice` (hourly)
- **RT Energy**: `SettlementPointPrice` (5-minute)
- **AS Prices**: `MCPC` by `AncillaryType`

## Notes and Assumptions

1. **Efficiency**: Assume 85% round-trip efficiency (92.5% one-way)
2. **Timeline**: Load Resource separation ends December 2025
3. **Intervals**: DAM is hourly, RT is 5-minute
4. **Units**: All power in MW, all prices in $/MWh
5. **Missing Data**: Some early years may not have Load Resource data

## Data Quality Checks

1. **Continuity**: Ensure no gaps in 5-minute intervals
2. **Consistency**: Verify charge + discharge don't exceed capacity
3. **Price Validation**: Check for negative or extreme prices
4. **Resource Matching**: Verify all _LD resources have corresponding gen resources
5. **SOC Physics**: Ensure SOC changes match charge/discharge with efficiency losses

---

**Document Version**: 1.0
**Last Updated**: 2024-11-17
**Author**: Power Market Pipeline Team