# Understanding Base Point in ERCOT: Gen vs Load Resources

## What is Base Point?

**Base Point** is ERCOT's dispatch instruction to a resource - it tells the resource exactly how many MW to produce (for generators) or consume (for loads) at a specific point in time.

Think of it as the "set point" or "target" that ERCOT's Security Constrained Economic Dispatch (SCED) algorithm calculates every 5 minutes to balance the grid while respecting all transmission constraints.

## Base Point for Generation Resources

### Definition
For generators, Base Point is the **MW output level** that ERCOT instructs the generator to produce.

### How it Works
- **Positive values**: Generate this many MW of power
- **Zero**: Don't generate (stay online but at zero output)
- **Negative values**: Rare for traditional generators, but for BESS it means charging (acting as load)

### Example
```
Resource: BATCAVE_BES1 (Battery Gen Resource)
Time: 10:00:00
Base Point: 75.0 MW
Meaning: Discharge 75 MW (sell power to grid)

Time: 11:00:00  
Base Point: 0.0 MW
Meaning: Neither charge nor discharge (idle)

Time: 12:00:00
Base Point: -50.0 MW (rare in data)
Meaning: Charge at 50 MW (buy power from grid)
```

### Data Location
- **File**: `60d_SCED_Gen_Resource_Data-*.csv`
- **Column**: `Base Point`
- **Update Frequency**: Every 5 minutes (SCED runs)

## Base Point for Load Resources

### Definition
For load resources, Base Point is the **MW consumption level** that ERCOT instructs the load to consume.

### How it Works
- **Positive values**: Consume this many MW of power (normal load operation)
- **Zero**: Don't consume power (load is offline or not dispatched)
- **Negative values**: Theoretically could mean generation, but not used in practice

### Example
```
Resource: BATCAVE_LD1 (Battery Load Resource)
Time: 10:00:00
Base Point: 0.0 MW
Meaning: Don't charge (battery is discharging via Gen Resource)

Time: 11:00:00
Base Point: 80.0 MW
Meaning: Charge at 80 MW (consume power from grid)

Time: 12:00:00
Base Point: 40.0 MW
Meaning: Charge at 40 MW (reduced charging rate)
```

### Data Location
- **File**: `60d_Load_Resource_Data_in_SCED-*.csv`
- **Column**: `Base Point`
- **Update Frequency**: Every 5 minutes (SCED runs)

## BESS (Battery) Special Case: Split Resources

### Why Split?
ERCOT (until December 2025) requires batteries to register as TWO separate resources:
1. **Generation Resource** (e.g., BATCAVE_BES1): For discharging/selling power
2. **Load Resource** (e.g., BATCAVE_LD1): For charging/buying power

### Coordination Between Gen and Load
At any given time, a BESS can only do one of three things:
1. **Discharge**: Gen Base Point > 0, Load Base Point = 0
2. **Charge**: Gen Base Point = 0, Load Base Point > 0  
3. **Idle**: Gen Base Point = 0, Load Base Point = 0

They should NEVER both be positive at the same time (can't charge and discharge simultaneously).

### Real Example from Data
```
BATCAVE at 06/17/2025 10:00:14:
- BATCAVE_BES1 (Gen): Base Point = 38.3 MW ← Discharging
- BATCAVE_LD1 (Load): Base Point = 0.0 MW ← Not charging
Result: Battery is discharging 38.3 MW

BATCAVE at 06/17/2025 01:00:14:
- BATCAVE_BES1 (Gen): Base Point = 0.0 MW ← Not discharging
- BATCAVE_LD1 (Load): Base Point = 21.6 MW ← Charging
Result: Battery is charging at 21.6 MW
```

## Base Point vs Other Dispatch Values

### Base Point vs HSL/LSL
- **HSL (High Sustained Limit)**: Maximum capability of the resource
- **LSL (Low Sustained Limit)**: Minimum stable operating level
- **Base Point**: Current dispatch instruction (must be between LSL and HSL)

Example:
```
Generator HSL: 100 MW (max capacity)
Generator LSL: 20 MW (minimum stable level)
Base Point: 75 MW (current dispatch)
```

### Base Point vs Telemetered Output
- **Base Point**: What ERCOT told the resource to do
- **Telemetered Net Output** (Gen) / **Real Power Consumption** (Load): What the resource actually did
- Difference = Deviation from dispatch instruction

Example:
```
Base Point: 50.0 MW (instruction)
Telemetered Output: 48.5 MW (actual)
Deviation: -1.5 MW (under-generation)
```

## DAM vs SCED Base Points

### DAM (Day-Ahead Market)
- **Frequency**: Hourly values for next day
- **File**: `60d_DAM_Gen_Resource_Data-*.csv`
- **Column**: Sometimes called `Base Point` or `Awarded Quantity`
- **Purpose**: Day-ahead commitment and scheduling

### SCED (Real-Time)
- **Frequency**: Every 5 minutes
- **File**: `60d_SCED_Gen_Resource_Data-*.csv` and `60d_Load_Resource_Data_in_SCED-*.csv`
- **Column**: `Base Point`
- **Purpose**: Real-time dispatch to balance the grid

## Revenue Implications

### For Generation Resources
```python
Revenue = Base Point × LMP (Locational Marginal Price)
# If Base Point = 50 MW and LMP = $30/MWh
# Revenue = 50 × $30 = $1,500 for that hour
```

### For Load Resources (Charging)
```python
Cost = Base Point × LMP
# If Base Point = 80 MW and LMP = $20/MWh  
# Cost = 80 × $20 = $1,600 for that hour
```

### BESS Arbitrage
```python
# Charge when prices are low
Night: Load Base Point = 100 MW, LMP = $15/MWh, Cost = $1,500

# Discharge when prices are high
Peak: Gen Base Point = 90 MW, LMP = $50/MWh, Revenue = $4,500

# Profit = Revenue - Cost = $4,500 - $1,500 = $3,000
# (Assuming 90% round-trip efficiency)
```

## Common Pitfalls

1. **Missing Load Resource Data**: Many analyses only look at Gen Resources and miss the Load Resource charging data

2. **Zero Base Point Interpretation**: 
   - For Gen: Zero means "don't generate" (could be charging as Load)
   - For Load: Zero means "don't consume" (could be discharging as Gen)

3. **Negative Base Points**: Very rare in ERCOT data
   - In theory, negative Gen Base Point = charging
   - In practice, charging happens through Load Resource positive Base Point

4. **DAM vs RT Confusion**: 
   - DAM Base Points are hourly forecasts/awards
   - SCED Base Points are 5-minute real-time instructions
   - They often differ due to real-time conditions

## Practical SQL Query Example

```sql
-- Find BESS charging and discharging periods
WITH battery_dispatch AS (
  SELECT 
    g.timestamp,
    g.resource_name as gen_resource,
    l.resource_name as load_resource,
    g.base_point as gen_base_point,
    l.base_point as load_base_point,
    CASE 
      WHEN g.base_point > 0 THEN 'DISCHARGING'
      WHEN l.base_point > 0 THEN 'CHARGING'
      ELSE 'IDLE'
    END as battery_state,
    GREATEST(g.base_point, l.base_point) as dispatch_mw
  FROM sced_gen_resources g
  JOIN sced_load_resources l 
    ON g.timestamp = l.timestamp
    AND REPLACE(g.resource_name, '_BES1', '') = REPLACE(l.resource_name, '_LD1', '')
  WHERE g.resource_type = 'PWRSTR'
)
SELECT 
  battery_state,
  COUNT(*) as periods,
  AVG(dispatch_mw) as avg_mw,
  SUM(dispatch_mw * 0.0833) as total_mwh -- 5-min = 1/12 hour
FROM battery_dispatch
GROUP BY battery_state;
```

## Summary

- **Base Point** = ERCOT's dispatch instruction (MW)
- **Gen Resource Base Point** = How much to generate (positive) or charge (negative, rare)
- **Load Resource Base Point** = How much to consume/charge (positive)
- **BESS uses both**: Gen for discharge, Load for charge (never simultaneously)
- **Critical for revenue**: Base Point × Price = Revenue (Gen) or Cost (Load)
- **Real-time data**: Updated every 5 minutes in SCED files
- **Day-ahead data**: Hourly awards in DAM files