# Complete Methodology: Reconstructing BESS Charging in ERCOT

## Executive Summary

BESS charging in ERCOT is asymmetrically documented:
- **RT (SCED)**: Explicitly recorded in Load Resource "Base Point" 
- **DAM**: Must be reconstructed from multiple sources using economic logic

## Data Architecture

### Real-Time (SCED) - EXPLICIT ✅

**Source**: `60d_Load_Resource_Data_in_SCED-*.csv`
**Key Column**: `Base Point` (MW)
**Meaning**: Direct dispatch instruction for load to consume

Example:
```
RRANCHES_LD1 at 07:30:11: Base Point = 75.1 MW
→ Battery is charging at 75.1 MW
```

### Day-Ahead (DAM) - IMPLICIT ⚠️

Must piece together from THREE sources:

1. **Gen Resource Awards** (`60d_DAM_Gen_Resource_Data-*.csv`)
   - Shows discharge schedule (positive awards)
   - When award = 0, battery is either idle or charging

2. **Energy Bid Awards** (`60d_DAM_EnergyBidAwards-*.csv`)
   - Some negative awards indicate charging
   - But incomplete - not all charging appears here

3. **Energy Bids** (`60d_DAM_EnergyBids-*.csv`)
   - Shows bid curves (willingness to buy at different prices)
   - Must compare to clearing prices to infer what would clear

## The Bid/Offer Mechanism

### How BESS Submits Bids/Offers

**Charging Bid** (I want to buy power):
```
MW    Price
-50   $500   (will pay up to $500/MWh to charge 50 MW)
-100  $100   (will pay up to $100/MWh to charge 100 MW)
-150  $20    (will pay up to $20/MWh to charge 150 MW)
```

**Discharging Offer** (I want to sell power):
```
MW    Price
50    $10    (will sell 50 MW if price ≥ $10/MWh)
100   $30    (will sell 100 MW if price ≥ $30/MWh)
150   $50    (will sell 150 MW if price ≥ $50/MWh)
```

### Clearing Logic

If clearing price = $25/MWh:
- Charging: All bids ≥ $25 clear → Charges 100 MW
- Discharging: All offers ≤ $25 clear → Discharges 50 MW

## Reconstruction Algorithm

### For Real-Time (Simple)
```python
def get_rt_charging(bess_name):
    load_name = bess_name.replace('_BES', '_LD')
    sced_load = read_sced_load_resources()
    
    charging = sced_load[
        (sced_load['Resource Name'] == load_name) &
        (sced_load['Base Point'] > 0)
    ]
    
    return charging[['timestamp', 'Base Point']]
```

### For Day-Ahead (Complex)
```python
def reconstruct_dam_charging(bess_name):
    # Step 1: Get discharge schedule
    gen_awards = read_dam_gen_awards()
    discharge_hours = gen_awards[
        (gen_awards['Resource Name'] == bess_name) &
        (gen_awards['Awarded Quantity'] > 0)
    ]['Hour Ending'].tolist()
    
    # Step 2: Get prices for all hours
    dam_prices = read_dam_prices()
    
    # Step 3: Apply economic logic
    charging_schedule = {}
    for hour in range(1, 25):
        if hour in discharge_hours:
            charging_schedule[hour] = 0  # Discharging
        else:
            # Check if economical to charge
            price = dam_prices[hour]
            if price < CHARGE_THRESHOLD:  # e.g., $20/MWh
                charging_schedule[hour] = MAX_CHARGE_MW
            else:
                charging_schedule[hour] = 0  # Idle
    
    return charging_schedule
```

## Why This Complexity Exists

### Historical Context
1. ERCOT originally designed for traditional generators and loads
2. BESS doesn't fit cleanly - it's both generator AND load
3. Split resource model (Gen + Load) creates data fragmentation

### Market Design
1. DAM optimizes day-ahead based on bids/offers
2. Only "cleared" quantities appear in awards
3. Charging often "self-scheduled" based on price signals

### Future (Post-Dec 2025)
ERCOT is moving to unified BESS resources, which should simplify this

## Validation Approach

### Energy Balance Check
```python
total_discharge = sum(gen_basepoints > 0)
total_charge = sum(load_basepoints > 0)
efficiency = 0.85

assert abs(total_discharge - total_charge * efficiency) < tolerance
```

### Price Arbitrage Check
```python
avg_charge_price = weighted_avg(prices when charging)
avg_discharge_price = weighted_avg(prices when discharging)

assert avg_discharge_price > avg_charge_price * (1/efficiency)
```

## Common Pitfalls

1. **Assuming DAM Load Resources have energy awards** - They don't
2. **Looking only at Gen Resources** - Misses all charging data
3. **Ignoring Energy Bid Awards** - Contains some charging data
4. **Not checking SCED Load Base Point** - This is the definitive RT charging
5. **Assuming negative Gen Base Point for charging** - Rare in practice

## Implementation Recommendations

### For Accurate BESS Revenue Calculation

1. **RT Revenue/Cost**:
   - Use SCED Gen Base Point for discharge revenue
   - Use SCED Load Base Point for charging cost
   - These are definitive

2. **DAM Revenue/Cost**:
   - Use DAM Gen Awards for discharge revenue (explicit)
   - Reconstruct charging from economic logic:
     - Check hours where Gen Award = 0
     - If price < threshold, assume charging
     - Use typical charge rate from RT data

3. **Ancillary Services**:
   - Both Gen and Load Resources can provide AS
   - Sum AS awards from both resource types

### SQL Query Example
```sql
-- Complete BESS position reconstruction
WITH bess_state AS (
  SELECT 
    timestamp,
    CASE 
      WHEN gen_base_point > 0 THEN 'DISCHARGE'
      WHEN load_base_point > 0 THEN 'CHARGE'
      ELSE 'IDLE'
    END as state,
    COALESCE(gen_base_point, 0) as discharge_mw,
    COALESCE(load_base_point, 0) as charge_mw,
    settlement_point_price as price
  FROM (
    SELECT 
      g.timestamp,
      g.base_point as gen_base_point,
      l.base_point as load_base_point,
      p.price as settlement_point_price
    FROM sced_gen_resources g
    LEFT JOIN sced_load_resources l 
      ON g.timestamp = l.timestamp 
      AND REPLACE(g.resource_name, '_BES1', '_LD1') = l.resource_name
    LEFT JOIN rt_prices p
      ON g.timestamp = p.timestamp
      AND g.settlement_point = p.settlement_point
    WHERE g.resource_type = 'PWRSTR'
  )
)
SELECT 
  state,
  COUNT(*) as periods,
  SUM(discharge_mw * price * 0.0833) as discharge_revenue,
  SUM(charge_mw * price * 0.0833) as charge_cost
FROM bess_state
GROUP BY state;
```

## Conclusion

BESS charging in ERCOT requires a multi-source reconstruction:
- **RT**: Use SCED Load Resource Base Point (explicit)
- **DAM**: Combine Gen Awards + Economic Logic (implicit)
- **Validation**: Check energy balance and price arbitrage
- **Future**: Unified BESS resources (Dec 2025) will simplify this