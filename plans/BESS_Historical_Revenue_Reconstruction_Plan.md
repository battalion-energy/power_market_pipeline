# BESS Historical Revenue Reconstruction Plan

## Mission Clarification
**We are NOT optimizing batteries.** These are real batteries that already operated in ERCOT. Our job is to **reconstruct what actually happened** and calculate the revenues they actually earned.

## What We're Actually Doing

### Historical Forensic Analysis
We're essentially doing accounting/auditing of BESS operations:

1. **What they bid** → DAM offer curves
2. **What they won** → DAM energy & AS awards  
3. **What they did** → SCED dispatch & telemetered output
4. **What they got paid** → Settlement calculations

This is **reconstruction**, not optimization!

## The Real Algorithm (Much Simpler!)

### Step 1: Identify When BESS Charged/Discharged
```python
def identify_bess_operations(resource_name, date):
    """
    Look at ACTUAL operations from disclosure data
    """
    # From DAM awards
    dam_awards = read_dam_gen_resource_data(resource_name, date)
    
    # From RT dispatch  
    sced_dispatch = read_sced_gen_resource_data(resource_name, date)
    
    # Actual operations
    for hour in range(24):
        dam_mw = dam_awards[hour]['Awarded Quantity']
        if dam_mw > 0:
            print(f"Hour {hour}: DISCHARGED {dam_mw} MW in DAM")
        elif dam_mw < 0:
            print(f"Hour {hour}: CHARGED {abs(dam_mw)} MW in DAM")
            
    # RT deviations
    for interval in sced_dispatch:
        base_point = interval['Base Point']
        telemetered = interval['Telemetered Net Output']
        print(f"{interval['time']}: Dispatched to {base_point} MW, Actually did {telemetered} MW")
```

### Step 2: Calculate What They Got Paid
```python
def calculate_actual_revenues(resource_name, date):
    """
    Simple accounting - what × price = revenue
    """
    revenues = {
        'dam_energy': 0,
        'rt_energy': 0,
        'regup': 0,
        'regdown': 0,
        'rrs': 0,
        'ecrs': 0,
        'nonspin': 0
    }
    
    # DAM Energy Revenue
    dam_awards = read_dam_gen_resource_data(resource_name, date)
    dam_prices = read_dam_prices(date)
    
    for hour, award in dam_awards.items():
        mw = award['Awarded Quantity']
        price = dam_prices[award['Settlement Point']][hour]
        revenues['dam_energy'] += mw * price
    
    # RT Energy Revenue (deviations from DAM)
    sced_data = read_sced_gen_resource_data(resource_name, date)
    rt_prices = read_rt_prices(date)
    
    for interval in sced_data:
        actual_mw = interval['Telemetered Net Output']
        dam_mw = get_dam_schedule(resource_name, interval['time'])
        deviation = actual_mw - dam_mw
        price = rt_prices[interval['Settlement Point']][interval['time']]
        revenues['rt_energy'] += deviation * price
    
    # AS Revenues - what they were awarded × clearing price
    as_awards = dam_awards['AS_Awards']
    as_prices = read_as_clearing_prices(date)
    
    revenues['regup'] = as_awards['RegUp'] * as_prices['RegUp']
    revenues['regdown'] = as_awards['RegDown'] * as_prices['RegDown']
    revenues['rrs'] = as_awards['RRS'] * as_prices['RRS']
    # ... etc
    
    return revenues
```

### Step 3: Track State of Charge (What Actually Happened)
```python
def track_actual_soc(resource_name, date):
    """
    Reconstruct SOC from actual charge/discharge operations
    """
    soc_timeline = []
    current_soc = get_initial_soc(resource_name, date)  # From COP snapshot
    
    for interval in get_all_intervals(date):
        telemetered_mw = get_telemetered_output(resource_name, interval)
        
        if telemetered_mw > 0:
            # Discharging
            energy_out = telemetered_mw * (5/60)  # 5-minute interval in hours
            current_soc -= energy_out
        elif telemetered_mw < 0:
            # Charging
            energy_in = abs(telemetered_mw) * (5/60)
            current_soc += energy_in * 0.85  # Assuming 85% efficiency
            
        soc_timeline.append({
            'time': interval,
            'soc_mwh': current_soc,
            'power_mw': telemetered_mw
        })
    
    return soc_timeline
```

## Data Sources for Reconstruction

### 1. DAM Awards (What They Won)
**File:** `60d_DAM_Gen_Resource_Data-{date}.csv`
```csv
Resource Name, Hour Ending, Awarded Quantity, RegUp Awarded, RegDown Awarded, ...
ANCHOR_BESS1, 1, -50.0, 0, 10.0, ...  # Charging 50MW, providing 10MW RegDown
ANCHOR_BESS1, 14, 100.0, 20.0, 0, ... # Discharging 100MW, providing 20MW RegUp
```

### 2. SCED Dispatch (What They Actually Did)
**File:** `60d_SCED_Gen_Resource_Data-{date}.csv`
```csv
SCED Time Stamp, Resource Name, Base Point, Telemetered Net Output
01/31/2025 13:00:00, ANCHOR_BESS1, 95.0, 94.8  # Told to do 95MW, actually did 94.8MW
01/31/2025 13:05:00, ANCHOR_BESS1, 95.0, 95.1  # Small deviation
```

### 3. COP Snapshots (State of Charge)
**File:** `60d_COP_Adjustment_Period_Snapshot-{date}.csv`
```csv
Resource Name, Hour Beginning Planned SOC, Minimum SOC, Maximum SOC
ANCHOR_BESS1, 150.0, 0.0, 200.0  # Started hour with 150MWh stored
```

## Key Differences from Optimization Approach

### What We DON'T Need to Do:
- ❌ Optimize charge/discharge schedules
- ❌ Predict optimal arbitrage points
- ❌ Solve MILP problems
- ❌ Forecast prices
- ❌ Make operational decisions

### What We DO Need to Do:
- ✅ Read what actually happened
- ✅ Match awards with prices
- ✅ Track actual SOC changes
- ✅ Calculate settlement revenues
- ✅ Validate our calculations

## Implementation Steps (Revised)

### Phase 1: Data Extraction (Week 1)
```rust
struct BessOperations {
    resource_name: String,
    date: NaiveDate,
    dam_awards: Vec<HourlyAward>,
    sced_dispatch: Vec<FiveMinuteDispatch>,
    initial_soc: f64,
    final_soc: f64,
}

impl BessOperations {
    fn from_disclosure_data(resource: &str, date: NaiveDate) -> Self {
        // Just read what happened, don't optimize anything
    }
}
```

### Phase 2: Revenue Calculation (Week 2)
```rust
struct RevenueCalculator {
    fn calculate_dam_energy_revenue(&self, awards: &[HourlyAward], prices: &DamPrices) -> f64 {
        // Simple: Σ(MW × Price)
    }
    
    fn calculate_rt_deviation_revenue(&self, dam: &[HourlyAward], actual: &[FiveMinuteDispatch], prices: &RtPrices) -> f64 {
        // Simple: Σ((Actual - DAM) × RT_Price)
    }
    
    fn calculate_as_revenue(&self, as_awards: &ASAwards, clearing_prices: &ASPrices) -> f64 {
        // Simple: Award × Clearing_Price for each service
    }
}
```

### Phase 3: Validation (Week 3)
```rust
struct Validator {
    fn validate_soc_feasibility(&self, operations: &BessOperations) -> bool {
        // Check: Did SOC stay within limits?
    }
    
    fn validate_power_limits(&self, operations: &BessOperations) -> bool {
        // Check: Did power stay within HSL/LSL?
    }
    
    fn cross_check_revenues(&self, calculated: &Revenue, settlement: &Settlement) -> f64 {
        // Compare our calc to ERCOT settlement
    }
}
```

## Questions (Now Much Simpler!)

### Data Questions:
1. **Do we have all the disclosure files we need?**
   - Yes, we have 28,775 files

2. **Can we identify all BESS resources?**
   - Yes, filter for Resource Type = 'PWRSTR'

3. **Do we have settlement statements to validate against?**
   - Need to check

### Technical Questions:
1. **How do we handle missing intervals?**
   - Interpolate or use last known value

2. **What efficiency do we use for SOC tracking?**
   - Can calculate from actual data: Energy_out / Energy_in

3. **How do we handle AS deployments?**
   - They show up in telemetered output

## The Bottom Line

This is **MUCH SIMPLER** than optimization:
1. Read what the BESS actually did (from disclosures)
2. Multiply by prices
3. Sum it up
4. Validate against settlements

No optimization needed. No predictions needed. Just good accounting of what actually happened.

## Next Steps

1. **Build simple data readers** for disclosure files
2. **Calculate revenues** from actual operations
3. **Track SOC** from actual charge/discharge
4. **Validate** against known totals
5. **Remove ALL optimization code** - we don't need it!

This is historical analysis, not forward-looking optimization. We're accountants, not traders!