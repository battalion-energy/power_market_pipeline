# TBX Revenue Calculation Units Documentation

## Overview
The TBX (TB2/TB4) calculations model battery arbitrage revenue for **1 MW batteries** with different storage durations.

## Battery Specifications

### TB2 (2-hour Battery)
- **Power Capacity**: 1 MW
- **Energy Capacity**: 2 MWh
- **Duration**: 2 hours
- **Operation**: Charges for 2 hours, discharges for 2 hours daily

### TB4 (4-hour Battery)  
- **Power Capacity**: 1 MW
- **Energy Capacity**: 4 MWh
- **Duration**: 4 hours
- **Operation**: Charges for 4 hours, discharges for 4 hours daily

## Revenue Units

### Daily Revenue
- **Unit**: `$/MW-day`
- **Meaning**: Revenue per megawatt of battery capacity per day
- **Example**: TB4 revenue of $523.80/MW-day means a 1 MW battery earns $523.80 that day

### Annual Revenue
- **Unit**: `$/MW-year`
- **Meaning**: Revenue per megawatt of battery capacity per year
- **Calculation**: Daily revenue × 365 days
- **Example**: $191,187.89/MW-year for top-performing nodes

## Calculation Method

For each day, the algorithm:

1. **Identifies optimal charge hours**: Finds the `n` hours with lowest prices (n=2 for TB2, n=4 for TB4)
2. **Identifies optimal discharge hours**: Finds the `n` hours with highest prices
3. **Calculates charge cost**: Sum of prices during charge hours ÷ efficiency
4. **Calculates discharge revenue**: Sum of prices during discharge hours × efficiency
5. **Net revenue**: Discharge revenue - Charge cost

### Mathematical Formula

For a 1 MW battery with `h` hour duration and efficiency `η`:

```
Daily Revenue ($/MW-day) = Σ(P_discharge × η) - Σ(P_charge ÷ η)
```

Where:
- `P_discharge`: Prices during h highest-price hours ($/MWh)
- `P_charge`: Prices during h lowest-price hours ($/MWh)
- `η`: Round-trip efficiency (0.9 or 90%)

## Scaling for Different Battery Sizes

The revenues scale linearly with battery power capacity:

| Battery Size | TB2 Daily Revenue | TB4 Daily Revenue |
|-------------|------------------|-------------------|
| 1 MW | $X/MW-day | $Y/MW-day |
| 100 MW | $100X/day | $100Y/day |
| 250 MW | $250X/day | $250Y/day |

## Example from Actual ERCOT Data

**Top Settlement Point**: BRP_PBL2_RN (Brownsville area)

- **TB4 Daily Average**: $523.80/MW-day
- **TB4 Annual**: $191,187.89/MW-year
- **TB2 Daily Average**: $308.27/MW-day
- **TB2 Annual**: $112,517.89/MW-year

### Real-world Application

For a 100 MW / 400 MWh battery at BRP_PBL2_RN:
- Daily TB4 revenue: $52,380/day
- Annual TB4 revenue: $19.1 million/year

## Important Notes

1. **Perfect Foresight**: These calculations assume perfect knowledge of daily prices (upper bound on achievable revenue)
2. **No Operating Constraints**: Doesn't account for:
   - State of charge constraints
   - Ramp rate limitations
   - Market participation rules
   - Degradation or maintenance
3. **90% Efficiency**: Accounts for 10% round-trip losses typical of lithium-ion batteries
4. **Price Source**: Uses ERCOT Day-Ahead Market (DAM) settlement point prices

## Verification

The units have been verified through:
1. Mathematical validation of the algorithm
2. Cross-checking annual = daily × 365
3. Comparison with industry-standard battery revenue models
4. Consistency between Python and Rust implementations