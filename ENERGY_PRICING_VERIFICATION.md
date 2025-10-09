# Energy Pricing Verification - DA and RT

## Executive Summary

✅ **DA Energy Pricing: CORRECT**
✅ **RT Energy Pricing: CORRECT**

Both Day-Ahead and Real-Time energy calculations use **resource-specific prices** (not system-wide averages). This is the CORRECT approach and does NOT have the same issues as the Ancillary Services calculations.

---

## DA Energy Pricing Analysis

### Implementation
```python
# bess_revenue_calculator.py:112-141
def calculate_dam_discharge_revenue(self, gen_resource: str) -> float:
    dam_gen_file = self.rollup_dir / f"DAM_Gen_Resources/{self.year}.parquet"

    df = pl.read_parquet(dam_gen_file).filter(
        pl.col("ResourceName") == gen_resource
    )

    # Calculate revenue: AwardedQuantity (MW) × Price ($/MWh)
    df = df.with_columns([
        (pl.col("AwardedQuantity") * pl.col("EnergySettlementPointPrice")).alias("revenue")
    ])

    total_revenue = df.select(pl.col("revenue").sum()).item()
```

### Data Source
- **File**: `DAM_Gen_Resources/{year}.parquet`
- **Quantity Column**: `AwardedQuantity` (MW awarded to THIS resource)
- **Price Column**: `EnergySettlementPointPrice` ($/MWh for THIS resource)

### Verification: Prices ARE Resource-Specific

**Sample: January 1, 2024, Hour 1**
| Resource | AwardedQuantity | EnergySettlementPointPrice |
|----------|----------------|---------------------------|
| BAIRDWND_UNIT1 | 12.8 MW | $9.64/MWh |
| BCATWIND_WIND_1 | 29.4 MW | $9.34/MWh |
| BOSQUESW_CC1_2 | 162.0 MW | $10.26/MWh |
| BRAUNIG_CC1_2 | 299.7 MW | $10.79/MWh |

**Conclusion**: Prices vary by resource (range: $9.34-10.79 at same time) ✅

---

## RT Energy Pricing Analysis

### Implementation
```python
# bess_revenue_calculator.py:315-495
def calculate_rt_net_revenue(self, gen_resource: str, load_resource: str, resource_node: str):
    sced_gen_file = self.rollup_dir / f"SCED_Gen_Resources/{self.year}.parquet"
    sced_load_file = self.rollup_dir / f"SCED_Load_Resources/{self.year}.parquet"
    rt_price_file = self.rollup_dir / f"RT_prices/{self.year}.parquet"

    # Load RT prices for THIS SPECIFIC resource node
    df_prices = pl.read_parquet(rt_price_file).filter(
        pl.col("SettlementPointName") == resource_node
    ).select([
        pl.col("datetime").alias("price_datetime"),
        pl.col("SettlementPointPrice").alias("rt_price")
    ])

    # Join discharge/charge data with resource-specific prices
    discharge_revenue = (pl.col("discharge_mw") * pl.col("rt_price") * (15.0 / 60.0)).sum()
    charge_cost = (pl.col("charge_mw") * pl.col("rt_price") * (15.0 / 60.0)).sum()
```

### Data Sources
- **Quantity Files**:
  - `SCED_Gen_Resources/{year}.parquet` (discharge MW)
  - `SCED_Load_Resources/{year}.parquet` (charge MW)
- **Price File**: `RT_prices/{year}.parquet`
- **Key Filter**: `SettlementPointName == resource_node` ← Resource-specific!
- **Price Column**: `SettlementPointPrice` ($/MWh for THIS settlement point)

### Verification: Prices ARE Resource-Specific

**Sample: July 15, 2024, 16:00 (Peak Hour)**
| SettlementPoint | SettlementPointPrice |
|----------------|---------------------|
| 7RNCHSLR_ALL | $59.37/MWh |
| ADL_RN | $59.38/MWh |
| AEEC | $61.58/MWh |
| AGUAYO_UNIT1 | $60.98/MWh |
| AJAXWIND_RN | $61.31/MWh |

**Price Distribution at 16:00:**
- Min: -$11.21/MWh
- Max: $191.95/MWh
- Mean: $57.42/MWh
- **Unique Prices: 336** ← Highly location-specific!

**Conclusion**: RT prices vary dramatically by location (range: -$11 to +$192) ✅

---

## Comparison with AS Pricing Issues

### What Was WRONG with AS Calculations

**Load AS (all products):**
- ❌ Joined with system-wide MCPC from `AS_prices` file
- ❌ Ignored embedded resource-specific MCPC in `DAM_Load_Resources`
- ❌ Result: 5-10% pricing error

**Gen RRS/ECRS:**
- ❌ Used wrong column names (old structure)
- ❌ Result: 100% missing revenue

### What Is CORRECT with Energy Calculations

**DA Energy:**
- ✅ Uses `EnergySettlementPointPrice` from `DAM_Gen_Resources`
- ✅ This is already resource-specific (embedded in file)
- ✅ No external join needed

**RT Energy:**
- ✅ Filters `RT_prices` to specific `SettlementPointName`
- ✅ Uses resource-specific `SettlementPointPrice`
- ✅ Joins by timestamp with resource-specific prices

---

## Why Energy Pricing is Different

### Data Structure Differences

| Aspect | Energy Pricing | AS Pricing |
|--------|---------------|------------|
| **DA Gen** | Embedded price in `DAM_Gen_Resources` | NO price, must join with `AS_prices` |
| **DA Load** | N/A (load doesn't sell energy) | Embedded MCPC in `DAM_Load_Resources` |
| **RT** | Resource-specific in `RT_prices` | N/A (no RT AS markets) |
| **Price Variation** | High (location-based congestion) | Moderate (locational + characteristics) |

### Why DA Gen Has Embedded Prices But Not AS Prices

**Energy markets:**
- Cleared at resource's specific location (settlement point)
- Transmission congestion directly affects price
- Every resource has unique locational marginal price (LMP)
- Price embedded in award file because it's fundamental to the award

**AS markets:**
- Cleared at system level with zonal/regional considerations
- System needs X MW of RegUp, bids stacked across ERCOT
- Most resources get system MCPC (marginal clearing price)
- Resource-specific MCPC only differs due to special cases (contracts, make-whole, local reliability)
- Price NOT embedded in Gen file because most resources just use system MCPC

**Load resources (special case):**
- Participating in AS markets is less common
- Often have bilateral contracts or special arrangements
- ERCOT includes resource-specific MCPC in `DAM_Load_Resources` for convenience
- This is why Load AS had the pricing bug - we ignored the embedded prices!

---

## Conclusion

✅ **No changes needed to energy pricing calculations**

The energy pricing calculations already use resource-specific prices:
- DA: `EnergySettlementPointPrice` from `DAM_Gen_Resources`
- RT: `SettlementPointPrice` filtered to `resource_node` from `RT_prices`

The AS pricing issues were:
1. Load AS: Using system MCPC instead of embedded resource-specific MCPC
2. Gen RRS/ECRS: Using wrong column names (missing disaggregated products)

These issues DO NOT affect energy calculations, which were implemented correctly from the start.

---

## Test Results

**Test Script**: `test_as_fixes.py`

**2024 Results (BATCAVE_BES1):**
- Gen RegUp: $1,265,020 ✅
- Gen RRS: $1,617,073 ✅ (was $0)
- Gen ECRS: $514,584 ✅ (was $0)
- Load ECRS: $9,772 ✅ (was $7,810 for full fleet)

**Backwards Compatibility:**
- 2020 tests passed without errors ✅
- Old column structure handled gracefully ✅
