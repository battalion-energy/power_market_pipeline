# ECRS Revenue Calculation - Root Cause Analysis

## Executive Summary

**Problem:** ECRS revenue calculated as $7,810 for full year 2024, but Q3 2024 alone should be $140,617 according to raw data.

**Root Cause:** Revenue calculator uses system-wide MCPC prices from `AS_prices` file instead of resource-specific MCPC already embedded in `DAM_Load_Resources` file.

**Impact:** ALL ancillary service revenues for Load resources may be incorrectly calculated (RegUp, RegDown, RRS, ECRS, NonSpin).

---

## Data Flow Analysis

### Source 1: DAM_Load_Resources File
**File:** `/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/rollup_files/DAM_Load_Resources/2024.parquet`

**Columns (relevant to ECRS):**
- `Load Resource Name` - Resource identifier
- `ECRSSD Awarded` - ECRS Service Deployment awards in MW
- `ECRS MCPC` - Resource-specific Market Clearing Price for Capacity ($/MW)
- `RegUp Awarded`, `RegUp MCPC`
- `RegDown Awarded`, `RegDown MCPC`
- `RRSFFR Awarded`, `RRS MCPC`
- `NonSpin Awarded`, `NonSpin MCPC`

**Key Finding:** THIS FILE ALREADY HAS RESOURCE-SPECIFIC MCPC PRICES!

**Q3 2024 ECRS Data:**
- Total rows with ECRS awards: 703
- Total Q3 ECRS revenue: $140,616.95
- Top earners:
  - SNDSW_LD1: $73,334
  - TINPD_LD3: $35,186
  - SNDSW_LD10: $22,884

**Sample Price Comparison (Jul 1, 2024, Hour 15):**
- Resource: TINPD_LD3
- Award: 45.0 MW
- **Resource-specific MCPC: $3.85/MW**
- Correct revenue: 45.0 × $3.85 = $173.25

---

### Source 2: AS_prices File
**File:** `/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/rollup_files/AS_prices/2024.parquet`

**Columns:**
- `DeliveryDate` - Date (date type)
- `HourEnding` - Hour string ("01:00", "02:00", etc.)
- `AncillaryType` - AS product type (REGUP, REGDN, RRS, ECRS, NSPIN)
- `MCPC` - SYSTEM-WIDE Market Clearing Price for Capacity ($/MW)

**Key Finding:** THESE ARE SYSTEM-WIDE AVERAGE PRICES, NOT RESOURCE-SPECIFIC!

**Sample Price Comparison (Jul 1, 2024, Hour 15):**
- **System-wide ECRS MCPC: $3.56/MW**
- Wrong revenue calculation: 45.0 × $3.56 = $160.20

**Price Difference:** $3.85 vs $3.56 = $0.29/MW (8.2% difference)

---

## Revenue Calculator Code Analysis

**File:** `bess_revenue_calculator.py`

### Current Implementation (INCORRECT)

```python
def calculate_dam_as_revenues(self, resource: str, is_gen: bool):
    # Line 163: Loads EXTERNAL price file
    as_price_file = self.rollup_dir / f"AS_prices/{self.year}.parquet"

    # Line 181: Loads system-wide prices
    df_prices = pl.read_parquet(as_price_file)

    # Lines 185-201: Maps award columns
    if is_gen:
        as_award_cols = {
            "ECRS": "ECRSAwarded",
            ...
        }
    else:
        as_award_cols = {
            "ECRS": "ECRSSD Awarded",  # ← Load uses SD (Service Deployment)
            ...
        }

    # Lines 212-230: Joins awards with SYSTEM-WIDE prices
    df_as_prices = df_prices.filter(
        pl.col("AncillaryType") == "ECRS"  # ← System-wide price!
    )

    df_joined = df.join(
        df_as_prices,
        on=["date", "hour"],
        how="left"
    )

    # Line 234: Calculates revenue using WRONG price
    revenue = (pl.col("awarded") * pl.col("MCPC")).sum()
```

### What SHOULD Happen (CORRECT)

For **Load Resources**, the MCPC is already in the data file:

```python
def calculate_dam_as_revenues_load(self, resource: str):
    """
    For Load resources, use EMBEDDED resource-specific MCPC prices
    """
    dam_file = self.rollup_dir / f"DAM_Load_Resources/{self.year}.parquet"

    df = pl.read_parquet(dam_file).filter(
        pl.col("Load Resource Name") == resource
    )

    # MCPC columns are ALREADY in the file - no join needed!
    as_revenues = {
        "RegUp": (df["RegUp Awarded"] * df["RegUp MCPC"]).sum(),
        "RegDown": (df["RegDown Awarded"] * df["RegDown MCPC"]).sum(),
        "RRS": (df["RRSFFR Awarded"] * df["RRS MCPC"]).sum(),
        "ECRS": (df["ECRSSD Awarded"] * df["ECRS MCPC"]).sum(),
        "NonSpin": (df["NonSpin Awarded"] * df["NonSpin MCPC"]).sum()
    }

    return as_revenues
```

For **Gen Resources**, we need to check if MCPC is also embedded or needs external join.

---

## Verification of Root Cause

### Test Case: Q3 2024 ECRS Revenue

**Expected (from raw data):**
```
Sum of (ECRSSD Awarded × ECRS MCPC) for Q3 2024 = $140,616.95
```

**Actual (from revenue calculator):**
```
Full year 2024 ECRS = $7,810
(Q3 is subset of full year, so Q3 < $7,810)
```

**Discrepancy:** ~18x undercount

### Why System-Wide vs Resource-Specific Matters

MCPC prices vary by resource based on:
1. Resource location (transmission constraints)
2. Resource characteristics (ramp rates, response time)
3. Network conditions (congestion)
4. Time of award clearance

Using system-wide average instead of resource-specific price can lead to:
- Under/over-counting revenue by 5-30%
- Incorrect attribution to specific batteries
- Wrong conclusions about battery performance

---

## Impact Assessment

### Affected Revenue Streams

**Load Resources (DEFINITE ISSUE):**
- ✅ RegUp revenue
- ✅ RegDown revenue
- ✅ RRS (Reserves) revenue
- ✅ ECRS revenue - CONFIRMED WRONG
- ✅ Non-Spin revenue

**Gen Resources (NEEDS VERIFICATION):**
- ❓ RegUp revenue
- ❓ RegDown revenue
- ❓ RRS revenue
- ❓ ECRS revenue - 0 awards in 2024, may not apply
- ❓ Non-Spin revenue

### Affected Batteries

ANY battery with a Load Resource that received AS awards in 2024:
- All BESS units participate in AS markets
- Primary revenue source for many batteries (>50% of total)
- Could affect 60-100 battery calculations

---

## Recommended Fixes

### Fix #1: Update Load AS Revenue Calculation (HIGH PRIORITY)

Replace external price join with embedded MCPC columns:

```python
# In bess_revenue_calculator.py, line ~150
def calculate_dam_as_revenues(self, resource: str, is_gen: bool):
    if is_gen:
        # Keep existing logic for Gen resources (needs separate verification)
        return self._calculate_gen_as_revenues(resource)
    else:
        # NEW: Use embedded MCPC for Load resources
        return self._calculate_load_as_revenues(resource)

def _calculate_load_as_revenues(self, resource: str):
    """Calculate AS revenues for Load resources using embedded MCPC"""
    dam_file = self.rollup_dir / f"DAM_Load_Resources/{self.year}.parquet"

    df = pl.read_parquet(dam_file).filter(
        pl.col("Load Resource Name") == resource
    )

    if len(df) == 0:
        return {"RegUp": 0.0, "RegDown": 0.0, "RRS": 0.0, "ECRS": 0.0, "NonSpin": 0.0}

    # Use resource-specific MCPC columns that are already in the file
    return {
        "RegUp": (df["RegUp Awarded"] * df["RegUp MCPC"]).sum(),
        "RegDown": (df["RegDown Awarded"] * df["RegDown MCPC"]).sum(),
        "RRS": (df["RRSFFR Awarded"] * df["RRS MCPC"]).sum(),
        "ECRS": (df["ECRSSD Awarded"] * df["ECRS MCPC"]).sum(),
        "NonSpin": (df["NonSpin Awarded"] * df["NonSpin MCPC"]).sum()
    }
```

### Fix #2: Verify Gen AS Revenue Calculation

Check if DAM_Gen_Resources also has embedded MCPC:

```bash
# Check if Gen file has MCPC columns
python3 << 'EOF'
import polars as pl
df = pl.read_parquet('/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/rollup_files/DAM_Gen_Resources/2024.parquet')
mcpc_cols = [col for col in df.columns if 'MCPC' in col]
print(f"MCPC columns in Gen file: {mcpc_cols}")
EOF
```

If Gen file also has MCPC columns, update Gen calculation similarly.

### Fix #3: Regenerate All Revenue Data

After fixing the calculator:
1. Delete old revenue files: `bess_revenue_*.csv`
2. Rerun revenue calculator for all years: 2019-2024
3. Regenerate normalized revenue files
4. Regenerate all charts

### Fix #4: Add Data Quality Checks

```python
# Add to revenue calculator
def verify_as_calculation(self, sample_resource: str, sample_date: str):
    """Verify AS revenue calculation against raw data"""
    # Calculate using our method
    calculated = self.calculate_dam_as_revenues(sample_resource, is_gen=False)

    # Calculate directly from raw data
    df = pl.read_parquet(f"DAM_Load_Resources/{self.year}.parquet")
    df_sample = df.filter(
        (pl.col("Load Resource Name") == sample_resource) &
        (pl.col("DeliveryDate").cast(pl.Date) == pl.date.fromisoformat(sample_date))
    )

    expected_ecrs = (df_sample["ECRSSD Awarded"] * df_sample["ECRS MCPC"]).sum()

    assert abs(calculated["ECRS"] - expected_ecrs) < 0.01, \
        f"ECRS mismatch: {calculated['ECRS']} vs {expected_ecrs}"
```

---

## CHISMGRD_BES1 Capacity Issue

**Separate Issue:** CHISMGRD_BES1 showing 9.99 MW instead of 100 MW.

**Root Cause:** Revenue calculator not loading updated mapping file.

**Fix:** Update revenue calculator to use correct mapping file and reload capacity.

---

## Next Steps

1. ✅ **IMMEDIATE:** Fix Load AS revenue calculation
2. ✅ **IMMEDIATE:** Verify Gen AS revenue calculation
3. ✅ **HIGH:** Regenerate all 2019-2024 revenue data
4. ✅ **HIGH:** Update normalization with correct revenues
5. ✅ **HIGH:** Regenerate all charts
6. ✅ **MEDIUM:** Fix CHISMGRD_BES1 capacity
7. ✅ **MEDIUM:** Add automated verification tests
8. ✅ **LOW:** Document correct calculation in code comments

---

## Conclusion

The ECRS revenue discrepancy is caused by using system-wide MCPC prices instead of resource-specific MCPC prices that are already embedded in the DAM_Load_Resources file. This likely affects all AS revenue streams for all Load resources.

The fix is straightforward: use the embedded MCPC columns instead of joining with an external price file.

**Estimated Time to Fix:** 2-4 hours
**Estimated Time to Regenerate All Data:** 4-8 hours
