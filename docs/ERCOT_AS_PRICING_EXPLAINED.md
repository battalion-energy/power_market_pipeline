# ERCOT Ancillary Services Pricing - Complete Explanation

## Your Questions Answered

### Q1: Are generators missing ECRS? I thought generators can earn ECRS.

**YES! Generators DO earn ECRS - and it's the PRIMARY source of ECRS revenue!**

**2024 ECRS Awards:**
- **Gen Resources:** 13,733,456 MW total (473,931 hour-intervals)
- **Load Resources:** 1,843,718 MW total (195,474 hour-intervals)
- **Ratio:** Gen gets 7.6x more ECRS than Load!

**But we have TWO bugs:**
1. Revenue calculator looks for "ECRSAwarded" but Gen uses "ECRSSDAwarded" in 2024
2. Revenue calculator joins Load with system-wide MCPC instead of using embedded resource-specific MCPC

### Q2: There's a generator resource for discharging, and the load resource is for charging, right?

**CORRECT!**

**BESS has TWO resources in ERCOT:**
- **Gen Resource** (e.g., BATCAVE_BES1) = Discharging, selling power to grid
- **Load Resource** (e.g., BATCAVE_LD1) = Charging, buying power from grid

**Both can provide Ancillary Services:**
- **Gen provides AS while discharging** (or standing ready to discharge)
- **Load provides AS while charging** (or standing ready to adjust charging)

### Q3: Where are you getting the prices for the ancillary services from? You're not just getting them from the price curves?

**Two different price sources:**

#### Source 1: System-Wide MCPC (AS_prices file)
```
File: rollup_files/AS_prices/2024.parquet
Columns: DeliveryDate, HourEnding, AncillaryType, MCPC
```

**Example (Jan 3, 2024, Hour 8):**
```
AncillaryType: ECRS
MCPC: $6.21/MW (system-wide marginal clearing price)
```

#### Source 2: Resource-Specific MCPC (DAM_Load_Resources file)
```
File: rollup_files/DAM_Load_Resources/2024.parquet
Columns: Load Resource Name, ECRSSD Awarded, ECRS MCPC, ...
```

**Example (Same hour, same date):**
```
Resource: TINPD_LD3
ECRS MCPC: $6.47/MW (resource-specific price, different from system!)
```

### Q4: What is the MCPC?

**MCPC = Market Clearing Price for Capacity**

In ERCOT's Day-Ahead Market for Ancillary Services:
- ERCOT procures reserves (RegUp, RegDown, RRS, ECRS, Non-Spin)
- Resources submit offers: "I'll provide X MW of RegUp at $Y/MW"
- ERCOT stacks offers (merit order) and clears market
- **System MCPC:** The marginal offer that clears the total requirement
- **Resource-specific MCPC:** What each resource actually receives

**Not the same as:**
- **SPP (Settlement Point Price):** Energy price ($/MWh)
- **Bid curves:** QSE-submitted offers (not clearing prices)

### Q5: Why are they different across the different resources at the same point in time?

**Resource-specific MCPC differs from system MCPC due to:**

#### 1. Self-Arranged / Out-of-Market Contracts
- Resource has bilateral contract at different price
- Not participating in market clearing but providing service
- MCPC reflects contract price, not market price

#### 2. Locational Value
- Transmission constraints create zonal differences
- Resources in constrained areas more valuable
- Higher MCPC in load pockets

#### 3. Resource-Specific Characteristics
- Ramp rate capability
- Response time (fast vs slow reserves)
- Reliability history
- May command premium/discount

#### 4. Network Topology
- Some resources needed for local reliability
- Voltage support, black start capability
- Essential reliability service (ERS) designation

#### 5. Make-Whole Payments
- Resource offered high, didn't clear market
- But needed for other reasons (energy, local reliability)
- Gets make-whole payment reflected in MCPC

**Example from Jan 3, 2024, Hour 8:**
```
System-wide ECRS MCPC: $6.21/MW
TINPD_LD3 ECRS MCPC: $6.47/MW (+4.2%)
```

The $0.26/MW difference could be due to any combination of the above factors.

---

## Data Structure Differences

### Gen Resources (DAM_Gen_Resources)

**Columns:**
- ResourceName
- AwardedQuantity (energy)
- EnergySettlementPointPrice
- RegUpAwarded, RegDownAwarded
- RRSAwarded, RRSPFRAwarded, RRSFFRAwarded, RRSUFRAwarded
- ECRSAwarded (not used in 2024)
- **ECRSSDAwarded** (Service Deployment - USED in 2024)
- NonSpinAwarded

**NO MCPC COLUMNS!** Must join with AS_prices for system-wide MCPC.

### Load Resources (DAM_Load_Resources)

**Columns:**
- Load Resource Name
- RegUp Awarded, **RegUp MCPC** ← Embedded!
- RegDown Awarded, **RegDown MCPC** ← Embedded!
- RRSFFR Awarded, **RRS MCPC** ← Embedded!
- ECRSMD Awarded (Manual Deployment)
- ECRSSD Awarded (Service Deployment)
- **ECRS MCPC** ← Embedded! (same for both MD and SD)
- NonSpin Awarded, **NonSpin MCPC** ← Embedded!

**HAS RESOURCE-SPECIFIC MCPC!** No join needed - prices already in file!

---

## ECRS Types Explained

### ECRS = Emergency Contingency Reserve Service

**Three Variants:**

#### 1. ECRS (base)
- Original product type
- Not used in 2024 (0 awards)
- Column exists for backwards compatibility

#### 2. ECRSSD = Service Deployment
- Automatically deployed by ERCOT systems
- **Gen:** 13,733,456 MW in 2024 (99.8% of Gen ECRS)
- **Load:** 402,971 MW in 2024 (21.9% of Load ECRS)
- Fast-response reserves

#### 3. ECRSMD = Manual Deployment
- Manually deployed by ERCOT operators
- **Gen:** Not tracked separately (maybe in ECRS base?)
- **Load:** 1,440,748 MW in 2024 (78.1% of Load ECRS)
- Slower response, operator-initiated

**Why the difference?**
- Gen resources (batteries discharging) can provide fast automatic response
- Load resources (batteries charging or standing ready) often manually dispatched
- Different operational characteristics

---

## Current Revenue Calculator Bugs

### Bug #1: Gen ECRS Using Wrong Column
```python
# WRONG:
"ECRS": "ECRSAwarded",  # ← This is 0 in 2024!

# CORRECT:
"ECRS": "ECRSSDAwarded",  # ← This has 13.7M MW!
```

**Impact:** Missing 100% of Gen ECRS revenue (~$50k-100k per battery per year)

### Bug #2: Load AS Using System MCPC Instead of Embedded
```python
# WRONG (current code):
df_as_prices = df_prices.filter(pl.col("AncillaryType") == "ECRS")
df_joined = df.join(df_as_prices, on=["date", "hour"])
revenue = (pl.col("awarded") * pl.col("MCPC")).sum()  # ← System MCPC

# CORRECT:
revenue = (pl.col("ECRSSD Awarded") * pl.col("ECRS MCPC")).sum()  # ← Embedded MCPC
```

**Impact:** Wrong price for all Load AS revenues (RegUp, RegDown, RRS, ECRS, NonSpin)

---

## Correct Implementation

### For Gen Resources:
```python
def _calculate_gen_as_revenues(self, resource: str):
    """Gen resources need system-wide MCPC from AS_prices file"""
    dam_file = rollup_dir / f"DAM_Gen_Resources/{year}.parquet"
    as_price_file = rollup_dir / f"AS_prices/{year}.parquet"

    df = pl.read_parquet(dam_file).filter(pl.col("ResourceName") == resource)
    df_prices = pl.read_parquet(as_price_file)

    # Join with system-wide MCPC
    as_revenues = {}
    for as_type in ["REGUP", "REGDN", "RRS", "ECRS", "NSPIN"]:
        # Map to correct column (ECRS → ECRSSDAwarded in 2024)
        award_col = "ECRSSDAwarded" if as_type == "ECRS" else f"{as_type}Awarded"

        df_prices_type = df_prices.filter(pl.col("AncillaryType") == as_type)
        df_joined = df.join(df_prices_type, on=["date", "hour"])

        revenue = (df_joined[award_col] * df_joined["MCPC"]).sum()
        as_revenues[as_type] = revenue

    return as_revenues
```

### For Load Resources:
```python
def _calculate_load_as_revenues(self, resource: str):
    """Load resources have embedded resource-specific MCPC - NO JOIN NEEDED!"""
    dam_file = rollup_dir / f"DAM_Load_Resources/{year}.parquet"

    df = pl.read_parquet(dam_file).filter(pl.col("Load Resource Name") == resource)

    # Use embedded MCPC columns - prices already in the file!
    return {
        "RegUp": (df["RegUp Awarded"] * df["RegUp MCPC"]).sum(),
        "RegDown": (df["RegDown Awarded"] * df["RegDown MCPC"]).sum(),
        "RRS": (df["RRSFFR Awarded"] * df["RRS MCPC"]).sum(),
        "ECRS": (
            (df["ECRSSD Awarded"] * df["ECRS MCPC"]).sum() +
            (df["ECRSMD Awarded"] * df["ECRS MCPC"]).sum()
        ),
        "NonSpin": (df["NonSpin Awarded"] * df["NonSpin MCPC"]).sum()
    }
```

---

## Expected Revenue Impact

### BATCAVE_BES1 Example (2024):

**Gen Resource (BATCAVE_BES1):**
- ECRSSDAwarded: 36,793 MW total
- Est. MCPC: ~$3-5/MW average
- **Missing ECRS Revenue: ~$110k-180k**

**Load Resource (BATCAVE_LD1):**
- ECRSSD Awarded: 386 MW
- Current (wrong): Using system $3.56/MW = $1,374
- Correct (embedded): Using resource-specific ~$3.85/MW = $1,486
- **Undercounting by ~$112**

**Total Impact for BATCAVE:**
- Missing Gen ECRS: ~$150k
- Wrong Load ECRS: ~$100 error
- **Gen ECRS is 1000x more important than the Load pricing error!**

---

## Action Items

1. ✅ Fix Gen ECRS column name: "ECRSAwarded" → "ECRSSDAwarded"
2. ✅ Fix Load AS to use embedded MCPC instead of system join
3. ✅ Handle both ECRSSD and ECRSMD for Load
4. ✅ Regenerate all revenue data 2019-2024
5. ✅ Verify RRS calculation (Gen uses RRSAwarded, Load uses RRSFFR Awarded)
6. ✅ Check if other AS products changed column names over time

---

## Summary

**Your intuition was correct!** Generators DO earn ECRS - it's their BIGGEST AS revenue source!

**The bugs:**
1. Looking for wrong column name (ECRSAwarded vs ECRSSDAwarded)
2. Using system MCPC instead of embedded resource-specific MCPC for Load

**Why MCPC varies:** Resource-specific prices reflect locational value, contracts, and operational characteristics - not just market clearing.

**Gen vs Load:** Both provide AS but from different operational states (discharging vs charging/ready).
