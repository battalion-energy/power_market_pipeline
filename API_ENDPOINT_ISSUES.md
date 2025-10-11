# ERCOT API Endpoint Issues - October 11, 2025

## Problem Summary

9 out of 13 datasets failed to download due to 404 errors from the ERCOT Public API.

## Failed Endpoints

| Dataset | Report Code | Endpoint | Error |
|---------|-------------|----------|-------|
| Load Forecast (Forecast Zones) | NP3-565-CD | `lf_by_fzones` | 404 Not Found |
| Load Forecast (Weather Zones) | NP3-566-CD | `lf_by_wzones` | 404 Not Found |
| Actual Load (Weather Zones) | NP6-345-CD | `act_sys_load_by_wzones` | 404 Not Found |
| Actual Load (Forecast Zones) | NP6-346-CD | `act_sys_load_by_fzones` | 404 Not Found |
| Fuel Mix | NP6-787-CD | `fuel_mix` | 404 Not Found |
| System Wide Demand | NP6-322-CD | `act_sys_load_5_min` | 404 Not Found |
| Unplanned Outages | NP3-233-CD | `unpl_res_outages` | 404 Not Found |
| DAM System Lambda | NP4-191-CD | `dam_sys_lambda` | 404 Not Found |
| RTM Prices | NP6-785-CD | `rtm_spp` | 404 Not Found |

## Successful Endpoints (For Reference)

| Dataset | Report Code | Endpoint | Status |
|---------|-------------|----------|--------|
| Wind Power | NP4-732-CD | `wpp_hrly_avrg_actl_fcast` | ✅ Works |
| Solar Power | NP4-745-CD | `spp_hrly_actual_fcast_geo` | ✅ Works |
| DAM Prices | NP4-190-CD | `dam_stlmnt_pnt_prices` | ✅ Works |
| AS Prices | NP4-188-CD | `as_prices` | ✅ Works |

## Root Cause Analysis

### Hypothesis 1: Endpoint Paths Changed
ERCOT may have updated their API structure and endpoint naming conventions.

**Evidence**:
- All failed endpoints use consistent naming patterns
- Successful endpoints also follow patterns
- May indicate systematic API redesign

### Hypothesis 2: Dataset Availability
Some reports may not be available via the Public API.

**Evidence**:
- Certain reports may require different API access
- Historical reports may be archived differently
- Some data may only be available via web portal

### Hypothesis 3: Authentication/Access Level
Failed endpoints may require different authentication or subscription level.

**Evidence**:
- Using standard Public API credentials
- No 401/403 errors (which would indicate auth issues)
- Getting 404 suggests endpoints don't exist, not access denied

## How to Fix

### Step 1: Verify Correct Endpoint Names

**Method A: Use ERCOT API Explorer**
1. Go to https://developer.ercot.com/
2. Navigate to API Specifications
3. Search for report codes (NP3-565-CD, etc.)
4. Find correct endpoint paths

**Method B: Use GridStatus Library**
```python
import gridstatus
ercot = gridstatus.ERCOT()
# GridStatus may have working implementations
```

**Method C: Check ERCOT API Swagger/OpenAPI Docs**
```bash
# Try to fetch API documentation
curl https://api.ercot.com/api-docs
```

### Step 2: Update Downloader Configurations

File: `ercot_ws_downloader/forecast_downloaders.py`

For each failed downloader, update the `get_endpoint()` method:

```python
# Before (BROKEN)
def get_endpoint(self) -> str:
    return "np3-565-cd/lf_by_fzones"

# After (FIX WITH CORRECT ENDPOINT)
def get_endpoint(self) -> str:
    return "np3-565-cd/correct_endpoint_name"  # Replace with actual endpoint
```

### Step 3: Test Each Endpoint

```bash
# Test individual endpoint
python3 << 'EOF'
import asyncio
from ercot_ws_downloader.client import ERCOTWebServiceClient

async def test():
    client = ERCOTWebServiceClient()
    try:
        result = await client._make_request(
            endpoint="np3-565-cd/ENDPOINT_TO_TEST",
            params={"page": 1, "size": 1}
        )
        print(f"✅ Endpoint works!")
        print(f"Sample response: {result[:200]}")
    except Exception as e:
        print(f"❌ Error: {e}")

asyncio.run(test())
EOF
```

### Step 4: Re-download Failed Datasets

```bash
# Re-download specific datasets after fixing endpoints
python download_all_forecast_data.py \
    --datasets load_fzone load_wzone actual_load_wzone \
    --start-date 2023-12-11 \
    --end-date 2025-10-10
```

## Alternative Data Sources

While fixing endpoints, consider these alternatives:

### 1. ERCOT Data Portal (Manual Download)
- URL: http://www.ercot.com/mp/data-products/data-product-details?id=NP3-565-CD
- Download CSVs manually
- Place in correct directories

### 2. GridStatus Library
```python
import gridstatus
ercot = gridstatus.ERCOT()

# Get load data
load = ercot.get_load(
    start="2023-12-11",
    end="2025-10-10"
)
```

### 3. EIA API
```python
# Energy Information Administration
# Has ERCOT load and generation data
import requests
eia_api_key = "YOUR_KEY"
```

### 4. ERCOT Website Archives
- Historical data available as ZIP files
- Download and extract manually
- May be easier than fixing API

## Impact Assessment

### Critical Impact (Must Fix)
- ❌ **RTM Prices** - Required for Model 2 (RT price forecasting)
- ⚠️ **Load Data** - Helpful but not critical (can work around)

### Medium Impact (Nice to Have)
- **Fuel Mix** - Helps understand generation mix
- **System Lambda** - Shadow price for system constraints
- **Outages** - Explains price anomalies

### Low Impact (Optional)
- Additional load granularity
- Duplicate/redundant datasets

## Workarounds (Use While Fixing)

### For Load Data
```python
# Option 1: Use DA prices as proxy for load
# High prices → high load
# Low prices → low load

# Option 2: Calculate net load
net_load = total_generation - wind - solar

# Option 3: Download from alternative source
import gridstatus
ercot = gridstatus.ERCOT()
load = ercot.get_load(...)
```

### For RT Prices
```python
# Option 1: Use GridStatus
rt_prices = ercot.get_lmp_realtime(...)

# Option 2: Download from ERCOT portal
# http://www.ercot.com/mp/data-products/data-product-details?id=NP6-785-CD
```

### For Fuel Mix
```python
# Option 1: Calculate from wind/solar/other data
# Wind + Solar + (Total - Wind - Solar) = approx fuel mix

# Option 2: Use EIA data
# EIA tracks ERCOT generation by fuel type
```

## Testing Protocol

After fixing endpoints, run comprehensive test:

```bash
# Test all endpoints
./test_all_endpoints.sh

# Expected output:
# ✅ Wind Power: OK
# ✅ Solar Power: OK
# ✅ DA Prices: OK
# ✅ AS Prices: OK
# ✅ Load Forecasts: OK  (after fix)
# ✅ Actual Load: OK     (after fix)
# ✅ Fuel Mix: OK        (after fix)
# ✅ RT Prices: OK       (after fix)
# ✅ Outages: OK         (after fix)
```

## Priority

**Phase 1 (Immediate)**:
- ✅ Use current data to train Model 3 (RT spike prediction)
- ✅ Use current data to train Model 1 (DA price forecasting)

**Phase 2 (Next Week)**:
- 🔧 Fix RTM_Prices endpoint (required for Model 2)
- 🔧 Fix Load data endpoints (improves model accuracy)

**Phase 3 (Optional)**:
- Fix remaining endpoints
- Add fuel mix, outages as features
- Compare model performance with/without these features

## Contact/Resources

- **ERCOT Developer Portal**: https://developer.ercot.com/
- **ERCOT Help Desk**: developer.support@ercot.com
- **GridStatus GitHub**: https://github.com/kmax12/gridstatus
- **ERCOT API Docs**: https://developer.ercot.com/api-docs

## Status

- **Last Updated**: October 11, 2025
- **Status**: DOCUMENTED - Ready to proceed with training using available data
- **Next Action**: Research correct endpoint names on ERCOT developer portal

---

**Bottom Line**: We have enough data to start training. Fix endpoints incrementally as time permits.
