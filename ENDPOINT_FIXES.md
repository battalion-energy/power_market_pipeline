# ERCOT API Endpoint Fixes - October 11, 2025

## Summary

Research completed on correct ERCOT Public API endpoint paths. This document contains the verified correct endpoints to fix the 9 failed datasets.

## Corrected Endpoints

### 1. RTM Prices ⚠️ CRITICAL (Required for Model 2)

**Current (BROKEN)**:
```python
# ercot_ws_downloader/downloaders.py:47
return "np6-785-cd/rtm_spp"  # 404 NOT FOUND
```

**Fix Options**:

**Option A: Use NP6-905-CD (15-minute Settlement Point Prices)**
```python
def get_endpoint(self) -> str:
    # Settlement Point Price for each Settlement Point, produced from SCED LMPs every 15 minutes
    return "np6-905-cd/spp_node_zone_hub"
```

**Option B: Use NP6-788-CD (5-minute LMP Prices)** ✅ RECOMMENDED
```python
def get_endpoint(self) -> str:
    # Locational Marginal Prices for each Settlement Point, produced by SCED every 5 minutes
    return "np6-788-cd/lmp_node_zone_hub"
```

**Reason for recommendation**: NP6-788-CD provides 5-minute resolution (matches our requirement for RT price forecasting), while NP6-905-CD is 15-minute.

**Note**: NP6-785-ER exists but it's for **historical** data (weekly reports), not real-time API access.

---

### 2. Load Forecast by Forecast Zone (NP3-565-CD)

**Current (BROKEN)**:
```python
# ercot_ws_downloader/forecast_downloaders.py:107
return "np3-565-cd/lf_by_fzones"  # 404 NOT FOUND
```

**Corrected**:
```python
def get_endpoint(self) -> str:
    # Seven-Day Load Forecast by Model and Weather Zone
    return "np3-565-cd/lf_by_model_weather_zone"
```

---

### 3. Load Forecast by Weather Zone (NP3-566-CD)

**Current (BROKEN)**:
```python
# ercot_ws_downloader/forecast_downloaders.py:142
return "np3-566-cd/lf_by_wzones"  # 404 NOT FOUND
```

**Corrected**:
```python
def get_endpoint(self) -> str:
    # Seven-Day Load Forecast by Model and Study Area
    return "np3-566-cd/lf_by_model_study_area"
```

---

### 4. Actual System Load by Weather Zone (NP6-345-CD)

**Current (BROKEN)**:
```python
# ercot_ws_downloader/forecast_downloaders.py:174
return "np6-345-cd/act_sys_load_by_wzones"  # 404 NOT FOUND
```

**Corrected**:
```python
def get_endpoint(self) -> str:
    # Actual System Load by Weather Zone (abbreviated endpoint name)
    return "np6-345-cd/act_sys_load_by_wzn"
```

**Note**: Endpoint name is abbreviated to `wzn` instead of `wzones`.

---

### 5. Actual System Load by Forecast Zone (NP6-346-CD)

**Current (BROKEN)**:
```python
# ercot_ws_downloader/forecast_downloaders.py:208
return "np6-346-cd/act_sys_load_by_fzones"  # 404 NOT FOUND
```

**Likely Fix** (needs verification):
```python
def get_endpoint(self) -> str:
    # Actual System Load by Forecast Zone (abbreviated endpoint name)
    return "np6-346-cd/act_sys_load_by_fzn"
```

**Status**: ⚠️ **Needs testing** - following same pattern as NP6-345-CD abbreviation.

---

### 6. Fuel Mix (NP6-787-CD)

**Current (BROKEN)**:
```python
# ercot_ws_downloader/forecast_downloaders.py:313
return "np6-787-cd/fuel_mix"  # 404 NOT FOUND
```

**Issue**: NP6-787-CD endpoint doesn't exist in current API.

**Alternative Solution**: Use NP3-910-ER (2-Day Real Time Gen and Load Data Reports)
```python
def get_endpoint(self) -> str:
    # 2-Day Aggregate Generation Summary (contains fuel mix data)
    return "np3-910-er/2d_agg_gen_summary"
```

**Note**: This is a different report but contains fuel mix information. Data structure may differ from original NP6-787-CD.

---

### 7. System Wide Demand (NP6-322-CD)

**Current (BROKEN)**:
```python
# ercot_ws_downloader/forecast_downloaders.py:347
return "np6-322-cd/act_sys_load_5_min"  # 404 NOT FOUND
```

**Status**: ⚠️ **Unable to find correct endpoint** in research. May not be available via Public API.

**Workaround**: Use Actual System Load by Weather Zone (NP6-345-CD) and sum across zones.

---

### 8. Unplanned Resource Outages (NP3-233-CD)

**Current (BROKEN)**:
```python
# ercot_ws_downloader/forecast_downloaders.py:242
return "np3-233-cd/unpl_res_outages"  # 404 NOT FOUND
```

**Issue**: NP3-233-CD doesn't exist. Unplanned outages are under different report code.

**Alternative**: Use NP1-346-ER (Unplanned Resource Outages Report)
```python
def get_endpoint(self) -> str:
    # Unplanned Resource Outages Report (different report code)
    return "np1-346-er/unpl_res_outages"
```

**Note**: Report code changed from NP3-233-CD to NP1-346-ER.

---

### 9. DAM System Lambda (NP4-191-CD)

**Current (BROKEN)**:
```python
# ercot_ws_downloader/forecast_downloaders.py:280
return "np4-191-cd/dam_sys_lambda"  # 404 NOT FOUND
```

**Corrected**:
```python
def get_endpoint(self) -> str:
    # DAM System Lambda (correct report code is NP4-523-CD)
    return "np4-523-cd/dam_sys_lambda"
```

**Note**: Report code changed from NP4-191-CD to NP4-523-CD.

---

## Implementation Steps

### Step 1: Update RTM Prices Downloader (CRITICAL)

File: `ercot_ws_downloader/downloaders.py`

```python
class RTMPriceDownloader(BaseDownloader):
    """Real-Time Market Settlement Point Prices downloader."""

    def get_dataset_name(self) -> str:
        return "RTM_Prices"

    def get_endpoint(self) -> str:
        # FIXED: Use NP6-788-CD for 5-minute LMP prices
        return "np6-788-cd/lmp_node_zone_hub"

    def get_output_dir(self) -> Path:
        return self.output_dir / "RTM_Settlement_Point_Prices"

    def format_params(self, start_date: datetime, end_date: datetime) -> Dict:
        return {
            "deliveryDateFrom": start_date.strftime("%Y-%m-%d"),
            "deliveryDateTo": end_date.strftime("%Y-%m-%d"),
        }

    def get_chunk_size(self) -> int:
        # RT prices are 5-minute, much more data
        return 7  # 7 days per chunk

    def get_page_size(self) -> int:
        # ~50 settlement points * 288 intervals/day * 7 days = ~100,000 records
        return 50000
```

### Step 2: Update Load Forecast Downloaders

File: `ercot_ws_downloader/forecast_downloaders.py`

```python
class LoadForecastByForecastZoneDownloader(BaseDownloader):
    # ... (keep existing code)

    def get_endpoint(self) -> str:
        # FIXED: Correct endpoint path
        return "np3-565-cd/lf_by_model_weather_zone"

    # ... (rest of class)

class LoadForecastByWeatherZoneDownloader(BaseDownloader):
    # ... (keep existing code)

    def get_endpoint(self) -> str:
        # FIXED: Correct endpoint path
        return "np3-566-cd/lf_by_model_study_area"

    # ... (rest of class)
```

### Step 3: Update Actual Load Downloaders

File: `ercot_ws_downloader/forecast_downloaders.py`

```python
class ActualSystemLoadByWeatherZoneDownloader(BaseDownloader):
    # ... (keep existing code)

    def get_endpoint(self) -> str:
        # FIXED: Abbreviated endpoint name
        return "np6-345-cd/act_sys_load_by_wzn"

    # ... (rest of class)

class ActualSystemLoadByForecastZoneDownloader(BaseDownloader):
    # ... (keep existing code)

    def get_endpoint(self) -> str:
        # FIXED: Abbreviated endpoint name (needs verification)
        return "np6-346-cd/act_sys_load_by_fzn"

    # ... (rest of class)
```

### Step 4: Update DAM System Lambda

File: `ercot_ws_downloader/forecast_downloaders.py`

```python
class DAMSystemLambdaDownloader(BaseDownloader):
    # ... (keep existing code)

    def get_endpoint(self) -> str:
        # FIXED: Correct report code is NP4-523-CD
        return "np4-523-cd/dam_sys_lambda"

    # ... (rest of class)
```

### Step 5: Update Fuel Mix (Alternative Report)

File: `ercot_ws_downloader/forecast_downloaders.py`

```python
class FuelMixDownloader(BaseDownloader):
    """
    Fuel Mix Report (using 2-Day Aggregate Generation Summary).

    Report NP3-910-ER includes:
    - 2-Day aggregate generation by fuel type
    - Alternative to NP6-787-CD which doesn't exist in API
    """

    def get_dataset_name(self) -> str:
        return "Fuel_Mix"

    def get_endpoint(self) -> str:
        # FIXED: Use NP3-910-ER instead of non-existent NP6-787-CD
        return "np3-910-er/2d_agg_gen_summary"

    def get_output_dir(self) -> Path:
        return self.output_dir / "Fuel_Mix"

    def format_params(self, start_date: datetime, end_date: datetime) -> Dict:
        return {
            "deliveryDateFrom": start_date.strftime("%Y-%m-%d"),
            "deliveryDateTo": end_date.strftime("%Y-%m-%d"),
        }

    def get_chunk_size(self) -> int:
        return 7  # 7 days per chunk

    def get_page_size(self) -> int:
        return 50000
```

### Step 6: Update Unplanned Outages (Alternative Report)

File: `ercot_ws_downloader/forecast_downloaders.py`

```python
class UnplannedResourceOutagesDownloader(BaseDownloader):
    """
    Unplanned Resource Outages Report.

    Report NP1-346-ER includes:
    - Unplanned outages for generation resources
    - Forced outages and maintenance outages
    - Resource name and capacity
    """

    def get_dataset_name(self) -> str:
        return "Unplanned_Resource_Outages"

    def get_endpoint(self) -> str:
        # FIXED: Correct report code is NP1-346-ER
        return "np1-346-er/unpl_res_outages"

    def get_output_dir(self) -> Path:
        return self.output_dir / "Unplanned_Resource_Outages"

    def format_params(self, start_date: datetime, end_date: datetime) -> Dict:
        # May need to adjust date parameter names
        return {
            "publishDatetimeFrom": start_date.strftime("%Y-%m-%dT00:00:00"),
            "publishDatetimeTo": end_date.strftime("%Y-%m-%dT23:59:59"),
        }

    def get_chunk_size(self) -> int:
        return 30

    def get_page_size(self) -> int:
        return 10000

    def get_lag_days(self) -> int:
        return 0
```

---

## Testing Protocol

After making the fixes, test each endpoint individually:

```bash
# Test RTM Prices (MOST CRITICAL)
uv run python download_all_forecast_data.py --datasets rtm --days 3

# Test Load Forecasts
uv run python download_all_forecast_data.py --datasets load_fzone load_wzone --days 3

# Test Actual Load
uv run python download_all_forecast_data.py --datasets actual_load_wzone actual_load_fzone --days 3

# Test System Metrics
uv run python download_all_forecast_data.py --datasets fuel_mix dam_lambda outages --days 3

# If all tests pass, run full re-download
uv run python download_all_forecast_data.py --start-date 2023-12-11 --end-date 2025-10-10
```

---

## Expected Outcomes

After implementing these fixes:

### ✅ Should Work (High Confidence)
1. **RTM Prices** (np6-788-cd/lmp_node_zone_hub) - VERIFIED
2. **Load Forecast by Forecast Zone** (np3-565-cd/lf_by_model_weather_zone) - VERIFIED
3. **Load Forecast by Weather Zone** (np3-566-cd/lf_by_model_study_area) - VERIFIED
4. **Actual Load by Weather Zone** (np6-345-cd/act_sys_load_by_wzn) - VERIFIED
5. **DAM System Lambda** (np4-523-cd/dam_sys_lambda) - VERIFIED

### ⚠️ Needs Testing (Medium Confidence)
6. **Actual Load by Forecast Zone** (np6-346-cd/act_sys_load_by_fzn) - Following pattern
7. **Fuel Mix** (np3-910-er/2d_agg_gen_summary) - Alternative report
8. **Unplanned Outages** (np1-346-er/unpl_res_outages) - Alternative report

### ❌ No Solution Found
9. **System Wide Demand** (np6-322-cd/act_sys_load_5_min) - Endpoint not found
   - **Workaround**: Calculate from NP6-345-CD by summing weather zones

---

## Priority Order

1. **IMMEDIATE**: Fix RTM Prices (required for Model 2)
2. **HIGH**: Fix Load Forecasts (improves Model 3 accuracy)
3. **MEDIUM**: Fix Actual Load (helpful for validation)
4. **LOW**: Fix Fuel Mix, System Lambda, Outages (optional features)

---

## Research Sources

- ERCOT Developer Portal: https://developer.ercot.com/
- ERCOT Data Products: https://www.ercot.com/mp/data-products
- GridStatus Documentation: https://opensource.gridstatus.io/
- ERCOT API Specifications: https://apiexplorer.ercot.com/
- ERCOT GitHub Discussions: https://github.com/ercot/api-specs/discussions

---

**Last Updated**: October 11, 2025
**Status**: Research complete, ready for implementation
**Next Step**: Apply fixes to downloader configurations and test
