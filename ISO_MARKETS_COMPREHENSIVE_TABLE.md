# Comprehensive ISO Market Data Download Table
## All ISOs - Energy, Ancillary Services, Load, Generation, and System Metrics

**Legend**: ✅ = Available/Implemented | ⚠️ = Partial/Working | ❌ = Not Available/Failed | 🔄 = In Progress

---

## ERCOT (Electric Reliability Council of Texas)

### Energy Markets

| Market Type | Resolution | Granularity | Hub Prices | Nodal Prices | Status | Report Code | Notes |
|------------|------------|-------------|------------|--------------|--------|-------------|-------|
| **Day-Ahead (DAM)** | Hourly | Settlement Point | ✅ | ✅ | ✅ Working | NP4-190-CD | All settlement points, hubs, zones |
| **Real-Time (RTM)** | 15-minute | Settlement Point | ✅ | ✅ | ✅ Working | NP6-905-CD | Settlement Point Prices (SPP) |
| **Real-Time (SCED)** | 5-minute | LMP | ✅ | ✅ | ⚠️ Archive only | NP6-788-CD | Requires archive download |

### Ancillary Services (DAM Clearing Prices)

| Service | Type | Resolution | Status | Report Code | Notes |
|---------|------|------------|--------|-------------|-------|
| **Regulation Up** | Frequency Regulation | Hourly | ✅ Working | NP4-188-CD | DAM clearing prices |
| **Regulation Down** | Frequency Regulation | Hourly | ✅ Working | NP4-188-CD | DAM clearing prices |
| **Responsive Reserve (RRS)** | Primary Reserve | Hourly | ✅ Working | NP4-188-CD | Spinning reserve equivalent |
| **Non-Spinning Reserve (NSPIN)** | Secondary Reserve | Hourly | ✅ Working | NP4-188-CD | 10-minute reserve |
| **ECRS** | Contingency Reserve | Hourly | ✅ Working | NP4-188-CD | ERCOT-specific |

### Load & Demand

| Dataset | Resolution | Granularity | Status | Report Code | Notes |
|---------|------------|-------------|--------|-------------|-------|
| **Load Forecast (Forecast Zones)** | Hourly | 7-day forecast | ✅ Working | NP3-565-CD | By model and weather zone |
| **Load Forecast (Weather Zones)** | Hourly | 7-day forecast | ⚠️ Not tested | NP3-566-CD | By model and study area |
| **Actual Load (Weather Zones)** | 5-minute | Actual demand | ❌ Param issue | NP6-345-CD | Endpoint exists, wrong params |
| **Actual Load (Forecast Zones)** | Hourly | Actual demand | ❌ Param issue | NP6-346-CD | Endpoint exists, wrong params |
| **System-Wide Demand** | 5-minute | Total system | ❌ Not found | NP6-322-CD | Workaround: sum zones |
| **Total Load (System)** | Varies | Aggregated | ⚠️ Calculated | - | Calculate from zones |

### Renewable Generation

| Resource | Actual | Forecast | Resolution | Status | Report Code | Notes |
|----------|--------|----------|------------|--------|-------------|-------|
| **Wind Power** | ✅ | ✅ STWPF | Hourly | ✅ Working | NP4-732-CD | System-wide + regional |
| **Solar Power** | ✅ | ✅ STPPF | Hourly | ✅ Working | NP4-745-CD | System-wide + regional, geo regions |

### Fuel Mix & Generation

| Dataset | Resolution | Fuel Types | Status | Report Code | Notes |
|---------|------------|------------|--------|-------------|-------|
| **Fuel Mix** | 15-minute | All fuel types | ❌ Endpoint issue | NP6-787-CD | Alternative: NP3-910-ER |
| **2-Day Gen Summary** | Daily | Aggregated | ⚠️ Alternative | NP3-910-ER | Replaces fuel mix |

### Battery Storage (BESS)

| Dataset | Resolution | Components | Status | Report Code | Notes |
|---------|------------|------------|--------|-------------|-------|
| **60-Day DAM Gen Resources** | Hourly | Discharge awards | ✅ Working | NP3-966-ER | Awards, prices, AS |
| **60-Day DAM Load Resources** | Hourly | Charge awards | ✅ Working | NP3-966-ER | Load side of BESS |
| **60-Day SCED Gen Resources** | 5-minute | Actual dispatch | ✅ Working | NP3-965-ER | Telemetered output, SOC |
| **60-Day SCED Load Resources** | 5-minute | Actual charge | ✅ Working | NP3-965-ER | Charging operations |

### System Metrics

| Metric | Resolution | Status | Report Code | Notes |
|--------|------------|--------|-------------|-------|
| **DAM System Lambda** | Hourly | ⚠️ Not tested | NP4-523-CD | Shadow price, system constraint |
| **Unplanned Outages** | Event-based | ❌ Not found | NP3-233-CD | Alternative: NP1-346-ER |
| **Available Reserves** | 5-minute | ❌ Not configured | - | Not yet implemented |
| **Reserve Margin** | Varies | ❌ Not configured | - | Not yet implemented |

### Curtailment

| Type | Status | Notes |
|------|--------|-------|
| **Generation Curtailment** | ❌ Not configured | Part of outages data |
| **Wind Curtailment** | ❌ Not configured | Can infer from HSL vs actual |
| **Solar Curtailment** | ❌ Not configured | Can infer from HSL vs actual |

---

## CAISO (California ISO)

### Energy Markets

| Market Type | Resolution | Granularity | Hub Prices | Nodal Prices | Status | Endpoint | Notes |
|------------|------------|-------------|------------|--------------|--------|----------|-------|
| **Day-Ahead (DAM)** | Hourly | LMP | ✅ | ✅ | 🔄 Configured | PRC_LMP | Trading hubs, DLAPs, PNodes |
| **Real-Time (RTM)** | 5-minute | LMP | ✅ | ✅ | 🔄 Configured | PRC_INTVL_LMP | 5-minute dispatch |
| **Hour-Ahead (HASP)** | 15-minute | LMP | ✅ | ✅ | 🔄 Configured | PRC_HASP_LMP | Hour-ahead schedule |

### Ancillary Services

| Service | Type | Resolution | Status | Endpoint | Notes |
|---------|------|------------|--------|----------|-------|
| **Regulation Up** | Frequency Regulation | Varies | 🔄 Configured | PRC_AS / PRC_INTVL_AS | DAM and RTM |
| **Regulation Down** | Frequency Regulation | Varies | 🔄 Configured | PRC_AS / PRC_INTVL_AS | DAM and RTM |
| **Spinning Reserve** | Primary Reserve | Varies | 🔄 Configured | PRC_AS / PRC_INTVL_AS | 10-minute |
| **Non-Spinning Reserve** | Secondary Reserve | Varies | 🔄 Configured | PRC_AS / PRC_INTVL_AS | 10-minute |
| **Regulation Mileage Up** | Performance Payment | Varies | 🔄 Configured | AS_MILEAGE_CALC | Movement-based |
| **Regulation Mileage Down** | Performance Payment | Varies | 🔄 Configured | AS_MILEAGE_CALC | Movement-based |

### Load & Demand

| Dataset | Resolution | Status | Notes |
|---------|------------|--------|-------|
| **Total Load** | 5-minute | 🔄 Configured | OASIS API |
| **Load Forecast** | Hourly | 🔄 Configured | Day-ahead forecast |

### Renewable Generation

| Resource | Status | Notes |
|----------|--------|-------|
| **Wind Power** | 🔄 Configured | OASIS API |
| **Solar Power** | 🔄 Configured | OASIS API |
| **Wind Forecast** | 🔄 Configured | Day-ahead |
| **Solar Forecast** | 🔄 Configured | Day-ahead |

### Battery Storage

| Dataset | Status | Notes |
|---------|--------|-------|
| **Battery Discharge/Charge** | ❌ Not configured | Available in OASIS |

---

## NYISO (New York ISO)

### Energy Markets

| Market Type | Resolution | Granularity | Hub Prices | Nodal Prices | Status | Notes |
|------------|------------|-------------|------------|--------------|--------|-------|
| **Day-Ahead (DAM)** | Hourly | LBMP | ✅ | ✅ | 🔄 Configured | Zones and nodes |
| **Real-Time (RTM)** | 5-minute | LBMP | ✅ | ✅ | 🔄 Configured | 5-minute dispatch |

### Ancillary Services

| Service | Type | Status | Notes |
|---------|------|--------|-------|
| **Regulation** | Frequency Regulation | 🔄 Configured | Combined Reg capacity |
| **Spinning Reserve** | 10-Minute Reserve | 🔄 Configured | Synchronized |
| **Non-Spinning Reserve** | 10-Minute Reserve | 🔄 Configured | Offline capability |
| **30-Minute Reserve** | Operating Reserve | 🔄 Configured | NYISO-specific |

### Load & Demand

| Dataset | Resolution | Status | Notes |
|---------|------------|--------|-------|
| **Total Load** | 5-minute | 🔄 Configured | Zonal load |
| **Load Forecast** | Hourly | 🔄 Configured | Day-ahead |

### Renewable Generation

| Resource | Status | Notes |
|----------|--------|-------|
| **Wind Power** | 🔄 Configured | Actual generation |
| **Wind Forecast** | 🔄 Configured | Day-ahead |

---

## ISO-NE (ISO New England)

### Energy Markets

| Market Type | Resolution | Granularity | Hub Prices | Nodal Prices | Status | Notes |
|------------|------------|-------------|------------|--------------|--------|-------|
| **Day-Ahead (DAM)** | Hourly | LMP | ✅ | ✅ | 🔄 Configured | Nodes and zones |
| **Real-Time (RTM)** | 5-minute | LMP | ✅ | ✅ | 🔄 Configured | 5-minute dispatch |

### Ancillary Services

| Service | Type | Status | Notes |
|---------|------|--------|-------|
| **Regulation** | Automatic Gen Control | 🔄 Configured | Real-time balancing |
| **Spinning Reserve** | 10-Minute Reserve | 🔄 Configured | TMSR |
| **Non-Spinning Reserve** | 10-Minute Reserve | 🔄 Configured | TMNSR |
| **30-Minute Reserve** | Operating Reserve | 🔄 Configured | TMOR |

### Load & Demand

| Dataset | Resolution | Status | Notes |
|---------|------------|--------|-------|
| **Total Load** | 5-minute | 🔄 Configured | System load |
| **Load Forecast** | Hourly | 🔄 Configured | Day-ahead |

### Renewable Generation

| Resource | Status | Notes |
|----------|--------|-------|
| **Wind Power** | 🔄 Configured | Actual generation |
| **Solar Power** | 🔄 Configured | Actual generation |

---

## SPP (Southwest Power Pool)

### Energy Markets

| Market Type | Resolution | Status | Notes |
|------------|------------|--------|-------|
| **Day-Ahead (DAM)** | Hourly | 🔄 Configured | Hub and nodal |
| **Real-Time (RTM)** | 5-minute | 🔄 Configured | SCED-based |

### Ancillary Services

| Service | Status | Notes |
|---------|--------|-------|
| **Regulation** | 🔄 Configured | Reg-Up and Reg-Down |
| **Spinning Reserve** | 🔄 Configured | Synchronized |
| **Supplemental Reserve** | 🔄 Configured | 10-minute |

---

## IESO (Ontario, Canada)

### Energy Markets

| Market Type | Resolution | Status | Notes |
|------------|------------|--------|-------|
| **Real-Time (CMSC)** | 5-minute | 🔄 Configured | Constrained Market Schedule |
| **Pre-Dispatch** | Hourly | 🔄 Configured | 3-hour ahead |

### Ancillary Services

| Service | Status | Notes |
|---------|--------|-------|
| **Regulation** | 🔄 Configured | AGC service |
| **Operating Reserve** | 🔄 Configured | 10-minute and 30-minute |

---

## AESO (Alberta, Canada)

### Energy Markets

| Market Type | Resolution | Status | Notes |
|------------|------------|--------|-------|
| **Pool Price** | Hourly | 🔄 Configured | Alberta pool price |
| **Real-Time** | 5-minute | 🔄 Configured | Dispatch pricing |

---

## Summary Status by ISO

### ERCOT - Most Comprehensive ✅
- **Energy**: ✅ DAM (hourly), ✅ RTM (15-min SPP working)
- **Ancillary Services**: ✅ All 5 services (RegUp, RegDown, RRS, NSPIN, ECRS)
- **Load**: ✅ Forecasts working, ⚠️ Actuals need param fixes
- **Renewables**: ✅ Wind/Solar actuals + forecasts
- **Battery**: ✅ Full BESS data (DAM awards, SCED dispatch)
- **Fuel Mix**: ⚠️ Alternative source needed
- **Outages**: ❌ Needs alternative endpoint
- **Reserve Margin**: ❌ Not configured
- **Curtailment**: ❌ Not configured (can infer from HSL data)

### CAISO - Configured, Not Fully Tested 🔄
- **Energy**: 🔄 DAM, RTM, HASP configured
- **Ancillary Services**: 🔄 6 services including mileage
- **Load**: 🔄 Configured
- **Renewables**: 🔄 Configured
- **Battery**: ❌ Not configured

### NYISO - Configured, Not Fully Tested 🔄
- **Energy**: 🔄 DAM, RTM configured
- **Ancillary Services**: 🔄 4 services
- **Load**: 🔄 Configured
- **Renewables**: 🔄 Wind only

### ISO-NE - Configured, Not Fully Tested 🔄
- **Energy**: 🔄 DAM, RTM configured
- **Ancillary Services**: 🔄 4 services
- **Load**: 🔄 Configured
- **Renewables**: 🔄 Wind and Solar

### SPP/IESO/AESO - Basic Configuration 🔄
- **Energy**: 🔄 Basic markets configured
- **Limited ancillary services data**

---

## Data Gaps & Recommendations

### Critical Gaps (Needs Implementation)
1. **Reserve Margin / Available Reserves** - None configured for any ISO
2. **Battery Discharge/Charge** - Only ERCOT has comprehensive BESS data
3. **Curtailment Data** - No direct curtailment tracking

### ERCOT Immediate Fixes Needed
1. ❌ **Actual Load parameter names** - Endpoints exist but wrong params
2. ❌ **System Lambda** - Need to test NP4-523-CD
3. ❌ **Fuel Mix** - Test alternative NP3-910-ER
4. ❌ **Outages** - Test alternative NP1-346-ER

### Recommended Additions
1. **Transmission Constraints** - Shadow prices, binding constraints
2. **Reserve Deployment** - Actual AS deployment events
3. **Interconnection Flows** - Import/export data
4. **Demand Response** - DR curtailment and pricing

---

**Last Updated**: October 11, 2025
**Primary Focus**: ERCOT (most complete implementation)
**Status**: ERCOT RTM prices fixed ✅ - All 3 models can be trained!
