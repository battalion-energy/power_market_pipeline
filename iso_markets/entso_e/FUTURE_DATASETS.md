# Additional German Grid Datasets - Future Implementation

**Last Updated:** 2025-10-29

This document lists additional high-value datasets available via the Netztransparenz OAuth API that can be implemented when needed.

---

## Currently Implemented ✅

1. **Day-Ahead Prices** (ENTSO-E API)
2. **reBAP Imbalance Prices** (OAuth, automated daily)
3. **FCR/aFRR/mFRR Capacity & Energy** (Regelleistung.net, automated daily)
4. **Redispatch** (OAuth, automated daily)
5. **Curtailment - Designated & Allocated** (OAuth, automated daily)

---

## Ready to Implement - High Priority

### 1. Value of Avoided Activation (VoAA) ⭐⭐⭐⭐
**Status:** Ready - API endpoint tested
**Endpoint:** `/data/NrvSaldo/VoAA/Qualitaetsgesichert/{dateFrom}/{dateTo}`
**Market Size:** €100-500M/year
**Implementation Time:** 30 minutes

**What it is:** Price signal for avoiding activation of expensive reserves. New revenue stream introduced in 2022.

**Value for BESS:**
- BESS can earn VoAA revenues by providing flexibility
- Incentive payment for reducing reserve activation needs
- Cutting-edge grid economics

**Technical Notes:**
- Requires date range splitting (API error on large ranges)
- Similar format to reBAP (German CSV with semicolon delimiter)
- 15-minute resolution

**Implementation:**
```python
# Add to download_grid_data_auto.py
def download_voaa_chunked(client, start_date, end_date, output_dir):
    # Split into monthly chunks to avoid 500 error
    # Parse German CSV format
    # Save to voaa/ directory
```

---

### 2. Spot Market Prices (All Exchanges) ⭐⭐⭐⭐
**Status:** Ready - API endpoint tested
**Endpoint:** `/data/Spotmarktpreise/{dateFrom}/{dateTo}`
**Market Size:** Benchmark for DA prices
**Implementation Time:** 30 minutes

**What it is:** Volume-weighted spot prices from EPEX, EXAA exchanges

**Value for BESS:**
- Compare with current day-ahead prices
- Identify exchange-specific arbitrage opportunities
- Intraday market dynamics
- Validate ENTSO-E DA price data

**Technical Notes:**
- Requires date range splitting (API error on large ranges)
- Hourly resolution
- Multiple exchanges per timestamp

**Implementation:**
```python
# Add to download_grid_data_auto.py
def download_spot_prices_chunked(client, start_date, end_date, output_dir):
    # Split into monthly chunks
    # Parse multi-exchange format
    # Save to spot_prices/ directory
```

---

### 3. Negative Price Events ⭐⭐⭐
**Status:** Ready
**Endpoints:**
- `/data/NegativePreise/1/{dateFrom}/{dateTo}` (1 hour consecutive)
- `/data/NegativePreise/3/{dateFrom}/{dateTo}` (3 hours consecutive)
- `/data/NegativePreise/4/{dateFrom}/{dateTo}` (4 hours consecutive)
- `/data/NegativePreise/6/{dateFrom}/{dateTo}` (6 hours consecutive)

**Implementation Time:** 45 minutes

**What it is:** Events when prices go negative for sustained periods

**Value for BESS:**
- Perfect charging opportunities (get paid to charge)
- Shows renewable oversupply events
- Revenue optimization indicator
- Risk management (avoid long-duration negative periods)

**Technical Notes:**
- 4 different logic types (1h, 3h, 4h, 6h)
- Daily resolution
- Shows start time, end time, duration, price

**Implementation:**
```python
# Add separate script for analysis use
def download_negative_price_events(client, logic=[1,3,4,6]):
    for hours in logic:
        endpoint = f"/data/NegativePreise/{hours}/{dateFrom}/{dateTo}"
        # Parse events
        # Useful for historical analysis
```

---

### 4. Index Balancing Energy Price (ID-AEP) ⭐⭐⭐
**Status:** Ready
**Endpoint:** `/data/IdAep/{dateFrom}/{dateTo}`
**Market Size:** Alternative imbalance indicator
**Implementation Time:** 30 minutes

**What it is:** Intraday index for imbalance settlement (alternative to reBAP)

**Value for BESS:**
- Real-time price indicator
- Intraday trading signal
- Alternative to reBAP for some applications
- Forward-looking imbalance indicator

**Technical Notes:**
- 15-minute resolution
- Can be compared with reBAP for arbitrage
- Useful for intraday strategy

---

## Lower Priority - Grid Operations

### 5. curative Redispatch (kRD) ⭐⭐⭐
**Status:** Tested - No current data available
**Endpoint:** `/data/VorhaltungkRD/{dateFrom}/{dateTo}`
**Market Size:** Part of redispatch market
**Implementation Time:** 20 minutes

**What it is:** Preventive redispatch reserves held to manage congestion

**Value for BESS:**
- BESS can provide curative redispatch services
- Shows forward-looking grid constraints
- Revenue opportunity

**Technical Notes:**
- May have limited historical data
- Daily resolution
- Check data availability before implementing

---

### 6. Capacity Reserve ⭐⭐⭐
**Status:** Tested - No current data available
**Endpoint:** `/data/Kapazitaetsreserve/{dateFrom}/{dateTo}`
**Market Size:** €50-100M/year
**Implementation Time:** 20 minutes

**What it is:** Strategic reserve for extreme scenarios (last resort power plants)

**Value for BESS:**
- Shows when grid is under extreme stress
- Identifies potential BESS capacity market opportunities
- Forward capacity market indicator
- Energy security context

**Technical Notes:**
- May have limited activation events
- Daily resolution
- Implement when needed for capacity market analysis

---

### 7. Generation Bans (Erzeugungsverbot) ⭐⭐⭐
**Status:** Ready
**Endpoint:** `/data/Erzeugungsverbot/{dateFromUtc}/{dateToUtc}`
**Market Size:** Grid stability indicator
**Implementation Time:** 30 minutes

**What it is:** When TSOs ban generation from specific units due to grid stability

**Value for BESS:**
- Extreme grid stress indicator
- High-value arbitrage opportunities
- Risk management (avoid constrained locations)
- Shows critical grid events

**Technical Notes:**
- Rare events (extreme situations only)
- Daily resolution
- Useful for risk analysis

**Implementation:**
```python
# Useful for historical analysis of extreme events
def download_generation_bans(client, start_date, end_date):
    endpoint = f"/data/Erzeugungsverbot/{dateFromUtc}/{dateToUtc}"
    # Parse ban events
    # Identify high-risk periods
```

---

## Future Priority - Renewable Forecasts

### 8. Wind/Solar Projections ⭐⭐
**Status:** Ready
**Endpoints:**
- `/data/hochrechnung/Wind`
- `/data/hochrechnung/Solar`
- `/data/onlinehochrechnung/Windonshore`
- `/data/onlinehochrechnung/Windoffshore`
- `/data/onlinehochrechnung/Solar`

**Implementation Time:** 1 hour (all 5 endpoints)

**What it is:** TSO forecasts of renewable generation

**Value for BESS:**
- Anticipate price movements
- Optimize charging/discharging schedule
- Price forecasting inputs
- Renewable integration analysis

**Technical Notes:**
- Hourly resolution
- 5 different data streams
- Useful for ML price forecasting models
- Less unique (public data)

**Implementation Priority:** When building price forecasting models

---

### 9. Renewable Marketing Data ⭐⭐
**Status:** Ready
**Endpoints:**
- `/data/vermarktung/DifferenzEinspeiseprognose` - Difference in feed-in forecast
- `/data/vermarktung/InanspruchnahmeAusgleichsenergie` - Utilization of balancing energy
- `/data/vermarktung/UntertaegigeStrommengen` - Intraday electricity volumes
- `/data/vermarktung/VermarktungEpex` - Marketing EPEX
- `/data/vermarktung/VermarktungExaa` - Marketing EXAA
- `/data/vermarktung/VermarktungsSolar` - Marketing Solar
- `/data/vermarktung/VermarktungsWind` - Marketing Wind
- `/data/vermarktung/VermarktungsSonstige` - Marketing Other

**Implementation Time:** 2 hours (all endpoints)

**What it is:** Renewable energy marketing and balancing data

**Value for BESS:**
- Understand renewable marketing strategies
- Identify imbalance patterns
- Intraday market dynamics
- Forecast error analysis

**Implementation Priority:** When analyzing renewable integration in detail

---

### 10. Grid Frequency/Stability Data ⭐⭐⭐⭐
**Status:** Research needed
**Potential Sources:**
- ENTSO-E Transparency Platform
- TSO real-time data
- Grid operation data

**What it is:** Real-time grid frequency, stability indicators

**Value for BESS:**
- FCR activation triggers (frequency deviations)
- Grid stress real-time monitoring
- Optimize reserve provision
- Critical for frequency response services

**Implementation Notes:**
- May require different data source
- High-frequency data (seconds/minutes)
- Large data volumes
- Implement when providing FCR services

---

### 11. Additional NRV-Saldo Data ⭐⭐⭐
**Status:** Ready
**Endpoints:**
- `/data/NrvSaldo/NRVSaldo/Betrieblich` - GCC balance operational
- `/data/NrvSaldo/NRVSaldo/Qualitaetsgesichert` - GCC balance quality-assured
- `/data/NrvSaldo/RZSaldo/Betrieblich` - LFC Area balance operational
- `/data/NrvSaldo/RZSaldo/Qualitaetsgesichert` - LFC Area balance quality-assured
- `/data/NrvSaldo/PRL/Betrieblich` - PRL (k * Delta f) operational
- `/data/NrvSaldo/PRL/Qualitaetsgesichert` - PRL (k * Delta f) quality-assured
- `/data/NrvSaldo/SRLOptimierung/Betrieblich` - aFRR Optimization operational
- `/data/NrvSaldo/SRLOptimierung/Qualitaetsgesichert` - aFRR Optimization quality-assured
- `/data/NrvSaldo/MRLOptimierung/Betrieblich` - mFRR Optimization operational
- `/data/NrvSaldo/MRLOptimierung/Qualitaetsgesichert` - mFRR Optimization quality-assured
- `/data/NrvSaldo/Difference/Betrieblich` - Difference operational
- `/data/NrvSaldo/Difference/Qualitaetsgesichert` - Difference quality-assured
- `/data/NrvSaldo/AbschaltbareLasten/Betrieblich` - Sheddable loads operational
- `/data/NrvSaldo/AbschaltbareLasten/Qualitaetsgesichert` - Sheddable loads quality-assured
- `/data/NrvSaldo/Zusatzmassnahmen/Betrieblich` - Additional measures operational
- `/data/NrvSaldo/Zusatzmassnahmen/Qualitaetsgesichert` - Additional measures quality-assured
- `/data/NrvSaldo/Nothilfe/Betrieblich` - Emergency aid abroad operational
- `/data/NrvSaldo/Nothilfe/Qualitaetsgesichert` - Emergency aid abroad quality-assured

**Implementation Time:** 4-6 hours (all endpoints)

**What it is:** Detailed grid balancing and operational data

**Value for BESS:**
- Deep understanding of grid operations
- Reserve optimization patterns
- System imbalance drivers
- Emergency measure triggers

**Implementation Priority:** When optimizing reserve participation

---

### 12. AEP Module Data ⭐⭐
**Status:** Ready
**Endpoints:**
- `/data/NrvSaldo/AEPModule/Qualitaetsgesichert` - AEP Module 1/2/3
- `/data/NrvSaldo/FinanzielleWirkungAEPModule/Qualitaetsgesichert` - Financial impact
- `/data/NrvSaldo/AepSchaetzer/Betrieblich` - AEP estimator

**Implementation Time:** 1 hour

**What it is:** Imbalance settlement mechanism components

**Value for BESS:**
- Understand imbalance pricing mechanism
- Optimize imbalance trading strategies
- Forecast imbalance prices

**Implementation Priority:** When deep-diving into imbalance settlement

---

### 13. Market Premiums/Values ⭐
**Status:** Ready
**Endpoints:**
- `/data/marktpraemie/{monthFrom}/{yearFrom}/{monthTo}/{yearTo}` - Monthly market values
- `/data/Jahresmarktpraemie` - Annual market values

**Implementation Time:** 30 minutes

**What it is:** Market value factors for renewable energy support

**Value for BESS:**
- Understand renewable economics
- Revenue context for renewable integration
- Market dynamics

**Implementation Priority:** Low - primarily for renewable generators

---

### 14. Traffic Light System ⭐⭐
**Status:** Ready
**Endpoint:** `/data/TrafficLight/{dateFrom}/{dateTo}`

**What it is:** Grid balance "traffic light" indicator (red/yellow/green)

**Value for BESS:**
- Real-time grid stress indicator
- Simple visualization of grid state
- Risk management signal

**Implementation Time:** 20 minutes
**Implementation Priority:** When building monitoring dashboards

---

## Multi-Zone Expansion (Future)

### European Markets - Same OAuth Client! 🌍
All available via same Netztransparenz OAuth credentials:

**Priority 1 Markets:**
- 🇫🇷 **France** - 2nd largest European power market
- 🇳🇱 **Netherlands** - High renewable penetration
- 🇧🇪 **Belgium** - Nuclear + renewables mix
- 🇦🇹 **Austria** - Hydro storage hub
- 🇨🇭 **Switzerland** - Pumped hydro leader

**Priority 2 Markets:**
- 🇩🇰 **Denmark** - Wind power leader (>50% wind)
- 🇳🇴 **Norway** - 100% hydro, price arbitrage
- 🇸🇪 **Sweden** - Nuclear + hydro
- 🇵🇱 **Poland** - Coal transition, growing renewables
- 🇨🇿 **Czech Republic** - Central European hub

**Implementation Time:** 1-2 weeks per market
**Value:** €50-100B/year additional addressable market across Europe

---

## Implementation Priorities

### Immediate (Next Week)
1. **VoAA** - New revenue stream, small data
2. **Spot Prices** - Validation of DA prices
3. **Generation Bans** - Extreme event analysis

### Short Term (Next Month)
4. **Negative Price Events** - Charging opportunity analysis
5. **ID-AEP** - Alternative imbalance indicator

### Medium Term (Next Quarter)
6. **Renewable Forecasts** - For price forecasting models
7. **Additional NRV-Saldo Data** - Deep grid operations

### Long Term (When Needed)
8. **Multi-zone expansion** - European markets
9. **Grid frequency data** - Real-time operations
10. **Market premiums** - Renewable economics context

---

## Technical Notes

### Date Range Handling
Some endpoints error with large date ranges (2019-2025):
- **VoAA:** 500 Internal Server Error
- **Spot Prices:** 400 Bad Request

**Solution:** Split into monthly or quarterly chunks

**Example:**
```python
def download_with_chunking(endpoint_func, start, end, chunk_months=3):
    chunks = split_date_range(start, end, chunk_months)
    all_data = []
    for chunk_start, chunk_end in chunks:
        data = endpoint_func(chunk_start, chunk_end)
        all_data.extend(data)
    return combine_and_deduplicate(all_data)
```

### Data Format Consistency
All Netztransparenz data uses German CSV format:
- Delimiter: `;` (semicolon)
- Decimal: `,` (comma)
- Encoding: UTF-8 with BOM
- Date format: `DD.MM.YYYY`
- Time format: `HH:MM`

Reuse existing parsing function from `download_rebap_auto.py`

### OAuth Token Management
- Token lifetime: 3600 seconds (1 hour)
- Auto-refresh implemented in `NetztransparenzOAuthClient`
- Rate limits: Unknown (haven't hit them yet)
- Same token works for all endpoints ✅

---

## Cost-Benefit Analysis

| Dataset | Implementation Time | Data Size | Client Value | Priority |
|---------|-------------------|-----------|--------------|----------|
| VoAA | 30 min | Small | High | 1 |
| Spot Prices | 30 min | Medium | High | 2 |
| Negative Prices | 45 min | Small | Medium | 3 |
| ID-AEP | 30 min | Medium | Medium | 4 |
| Generation Bans | 30 min | Small | Medium | 5 |
| NRV-Saldo (all) | 4-6 hours | Large | Medium | 6 |
| Renewable Forecasts | 1 hour | Large | Medium | 7 |
| Multi-zone | 1-2 weeks | Very Large | Very High | 8 |

---

## Documentation

**API Documentation:** https://api-portal.netztransparenz.de/
**Swagger UI:** https://extranet.netztransparenz.de/swagger
**OAuth Guide:** `iso_markets/entso_e/OAUTH_SETUP_GUIDE.md`
**Available Datasets:** This file

---

**Last Updated:** 2025-10-29
**Status:** Ready for selective implementation based on client needs
**OAuth Client:** Already configured and working ✅
