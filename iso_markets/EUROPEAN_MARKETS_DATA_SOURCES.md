# European Power Market Data Sources for BESS Operations

**Last Updated:** 2025-10-29

## Summary of Downloaded Data

### ✅ **Germany (DE)** - COMPLETE
**Total Records:** 741,304+ records (2019-2025)
**Market Size:** €50-60B/year
**BESS Potential:** ⭐⭐⭐⭐⭐ (Very High)

**Data Sources:**
1. **ENTSO-E Transparency Platform** - DA prices, ancillary services ✅
2. **Netztransparenz.de OAuth API** - reBAP, redispatch, curtailment ✅

**Downloaded Datasets:**
- Day-Ahead Prices: 61,563 records (2019-2025) via ENTSO-E
- reBAP Imbalance Prices: 237,979 records (2019-2025) via Netztransparenz
- FCR/aFRR/mFRR Capacity & Energy: 241,000+ records via Regelleistung.net
- Redispatch: 69,410 records (2019-2025) via Netztransparenz
- Curtailment (Designated): 37,828 records via Netztransparenz
- Curtailment (Allocated): 37,828 records via Netztransparenz
- **Spot Market Prices: 39,484 records** (2021-2025) via Netztransparenz ✅
- **ID-AEP (Intraday index): 186,820 records** (2019-2025) via Netztransparenz ✅

**Automation:** ✅ Daily updates at 6:00, 6:05, 6:10 AM via cron

**Revenue Streams Covered:** €9-16B/year addressable market

---

### ✅ **Spain (ES)** - PARTIAL
**Total Records:** 231,772+ records (2019-2025)
**Market Size:** €20-25B/year
**BESS Potential:** ⭐⭐⭐⭐⭐ (Very High - 60%+ renewable penetration)

**Data Sources:**
1. **ENTSO-E** - Imbalance prices, generation mix ✅
2. **OMIE** (Operador del Mercado Ibérico) - DA prices ⚠️ NOT YET IMPLEMENTED

**Downloaded Datasets:**
- Imbalance Prices: 81,605 records (2019-2025) via ENTSO-E ✅
- Generation by Type: 150,167 records (2019-2025) via ENTSO-E ✅

**Missing Data:**
- ❌ Day-Ahead Prices - Requires OMIE API
  - OMIE API: https://www.omie.es/es/file-access-list
  - Alternative: https://www.esios.ree.es/ (REE - Red Eléctrica de España)

**Automation:** ⚠️ Not automated yet (manual OMIE API implementation needed)

---

## 🚫 Markets Requiring Local Exchange APIs

**Key Finding:** ENTSO-E Transparency Platform does NOT provide DA prices for most European countries. Each country publishes via local power exchanges.

### **France (FR)**
**Market Size:** €40-45B/year
**BESS Potential:** ⭐⭐⭐⭐ (High)

**Required Data Sources:**
1. **EPEX SPOT** - Day-ahead and intraday prices
   - API: https://www.epexspot.com/en/market-data
   - Requires: Paid subscription or data license
2. **RTE** (Réseau de Transport d'Électricité) - Imbalance, balancing, transparency data
   - API: https://data.rte-france.com/
   - Free registration available
3. **ENTSO-E** - Some ancillary services data (limited)

**Status:** ❌ NOT IMPLEMENTED

---

### **Italy (IT)**
**Market Size:** €30-35B/year
**BESS Potential:** ⭐⭐⭐⭐⭐ (Very High - 6 price zones, high solar)

**Required Data Sources:**
1. **GME** (Gestore dei Mercati Energetici) - DA prices, intraday
   - Website: https://www.mercatoelettrico.org/
   - API: Requires registration and data license
   - Note: Italy has 6 price zones (CNOR, CSUD, NORD, SARD, SICI, SUD)
2. **Terna** - Grid data, generation, imbalance
   - Transparency: https://www.terna.it/it/sistema-elettrico/transparency-report
3. **ENTSO-E** - Some generation data (limited)

**Status:** ❌ NOT IMPLEMENTED

---

### **Netherlands (NL)**
**Market Size:** €10-12B/year
**BESS Potential:** ⭐⭐⭐⭐ (High - 30%+ renewables)

**Required Data Sources:**
1. **EPEX SPOT** or **APX** (now part of EPEX) - DA prices
   - Same as France - paid subscription
2. **TenneT NL** - Dutch TSO data
   - https://www.tennet.eu/electricity-market/
3. **ENTSO-E** - Some ancillary data (limited)

**Status:** ❌ NOT IMPLEMENTED

---

### **Belgium (BE)**
**Market Size:** €8-10B/year
**BESS Potential:** ⭐⭐⭐ (Medium-High)

**Required Data Sources:**
1. **EPEX SPOT** - DA prices
2. **Elia** - Belgian TSO data
   - https://www.elia.be/en/grid-data
   - Open data portal available
3. **ENTSO-E** - Some data (limited)

**Status:** ❌ NOT IMPLEMENTED

---

### **Austria (AT)**
**Market Size:** €8-10B/year
**BESS Potential:** ⭐⭐⭐ (Medium - competes with pumped hydro)

**Required Data Sources:**
1. **EXAA** (Energy Exchange Austria) - DA prices
   - https://www.exaa.at/en/
2. **APG** (Austrian Power Grid) - TSO data
   - https://www.apg.at/en/
3. **ENTSO-E** - Some data (limited)

**Status:** ❌ NOT IMPLEMENTED

---

### **Denmark (DK)**
**Market Size:** €5-6B/year (2 zones)
**BESS Potential:** ⭐⭐⭐⭐⭐ (Very High - 50%+ wind penetration)

**Required Data Sources:**
1. **Nord Pool** - DA and intraday prices (DK1 West, DK2 East)
   - https://www.nordpoolgroup.com/
   - API: https://data.nordpoolgroup.com/
   - Requires: Free registration for basic data
2. **Energinet** - Danish TSO
   - https://en.energinet.dk/
3. **ENTSO-E** - Some data (limited)

**Status:** ❌ NOT IMPLEMENTED

---

### **United Kingdom (GB)**
**Market Size:** €35-40B/year
**BESS Potential:** ⭐⭐⭐⭐⭐ (Very High - 40%+ wind)

**Required Data Sources:**
1. **EPEX SPOT UK** or **N2EX** - DA prices
2. **National Grid ESO** - Imbalance, balancing mechanism
   - https://www.nationalgrideso.com/
   - BMRS (Balancing Mechanism Reporting Service): https://www.bmreports.com/
   - API: https://www.elexon.co.uk/data/
3. **Elexon** - Settlement data

**Note:** UK is NOT on ENTSO-E Transparency Platform (post-Brexit)

**Status:** ❌ NOT IMPLEMENTED

---

## Implementation Priority & Cost-Benefit

| Market | Priority | Implementation Time | Data Cost | BESS Value | Status |
|--------|----------|-------------------|-----------|------------|--------|
| Germany | 1 | DONE | Free | Very High | ✅ Complete |
| Spain (OMIE) | 2 | 1-2 days | Free | Very High | ⚠️ Partial |
| Denmark (Nord Pool) | 3 | 2-3 days | Free (basic) | Very High | ❌ Todo |
| UK (BMRS) | 4 | 3-5 days | Free | Very High | ❌ Todo |
| Italy (GME) | 5 | 3-5 days | Paid? | Very High | ❌ Todo |
| France (EPEX) | 6 | 3-5 days | Paid | High | ❌ Todo |
| Netherlands (EPEX) | 7 | 3-5 days | Paid | High | ❌ Todo |
| Belgium (Elia) | 8 | 2-3 days | Mixed | Medium | ❌ Todo |
| Austria (EXAA) | 9 | 2-3 days | Paid? | Medium | ❌ Todo |

---

## Recommended Next Steps

### Immediate (This Week)
1. **Implement OMIE API for Spain DA prices**
   - Registration: https://www.omie.es/
   - API documentation available
   - Free access (public market data)
   - Impact: Complete Spain dataset (60%+ renewable market)

### Short Term (Next 2 Weeks)
2. **Implement Nord Pool API for Denmark**
   - Free registration: https://data.nordpoolgroup.com/
   - Covers: DK1, DK2, and potentially NO, SE, FI
   - Impact: Access to Nordics (highest wind penetration in Europe)

3. **Implement UK BMRS/Elexon API**
   - Free access: https://www.elexon.co.uk/data/
   - Comprehensive dataset (imbalance, balancing mechanism, DA via N2EX)
   - Impact: 2nd largest European market after Germany

### Medium Term (Next Month)
4. **Evaluate EPEX SPOT data license**
   - Covers: France, Netherlands, Belgium, Luxembourg, Austria, parts of Germany
   - Cost: TBD (contact EPEX for pricing)
   - Impact: Access to 5+ markets with single license

5. **Implement Italy GME API**
   - Registration required
   - 6 price zones = high arbitrage opportunities
   - Cost: TBD

---

## API Implementation Status

### Working APIs ✅
- **ENTSO-E Transparency Platform** (free, registered) ✅
  - API Key: `b8ee80cc-fcf6-4700-ad47-5367dbd79ab7`
  - Good for: Generation data, some imbalance, limited DA prices
  - Works for: Germany (DE_LU), Spain (partial)

- **Netztransparenz.de OAuth** (free, registered) ✅
  - Client ID: `cm_app_ntp_id_df8c6db0401c481d968b155e7b1635c5`
  - Good for: German grid data (reBAP, redispatch, curtailment, spot prices, ID-AEP)
  - Automated daily updates ✅

- **Regelleistung.net** (free, no registration) ✅
  - Good for: German reserve markets (FCR, aFRR, mFRR)
  - Automated daily updates ✅

### Pending APIs ⚠️
- **OMIE** (Spain) - Free registration needed
- **Nord Pool** - Free registration available
- **UK BMRS/Elexon** - Free registration available
- **RTE France** - Free registration available
- **Elia Belgium** - Free/open data

### Unknown/Paid APIs ❌
- **EPEX SPOT** - Likely paid, contact for pricing
- **GME Italy** - Registration required, cost TBD
- **EXAA Austria** - Registration required, cost TBD

---

## Data Coverage Summary

**Current Status (2025-10-29):**

| Market | DA Prices | Imbalance | Reserves | Grid Ops | Generation | Total Records |
|--------|-----------|-----------|----------|----------|------------|---------------|
| **Germany** | ✅ (61K) | ✅ (238K) | ✅ (241K) | ✅ (145K) | ✅ | **741K+** |
| **Spain** | ❌ OMIE | ✅ (82K) | ❌ | ❌ | ✅ (150K) | **232K** |
| France | ❌ EPEX | ❌ | ❌ | ❌ | ❌ | **0** |
| Italy | ❌ GME | ❌ | ❌ | ❌ | ❌ | **0** |
| Netherlands | ❌ EPEX | ❌ | ❌ | ❌ | ❌ | **0** |
| Belgium | ❌ EPEX | ❌ | ❌ | ❌ | ❌ | **0** |
| UK | ❌ N2EX | ❌ | ❌ | ❌ | ❌ | **0** |
| Denmark | ❌ Nord Pool | ❌ | ❌ | ❌ | ❌ | **0** |

**Total Records:** 973,076+ records
**Market Coverage:** €70-85B/year (Germany + partial Spain)
**Target Market:** €200-250B/year (full Europe)

---

## Technical Notes

### Why ENTSO-E DA Prices Don't Work for Most Markets
- **ENTSO-E Transparency Platform** is primarily for generation, load, cross-border flows
- **Day-ahead prices** are published by power exchanges, not TSOs
- Each exchange has its own API:
  - EPEX SPOT → DE, FR, BE, NL, AT, CH
  - OMIE → ES, PT
  - GME → IT
  - Nord Pool → NO, SE, DK, FI, EE, LT, LV
  - N2EX/APX → UK

### Data Availability
- Most power exchanges provide free historical data via web downloads
- Real-time and automated API access often requires paid licenses
- TSO data (imbalance, reserves, grid operations) is typically free

### BESS Revenue Optimization Requirements
**Minimum dataset for each market:**
1. Day-ahead prices (hourly) - Primary revenue signal
2. Imbalance prices (15-min or finer) - Balancing revenue
3. Reserve capacity prices (FCR/aFRR/mFRR) - Capacity payments
4. Reserve energy prices - Activation revenues
5. Generation mix (optional) - Context for forecasting

**Germany is the only market where we have all 5** ✅

---

## Contact Information for API Access

**Free APIs:**
- OMIE (Spain): https://www.omie.es/ → Contact for API access
- Nord Pool: https://data.nordpoolgroup.com/ → Self-service registration
- UK BMRS: https://www.elexon.co.uk/data/ → Self-service registration
- RTE France: https://data.rte-france.com/ → Self-service registration
- Elia Belgium: https://www.elia.be/en/grid-data/open-data → Direct download

**Paid/TBD APIs:**
- EPEX SPOT: https://www.epexspot.com/en/market-data → Contact: marketdata@epexspot.com
- GME Italy: https://www.mercatoelettrico.org/ → Contact for API access
- EXAA Austria: https://www.exaa.at/en/ → Contact for API access

---

**Last Updated:** 2025-10-29
**Status:** Germany complete, Spain partial, other markets require local exchange APIs
**Next Action:** Implement OMIE API for Spain DA prices
