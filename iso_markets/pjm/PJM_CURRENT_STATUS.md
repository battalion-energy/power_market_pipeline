# PJM Data Download - Current Status Report
**Date:** 2025-10-07
**Time:** 16:30 ET

## ✅ Completed Tasks

### 1. Day-Ahead Hub LMP Prices (2023-10-07 to 2025-10-06)
- **Status:** ✓ Complete
- **Files:** 77 CSV files
- **Size:** 15MB
- **Hubs:** 24 major trading hubs
- **Location:** `/home/enrico/data/PJM_data/csv_files/da_hubs/`
- **Coverage:** 2 years of recent data

### 2. Day-Ahead Ancillary Services (2023-10-07 to 2025-10-06)
- **Status:** ✓ Complete
- **Files:** 9 CSV files
- **Size:** 7.6MB
- **Records:** ~77,000 records
- **Services:** RegA, RegD, Sync_Reserve, Non-Sync_Reserve, Primary_Reserve
- **Location:** `/home/enrico/data/PJM_data/csv_files/da_ancillary_services/`
- **Coverage:** 2 years of recent data

### 3. PJM Pnodes Reference Data
- **Status:** ✓ Complete
- **Total Pnodes:** 22,528
- **File:** `/home/enrico/data/PJM_data/reference/pnodes.csv`
- **Breakdown:**
  - BUS: 21,009
  - AGGREGATE: 1,517
  - LOCALE: 2
- **Subtypes:**
  - LOAD: 16,357
  - GEN: 4,296
  - AGGREGATE: 1,246
  - HUB: 12 (trading hubs)
  - ZONE: 34 (transmission zones)

### 4. Real-Time Hourly Hub LMP Prices (2023-10-07 to 2025-10-06)
- **Status:** ✓ Complete
- **Files:** 74 CSV files
- **Size:** 15MB
- **Hubs with data:** 8 (AEP, APS, BGE, DAY, DEOK, DOMINION, DPL, RECO)
- **Endpoint:** `rt_hrl_lmps` (settlement roll-up of 5-minute runs)
- **Location:** `/home/enrico/data/PJM_data/csv_files/rt_hourly/`
- **Data Quality:** ✓ Verified - 24 hours × 9 quarters per hub
- **Note:** Many hubs don't report RT hourly data (normal behavior)

### 5. Parquet Conversion Pipeline (4 Stages)
- **Status:** ✓ Complete
- **Script:** `convert_to_parquet.py`
- **Implementation:** Python-based, follows ERCOT-style architecture
- **Stages:**
  - Stage 1: CSV organization ✓ (already organized)
  - Stage 2: Raw Parquet conversion ✓ (year-partitioned, type-enforced)
  - Stage 3: Flattened Parquet ✓ (wide format, nodes as columns)
  - Stage 4: Combined datasets ✓ (DA + RT + AS joins)
- **Output:**
  - 2023: 2,064 timestamps × 24 columns (0.4 MB)
  - 2024: 8,783 timestamps × 24 columns (1.5 MB)
  - 2025: 6,695 timestamps × 22 columns (1.1 MB)
- **Location:** `/home/enrico/data/PJM_data/parquet_files/` & `flattened/` & `combined/`

### 6. Automation Scripts
- **Daily Update Script:** ✓ Created and tested (`daily_update_pjm.py`)
  - Downloads yesterday's data for DA LMP, RT hourly, ancillary services
  - Handles all 24 hubs automatically
  - Proper rate limiting (6 req/min)
  - Error handling for missing data
  - Test run successful: 28 files created for 2025-10-06
- **Cron Setup Script:** ✓ Created (`setup_cron.sh`)
  - Installs cron job for 3x daily execution (8am, 2pm, 8pm ET)
  - Automated logging to `logs/daily_update.log`
  - Safe installation with duplicate detection
- **Nodal Download Scripts:** ✓ Created (`download_nodal_da_lmps.py`)
  - Downloads ALL 22,528 pnodes without filtering (required for archived data)
  - Quarter-based downloads with API limit warnings

## 🔄 In Progress Tasks

None currently running

## 📋 Pending Tasks - Hub Level Data

### Immediate Next Steps (Hub Data)
1. ✅ Complete RT hourly hub download - DONE
2. ☐ Download RT hourly ancillary services (if endpoint exists)
3. ☐ Download frequency regulation data (2019-01-01 to 2025-10-06)
4. ☐ Attempt historical backfill for all hub markets (2019-2023)
   - ⚠️ **BLOCKED:** Network connectivity issue (DNS resolution failure for api.pjm.com)
   - Retry when network is restored

### Historical Hub Data Challenges (2019-2023)
**Problem:** Data older than 731 days is "archived" with restrictions:
- ❌ Cannot filter by specific pnode_id for archived data
- ❌ Date range must be within same calendar year
- ❌ Limited filtering options

**Current Issue:**
- ⚠️ DNS resolution failure for api.pjm.com (network connectivity)
- Historical download script ready, will run when network restored

**Options:**
1. Download ALL pnodes together for historical period (massive dataset)
2. Focus on recent data only (2023-2025) ✓ Current approach completed
3. Use PJM bulk historical downloads if available

## 📋 Pending Tasks - Nodal Level Data

### Large-Scale Nodal Downloads
⚠️ **WARNING:** Nodal data is ~1000x larger than hub data

1. ✅ Create nodal download scripts (download ALL pnodes, no filtering) - DONE
2. ☐ Download all nodal DA LMP (22,528 pnodes) - ready to run
3. ☐ Download all nodal RT hourly LMP - script needs creation
4. ☐ Download all nodal RT 5-minute LMP (MASSIVE - 40B+ rows for 6 years)
5. ☐ Download nodal ancillary services (subset of pnodes)

**Strategy for Nodal Data:**
- Download by date range WITHOUT pnode_id filter (required for archived data)
- Process quarterly chunks (90 days)
- Store as large files, filter later
- Consider recent data only (2023-2025) vs full historical
- ⚠️ 1 day of all nodes = ~540K rows (exceeds 50K API limit, will be truncated)

## 📋 Pending Tasks - Data Processing

### Parquet Conversion (Python-based, ERCOT-style)
Following 4-stage pipeline from ISO_DATA_PIPELINE_STRATEGY.md:

1. ✅ **Stage 1: CSV Organization** - DONE
   - Already organized in csv_files/ directories

2. ✅ **Stage 2: Raw Parquet Conversion** - DONE
   - Convert CSVs to year-partitioned Parquet ✓
   - Enforce Float64 for all price columns ✓
   - Compression (Snappy) ✓

3. ✅ **Stage 3: Flattened Parquet** - DONE
   - Transform to wide format (nodes as columns) ✓
   - Time as rows ✓
   - Separate files for DA, RT hourly ✓
   - Fixed duplicate handling ✓

4. ✅ **Stage 4: Combined Datasets** - DONE
   - Multi-market combinations (DA + AS + RT) ✓
   - Time-aligned joins ✓
   - Analysis-ready format ✓

### Database Preparation
1. ☐ Organize data for database upload
2. ☐ Ensure proper hub names and columns
3. ☐ Create standardized schema
4. ☐ Add metadata and indexing

## 📋 Pending Tasks - Automation

1. ✅ Create daily update script (all data types) - DONE & TESTED
2. ✅ Create cron setup script (3x daily: 8am, 2pm, 8pm) - DONE
3. ☐ Install cron job (ready to install)
4. ✅ Add error handling and logging - DONE
5. ☐ Data quality monitoring

## 📊 Data Summary

### Current Storage
```
/home/enrico/data/PJM_data/
├── csv_files/
│   ├── da_hubs/              15MB, 77 files ✓
│   ├── da_ancillary_services/ 7.6MB, 9 files ✓
│   ├── rt_hourly/            15MB, 74 files ✓
│   ├── rt_ancillary_services/ (pending)
│   ├── frequency_regulation/ (pending)
│   ├── da_nodal/            (pending - will be massive)
│   ├── rt_hourly_nodal/     (pending - will be massive)
│   └── rt_5min_nodal/       (pending - HUGE)
├── parquet_files/           ~3MB (da_hubs, rt_hourly, da_ancillary_services) ✓
├── flattened/               ~3MB (da_hubs, rt_hourly) ✓
├── combined/                ~3MB (yearly combined datasets) ✓
└── reference/
    └── pnodes.csv            ~2MB, 22,528 rows ✓
```

**Total Data Volume:** ~40MB CSV + ~10MB Parquet = ~50MB (hub-level only, 2 years)

### Coverage Summary
- **Hub-Level DA LMP:** 2023-10-07 to 2025-10-06 ✓ COMPLETE
- **Hub-Level DA Ancillary:** 2023-10-07 to 2025-10-06 ✓ COMPLETE
- **Hub-Level RT Hourly:** 2023-10-07 to 2025-10-06 ✓ COMPLETE
- **Hub-Level RT Ancillary:** Not started
- **Frequency Regulation:** Not started
- **Nodal-Level Data:** Not started (scripts ready)
- **Historical (2019-2023):** Blocked by network connectivity
- **Parquet Pipeline:** ✓ COMPLETE (all 4 stages)
- **Automation:** ✓ Scripts ready (cron not yet installed)

## 🔧 Scripts Created

### Download Scripts
1. `download_da_hub_prices.py` - Day-ahead hub LMP prices ✓
2. `download_rt_hourly_lmps.py` - Real-time hourly hub LMPs ✓
3. `download_da_ancillary_services.py` - Day-ahead ancillary services ✓
4. `download_nodal_da_lmps.py` - ALL nodal DA LMPs (22,528 pnodes) ✓ NEW
5. `download_pnodes.py` - Pnode reference data ✓
6. `pjm_api_client.py` - API client with rate limiting (6 req/min) ✓
7. `test_api_key.py` - Test API connection ✓

### Processing Scripts
1. `convert_to_parquet.py` - 4-stage Parquet pipeline ✓ NEW
   - Stage 2: Raw Parquet conversion (year-partitioned)
   - Stage 3: Flattened transformation (wide format)
   - Stage 4: Combined datasets (multi-market joins)

### Automation Scripts
1. `daily_update_pjm.py` - Daily automated data collection ✓ NEW
   - Updates DA LMP, RT hourly, ancillary services
   - Handles all 24 hubs automatically
   - Rate limiting and error handling
2. `setup_cron.sh` - Cron job installation ✓ NEW
   - 3x daily execution (8am, 2pm, 8pm ET)
   - Automated logging

### API Client Methods
- `get_day_ahead_lmps()` - DA hourly LMPs ✓
- `get_rt_hourly_lmps()` - RT hourly LMPs ✓
- `get_rt_fivemin_lmps()` - RT 5-minute LMPs (not yet used)
- `get_ancillary_services()` - DA ancillary services ✓
- `get_pnodes()` - Pnode reference data ✓

## 📝 Documentation Files

1. `README.md` - Quick start guide and script reference ✓ NEW
2. `PJM_CURRENT_STATUS.md` - This file (current status) ✓ UPDATED
3. `PJM_DOWNLOAD_PLAN.md` - Original download plan
4. `PJM_COMPREHENSIVE_PLAN.md` - Comprehensive plan with nodal data
5. API documentation files (txt/pdf in directory)

## 💡 Next Immediate Steps

### Completed Today ✅
1. ✅ RT hourly hub download complete (74 files, 15MB)
2. ✅ Verified RT hourly data quality (24 hours × 9 quarters)
3. ✅ Created Parquet conversion pipeline (all 4 stages)
4. ✅ Tested full pipeline on existing data
5. ✅ Created daily update script and tested successfully
6. ✅ Created cron setup script
7. ✅ Created nodal download scripts

### Remaining Tasks
1. ☐ Install cron job for automated daily updates
2. ☐ Investigate RT ancillary services endpoint
3. ☐ Investigate frequency regulation endpoint
4. ☐ Historical hub backfill (2019-2023) - blocked by network
5. ☐ Download nodal DA LMP data (when network restored)
6. ☐ Create nodal RT download scripts
7. ☐ Database preparation and schema mapping

## 📈 Estimated Data Volumes

### Hub-Level (24 Hubs, 2 Years Recent)
- DA Hourly: ~400K rows → ~150MB CSV, ~15MB Parquet
- RT Hourly: ~400K rows → ~150MB CSV, ~15MB Parquet
- RT 5-min: ~5M rows → ~2GB CSV, ~150MB Parquet
- Ancillary Services: ~70K rows → ~30MB CSV, ~3MB Parquet

### Nodal-Level (22,528 Pnodes, 2 Years Recent)
- DA Hourly: ~400M rows → ~150GB CSV, ~15GB Parquet
- RT Hourly: ~400M rows → ~150GB CSV, ~15GB Parquet
- RT 5-min: ~5B rows → ~2TB CSV, ~200GB Parquet

### Nodal-Level (22,528 Pnodes, 6 Years Full)
- DA Hourly: ~3.5B rows → ~1.5TB CSV, ~150GB Parquet
- RT Hourly: ~3.5B rows → ~1.5TB CSV, ~150GB Parquet
- RT 5-min: ~42B rows → ~18TB CSV, ~1.8TB Parquet

## ⚙️ System Configuration

### API Settings
- Rate limit: 6 requests/minute (non-member)
- Max rows per request: 50,000
- Max date range: 365 days
- Archived data cutoff: 731 days (DA), 186 days (RT 5-min)

### Download Strategy
- Quarter-based downloads (90-day chunks)
- Automatic rate limiting with wait periods
- Retry logic for transient failures
- Continues on individual hub failures

### Network Requirements
- WiFi must be OFF to reach api.pjm.com (DNS issue)
- Uses ethernet connection

## 📝 Notes

- All hub-level downloads working correctly
- RT hourly data quality verified
- Nodal downloads will require different approach (no pnode filtering)
- Historical data (2019-2023) may require bulk download approach
- Consider PJM membership for 600 req/min if doing massive nodal downloads
- Python-based processing per user requirement (not Rust)
- Follow ERCOT-style 4-stage Parquet pipeline
