# PJM Data Download - Comprehensive Plan
**Last Updated:** 2025-10-07

## Overview
Complete data collection for PJM markets including:
1. Hub-level data (24 major hubs)
2. Nodal-level data (all 22,528 pnodes)
3. Ancillary services including frequency regulation
4. Historical backfill to 2019-01-01
5. Parquet file generation (similar to ERCOT structure)
6. Database-ready organization

## Data Hierarchy

### Hub-Level Data (24 Hubs)
Priority: Complete hub data first before nodal data

#### Markets to Download:
1. **Day-Ahead LMP (Hourly)** - `da_hrl_lmps`
   - Status: ✓ 2023-2025 complete
   - Pending: 2019-01-01 to 2023-10-06 (archived data restrictions)

2. **Real-Time Hourly LMP** - `rt_hrl_lmps`
   - Status: 🔄 Currently downloading 2023-2025
   - Pending: 2019-01-01 to 2023-10-06

3. **Day-Ahead Ancillary Services** - `da_ancillary_services`
   - Status: ✓ 2023-2025 complete
   - Pending: 2019-01-01 to 2023-10-06
   - Services: RegA, RegD, Sync_Reserve, Non-Sync_Reserve, Primary_Reserve

4. **Real-Time Ancillary Services** - `rt_ancillary_services` (if exists)
   - Status: ☐ Not started
   - Need to verify endpoint exists

5. **Frequency Regulation** - TBD specific endpoint
   - Status: ☐ Not started
   - Historical: 2019-01-01 to 2025-10-06

### Nodal-Level Data (All 22,528 Pnodes)
⚠️ **MUCH LARGER DATASET** - Start after hub data complete

#### Markets to Download:
1. **Day-Ahead LMP (Hourly)** - `da_hrl_lmps` (no pnode_id filter)
   - ~22,528 pnodes × 24 hours × 365 days × 6 years = massive dataset
   - Estimated: 3.5 BILLION rows for full historical
   - Requires different download strategy (by date range, no pnode filter)

2. **Real-Time Hourly LMP** - `rt_hrl_lmps` (no pnode filter)
   - Same magnitude as DA hourly

3. **Real-Time 5-Minute LMP** - `rt_fivemin_lmps` or `five_min_itsced_lmps`
   - 288 intervals per day (5-min)
   - ~22,528 pnodes × 288 intervals × 365 days × 6 years = 40+ BILLION rows
   - **Largest dataset** - will require staged downloads

4. **Ancillary Services (Nodal)** - if applicable
   - Not all pnodes participate in ancillary markets
   - Need to identify which pnodes are relevant

## API Constraints

### Rate Limits
- Non-member: 6 requests/minute
- Member: 600 requests/minute
- Current: 6 requests/minute

### Data Limits
- Max 50,000 rows per request
- Max 365-day date range per request
- Archived data (>731 days for DA, >186 days for RT 5-min):
  - ❌ Cannot filter by specific pnode_id
  - ❌ Must be within same calendar year
  - ❌ Limited sorting/filtering

## Download Strategy

### Phase 1: Hub-Level Data (Current)
**Estimated Time:** ~12-20 hours total

1. ✓ DA Hub LMP (2023-2025) - COMPLETE
2. 🔄 RT Hourly Hub LMP (2023-2025) - IN PROGRESS
3. ☐ RT Hourly Ancillary Services (2023-2025)
4. ☐ Frequency Regulation (2019-2025)
5. ✓ DA Ancillary Services (2023-2025) - COMPLETE
6. ☐ Historical backfill attempt (2019-2023) - may fail due to archived restrictions

### Phase 2: Nodal Data Strategy
**Estimated Time:** Days to weeks depending on approach

#### Option A: Full Nodal Download (No Pnode Filter)
- Download ALL pnodes together for each date range
- Process quarterly (90-day chunks)
- Store as large files, filter later
- Pros: Works with archived data
- Cons: Massive files, lots of data we may not need

#### Option B: Recent Nodal Download Only (Last 2 Years)
- Download with pnode filters for recent data only
- Focus on most important nodes
- Pros: Smaller dataset, targeted
- Cons: No historical before 2023

#### Recommended: Hybrid Approach
1. Recent data (2023-2025): Download specific nodes of interest
2. Historical data (2019-2023): Download full dataset or skip
3. Focus on hubs + important nodes first

### Phase 3: Data Processing
**Python-based (NOT Rust)**

1. **CSV Organization**
   - Organize by market type and date
   - Ensure consistent column naming
   - Hub names properly mapped

2. **Parquet Conversion** (similar to ERCOT)
   - Review ERCOT Parquet structure
   - Convert CSVs to Parquet
   - Compression and partitioning strategy

3. **Flattened Datasets** (like ERCOT)
   - Nodes as columns
   - Time as rows
   - Separate files for DA, RT hourly, RT 5-min
   - Combined market views

4. **Database Preparation**
   - Standardized schema
   - Proper hub/node metadata
   - Indexing strategy

### Phase 4: Automation
1. Daily update script (all markets)
2. Cron job (3x daily: 8am, 2pm, 8pm)
3. Error handling and logging
4. Data quality checks

## File Organization

```
/home/enrico/data/PJM_data/
├── csv_files/
│   ├── da_hubs/                    # ✓ Complete (2023-2025)
│   ├── da_ancillary_services/      # ✓ Complete (2023-2025)
│   ├── rt_hourly/                  # 🔄 In progress (2023-2025)
│   ├── rt_ancillary_services/      # ☐ Pending
│   ├── frequency_regulation/       # ☐ Pending
│   ├── da_nodal/                   # ☐ Pending (massive)
│   ├── rt_hourly_nodal/            # ☐ Pending (massive)
│   └── rt_5min_nodal/              # ☐ Pending (huge)
├── parquet_files/
│   ├── da_lmp/                     # ☐ Pending
│   ├── rt_hourly/                  # ☐ Pending
│   ├── rt_5min/                    # ☐ Pending
│   └── ancillary_services/         # ☐ Pending
├── flattened/
│   ├── da_lmp_flattened.parquet    # ☐ Pending
│   ├── rt_hourly_flattened.parquet # ☐ Pending
│   └── combined/                   # ☐ Pending
└── reference/
    └── pnodes.csv                  # ✓ Complete (22,528 nodes)
```

## Endpoint Reference

### Known Working Endpoints:
- `da_hrl_lmps` - Day-ahead hourly LMPs
- `rt_hrl_lmps` - Real-time hourly LMPs (settlement roll-up)
- `rt_fivemin_lmps` - Real-time 5-minute LMPs
- `da_ancillary_services` - Day-ahead ancillary services
- `pnode` - Pnode reference data

### Endpoints to Investigate:
- `rt_ancillary_services` - Real-time ancillary (verify exists)
- `five_min_itsced_lmps` - Raw 5-minute IT-SCED LMPs (node-level)
- `rt_fivemin_hrl_lmps` - Aggregated 5-min reporting (verify)
- `rt_fivemin_mnt_lmps` - Aggregated 5-min reporting (verify)
- Frequency regulation specific endpoint

## Estimated Data Volumes

### Hub-Level (24 Hubs, 6 Years)
- DA Hourly: ~1.3M rows → ~500MB CSV, ~50MB Parquet
- RT Hourly: ~1.3M rows → ~500MB CSV, ~50MB Parquet
- RT 5-min: ~15.5M rows → ~6GB CSV, ~500MB Parquet
- Ancillary Services: ~500K rows → ~200MB CSV, ~20MB Parquet

### Nodal-Level (22,528 Pnodes, 6 Years)
- DA Hourly: ~3.5B rows → ~1.5TB CSV, ~150GB Parquet
- RT Hourly: ~3.5B rows → ~1.5TB CSV, ~150GB Parquet
- RT 5-min: ~42B rows → ~18TB CSV, ~1.8TB Parquet

⚠️ **Note:** Nodal-level data will require careful planning and significant storage.

## Next Immediate Steps

1. ☐ Wait for RT hourly hub download to complete (~30 more minutes)
2. ☐ Spot check RT hourly data quality
3. ☐ Download RT ancillary services (if endpoint exists)
4. ☐ Download frequency regulation data
5. ☐ Attempt historical hub backfill (2019-2023)
6. ☐ Review ERCOT Parquet structure
7. ☐ Design nodal download strategy
8. ☐ Create Parquet conversion scripts
9. ☐ Create flattened dataset scripts
10. ☐ Setup automation and cron jobs

## Success Criteria

### Hub Data:
- ✓ All 24 hubs downloaded for all markets
- ✓ Coverage: 2019-01-01 to 2025-10-06 (or explain gaps)
- ✓ Data quality verified
- ✓ Organized for database upload

### Nodal Data:
- ✓ All relevant pnodes downloaded
- ✓ Same time coverage as hub data
- ✓ Compressed as Parquet files
- ✓ Flattened datasets created

### Processing:
- ✓ Parquet files created (ERCOT-style)
- ✓ Flattened datasets with nodes as columns
- ✓ Database-ready format
- ✓ Documentation of schema and organization

### Automation:
- ✓ Daily update script working
- ✓ Cron job tested and running
- ✓ Error handling in place
- ✓ Data quality monitoring

## Notes

- Start with hub data before tackling nodal data
- Nodal data is 1000x larger than hub data
- May need to be selective about which nodes to download
- Archived data restrictions may prevent full historical backfill
- Consider PJM membership for 600 req/min rate limit if doing massive downloads
