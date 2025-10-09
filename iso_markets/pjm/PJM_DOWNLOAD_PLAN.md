# PJM Data Download Comprehensive Plan

## Overview
Complete historical and ongoing data collection for PJM markets:
- **Day-Ahead LMP Hub Prices** (2019-01-01 to present)
- **Day-Ahead Ancillary Services** (2019-01-01 to present)
- **Real-Time 5-Minute LMP** (2020-10-07 to present, last 5 years)

## Current Status (2025-10-07)

### ✓ Completed
1. **LMP Hub Prices: 2023-10-07 to 2025-10-06**
   - 24 hubs × 8 quarters = 77 CSV files
   - Size: 15MB
   - Location: `/home/enrico/data/PJM_data/csv_files/da_hubs/`

2. **Ancillary Services: 2023-10-07 to 2025-10-06**
   - Status: Running (in progress)
   - 8 quarters of data
   - Location: `/home/enrico/data/PJM_data/csv_files/da_ancillary_services/`

### 🔄 In Progress
- Ancillary Services download (Q2 2025 downloading)

### 📋 Pending Tasks

#### Phase 1: Historical Data Backfill (2019-2023)
1. **LMP Hub Prices: 2019-01-01 to 2023-10-06**
   - Date range: 4 years, 9 months
   - 19 quarters × 24 hubs
   - Estimated time: ~6 hours (with rate limiting)
   - Command: `python download_da_hub_prices.py --start-date 2019-01-01 --end-date 2023-10-06`

2. **Ancillary Services: 2019-01-01 to 2023-10-06**
   - Date range: 4 years, 9 months
   - 19 quarters
   - Estimated time: ~1 hour
   - Command: `python download_da_ancillary_services.py --start-date 2019-01-01 --end-date 2023-10-06`

#### Phase 2: Real-Time Data
3. **Real-Time 5-Min LMP: 2020-10-07 to 2025-10-06**
   - Date range: 5 years
   - WARNING: Much larger dataset (5-min vs hourly = 12x more data)
   - Estimated size: ~500MB-1GB
   - Command: TBD (script to be created)

#### Phase 3: Automated Daily Updates
4. **Create daily update script** (`daily_update_pjm.sh`)
   - Download yesterday's data
   - LMP hubs
   - Ancillary services
   - Real-time data

5. **Setup cron job** (3x daily at 8am, 2pm, 8pm)
   ```cron
   0 8,14,20 * * * /home/enrico/projects/power_market_pipeline/iso_markets/pjm/daily_update_pjm.sh
   ```

6. **Test cron execution**

## Data Specifications

### Day-Ahead Hub LMP Prices
- **Endpoint:** `da_hrl_lmps`
- **Frequency:** Hourly (24 records/day per hub)
- **Hubs:** 24 total
- **Fields:** datetime_beginning_utc, datetime_beginning_ept, pnode_id, pnode_name, type, zone, system_energy_price_da, total_lmp_da, congestion_price_da, marginal_loss_price_da

### Day-Ahead Ancillary Services
- **Endpoint:** `da_ancillary_services`
- **Frequency:** Hourly (varies by service)
- **Services:** RegA, RegD, Sync_Reserve, Non-Sync_Reserve, Primary_Reserve
- **Fields:** datetime_beginning_utc, datetime_beginning_ept, ancillary_service, unit, value

### Real-Time 5-Minute LMP
- **Endpoint:** `rt_fivemin_lmps`
- **Frequency:** 5-minute intervals (288 records/day per pnode)
- **Data Volume:** MUCH LARGER than day-ahead
- **Archive Cutoff:** 186 days (data older than ~6 months has restrictions)

## API Constraints

### Rate Limits
- **Non-member accounts:** 6 requests/minute
- **Member accounts:** 600 requests/minute
- **Current config:** 6 requests/minute

### Data Constraints
- **Max rows per request:** 50,000
- **Max date range per request:** 365 days (actually enforced as ~366)
- **Archived data cutoff:** 731 days (2 years)
  - DA Hub LMPs: archived after 731 days
  - RT 5-min LMPs: archived after 186 days

### Archived Data Restrictions
When requesting data older than cutoff:
- Date range must be within same calendar year
- No custom sorting
- Limited filtering (dates, type, row_is_current, version_nbr only)
- Cannot filter by specific pnode_id for archived data

## Download Strategy

### Quarter-Based Downloads
- Each quarter = ~90 days
- Well under 365-day limit
- Manageable file sizes
- Clean organization

### Rate Limiting
- 6 requests/minute enforced
- Automatic retry on 429 errors
- Wait periods between requests

### Error Handling
- Continue on individual hub failures
- Log all errors
- Retry logic for transient failures

## File Organization

```
/home/enrico/data/PJM_data/
├── csv_files/
│   ├── da_hubs/
│   │   ├── AEP_2023-10-07_2023-12-31.csv
│   │   ├── AEP_2024-01-01_2024-03-31.csv
│   │   └── ...
│   ├── da_ancillary_services/
│   │   ├── ancillary_services_2023-10-07_2023-12-31.csv
│   │   └── ...
│   └── rt_5min_lmps/
│       └── (to be created)
└── combined/
    └── (optional: combined files per hub/service)
```

## Next Steps

1. ✅ Wait for current ancillary services download to complete
2. 📥 Download historical LMP data (2019-2023)
3. 📥 Download historical ancillary data (2019-2023)
4. ✅ Verify data completeness
5. 📝 Create real-time download script
6. 📥 Download 5 years of RT data
7. 📝 Create daily update automation
8. ⚙️ Setup and test cron job
9. 🔄 Monitor automated updates

## Estimated Timeline

- Current ancillary services: ~15 minutes (in progress)
- Historical LMP backfill: ~6 hours
- Historical ancillary backfill: ~1 hour
- RT data download: ~4-6 hours (large dataset)
- Automation setup: ~1 hour
- **Total:** ~12-14 hours of download time + setup

## Notes

- Downloads respect PJM's 6 req/min rate limit
- Some rate limit errors are normal and handled automatically
- Data is saved incrementally (quarter by quarter)
- Each script can be rerun to fill gaps
- Cron job will ensure daily updates going forward
