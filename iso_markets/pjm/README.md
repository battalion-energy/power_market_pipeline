# PJM Data Collection and Processing Pipeline

Complete pipeline for downloading, processing, and analyzing PJM electricity market data.

## Quick Start

```bash
# 1. Set up environment
# Add PJM_API_KEY to your .env file

# 2. Download hub-level data (last 2 years)
python download_da_hub_prices.py --start-date 2023-10-07 --end-date 2025-10-06
python download_rt_hourly_lmps.py --start-date 2023-10-07 --end-date 2025-10-06
python download_da_ancillary_services.py --start-date 2023-10-07 --end-date 2025-10-06

# 3. Convert to Parquet
python convert_to_parquet.py --all-markets --all-years --all-stages

# 4. Setup daily updates
./setup_cron.sh
```

## Scripts Created

### Download Scripts
1. `download_da_hub_prices.py` - Day-ahead hub LMP prices (24 hubs)
2. `download_rt_hourly_lmps.py` - Real-time hourly LMPs (settlement roll-up)
3. `download_da_ancillary_services.py` - Ancillary services pricing
4. `download_nodal_da_lmps.py` - ALL nodal DA LMPs (22,528 pnodes)
5. `download_pnodes.py` - Pnode reference data

### Processing Scripts
6. `convert_to_parquet.py` - CSV → Parquet conversion (4-stage pipeline)
7. `daily_update_pjm.py` - Daily automated updates
8. `setup_cron.sh` - Cron job installation

### Core Module
9. `pjm_api_client.py` - API client with rate limiting

## Documentation

- `README.md` - This file
- `PJM_CURRENT_STATUS.md` - Current progress and data inventory
- `PJM_COMPREHENSIVE_PLAN.md` - Complete strategy
- `ISO_DATA_PIPELINE_STRATEGY.md` - Parent pipeline architecture

See PJM_CURRENT_STATUS.md for current download progress and data volumes.
