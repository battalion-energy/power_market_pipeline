#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="${1:-/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data}"
export PMP_MAX_THREADS="${PMP_MAX_THREADS:-22}"

echo "Using BASE_DIR=$BASE_DIR"
echo "PMP_MAX_THREADS=$PMP_MAX_THREADS"

log() { echo "[$(date +'%F %T')] $*"; }

log "Starting 5-year revenue calculation"
nice -n 5 python run_bess_revenue_5_years.py || true

log "Building exact rollups from hourly/15m time-series"
nice -n 5 python tools/rollup_bess_revenue_timeseries.py --base-dir "$BASE_DIR" --years 2020,2021,2022,2023,2024,2025 --mapping bess_mapping/BESS_UNIFIED_MAPPING_V3_CLARIFIED.csv || true

log "Generating quarterly/monthly charts (exact aggregation)"
nice -n 5 python create_bess_revenue_charts_quarterly_monthly.py || true

log "Generating fleet-by-year bar chart"
nice -n 5 python tools/plot_fleet_by_year.py || true

log "Batch daily plots (2020-2024, all BESS, 5 days/month, hourly+15m+advanced)"
for Y in 2020 2021 2022 2023 2024 2025; do
  nice -n 5 python tools/batch_daily_plots.py --base-dir "$BASE_DIR" --year $Y --all --per-month 5 --with-15min --with-advanced || true
done

log "Pipeline complete"
