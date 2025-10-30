#!/bin/bash
#
# Automated daily grid data update via Netztransparenz OAuth API
# Downloads: Redispatch, Curtailment
# Run via cron for automated updates
#

set -e

PROJECT_DIR="/home/enrico/projects/power_market_pipeline"
SCRIPT_DIR="$PROJECT_DIR/iso_markets/entso_e"
LOG_DIR="$PROJECT_DIR/logs/grid_data"

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Log file with timestamp
LOG_FILE="$LOG_DIR/grid_data_update_$(date +%Y%m%d_%H%M%S).log"

cd "$PROJECT_DIR"

echo "================================================================================" | tee -a "$LOG_FILE"
echo "Grid Data Automated Daily Update (Redispatch + Curtailment)" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "Date: $(date)" | tee -a "$LOG_FILE"
echo "Log: $LOG_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Run Python download script for last 30 days (ensures we catch any late updates)
echo "Running automated download (last 30 days)..." | tee -a "$LOG_FILE"
uv run python "$SCRIPT_DIR/download_grid_data_incremental.py" --days 30 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "✅ Daily grid data update complete!" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"

# Keep only last 30 days of logs
find "$LOG_DIR" -name "grid_data_update_*.log" -type f -mtime +30 -delete

exit 0
