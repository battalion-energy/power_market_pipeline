#!/bin/bash
#
# Automated daily reBAP data update via Netztransparenz OAuth API
# Run this script via cron for automated updates
#

set -e

PROJECT_DIR="/home/enrico/projects/power_market_pipeline"
SCRIPT_DIR="$PROJECT_DIR/iso_markets/entso_e"
LOG_DIR="$PROJECT_DIR/logs/rebap"

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Log file with timestamp
LOG_FILE="$LOG_DIR/rebap_update_$(date +%Y%m%d_%H%M%S).log"

cd "$PROJECT_DIR"

echo "================================================================================" | tee -a "$LOG_FILE"
echo "reBAP Automated Daily Update" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "Date: $(date)" | tee -a "$LOG_FILE"
echo "Log: $LOG_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Run Python download script
echo "Running automated download..." | tee -a "$LOG_FILE"
uv run python "$SCRIPT_DIR/download_rebap_auto.py" --days 7 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "✅ Daily reBAP update complete!" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"

# Keep only last 30 days of logs
find "$LOG_DIR" -name "rebap_update_*.log" -type f -mtime +30 -delete

exit 0
