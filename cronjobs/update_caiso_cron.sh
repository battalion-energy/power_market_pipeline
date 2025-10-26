#!/bin/bash
# CAISO Daily Data Update Cron Job
# Downloads yesterday's data for:
#   - Day-Ahead Nodal LMPs (all nodes, hourly)
#   - Real-Time 5-Minute Nodal LMPs (all nodes, 5-min intervals)
#   - Day-Ahead Ancillary Services (RU, RD, SR, NR)
#
# Conservative throttling: 6-second delays between requests (CAISO limit: 1 request per 5 seconds)
# Exponential backoff retry on failures (30s, 60s, 120s, 240s, 480s)

# Change to project directory
cd /home/enrico/projects/power_market_pipeline || exit 1

# Load environment variables
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Setup logging - use LOGS_DIR from .env or default
LOG_DIR="${LOGS_DIR:-/pool/ssd8tb/logs}"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/caiso_update_${TIMESTAMP}.log"
LATEST_LOG="${LOG_DIR}/caiso_update_latest.log"

echo "================================================================================" | tee "$LOG_FILE"
echo "CAISO Data Update - $(date)" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Run CAISO updater with auto-resume capability
# Downloads 3 data types:
#   • DA nodal LMPs (all nodes, hourly)
#   • RT 5-minute nodal LMPs (all nodes, 5-min intervals)
#   • DA ancillary services (RU, RD, SR, NR)
# Auto-throttle with conservative 6-second delays between requests
# Handle failures with exponential backoff
#
# Note: RT 5-min downloads are LARGE (~1.8 GB per day)
# Timeout set to 4 hours to accommodate RT 5-min chunked downloads
nice -n 19 timeout 14400 /home/enrico/.local/bin/uv run python -m iso_markets.caiso.update_caiso_with_resume 2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

echo "" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "Update completed at $(date) with exit code: $EXIT_CODE" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"

# Create symlink to latest log
ln -sf "$LOG_FILE" "$LATEST_LOG"

exit $EXIT_CODE
