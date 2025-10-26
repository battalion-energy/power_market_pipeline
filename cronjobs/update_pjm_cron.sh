#!/bin/bash
#
# PJM Data Update Cron Wrapper
# Runs daily data updates with low priority and logging
#
# Auto-resumes from last downloaded date for each data type
# Downloads: DA nodal, RT hourly nodal, and ancillary services
#

# Set working directory
cd /home/enrico/projects/power_market_pipeline || exit 1

# Load environment variables from .env
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Set up environment
export PATH="/home/enrico/.cargo/bin:/home/enrico/.local/bin:$PATH"

# Log file with timestamp - use LOGS_DIR from .env or default
LOG_DIR="${LOGS_DIR:-/pool/ssd8tb/logs}"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/pjm_update_${TIMESTAMP}.log"

# Keep only last 30 days of logs
find "${LOG_DIR}" -name "pjm_update_*.log" -mtime +30 -delete 2>/dev/null

echo "================================================================================" | tee -a "${LOG_FILE}"
echo "PJM Data Update - Started at $(date)" | tee -a "${LOG_FILE}"
echo "================================================================================" | tee -a "${LOG_FILE}"

# Calculate yesterday's date for reference
YESTERDAY=$(date -d "yesterday" +%Y-%m-%d)
TODAY=$(date +%Y-%m-%d)

echo "📅 Auto-resuming from last downloaded date for each data type..." | tee -a "${LOG_FILE}"
echo "   (Will catch up any gaps if cron job failed for a few days)" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

# Run PJM updater with auto-resume capability
# This will automatically:
# - Find the last downloaded date for each data type:
#   • DA nodal LMPs (all ~22K nodes)
#   • RT hourly nodal LMPs (all ~14K-22K nodes)
#   • RT 5-minute nodal LMPs (last 6 months, all nodes)
#   • DA ancillary services
# - Resume from that point
# - Catch up any gaps if cron job failed
# - Auto-throttle with conservative 2-second delays between requests
# - Handle rate limits with exponential backoff
nice -n 19 timeout 10800 /home/enrico/.local/bin/uv run python -m iso_markets.pjm.update_pjm_with_resume >> "${LOG_FILE}" 2>&1

OVERALL_SUCCESS=$?

echo "" | tee -a "${LOG_FILE}"
echo "================================================================================" | tee -a "${LOG_FILE}"
if [ $OVERALL_SUCCESS -eq 0 ]; then
    echo "✅ PJM update completed successfully at $(date)" | tee -a "${LOG_FILE}"
else
    echo "⚠️  PJM update completed with some errors at $(date)" | tee -a "${LOG_FILE}"
    echo "    Check log for details: ${LOG_FILE}" | tee -a "${LOG_FILE}"
fi
echo "================================================================================" | tee -a "${LOG_FILE}"

# Create symlink to latest log
ln -sf "${LOG_FILE}" "${LOG_DIR}/pjm_update_latest.log"

exit $OVERALL_SUCCESS
