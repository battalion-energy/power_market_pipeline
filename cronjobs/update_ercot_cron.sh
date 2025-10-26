#!/bin/bash
#
# ERCOT Data Update Cron Wrapper
# Runs unified dataset updater with low priority and logging
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
LOG_FILE="${LOG_DIR}/ercot_update_${TIMESTAMP}.log"

# Keep only last 30 days of logs
find "${LOG_DIR}" -name "ercot_update_*.log" -mtime +30 -delete

echo "================================================================================" | tee -a "${LOG_FILE}"
echo "ERCOT Data Update - Started at $(date)" | tee -a "${LOG_FILE}"
echo "================================================================================" | tee -a "${LOG_FILE}"

# Run update with nice (low priority: 19 = lowest)
# Use timeout to prevent hanging (4 hours max)
nice -n 19 timeout 14400 /home/enrico/.local/bin/uv run python scripts/update_all_datasets.py >> "${LOG_FILE}" 2>&1
EXIT_CODE=$?

echo "" | tee -a "${LOG_FILE}"
echo "================================================================================" | tee -a "${LOG_FILE}"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Update completed successfully at $(date)" | tee -a "${LOG_FILE}"
elif [ $EXIT_CODE -eq 124 ]; then
    echo "⏱️  Update timed out after 4 hours at $(date)" | tee -a "${LOG_FILE}"
else
    echo "❌ Update failed with exit code $EXIT_CODE at $(date)" | tee -a "${LOG_FILE}"
fi
echo "================================================================================" | tee -a "${LOG_FILE}"

# Create symlink to latest log
ln -sf "${LOG_FILE}" "${LOG_DIR}/ercot_update_latest.log"

exit $EXIT_CODE
