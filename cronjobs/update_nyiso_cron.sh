#!/bin/bash
#
# NYISO Data Update Cron Wrapper
# Runs daily data updates with auto-resume capability
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
LOG_FILE="${LOG_DIR}/nyiso_update_${TIMESTAMP}.log"

# Keep only last 30 days of logs
find "${LOG_DIR}" -name "nyiso_update_*.log" -mtime +30 -delete

echo "================================================================================" | tee -a "${LOG_FILE}"
echo "NYISO Data Update - Started at $(date)" | tee -a "${LOG_FILE}"
echo "================================================================================" | tee -a "${LOG_FILE}"

# Calculate date range (yesterday to today for incremental updates)
YESTERDAY=$(date -d "yesterday" +%Y-%m-%d)
TODAY=$(date +%Y-%m-%d)

echo "📅 Auto-resuming from last downloaded date..." | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

# Run NYISO gridstatus downloader with auto-resume
# This will automatically start from the last downloaded date
# Downloads all data types: LMP (DA & RT), AS (DA & RT), Load, Fuel Mix
nice -n 19 timeout 3600 /home/enrico/.local/bin/uv run python scripts/download_nyiso_gridstatus.py \
  --auto-resume \
  --output-dir "${ISO_DATA_DIR}" >> "${LOG_FILE}" 2>&1

EXIT_CODE=$?

echo "" | tee -a "${LOG_FILE}"
echo "================================================================================" | tee -a "${LOG_FILE}"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ NYISO update completed successfully at $(date)" | tee -a "${LOG_FILE}"
elif [ $EXIT_CODE -eq 124 ]; then
    echo "⏱️  NYISO update timed out after 1 hour at $(date)" | tee -a "${LOG_FILE}"
else
    echo "⚠️  NYISO update completed with exit code $EXIT_CODE at $(date)" | tee -a "${LOG_FILE}"
fi
echo "================================================================================" | tee -a "${LOG_FILE}"

# Create symlink to latest log
ln -sf "${LOG_FILE}" "${LOG_DIR}/nyiso_update_latest.log"

exit $EXIT_CODE
