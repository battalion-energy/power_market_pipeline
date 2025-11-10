#!/bin/bash
#
# ERCOT Energy Storage Resources (ESR) Daily Download Cron Wrapper
# Downloads and archives ESR data from ERCOT dashboard API
#

# Set working directory
cd /home/enrico/projects/power_market_pipeline || exit 1

# Load environment variables from .env if exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Set up environment
export PATH="/home/enrico/.cargo/bin:/home/enrico/.local/bin:$PATH"

# Log file with timestamp - use LOGS_DIR from .env or default
LOG_DIR="${LOGS_DIR:-/pool/ssd8tb/logs}"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/esr_download_${TIMESTAMP}.log"

# Keep only last 30 days of logs
find "${LOG_DIR}" -name "esr_download_*.log" -mtime +30 -delete

echo "================================================================================" | tee -a "${LOG_FILE}"
echo "ERCOT ESR Data Download - Started at $(date)" | tee -a "${LOG_FILE}"
echo "================================================================================" | tee -a "${LOG_FILE}"

# Run the CSV download script (appends to contiguous CSV file)
python3 /home/enrico/projects/power_market_pipeline/ercot_battery_storage_data/download_esr_daily_csv.py >> "${LOG_FILE}" 2>&1
EXIT_CODE=$?

echo "" | tee -a "${LOG_FILE}"
echo "================================================================================" | tee -a "${LOG_FILE}"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ ESR download completed successfully at $(date)" | tee -a "${LOG_FILE}"
else
    echo "❌ ESR download failed with exit code $EXIT_CODE at $(date)" | tee -a "${LOG_FILE}"
fi
echo "================================================================================" | tee -a "${LOG_FILE}"

# Create symlink to latest log
ln -sf "${LOG_FILE}" "${LOG_DIR}/esr_download_latest.log"

exit $EXIT_CODE
