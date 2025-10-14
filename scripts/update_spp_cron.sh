#!/bin/bash
#
# SPP Data Update Cron Wrapper
# Runs daily data updates with low priority and logging
#

# Set working directory
cd /home/enrico/projects/power_market_pipeline || exit 1

# Set up environment
export PATH="/home/enrico/.cargo/bin:/home/enrico/.local/bin:$PATH"
export SPP_DATA_DIR="/pool/ssd8tb/data/iso/SPP"

# Log file with timestamp
LOG_DIR="/home/enrico/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/spp_update_${TIMESTAMP}.log"

# Keep only last 30 days of logs
find "${LOG_DIR}" -name "spp_update_*.log" -mtime +30 -delete

echo "================================================================================" | tee -a "${LOG_FILE}"
echo "SPP Data Update - Started at $(date)" | tee -a "${LOG_FILE}"
echo "================================================================================" | tee -a "${LOG_FILE}"

echo "📅 Auto-resuming from last downloaded date (with catchup for any gaps)..." | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

# Run SPP updater with auto-resume capability
# This will automatically:
# - Find the last downloaded date for each data type
# - Resume from that point
# - Catch up any gaps if cron job failed for a few days
nice -n 19 timeout 3600 /home/enrico/.local/bin/uv run python iso_markets/spp/update_spp_with_resume.py >> "${LOG_FILE}" 2>&1

OVERALL_SUCCESS=$?

echo "" | tee -a "${LOG_FILE}"
echo "================================================================================" | tee -a "${LOG_FILE}"
if [ $OVERALL_SUCCESS -eq 0 ]; then
    echo "✅ SPP update completed successfully at $(date)" | tee -a "${LOG_FILE}"
else
    echo "⚠️  SPP update completed with some errors at $(date)" | tee -a "${LOG_FILE}"
fi
echo "================================================================================" | tee -a "${LOG_FILE}"

# Create symlink to latest log
ln -sf "${LOG_FILE}" "${LOG_DIR}/spp_update_latest.log"

exit $OVERALL_SUCCESS
