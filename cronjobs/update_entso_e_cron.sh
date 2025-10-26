#!/bin/bash
#
# ENTSO-E / Germany Data Update Cron Wrapper
# Runs daily data updates with low priority and logging
# Updates: DA prices, imbalance prices, FCR, aFRR, mFRR (capacity & energy)
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
LOG_FILE="${LOG_DIR}/entso_e_update_${TIMESTAMP}.log"

# Keep only last 30 days of logs
find "${LOG_DIR}" -name "entso_e_update_*.log" -mtime +30 -delete

echo "================================================================================" | tee -a "${LOG_FILE}"
echo "ENTSO-E / Germany Data Update - Started at $(date)" | tee -a "${LOG_FILE}"
echo "================================================================================" | tee -a "${LOG_FILE}"
echo "📍 Data sources: ENTSO-E Transparency Platform + Regelleistung.net" | tee -a "${LOG_FILE}"
echo "📊 Products: DA prices, Imbalance prices, FCR, aFRR, mFRR" | tee -a "${LOG_FILE}"
echo "🔄 Mode: Auto-resume from last downloaded date" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

# Run Germany updater with auto-resume capability
# This will automatically:
# - Find the last downloaded date for each data type (7 types total)
# - Resume from that point for each independently
# - Catch up any gaps if cron job failed for a few days
# - Download: DA prices, imbalance prices, FCR/aFRR/mFRR capacity & energy

# Run as low priority background task
# nice -n 19: lowest CPU priority (least impact on system)
# ionice -c 3: idle IO priority (only uses IO when system is idle)
# timeout 7200: kill after 2 hours if hanging

nice -n 19 ionice -c 3 timeout 7200 /home/enrico/.local/bin/uv run python -m iso_markets.entso_e.update_germany_with_resume >> "${LOG_FILE}" 2>&1

OVERALL_SUCCESS=$?

echo "" | tee -a "${LOG_FILE}"
echo "================================================================================" | tee -a "${LOG_FILE}"
if [ $OVERALL_SUCCESS -eq 0 ]; then
    echo "✅ ENTSO-E / Germany update completed successfully at $(date)" | tee -a "${LOG_FILE}"
elif [ $OVERALL_SUCCESS -eq 124 ]; then
    echo "⏱️  ENTSO-E / Germany update timed out after 2 hours at $(date)" | tee -a "${LOG_FILE}"
else
    echo "⚠️  ENTSO-E / Germany update completed with some errors at $(date)" | tee -a "${LOG_FILE}"
    echo "    Check log for details: ${LOG_FILE}" | tee -a "${LOG_FILE}"
fi
echo "================================================================================" | tee -a "${LOG_FILE}"

# Create symlink to latest log
ln -sf "${LOG_FILE}" "${LOG_DIR}/entso_e_update_latest.log"

# Optional: Send notification on failure (uncomment if using ntfy or similar)
# if [ $OVERALL_SUCCESS -ne 0 ]; then
#     curl -d "ENTSO-E update failed - check logs" ntfy.sh/power_market_updates 2>/dev/null || true
# fi

exit $OVERALL_SUCCESS
