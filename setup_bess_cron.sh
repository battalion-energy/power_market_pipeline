#!/bin/bash
##
# Setup ERCOT BESS Data Cron Job
#
# This script installs a cron job that runs every 5 minutes to update
# the ERCOT battery storage data catalog.
#
# Usage: bash setup_bess_cron.sh
##

# Configuration
SCRIPT_PATH="/home/enrico/projects/power_market_pipeline/ercot_bess_cron_updater.py"
LOG_PATH="/home/enrico/projects/power_market_pipeline/ercot_battery_storage_data/bess_updater.log"
PYTHON_BIN="$(which python3)"

# Ensure script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Script not found at $SCRIPT_PATH"
    exit 1
fi

# Make script executable
chmod +x "$SCRIPT_PATH"

# Create log directory
mkdir -p "$(dirname "$LOG_PATH")"

# Create cron job entry
CRON_ENTRY="*/5 * * * * nice -n 19 $PYTHON_BIN $SCRIPT_PATH >> $LOG_PATH 2>&1"

# Check if cron job already exists
if crontab -l 2>/dev/null | grep -F "$SCRIPT_PATH" > /dev/null; then
    echo "Cron job already exists. Removing old entry..."
    crontab -l 2>/dev/null | grep -v -F "$SCRIPT_PATH" | crontab -
fi

# Add new cron job
(crontab -l 2>/dev/null; echo "$CRON_ENTRY") | crontab -

echo "Cron job installed successfully!"
echo ""
echo "Configuration:"
echo "  Script: $SCRIPT_PATH"
echo "  Schedule: Every 5 minutes"
echo "  Priority: Low (nice -n 19)"
echo "  Log file: $LOG_PATH"
echo ""
echo "Verify installation:"
echo "  crontab -l | grep bess"
echo ""
echo "Monitor logs:"
echo "  tail -f $LOG_PATH"
echo ""
echo "Remove cron job:"
echo "  crontab -l | grep -v bess_cron_updater | crontab -"
