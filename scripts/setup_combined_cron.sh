#!/bin/bash
##
# Setup Combined ERCOT Data Collection Cron Job
#
# This installs a cron job that runs every 5 minutes to collect:
# 1. BESS operational data (charging/discharging)
# 2. SCED LMP forecasts (with vintage preservation)
#
# Usage: bash setup_combined_cron.sh
##

# Configuration
SCRIPT_PATH="/home/enrico/projects/power_market_pipeline/ercot_combined_updater.sh"
LOG_PATH="/home/enrico/projects/power_market_pipeline/ercot_battery_storage_data/combined_updater.log"

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
CRON_ENTRY="*/5 * * * * nice -n 19 bash $SCRIPT_PATH"

# Check if cron job already exists
if crontab -l 2>/dev/null | grep -F "$SCRIPT_PATH" > /dev/null; then
    echo "Cron job already exists. Removing old entry..."
    crontab -l 2>/dev/null | grep -v -F "$SCRIPT_PATH" | crontab -
fi

# Remove old BESS-only cron job if it exists
OLD_BESS_SCRIPT="/home/enrico/projects/power_market_pipeline/ercot_bess_cron_updater.py"
if crontab -l 2>/dev/null | grep -F "$OLD_BESS_SCRIPT" > /dev/null; then
    echo "Removing old BESS-only cron job..."
    crontab -l 2>/dev/null | grep -v -F "$OLD_BESS_SCRIPT" | crontab -
fi

# Add new cron job
(crontab -l 2>/dev/null; echo "$CRON_ENTRY") | crontab -

echo "Combined cron job installed successfully!"
echo ""
echo "Configuration:"
echo "  Script: $SCRIPT_PATH"
echo "  Schedule: Every 5 minutes"
echo "  Priority: Low (nice -n 19)"
echo "  Main log: $LOG_PATH"
echo ""
echo "Data collected:"
echo "  1. BESS operational data → ercot_battery_storage_data/bess_catalog.csv"
echo "     Log: ercot_battery_storage_data/bess_updater.log"
echo ""
echo "  2. SCED forecasts → ercot_battery_storage_data/sced_forecasts/forecast_catalog.csv"
echo "     Log: ercot_battery_storage_data/sced_forecasts/forecast_updater.log"
echo ""
echo "Verify installation:"
echo "  crontab -l | grep ercot"
echo ""
echo "Monitor logs:"
echo "  tail -f $LOG_PATH"
echo "  tail -f ercot_battery_storage_data/bess_updater.log"
echo "  tail -f ercot_battery_storage_data/sced_forecasts/forecast_updater.log"
echo ""
echo "Remove cron job:"
echo "  crontab -l | grep -v ercot_combined_updater | crontab -"
