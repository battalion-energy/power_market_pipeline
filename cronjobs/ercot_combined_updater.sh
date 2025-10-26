#!/bin/bash
##
# ERCOT Combined Data Updater - Runs every 5 minutes
#
# This script collects both:
# 1. BESS operational data (charging/discharging)
# 2. SCED LMP forecasts (with vintage preservation)
#
# Each script maintains its own data catalog with automatic deduplication.
##

# Initialize pyenv (required for cron environment)
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
if command -v pyenv 1>/dev/null 2>&1; then
    eval "$(pyenv init --path)"
    eval "$(pyenv init -)"
fi

# Project directory
PROJ_DIR="/home/enrico/projects/power_market_pipeline"

# Ensure we're in the project directory
cd "$PROJ_DIR" || exit 1

# Load environment variables from .env
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Use LOGS_DIR from .env or default
LOG_DIR="${LOGS_DIR:-/pool/ssd8tb/logs}"
mkdir -p "$LOG_DIR"

# Log start time
echo "=== ERCOT Combined Updater Started at $(date) ===" >> "$LOG_DIR/combined_updater.log"

# 1. Update BESS operational data (existing script)
python3 "$PROJ_DIR/scripts/ercot_bess_cron_updater.py" >> "$LOG_DIR/bess_updater.log" 2>&1
BESS_EXIT=$?

# 2. Update SCED forecasts (new script)
python3 "$PROJ_DIR/scripts/ercot_sced_forecast_collector.py" --continuous >> "$LOG_DIR/sced_forecasts/forecast_updater.log" 2>&1
SCED_EXIT=$?

# Log completion
if [ $BESS_EXIT -eq 0 ] && [ $SCED_EXIT -eq 0 ]; then
    echo "✓ Both updates completed successfully at $(date)" >> "$LOG_DIR/combined_updater.log"
elif [ $BESS_EXIT -ne 0 ]; then
    echo "✗ BESS update failed (exit: $BESS_EXIT) at $(date)" >> "$LOG_DIR/combined_updater.log"
elif [ $SCED_EXIT -ne 0 ]; then
    echo "✗ SCED update failed (exit: $SCED_EXIT) at $(date)" >> "$LOG_DIR/combined_updater.log"
fi

exit 0
