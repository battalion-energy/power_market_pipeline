#!/bin/bash
#
# Full ISO Parquet Conversion Script
# Runs all ISOs sequentially to avoid memory issues
#

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

LOG_FILE="/pool/ssd8tb/data/iso/unified_iso_data/logs/full_conversion_$(date +%Y%m%d_%H%M%S).log"

echo "=============================================================================" | tee -a "$LOG_FILE"
echo "ISO PARQUET FULL CONVERSION STARTED: $(date)" | tee -a "$LOG_FILE"
echo "=============================================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Phase 1: Convert 2024 data for all ISOs (DA only)
echo "PHASE 1: Converting 2024 data (DA only)..." | tee -a "$LOG_FILE"
echo "---------------------------------------------" | tee -a "$LOG_FILE"
python3 run_all_iso_converters.py --year 2024 --da-only 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "=============================================================================" | tee -a "$LOG_FILE"
echo "PHASE 1 COMPLETE - Spot checking results..." | tee -a "$LOG_FILE"
echo "=============================================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Spot check
python3 monitor_conversion.py 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "Press ENTER to continue with full historical conversion (2019-2025)..."
echo "Or Ctrl+C to stop here and review results."
read

# Phase 2: Convert all historical data
echo "" | tee -a "$LOG_FILE"
echo "=============================================================================" | tee -a "$LOG_FILE"
echo "PHASE 2: Converting ALL historical data (2019-2025, DA only)..." | tee -a "$LOG_FILE"
echo "=============================================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

python3 run_all_iso_converters.py --da-only 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "=============================================================================" | tee -a "$LOG_FILE"
echo "FULL CONVERSION COMPLETE: $(date)" | tee -a "$LOG_FILE"
echo "=============================================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Final summary
python3 monitor_conversion.py 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "Full log saved to: $LOG_FILE" | tee -a "$LOG_FILE"
