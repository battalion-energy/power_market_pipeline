#!/bin/bash
# ISO-NE Daily Data Update Script
# Downloads latest LMP (hubs + nodal) and ancillary services data
#
# This script is designed to run as a daily cronjob and will:
# - Download yesterday's and today's data (to catch late postings)
# - Include hub-level LMPs (DA + RT)
# - Include nodal LMPs (DA + RT)
# - Include all ancillary services (freq_reg + reserves)
# - Log all output for debugging

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="/home/enrico/logs/isone_daily_update"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Calculate date range (yesterday and today to catch late postings)
YESTERDAY=$(date -d "yesterday" +%Y-%m-%d)
TODAY=$(date +%Y-%m-%d)

echo "=================================================================================="
echo "ISO-NE Daily Update"
echo "Started: $(date)"
echo "Date range: $YESTERDAY to $TODAY"
echo "=================================================================================="

# Change to project directory
cd "$PROJECT_ROOT"

# Function to run a download with logging
run_download() {
    local name=$1
    local script=$2
    shift 2
    local args=("$@")

    local log_file="$LOG_DIR/${name}_${TIMESTAMP}.log"

    echo ""
    echo "Starting: $name"
    echo "Log: $log_file"
    echo "Command: uv run python $script ${args[*]}"

    if uv run python "$script" "${args[@]}" 2>&1 | tee "$log_file"; then
        echo "✓ Completed: $name"
        return 0
    else
        echo "✗ Failed: $name (see $log_file)"
        return 1
    fi
}

# Track failures
FAILURES=0

# Download hub-level LMPs (DA + RT)
if ! run_download \
    "hub_lmps" \
    "iso_markets/isone/download_lmp.py" \
    --start-date "$YESTERDAY" \
    --end-date "$TODAY" \
    --market-types da rt \
    --hubs-only \
    --max-concurrent 1; then
    ((FAILURES++))
fi

# Download nodal LMPs (DA + RT)
if ! run_download \
    "nodal_lmps" \
    "iso_markets/isone/download_lmp.py" \
    --start-date "$YESTERDAY" \
    --end-date "$TODAY" \
    --market-types da rt \
    --all-nodes \
    --max-concurrent 1; then
    ((FAILURES++))
fi

# Download ancillary services (freq_reg + reserves for all zones)
if ! run_download \
    "ancillary_services" \
    "iso_markets/isone/download_ancillary_services.py" \
    --start-date "$YESTERDAY" \
    --end-date "$TODAY" \
    --data-types freq_reg reserves \
    --reserve-zones 7000 7001 7002 7003 7004 7005 7006 7007 7008 7009 7010 7011 \
    --max-concurrent 1; then
    ((FAILURES++))
fi

# Download operational data (load forecasts, fuel mix, demand, reserves, constraints, outages)
if ! run_download \
    "operational_data" \
    "iso_markets/isone/download_operational_data.py" \
    --start-date "$YESTERDAY" \
    --end-date "$TODAY" \
    --data-types all \
    --max-concurrent 1; then
    ((FAILURES++))
fi

# Print summary
echo ""
echo "=================================================================================="
echo "ISO-NE Daily Update Complete"
echo "Completed: $(date)"
echo "Failures: $FAILURES"
echo "Logs: $LOG_DIR/*_${TIMESTAMP}.log"
echo "=================================================================================="

# Exit with error if any downloads failed
if [ $FAILURES -gt 0 ]; then
    echo "⚠ Warning: $FAILURES download(s) failed"
    exit 1
fi

exit 0
