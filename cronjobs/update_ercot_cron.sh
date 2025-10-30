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

# Resource limits to prevent system crashes:
# - Memory: 40GB hard limit (will kill process if exceeded)
# - CPU: 10 cores maximum
# - Priority: nice 19 (lowest priority)
# - Timeout: 4 hours maximum

# Set memory limit (40GB = 41943040 KB)
ulimit -v 41943040  # Virtual memory limit
ulimit -m 41943040  # Physical memory limit (if supported)

echo "Resource Limits:" | tee -a "${LOG_FILE}"
echo "  Max Memory: 40GB" | tee -a "${LOG_FILE}"
echo "  Max CPU Cores: 10" | tee -a "${LOG_FILE}"
echo "  Nice Priority: 19" | tee -a "${LOG_FILE}"
echo "  Timeout: 4 hours" | tee -a "${LOG_FILE}"

# Export environment variable for Rust to limit CPU cores
export RAYON_NUM_THREADS=10

# Function to set OOM killer priority (kill this process first if system runs out of memory)
set_oom_score() {
    local pid=$1
    # OOM score 1000 = most likely to be killed (0-1000 scale)
    if [ -f "/proc/${pid}/oom_score_adj" ]; then
        echo 900 > "/proc/${pid}/oom_score_adj" 2>/dev/null || true
    fi
}

# Run update with systemd-run for hard resource limits (if available)
if command -v systemd-run &> /dev/null; then
    echo "Using systemd-run for resource control" | tee -a "${LOG_FILE}"
    # systemd-run provides hard memory limits and CPU quota
    # MemoryMax will kill the process if it exceeds 40GB
    systemd-run --user --scope \
        --property=MemoryMax=40G \
        --property=MemoryHigh=35G \
        --property=CPUQuota=1000% \
        --nice=19 \
        timeout 14400 /home/enrico/.local/bin/uv run python scripts/update_all_datasets.py >> "${LOG_FILE}" 2>&1
    EXIT_CODE=$?
else
    # Fallback: use nice and ulimit only
    echo "Using nice/ulimit for resource control (systemd-run not available)" | tee -a "${LOG_FILE}"
    nice -n 19 timeout 14400 /home/enrico/.local/bin/uv run python scripts/update_all_datasets.py >> "${LOG_FILE}" 2>&1 &
    PID=$!
    set_oom_score $PID
    wait $PID
    EXIT_CODE=$?
fi

echo "" | tee -a "${LOG_FILE}"
echo "================================================================================" | tee -a "${LOG_FILE}"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Update completed successfully at $(date)" | tee -a "${LOG_FILE}"
elif [ $EXIT_CODE -eq 124 ]; then
    echo "⏱️  Update timed out after 4 hours at $(date)" | tee -a "${LOG_FILE}"
elif [ $EXIT_CODE -eq 137 ]; then
    echo "💀 Process killed (likely by OOM killer due to memory limit) at $(date)" | tee -a "${LOG_FILE}"
    echo "   Consider reducing batch sizes or processing fewer datasets" | tee -a "${LOG_FILE}"
elif [ $EXIT_CODE -eq 143 ]; then
    echo "🛑 Process terminated (SIGTERM) at $(date)" | tee -a "${LOG_FILE}"
else
    echo "❌ Update failed with exit code $EXIT_CODE at $(date)" | tee -a "${LOG_FILE}"
fi
echo "================================================================================" | tee -a "${LOG_FILE}"

# Create symlink to latest log
ln -sf "${LOG_FILE}" "${LOG_DIR}/ercot_update_latest.log"

exit $EXIT_CODE
