#!/bin/bash
# SAFE converter testing - ONE AT A TIME with memory monitoring
# This prevents system crashes by running sequentially with HARD MEMORY LIMITS

set -e

LOG_DIR="/pool/ssd8tb/data/iso/unified_iso_data/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Memory limits (adjust based on your system)
MEMORY_CAP_GB=60        # Hard cap via systemd cgroup (kills process if exceeded)
MEMORY_HIGH_GB=55       # Soft throttle to prevent thrashing
MIN_AVAILABLE_GB=20     # Safety check - stop if system RAM too low

# Check if systemd-run is available
HAVE_SYSTEMD_RUN=false
if command -v systemd-run &> /dev/null; then
    HAVE_SYSTEMD_RUN=true
    echo "✅ systemd-run available - using cgroup memory limits"
else
    echo "⚠️  systemd-run not found - using Python-only memory limits"
fi

# Function to check available memory
check_memory() {
    available_gb=$(free -g | awk '/^Mem:/{print $7}')
    if [ "$available_gb" -lt "$MIN_AVAILABLE_GB" ]; then
        echo "ERROR: Low memory! Only ${available_gb}GB available (threshold: ${MIN_AVAILABLE_GB}GB)"
        echo "Stopping to prevent system crash"
        exit 1
    fi
    echo "Memory check OK: ${available_gb}GB available"
}

# Function to run command with memory limits
run_with_memlimit() {
    local log_file="$1"
    shift  # remaining args are the command

    if [ "$HAVE_SYSTEMD_RUN" = true ]; then
        # Use systemd-run for hard cgroup memory limit
        systemd-run --user --scope \
            -p "MemoryMax=${MEMORY_CAP_GB}G" \
            -p "MemoryHigh=${MEMORY_HIGH_GB}G" \
            -p "MemorySwapMax=4G" \
            --collect \
            "$@" > "$log_file" 2>&1
    else
        # Fallback to direct execution (Python has built-in limit)
        "$@" > "$log_file" 2>&1
    fi
}

echo "========================================="
echo "SAFE ISO Converter Testing"
echo "Running ONE converter at a time"
echo "Memory Cap: ${MEMORY_CAP_GB}GB (systemd cgroup)"
echo "Python Limit: 50GB (built-in self-limit)"
echo "Min Available: ${MIN_AVAILABLE_GB}GB"
echo "Logs: $LOG_DIR"
echo "========================================="

# Pre-flight memory check
check_memory

# Test each converter sequentially
CONVERTERS=("nyiso" "isone" "spp" "miso" "ercot")

for converter in "${CONVERTERS[@]}"; do
    echo ""
    echo "========================================="
    echo "Testing: ${converter^^}"
    echo "========================================="

    # Check memory before starting
    check_memory

    # Run converter with memory limits
    LOG_FILE="$LOG_DIR/${converter}_safe_test_${TIMESTAMP}.log"
    echo "Starting ${converter} converter with memory protection..."
    echo "Log: $LOG_FILE"

    if run_with_memlimit "$LOG_FILE" python3 "${converter}_parquet_converter.py" --year 2024 --da-only; then
        echo "✅ ${converter^^} completed successfully"
        # Show last few lines of success
        tail -5 "$LOG_FILE" | grep -E "rows|Successfully|complete" || true
    else
        exit_code=$?
        echo "❌ ${converter^^} FAILED (exit code: $exit_code)"
        echo "Check log: $LOG_FILE"
        echo ""
        echo "Last 30 lines of log:"
        tail -30 "$LOG_FILE"

        # Check if it was OOM killed
        if grep -q "MemoryError\|Killed\|out of memory" "$LOG_FILE"; then
            echo ""
            echo "⚠️  MEMORY ERROR DETECTED"
            echo "The converter hit its memory limit. This is GOOD - it prevented a system crash!"
            echo "Consider increasing batch size or processing fewer files at once."
        fi
    fi

    # Check memory after completion
    check_memory

    # Brief pause to let system stabilize
    echo "Waiting 10 seconds before next converter..."
    sleep 10
done

echo ""
echo "========================================="
echo "All converters tested!"
echo "========================================="
