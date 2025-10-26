#!/bin/bash
# Run ALL ISO converters sequentially with full memory protection
# This processes ALL available data for each ISO (all years, all market types)
#
# Usage: ./run_all_converters_safe.sh [--year YYYY]
#
# Examples:
#   ./run_all_converters_safe.sh              # Process all years
#   ./run_all_converters_safe.sh --year 2024  # Process only 2024

set -e

LOG_DIR="/pool/ssd8tb/data/iso/unified_iso_data/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Memory limits (adjust based on your system)
MEMORY_CAP_GB=${MEMORY_CAP_GB:-80}        # Hard cap via systemd cgroup
MEMORY_HIGH_GB=${MEMORY_HIGH_GB:-75}      # Soft throttle
MIN_AVAILABLE_GB=30                        # Safety check

# Parse arguments
YEAR_ARG=""
if [ "$1" = "--year" ] && [ -n "$2" ]; then
    YEAR_ARG="--year $2"
    echo "Processing year: $2"
fi

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
        echo "❌ ERROR: Low memory! Only ${available_gb}GB available (threshold: ${MIN_AVAILABLE_GB}GB)"
        echo "Stopping to prevent system crash"
        exit 1
    fi
    echo "✅ Memory check OK: ${available_gb}GB available"
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
            -p "MemorySwapMax=8G" \
            --collect \
            "$@" > "$log_file" 2>&1
    else
        # Fallback to direct execution (Python has built-in limit)
        "$@" > "$log_file" 2>&1
    fi
}

echo "========================================="
echo "ISO CONVERTER PRODUCTION RUN"
echo "Running ALL converters sequentially"
echo "========================================="
echo "Memory Cap:      ${MEMORY_CAP_GB}GB (systemd cgroup)"
echo "Python Limit:    50GB (built-in self-limit)"
echo "Min Available:   ${MIN_AVAILABLE_GB}GB"
echo "Log Directory:   $LOG_DIR"
echo "Timestamp:       $TIMESTAMP"
if [ -n "$YEAR_ARG" ]; then
    echo "Year Filter:     $2"
else
    echo "Year Filter:     ALL YEARS"
fi
echo "========================================="
echo ""

# Pre-flight memory check
check_memory
echo ""

# Define converters and their arguments
# Format: "name|args"
declare -a CONVERTERS=(
    "pjm|$YEAR_ARG --da-only"
    "caiso|$YEAR_ARG --da-only"
    "nyiso|$YEAR_ARG --da-only"
    "spp|$YEAR_ARG --da-only"
    "isone|$YEAR_ARG --da-only"
    "miso|$YEAR_ARG --da-only"
    "ercot|$YEAR_ARG --da-only"
)

# Track statistics
TOTAL_CONVERTERS=${#CONVERTERS[@]}
SUCCESSFUL=0
FAILED=0
START_TIME=$(date +%s)

# Run each converter
for converter_spec in "${CONVERTERS[@]}"; do
    # Parse converter name and args
    IFS='|' read -r converter args <<< "$converter_spec"

    echo ""
    echo "========================================="
    echo "[$((SUCCESSFUL + FAILED + 1))/$TOTAL_CONVERTERS] Processing: ${converter^^}"
    echo "========================================="
    echo "Arguments: $args"
    echo ""

    # Check memory before starting
    check_memory
    echo ""

    # Run converter with memory limits
    LOG_FILE="$LOG_DIR/${converter}_production_${TIMESTAMP}.log"
    echo "⏳ Starting ${converter} converter..."
    echo "📝 Log: $LOG_FILE"
    echo ""

    CONVERTER_START=$(date +%s)

    if run_with_memlimit "$LOG_FILE" python3 "${converter}_parquet_converter.py" $args; then
        CONVERTER_END=$(date +%s)
        CONVERTER_DURATION=$((CONVERTER_END - CONVERTER_START))
        SUCCESSFUL=$((SUCCESSFUL + 1))

        echo "✅ ${converter^^} COMPLETED SUCCESSFULLY in ${CONVERTER_DURATION}s"
        echo ""

        # Show summary from log
        echo "📊 Summary:"
        tail -20 "$LOG_FILE" | grep -E "rows|Successfully|complete|Writing|parquet" || echo "   (check log for details)"

    else
        CONVERTER_END=$(date +%s)
        CONVERTER_DURATION=$((CONVERTER_END - CONVERTER_START))
        exit_code=$?
        FAILED=$((FAILED + 1))

        echo "❌ ${converter^^} FAILED after ${CONVERTER_DURATION}s (exit code: $exit_code)"
        echo "📝 Check log: $LOG_FILE"
        echo ""
        echo "Last 30 lines of log:"
        tail -30 "$LOG_FILE"
        echo ""

        # Check if it was OOM killed
        if grep -q "MemoryError\|Killed\|out of memory" "$LOG_FILE"; then
            echo "⚠️  MEMORY ERROR DETECTED"
            echo "The converter hit its memory limit. This prevented a system crash!"
            echo ""
            echo "Solutions:"
            echo "  1. Increase memory limit: export MEMORY_CAP_GB=120"
            echo "  2. Process one year at a time: --year 2024"
            echo "  3. Reduce BATCH_SIZE in ${converter}_parquet_converter.py"
            echo ""
        fi

        # Ask whether to continue
        read -p "Continue with remaining converters? (y/N) " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "❌ Stopping conversion process"
            exit 1
        fi
    fi

    # Check memory after completion
    echo ""
    check_memory

    # Brief pause to let system stabilize
    echo ""
    echo "⏸️  Waiting 15 seconds before next converter..."
    sleep 15
done

END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
HOURS=$((TOTAL_DURATION / 3600))
MINUTES=$(((TOTAL_DURATION % 3600) / 60))
SECONDS=$((TOTAL_DURATION % 60))

echo ""
echo "========================================="
echo "CONVERSION COMPLETE!"
echo "========================================="
echo "Total Duration:  ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "Successful:      $SUCCESSFUL / $TOTAL_CONVERTERS"
echo "Failed:          $FAILED / $TOTAL_CONVERTERS"
echo "Logs:            $LOG_DIR/*_production_${TIMESTAMP}.log"
echo "========================================="

if [ $FAILED -eq 0 ]; then
    echo "🎉 All converters completed successfully!"
    exit 0
else
    echo "⚠️  Some converters failed - check logs above"
    exit 1
fi
