#!/bin/bash
# Rebuild 2025 parquet files for all ISOs with atomic updates
#
# This script:
# - Processes ONLY 2025 data for each ISO
# - Uses atomic file moves (temp file → mv) to prevent corruption
# - Runs one ISO at a time with memory protection
# - Safe to run while other processes read the files
#
# The atomic update process (built into each converter):
#   1. Create temp file: da_energy_hourly_2025.parquet.tmp
#   2. Write all data to temp file
#   3. Atomic move: mv temp → final (replaces old file instantly)
#
# Usage: ./rebuild_2025_parquet.sh [--da-only | --rt-only | --as-only]

set -e

LOG_DIR="/pool/ssd8tb/data/iso/unified_iso_data/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
YEAR=2025

# Memory limits
MEMORY_CAP_GB=${MEMORY_CAP_GB:-80}
MEMORY_HIGH_GB=${MEMORY_HIGH_GB:-75}
MIN_AVAILABLE_GB=30

# Parse market type arguments
MARKET_ARGS="--da-only"  # Default: DA only
if [ "$1" = "--rt-only" ]; then
    MARKET_ARGS="--rt-only"
    echo "Processing: Real-Time markets only"
elif [ "$1" = "--as-only" ]; then
    MARKET_ARGS="--as-only"
    echo "Processing: Ancillary Services only"
elif [ "$1" = "--all" ]; then
    MARKET_ARGS=""
    echo "Processing: ALL market types (DA + RT + AS)"
else
    echo "Processing: Day-Ahead markets only (default)"
fi

# Check if systemd-run is available
HAVE_SYSTEMD_RUN=false
if command -v systemd-run &> /dev/null; then
    HAVE_SYSTEMD_RUN=true
fi

# Function to check available memory
check_memory() {
    available_gb=$(free -g | awk '/^Mem:/{print $7}')
    if [ "$available_gb" -lt "$MIN_AVAILABLE_GB" ]; then
        echo "❌ ERROR: Low memory! Only ${available_gb}GB available"
        exit 1
    fi
    echo "✅ Memory: ${available_gb}GB available"
}

# Function to run command with memory limits
run_with_memlimit() {
    local log_file="$1"
    shift

    if [ "$HAVE_SYSTEMD_RUN" = true ]; then
        systemd-run --user --scope \
            -p "MemoryMax=${MEMORY_CAP_GB}G" \
            -p "MemoryHigh=${MEMORY_HIGH_GB}G" \
            -p "MemorySwapMax=8G" \
            --collect \
            "$@" > "$log_file" 2>&1
    else
        "$@" > "$log_file" 2>&1
    fi
}

# Function to verify parquet file was created/updated
verify_parquet() {
    local iso=$1
    local market_type=$2
    local parquet_dir="/pool/ssd8tb/data/iso/unified_iso_data/parquet/${iso}/${market_type}"
    local parquet_file="${parquet_dir}/${market_type}_${YEAR}.parquet"

    if [ -f "$parquet_file" ]; then
        local size=$(du -h "$parquet_file" | cut -f1)
        local mtime=$(stat -c %y "$parquet_file" | cut -d'.' -f1)
        echo "   ✅ $parquet_file"
        echo "      Size: $size, Modified: $mtime"
        return 0
    else
        echo "   ⚠️  File not found: $parquet_file"
        return 1
    fi
}

echo "========================================="
echo "REBUILD 2025 PARQUET FILES"
echo "========================================="
echo "Year:            $YEAR"
echo "Market Types:    $MARKET_ARGS"
echo "Memory Cap:      ${MEMORY_CAP_GB}GB"
echo "Atomic Updates:  ✅ Enabled (built-in)"
echo "Log Directory:   $LOG_DIR"
echo "Timestamp:       $TIMESTAMP"
echo "========================================="
echo ""

# Pre-flight check
check_memory
echo ""

# Define all ISOs
ISOS=("pjm" "caiso" "nyiso" "spp" "isone" "miso" "ercot")

SUCCESSFUL=0
FAILED=0
START_TIME=$(date +%s)

for iso in "${ISOS[@]}"; do
    echo ""
    echo "========================================="
    echo "[$((SUCCESSFUL + FAILED + 1))/${#ISOS[@]}] Rebuilding: ${iso^^} (Year $YEAR)"
    echo "========================================="

    check_memory

    LOG_FILE="$LOG_DIR/${iso}_rebuild_${YEAR}_${TIMESTAMP}.log"
    echo "⏳ Processing..."
    echo "📝 Log: $LOG_FILE"

    CONVERTER_START=$(date +%s)

    if run_with_memlimit "$LOG_FILE" python3 "${iso}_parquet_converter.py" --year $YEAR $MARKET_ARGS; then
        CONVERTER_END=$(date +%s)
        DURATION=$((CONVERTER_END - CONVERTER_START))
        SUCCESSFUL=$((SUCCESSFUL + 1))

        echo "✅ ${iso^^} completed in ${DURATION}s"
        echo ""

        # Show what was created/updated
        echo "📦 Parquet files updated:"
        if [[ "$MARKET_ARGS" == *"da"* ]] || [ -z "$MARKET_ARGS" ]; then
            verify_parquet "$iso" "da_energy_hourly" || true
            verify_parquet "$iso" "da_energy_hourly_hub" || true
            verify_parquet "$iso" "da_energy_hourly_nodal" || true
        fi
        if [[ "$MARKET_ARGS" == *"rt"* ]] || [ -z "$MARKET_ARGS" ]; then
            verify_parquet "$iso" "rt_energy_5min" || true
        fi
        if [[ "$MARKET_ARGS" == *"as"* ]] || [ -z "$MARKET_ARGS" ]; then
            verify_parquet "$iso" "as_prices" || true
        fi

    else
        CONVERTER_END=$(date +%s)
        DURATION=$((CONVERTER_END - CONVERTER_START))
        exit_code=$?
        FAILED=$((FAILED + 1))

        echo "❌ ${iso^^} FAILED after ${DURATION}s (exit code: $exit_code)"
        echo ""
        echo "Last 30 lines:"
        tail -30 "$LOG_FILE"
        echo ""

        read -p "Continue with remaining ISOs? (y/N) " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi

    check_memory
    echo "⏸️  Waiting 10 seconds..."
    sleep 10
done

END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
MINUTES=$((TOTAL_DURATION / 60))
SECONDS=$((TOTAL_DURATION % 60))

echo ""
echo "========================================="
echo "REBUILD COMPLETE!"
echo "========================================="
echo "Duration:   ${MINUTES}m ${SECONDS}s"
echo "Successful: $SUCCESSFUL / ${#ISOS[@]}"
echo "Failed:     $FAILED / ${#ISOS[@]}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo "🎉 All 2025 parquet files rebuilt successfully!"
    echo ""
    echo "Files are safe to read during updates due to atomic mv operation:"
    echo "  1. New data written to: file_2025.parquet.tmp"
    echo "  2. Atomic replace: mv file_2025.parquet.tmp file_2025.parquet"
    echo "  3. Readers never see partial/corrupted data"
    echo ""
    echo "Output location: /pool/ssd8tb/data/iso/unified_iso_data/parquet/"
    exit 0
else
    echo "⚠️  Some ISOs failed - check logs"
    exit 1
fi
