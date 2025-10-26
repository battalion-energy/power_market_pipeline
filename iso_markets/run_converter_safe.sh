#!/bin/bash
# Safe wrapper for running any ISO converter with memory protection
# Usage: ./run_converter_safe.sh <converter_name> [additional args...]
# Example: ./run_converter_safe.sh miso --year 2024 --da-only

if [ $# -lt 1 ]; then
    echo "Usage: $0 <converter> [args...]"
    echo ""
    echo "Available converters: pjm, caiso, nyiso, spp, isone, miso, ercot"
    echo ""
    echo "Examples:"
    echo "  $0 miso --year 2024 --da-only"
    echo "  $0 ercot --all"
    echo "  $0 pjm --year 2023 --rt-only"
    echo ""
    echo "Memory limits:"
    echo "  - systemd cgroup: 60GB hard cap (if available)"
    echo "  - Python built-in: 50GB self-limit"
    echo "  - Set ISO_CONVERTER_MEMORY_LIMIT_GB env var to change Python limit"
    exit 1
fi

CONVERTER="$1"
shift  # Remove converter name, keep other args

# Memory limits
MEMORY_CAP_GB=${MEMORY_CAP_GB:-60}
MEMORY_HIGH_GB=${MEMORY_HIGH_GB:-55}

# Check if systemd-run is available
if command -v systemd-run &> /dev/null; then
    echo "🛡️  Running ${CONVERTER} with systemd cgroup memory limit: ${MEMORY_CAP_GB}GB"
    exec systemd-run --user --scope \
        -p "MemoryMax=${MEMORY_CAP_GB}G" \
        -p "MemoryHigh=${MEMORY_HIGH_GB}G" \
        -p "MemorySwapMax=4G" \
        --collect \
        python3 "${CONVERTER}_parquet_converter.py" "$@"
else
    echo "🛡️  Running ${CONVERTER} with Python built-in memory limit (50GB default)"
    echo "ℹ️  Install systemd for additional cgroup protection"
    exec python3 "${CONVERTER}_parquet_converter.py" "$@"
fi
