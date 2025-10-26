#!/bin/bash
#
# Test all fixed converters for 2024 DA-only
#

cd "$(dirname "$0")"

LOG_DIR="/pool/ssd8tb/data/iso/unified_iso_data/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=========================================="
echo "Testing Fixed Converters - $TIMESTAMP"
echo "=========================================="
echo ""

# Test NYISO
echo "Testing NYISO..."
python3 nyiso_parquet_converter.py --year 2024 --da-only > "$LOG_DIR/nyiso_fixed_$TIMESTAMP.log" 2>&1
echo "NYISO exit code: $?"

# Test ISONE
echo "Testing ISONE..."
python3 isone_parquet_converter.py --year 2024 --da-only > "$LOG_DIR/isone_fixed_$TIMESTAMP.log" 2>&1
echo "ISONE exit code: $?"

# Test SPP
echo "Testing SPP..."
python3 spp_parquet_converter.py --year 2024 --da-only > "$LOG_DIR/spp_fixed_$TIMESTAMP.log" 2>&1
echo "SPP exit code: $?"

# Test MISO
echo "Testing MISO..."
python3 miso_parquet_converter.py --year 2024 --da-only > "$LOG_DIR/miso_fixed_$TIMESTAMP.log" 2>&1
echo "MISO exit code: $?"

# Test ERCOT
echo "Testing ERCOT..."
python3 ercot_parquet_converter.py --year 2024 --da-only > "$LOG_DIR/ercot_fixed_$TIMESTAMP.log" 2>&1
echo "ERCOT exit code: $?"

echo ""
echo "=========================================="
echo "All tests complete!"
echo "Check logs in: $LOG_DIR"
echo "=========================================="

# Show summary
python3 monitor_conversion.py
