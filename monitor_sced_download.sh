#!/bin/bash
# Monitor SCED Gen Resources Download Progress

echo "=========================================================================="
echo "60-Day SCED Gen Resources Download Monitor"
echo "=========================================================================="
echo ""

# Check if process is running
if pgrep -f "60d_SCED_Gen_Resources" > /dev/null; then
    echo "✓ Download process is RUNNING"
    PID=$(pgrep -f "60d_SCED_Gen_Resources" | head -1)
    echo "  PID: $PID"
    echo ""
else
    echo "✗ No download process detected"
    echo ""
fi

# Show recent log activity
if [ -f sced_gen_download.log ]; then
    echo "Recent Activity (last 15 lines):"
    echo "--------------------------------------------------------------------------"
    tail -15 sced_gen_download.log 2>/dev/null || echo "Log file not readable"
    echo ""
fi

# Check downloaded files
echo "=========================================================================="
echo "Downloaded Files:"
echo "=========================================================================="
SCED_DIR="/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/60-Day_SCED_Disclosure_Reports/Gen_Resources"
if [ -d "$SCED_DIR" ]; then
    FILE_COUNT=$(ls -1 "$SCED_DIR"/*.csv 2>/dev/null | wc -l)
    echo "CSV files created: $FILE_COUNT"

    if [ $FILE_COUNT -gt 0 ]; then
        echo ""
        echo "Most recent files:"
        ls -lht "$SCED_DIR"/*.csv 2>/dev/null | head -5

        echo ""
        echo "Total size:"
        du -sh "$SCED_DIR" 2>/dev/null
    fi
else
    echo "Directory not yet created"
fi

echo ""
echo "=========================================================================="
echo "To follow live progress: tail -f sced_gen_download.log"
echo "To check state: cat ercot_download_state.json | jq '.datasets.\"60d_SCED_Gen_Resources\"'"
echo "=========================================================================="
