#!/bin/bash
# Monitor ERCOT Web Service Downloads

echo "=========================================================================="
echo "ERCOT Download Monitor"
echo "=========================================================================="
echo ""

# Check if download is running
if pgrep -f "ercot_ws_download_all.py" > /dev/null; then
    echo "✓ Download process is RUNNING"
    echo ""
else
    echo "✗ No download process detected"
    echo ""
fi

# Show recent log activity
echo "Recent Activity (last 20 lines):"
echo "--------------------------------------------------------------------------"
tail -20 ercot_ws_download.log
echo ""

# Check state file
if [ -f ercot_download_state.json ]; then
    echo "=========================================================================="
    echo "Download Progress:"
    echo "=========================================================================="
    python3 -c "
import json
from datetime import datetime

with open('ercot_download_state.json', 'r') as f:
    state = json.load(f)

print(f\"Last updated: {state.get('last_updated', 'Unknown')}\")
print(f\"Datasets tracked: {len(state.get('datasets', {}))}\")
print(\"\")

for dataset_name, info in state.get('datasets', {}).items():
    print(f\"📊 {dataset_name}:\")
    if 'last_timestamp' in info:
        print(f\"   Last data: {info['last_timestamp']}\")
    if 'last_download' in info:
        print(f\"   Last download: {info['last_download']}\")
    if 'last_records_count' in info:
        print(f\"   Records in last chunk: {info['last_records_count']:,}\")
    print(\"\")
" 2>/dev/null || echo "No state file data available yet"
fi

# Check CSV files created
echo "=========================================================================="
echo "Downloaded CSV Files:"
echo "=========================================================================="
find /pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/ -name "*.csv" -newer ercot_download_state.json -type f 2>/dev/null | head -20

echo ""
echo "=========================================================================="
echo "To follow live progress: tail -f ercot_ws_download.log"
echo "=========================================================================="
