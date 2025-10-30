#!/bin/bash
#
# Update reBAP data with new download
# Finds latest CSV in Downloads, merges with existing data, removes duplicates
#

set -e

PROJECT_DIR="/home/enrico/projects/power_market_pipeline"
DOWNLOAD_DIR="/home/enrico/Downloads/GermanPowerMarket"
DATA_DIR="/pool/ssd8tb/data/iso/ENTSO_E/csv_files/rebap"
CONVERT_SCRIPT="$PROJECT_DIR/iso_markets/entso_e/scripts/convert_rebap_update.py"

cd "$PROJECT_DIR"

echo "================================================================================"
echo "reBAP Data Update"
echo "================================================================================"
echo "Date: $(date)"
echo ""

# Find latest reBAP CSV in Downloads
echo "Looking for new reBAP CSV in Downloads..."
NEW_FILE=$(ls -t "$DOWNLOAD_DIR"/reBAP*.csv 2>/dev/null | head -1)

if [ -z "$NEW_FILE" ]; then
    echo "❌ No reBAP CSV files found in $DOWNLOAD_DIR"
    echo ""
    echo "Please download reBAP data from:"
    echo "  https://www.netztransparenz.de/de-de/Regelenergie/Ausgleichsenergiepreis/reBAP"
    echo ""
    echo "Save to: $DOWNLOAD_DIR/"
    exit 1
fi

echo "✅ Found: $(basename "$NEW_FILE")"
echo "   Size: $(du -h "$NEW_FILE" | cut -f1)"
echo "   Modified: $(date -r "$NEW_FILE")"
echo ""

# Find existing data file
EXISTING_FILE=$(ls -t "$DATA_DIR"/rebap_de_*.csv 2>/dev/null | grep -v raw | head -1)

if [ -z "$EXISTING_FILE" ]; then
    echo "⚠️  No existing reBAP data found - this will be the first file"
    EXISTING_FILE=""
else
    echo "Existing data: $(basename "$EXISTING_FILE")"
    EXISTING_RECORDS=$(wc -l < "$EXISTING_FILE")
    echo "   Records: $((EXISTING_RECORDS - 1))"
    echo "   Last date: $(tail -1 "$EXISTING_FILE" | cut -d',' -f1)"
    echo ""
fi

# Run Python update script
echo "Processing and merging data..."
uv run python "$CONVERT_SCRIPT" "$NEW_FILE" "$EXISTING_FILE"

echo ""
echo "================================================================================"
echo "✅ reBAP update complete!"
echo "================================================================================"
echo ""
echo "Verify with:"
echo "  tail -5 $DATA_DIR/rebap_de_*.csv"
echo ""
