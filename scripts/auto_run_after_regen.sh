#!/bin/bash
# Wait for SCED regeneration to complete, then run 5-year calculation

echo "Waiting for SCED regeneration to complete..."
echo "Monitor: tail -f /tmp/sced_regen_fixed.log"
echo ""

# Wait for the process to finish
while ps aux | grep -q "[e]rcot_data_processor.*SCED_Gen"; do
    sleep 30
done

echo "✅ SCED regeneration complete!"
echo ""

# Verify telemetry column exists
python3 << 'PYEOF'
import polars as pl
from pathlib import Path

df = pl.read_parquet(
    Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/rollup_files/SCED_Gen_Resources/2024.parquet"),
    n_rows=0
)

if "TelemeteredNetOutput" in df.columns:
    print("✅ TelemeteredNetOutput confirmed in parquet!")
    exit(0)
else:
    print("✗ TelemeteredNetOutput still missing!")
    print("Available columns:", df.columns)
    exit(1)
PYEOF

if [ $? -eq 0 ]; then
    echo ""
    echo "Starting 5-year revenue calculation (2020-2024)..."
    echo ""
    python3 /home/enrico/projects/power_market_pipeline/run_bess_revenue_5_years.py 2>&1 | tee bess_revenue_5_years.log
else
    echo "ERROR: Telemetry column not found. Check regeneration."
    exit 1
fi
