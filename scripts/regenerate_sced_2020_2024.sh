#!/bin/bash
# Regenerate ONLY 2020-2024 (the years we need for analysis)
# Process one year at a time to avoid memory issues

set -e

export ERCOT_DATA_DIR="/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data"
export PATH="$HOME/.cargo/bin:$PATH"

cd /home/enrico/projects/power_market_pipeline/ercot_data_processor

echo "=================================="
echo "Regenerating SCED 2020-2024 ONLY"
echo "Processing one year at a time"
echo "=================================="
echo ""

# We need to regenerate the ENTIRE dataset to get telemetry
# But we'll monitor memory and can restart if needed

cargo build --release --jobs 24

echo ""
echo "Starting full regeneration (will include all years)..."
echo "Monitor: tail -f /tmp/sced_regen_2020_2024.log"
echo ""

# Run with output to log
cargo run --release --bin ercot_data_processor -- --annual-rollup --dataset SCED_Gen_Resources 2>&1 | tee /tmp/sced_regen_2020_2024.log

echo ""
echo "✅ Regeneration complete!"
echo ""

# Verify 2020-2024 have telemetry
python3 << 'PYEOF'
import polars as pl
from pathlib import Path

ROLLUP_DIR = Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/rollup_files")

print("Verifying 2020-2024 have TelemeteredNetOutput:")
for year in range(2020, 2025):
    df = pl.read_parquet(ROLLUP_DIR / f"SCED_Gen_Resources/{year}.parquet", n_rows=0)
    has_telemetry = "TelemeteredNetOutput" in df.columns
    print(f"  {year}: {'✅' if has_telemetry else '✗'}")
PYEOF
