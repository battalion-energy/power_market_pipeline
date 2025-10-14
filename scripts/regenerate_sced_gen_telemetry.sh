#!/bin/bash
set -e

export ERCOT_DATA_DIR="/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data"
export PATH="$HOME/.cargo/bin:$PATH"

cd /home/enrico/projects/power_market_pipeline/ercot_data_processor

echo "Building Rust processor with telemetry trailing space fix..."
cargo build --release --jobs 24

echo ""
echo "Regenerating SCED_Gen_Resources with Telemetered Net Output (with trailing space fix)..."
cargo run --release --bin ercot_data_processor -- --annual-rollup --dataset SCED_Gen_Resources

echo ""
echo "Done! Checking if TelemeteredNetOutput is now in the parquet..."
python3 << 'PYEOF'
import polars as pl
from pathlib import Path

df = pl.read_parquet(
    Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/rollup_files/SCED_Gen_Resources/2024.parquet"),
    n_rows=0
)

if "TelemeteredNetOutput" in df.columns:
    print("✓ SUCCESS - TelemeteredNetOutput is now in the parquet!")
else:
    print("✗ FAIL - TelemeteredNetOutput still missing")
    print("Available columns:", df.columns)
PYEOF
