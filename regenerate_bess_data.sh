#!/bin/bash
set -e

echo "=========================================="
echo "BESS Data Regeneration Script"
echo "=========================================="
echo ""

# Set environment variables
export ERCOT_DATA_DIR="/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data"

cd /home/enrico/projects/power_market_pipeline/ercot_data_processor

echo "Step 1: Building Rust processor with fixes..."
echo "----------------------------------------------"
PATH="$HOME/.cargo/bin:$PATH" cargo build --release --jobs 24
echo "✅ Build complete"
echo ""

echo "Step 2: Regenerating SCED Load Resources (with ResourceName column)..."
echo "----------------------------------------------------------------------"
PATH="$HOME/.cargo/bin:$PATH" cargo run --release --bin ercot_data_processor -- \
    --annual-rollup \
    --dataset SCED_Load_Resources

echo "✅ SCED Load Resources complete"
echo ""

echo "Step 3: Processing DAM Energy Bid Awards (for DA charging)..."
echo "--------------------------------------------------------------"
PATH="$HOME/.cargo/bin:$PATH" cargo run --release --bin ercot_data_processor -- \
    --annual-rollup \
    --dataset DAM_Energy_Bid_Awards

echo "✅ DAM Energy Bid Awards complete"
echo ""

echo "=========================================="
echo "Verification"
echo "=========================================="
echo ""

echo "SCED Load Resources columns:"
python3 << 'EOF'
import os
import pyarrow.parquet as pq
path = os.path.join(os.environ['ERCOT_DATA_DIR'], 'rollup_files/SCED_Load_Resources/2024.parquet')
table = pq.read_table(path)
for name in table.schema.names:
    if 'Resource' in name or 'Name' in name:
        print(f"  ✓ {name}")
EOF

echo ""
echo "DAM Energy Bid Awards files:"
ls -lh "$ERCOT_DATA_DIR/rollup_files/DAM_Energy_Bid_Awards/" 2>/dev/null || echo "  Directory not yet created"

echo ""
echo "=========================================="
echo "✅ ALL DONE!"
echo "=========================================="
