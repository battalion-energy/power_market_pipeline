#!/bin/bash

# Simple script to reprocess COP files using existing processor
# Handles both CompleteCOP_*.csv (before Dec 13, 2022) and 60d_COP_*.csv (after)

set -e  # Exit on error

echo "=========================================="
echo "COP Files Reprocessing Script"
echo "=========================================="
echo ""

# Configuration
BASE_DIR="/Users/enrico/data/ERCOT_data"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROCESSOR="$SCRIPT_DIR/target/release/rt_rust_processor"

# Check if processor exists, build if not
if [ ! -f "$PROCESSOR" ]; then
    echo "Building processor..."
    cd "$SCRIPT_DIR"
    cargo build --release
fi

# Count COP files
echo "📊 Counting COP files..."
python3 -c "
import glob
import os
cop_dir = '$BASE_DIR/60-Day_COP_Adjustment_Period_Snapshot/csv'
cop_60d = glob.glob(os.path.join(cop_dir, '60d_COP_*.csv'))
cop_complete = glob.glob(os.path.join(cop_dir, 'CompleteCOP_*.csv'))
print(f'  CompleteCOP files: {len(cop_complete)}')
print(f'  60d_COP files: {len(cop_60d)}')
print(f'  Total files: {len(cop_60d) + len(cop_complete)}')
"
echo ""

# Run the enhanced annual processor - it will process COP files with both patterns
echo "🚀 Processing COP files..."
echo ""

# Use the enhanced processor which already handles both file patterns
$PROCESSOR --annual-rollup "$BASE_DIR" 2>&1 | \
    tee /tmp/cop_rollup.log | \
    grep -E "COP Snapshot files|Processing year.*files|Saved.*rows.*parquet|Complete:.*COP" || true

echo ""
echo "✅ COP reprocessing complete!"
echo ""

# Show results
echo "📁 Output files:"
ls -lh "$BASE_DIR/rollup_files/COP_Snapshots/"*.parquet 2>/dev/null || echo "No output files found"

echo ""
echo "📄 Full log saved to: /tmp/cop_rollup.log"