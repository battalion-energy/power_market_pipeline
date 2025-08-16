#!/bin/bash

# Minimal script to process ONLY COP files (faster than full rollup)
# Can be run from any directory

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROCESSOR="$SCRIPT_DIR/target/release/rt_rust_processor"

echo "🔄 Processing COP files only..."

# Just run the processor with COP data type
"$PROCESSOR" \
    --process-annual \
    --data-type COP_Snapshots \
    2>&1 | tee /tmp/cop_only.log

echo "✅ Done! Log saved to /tmp/cop_only.log"