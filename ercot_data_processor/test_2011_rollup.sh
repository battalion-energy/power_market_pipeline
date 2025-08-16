#!/bin/bash

# Test annual rollup on 2011 data with DST flag evolution

# Create test directory with only 2011 files
TEST_DIR="/tmp/test_2011_ercot"
mkdir -p "$TEST_DIR/Settlement_Point_Prices_at_Resource_Nodes,_Hubs_and_Load_Zones/csv"

# Copy only 2011 files
echo "Copying 2011 files..."
cp /Users/enrico/data/ERCOT_data/Settlement_Point_Prices_at_Resource_Nodes,_Hubs_and_Load_Zones/csv/*2011* "$TEST_DIR/Settlement_Point_Prices_at_Resource_Nodes,_Hubs_and_Load_Zones/csv/"

# Count files
echo "Files copied: $(ls $TEST_DIR/Settlement_Point_Prices_at_Resource_Nodes,_Hubs_and_Load_Zones/csv/*.csv | wc -l)"

# Run the enhanced annual processor
echo "Running annual rollup on 2011 data..."
./target/debug/rt_rust_processor --annual-rollup "$TEST_DIR"

echo "Done!"