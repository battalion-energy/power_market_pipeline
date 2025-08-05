#!/bin/bash

# Run BESS daily revenue processor for 2024 only as a test

echo "🔍 Running BESS Daily Revenue Processor for 2024"
echo "================================================"

# Create a temporary directory with only 2024 files for testing
echo "Setting up test environment..."

# Run the processor
echo ""
echo "Running processor..."
./target/release/rt_rust_processor --bess-daily-revenue 2>&1 | grep -E "(2024|Processing|Error|✅|💰|📊)"

# Check results
echo ""
echo "Checking output..."
if [ -d "bess_daily_revenues" ]; then
    echo "✅ Output directory created"
    
    # Count files
    daily_count=$(find bess_daily_revenues/daily -name "*.parquet" 2>/dev/null | wc -l)
    monthly_count=$(find bess_daily_revenues/monthly -name "*.parquet" 2>/dev/null | wc -l)
    
    echo "📊 Files created:"
    echo "  - Daily parquet files: $daily_count"
    echo "  - Monthly parquet files: $monthly_count"
    
    # List 2024 files
    echo ""
    echo "📁 2024 files:"
    ls -la bess_daily_revenues/daily/*2024* 2>/dev/null || echo "  No 2024 daily files found"
    ls -la bess_daily_revenues/monthly/*2024* 2>/dev/null || echo "  No 2024 monthly files found"
fi