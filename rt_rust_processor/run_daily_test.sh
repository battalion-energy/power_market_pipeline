#!/bin/bash

echo "🔍 Testing BESS Daily Revenue Processor"
echo "========================================"

# First, ensure binary is up to date
echo "Building release binary..."
cargo build --release --quiet

# Check if build succeeded
if [ $? -ne 0 ]; then
    echo "❌ Build failed"
    exit 1
fi

echo "✅ Build successful"

# Run the processor
echo ""
echo "Running BESS daily revenue processor..."
./target/release/rt_rust_processor --bess-daily-revenue

# Check if output directories were created
if [ -d "bess_daily_revenues/daily" ] && [ -d "bess_daily_revenues/monthly" ]; then
    echo ""
    echo "✅ Output directories created:"
    echo "  - bess_daily_revenues/daily"
    echo "  - bess_daily_revenues/monthly"
    
    # Count files
    daily_count=$(find bess_daily_revenues/daily -name "*.parquet" | wc -l)
    monthly_count=$(find bess_daily_revenues/monthly -name "*.parquet" | wc -l)
    
    echo ""
    echo "📊 Files created:"
    echo "  - Daily parquet files: $daily_count"
    echo "  - Monthly parquet files: $monthly_count"
else
    echo "❌ Output directories not created"
fi