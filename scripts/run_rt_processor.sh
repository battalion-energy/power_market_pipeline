#!/bin/bash
# Optimized RT price processor runner
# Handles massive file counts (500K+ files) without hitting system limits

echo "🚀 ERCOT RT Price Processor - Optimized for Large Datasets"
echo "==========================================================="
echo ""
echo "System Configuration:"
echo "  CPU Cores: $(nproc --all) threads (24 physical cores)"
echo "  Available RAM: $(free -h | awk '/^Mem:/ {print $2}')"
echo "  Current file descriptor limit: $(ulimit -n)"
echo ""

# Set optimal environment variables
export RAYON_NUM_THREADS=16  # Reduced to prevent file descriptor exhaustion
export POLARS_MAX_THREADS=24  # Use physical cores for computation
export RUST_MIN_STACK=8388608  # 8MB stack per thread

# Optional: Increase file descriptor limit if needed (requires sudo)
# sudo sysctl -w fs.file-max=2000000
# ulimit -n 65536

echo "Processing Configuration:"
echo "  RAYON_NUM_THREADS: $RAYON_NUM_THREADS (for file I/O)"
echo "  POLARS_MAX_THREADS: $POLARS_MAX_THREADS (for computation)"
echo ""

# Change to processor directory
cd /home/enrico/projects/power_market_pipeline/ercot_data_processor

# Run the processor
echo "Starting RT price processing..."
echo "This will process 515,814 files - expect this to take some time."
echo ""

./target/release/ercot_data_processor \
    --annual-rollup /home/enrico/data/ERCOT_data \
    --dataset RT_prices

echo ""
echo "✅ Processing complete!"