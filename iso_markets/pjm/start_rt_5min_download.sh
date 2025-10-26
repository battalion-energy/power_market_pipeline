#!/bin/bash
# Start RT 5-Min Nodal Download
# Downloads all 5-minute RT nodal data working backwards from 2025 to 2019

cd /home/enrico/projects/power_market_pipeline/iso_markets/pjm

echo "================================================================================"
echo "PJM RT 5-Minute Nodal LMP Download"
echo "================================================================================"
echo ""
echo "This will download RT 5-minute nodal data for 2019-2025:"
echo "  • ~2,477 days of data"
echo "  • ~350 GB total size"
echo "  • ~140 MB per day"
echo "  • Estimated time: ~41 days"
echo ""
echo "Strategy: Start with 2025 (most recent) and work backwards to 2019"
echo ""
echo "================================================================================"
echo ""

# Start with 2025
echo "Starting 2025 RT 5-min download..."
nohup nice -n 19 python download_historical_nodal_rt_lmps.py \
    --granularity 5min \
    --year 2025 \
    --quick-skip \
    > rt_5min_2025.log 2>&1 &

PID_2025=$!
echo "✓ Started 2025 download, PID: $PID_2025"
echo "  Log: rt_5min_2025.log"
echo ""
echo "Monitoring progress..."
sleep 5

# Check if it's running
if ps -p $PID_2025 > /dev/null; then
    echo "✓ 2025 download is running"
    echo ""
    echo "To monitor progress:"
    echo "  tail -f rt_5min_2025.log"
    echo ""
    echo "To check file count:"
    echo "  ls /home/enrico/data/PJM_data/csv_files/rt_5min_nodal/*.csv | wc -l"
    echo ""
    echo "After 2025 completes, manually start 2024:"
    echo "  nice -n 19 python download_historical_nodal_rt_lmps.py --granularity 5min --year 2024 --quick-skip > rt_5min_2024.log 2>&1 &"
    echo ""
else
    echo "✗ Failed to start download. Check rt_5min_2025.log for errors"
    exit 1
fi

exit 0
