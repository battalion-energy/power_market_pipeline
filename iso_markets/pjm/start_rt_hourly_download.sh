#!/bin/bash
# Start RT Hourly Nodal Download for 2025
# Run this AFTER the DA nodal download completes

cd /home/enrico/projects/power_market_pipeline/iso_markets/pjm

echo "Starting RT Hourly Nodal Download for 2025..."
echo "This will download 285 days of data (~9.7 GB)"
echo "Estimated time: ~9.5 hours"
echo ""

# Start the download in background with nohup
nohup nice -n 19 python download_historical_nodal_rt_lmps.py \
    --granularity hourly \
    --year 2025 \
    --quick-skip \
    > rt_hourly_2025.log 2>&1 &

PID=$!
echo "Started RT hourly download for 2025, PID: $PID"
echo "Log file: rt_hourly_2025.log"
echo ""
echo "To check status:"
echo "  tail -f rt_hourly_2025.log"
echo ""
echo "To check progress:"
echo "  ls /home/enrico/data/PJM_data/csv_files/rt_hourly_nodal/ | wc -l"
echo ""
echo "To restart after reboot (it will auto-resume):"
echo "  ./start_rt_hourly_download.sh"
