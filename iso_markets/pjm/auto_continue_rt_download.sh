#!/bin/bash
# Auto-monitor and continue RT hourly downloads
# Checks every hour if 2024 is done, then starts 2019-2023

LOG_FILE="auto_continue.log"
PID_2024=739233
RT_DIR="/home/enrico/data/PJM_data/csv_files/rt_hourly_nodal"

echo "===========================================================" | tee -a $LOG_FILE
echo "$(date): Starting auto-monitor for RT downloads" | tee -a $LOG_FILE
echo "Monitoring PID: $PID_2024" | tee -a $LOG_FILE
echo "===========================================================" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

while true; do
    echo "$(date): Checking status..." | tee -a $LOG_FILE

    # Check if 2024 process is still running
    if ps -p $PID_2024 > /dev/null 2>&1; then
        # Count current files
        COUNT_2024=$(ls $RT_DIR/nodal_rt_hourly_lmp_2024-*.csv 2>/dev/null | wc -l)
        TOTAL_FILES=$(ls $RT_DIR/*.csv 2>/dev/null | wc -l)
        SIZE=$(du -sh $RT_DIR | awk '{print $1}')

        echo "  2024 download still running (PID $PID_2024)" | tee -a $LOG_FILE
        echo "  Progress: $COUNT_2024/366 files (2024), $TOTAL_FILES total, $SIZE" | tee -a $LOG_FILE
        echo "  Sleeping for 1 hour..." | tee -a $LOG_FILE
        echo "" | tee -a $LOG_FILE
        sleep 3600  # 1 hour
    else
        echo "  2024 download process completed!" | tee -a $LOG_FILE

        # Verify completion
        COUNT_2024=$(ls $RT_DIR/nodal_rt_hourly_lmp_2024-*.csv 2>/dev/null | wc -l)
        echo "  Final 2024 file count: $COUNT_2024/366" | tee -a $LOG_FILE

        if [ $COUNT_2024 -ge 365 ]; then
            echo "  ✓ 2024 download complete!" | tee -a $LOG_FILE
            echo "" | tee -a $LOG_FILE
            echo "===========================================================" | tee -a $LOG_FILE
            echo "$(date): Starting 2019-2023 download..." | tee -a $LOG_FILE
            echo "===========================================================" | tee -a $LOG_FILE

            # Start 2019-2023 download
            cd /home/enrico/projects/power_market_pipeline/iso_markets/pjm
            nohup nice -n 19 python download_historical_nodal_rt_lmps.py \
                --granularity hourly \
                --start-date 2019-01-01 \
                --end-date 2023-12-31 \
                --quick-skip \
                > rt_hourly_2019-2023.log 2>&1 &

            NEW_PID=$!
            echo "Started 2019-2023 download, PID: $NEW_PID" | tee -a $LOG_FILE
            echo "Log file: rt_hourly_2019-2023.log" | tee -a $LOG_FILE
            echo "" | tee -a $LOG_FILE
            echo "Monitor complete. Check rt_hourly_2019-2023.log for progress." | tee -a $LOG_FILE

            exit 0
        else
            echo "  ⚠️  Warning: Only $COUNT_2024 files found, expected 366" | tee -a $LOG_FILE
            echo "  Starting 2019-2023 anyway..." | tee -a $LOG_FILE
            echo "" | tee -a $LOG_FILE

            # Start 2019-2023 download anyway
            cd /home/enrico/projects/power_market_pipeline/iso_markets/pjm
            nohup nice -n 19 python download_historical_nodal_rt_lmps.py \
                --granularity hourly \
                --start-date 2019-01-01 \
                --end-date 2023-12-31 \
                --quick-skip \
                > rt_hourly_2019-2023.log 2>&1 &

            NEW_PID=$!
            echo "Started 2019-2023 download, PID: $NEW_PID" | tee -a $LOG_FILE

            exit 0
        fi
    fi
done
