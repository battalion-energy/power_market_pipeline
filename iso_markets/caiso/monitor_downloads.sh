#!/bin/bash
# Monitor CAISO download progress

LOG_2023="/home/enrico/logs/caiso_csv_2023_2025.log"
LOG_2022="/home/enrico/logs/caiso_csv_2022.log"
LOG_RT5MIN="/home/enrico/logs/caiso_rt_5min_2022_2025.log"

DATA_DIR="/home/enrico/data/CAISO_data/csv_files"

echo "================================================================================"
echo "CAISO Download Monitor - $(date)"
echo "================================================================================"
echo ""

# Function to check if process is running
check_process() {
    local log_file=$1
    local description=$2

    # Get last few lines of log
    if [ -f "$log_file" ]; then
        echo "### $description ###"
        echo "Last 5 lines of log:"
        tail -5 "$log_file" 2>/dev/null
        echo ""

        # Check if completed
        if tail -20 "$log_file" 2>/dev/null | grep -q "All data updated successfully"; then
            echo "✅ STATUS: COMPLETED"
        elif tail -20 "$log_file" 2>/dev/null | grep -q "Total:.*Success:"; then
            echo "✅ STATUS: COMPLETED WITH SUMMARY"
        else
            echo "⏳ STATUS: IN PROGRESS"
        fi
        echo ""
    else
        echo "### $description ###"
        echo "⚠️  Log file not found: $log_file"
        echo ""
    fi
}

# Check each download
check_process "$LOG_2023" "2023-2025 DA Nodal + AS"
check_process "$LOG_2022" "2022 DA Nodal + AS"
check_process "$LOG_RT5MIN" "2022-2025 RT 5-Min Nodal"

# Count downloaded files
echo "================================================================================"
echo "File Counts"
echo "================================================================================"
echo "DA Nodal files:     $(ls -1 $DATA_DIR/da_nodal/*.csv 2>/dev/null | wc -l)"
echo "RT 5-Min files:     $(ls -1 $DATA_DIR/rt_5min_nodal/*.csv 2>/dev/null | wc -l)"
echo "AS files:           $(ls -1 $DATA_DIR/da_ancillary_services/*.csv 2>/dev/null | wc -l)"
echo ""

# Disk usage
echo "================================================================================"
echo "Disk Usage"
echo "================================================================================"
du -sh $DATA_DIR/da_nodal 2>/dev/null || echo "DA Nodal: No data yet"
du -sh $DATA_DIR/rt_5min_nodal 2>/dev/null || echo "RT 5-Min: No data yet"
du -sh $DATA_DIR/da_ancillary_services 2>/dev/null || echo "AS: No data yet"
echo ""

# Check for running processes
echo "================================================================================"
echo "Running Processes"
echo "================================================================================"
ps aux | grep "update_caiso_with_resume" | grep -v grep | awk '{print "PID "$2": "$11" "$12" "$13" "$14" "$15}'
echo ""

echo "================================================================================"
echo "Monitor completed at $(date)"
echo "================================================================================"
