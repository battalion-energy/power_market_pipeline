#!/bin/bash
# Monitor ISO-NE download and run audit when complete

echo "Monitoring ISO-NE historical download..."
echo "Log file: isone_historical_full_download.log"
echo ""

# Wait for download to complete by monitoring the log file
# The download is complete when we see "DOWNLOAD COMPLETE" in the log

LOG_FILE="isone_historical_full_download.log"
CHECK_INTERVAL=60  # Check every 60 seconds

while true; do
    if [ -f "$LOG_FILE" ]; then
        # Check if download is complete
        if grep -q "DOWNLOAD COMPLETE" "$LOG_FILE"; then
            echo ""
            echo "✓ Download complete! Starting data quality audit..."
            echo ""

            # Run the audit
            python3 audit_isone_data_quality.py --start-date 2019-01-01 --end-date $(date +%Y-%m-%d)

            echo ""
            echo "✓ Audit complete! Check isone_data_quality_audit.log for details"
            exit 0
        else
            # Show progress
            LAST_DATE=$(grep "Processing 20" "$LOG_FILE" | tail -1 | sed 's/.*Processing \([0-9-]*\).*/\1/')
            DOWNLOADED=$(grep "✓ Saved" "$LOG_FILE" | wc -l)
            FAILED=$(grep "✗ Failed" "$LOG_FILE" | wc -l)

            echo "$(date '+%H:%M:%S') - Still downloading... Last: $LAST_DATE | Downloaded: $DOWNLOADED | Failed: $FAILED"
        fi
    else
        echo "$(date '+%H:%M:%S') - Waiting for log file to appear..."
    fi

    sleep $CHECK_INTERVAL
done
