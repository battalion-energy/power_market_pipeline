#!/bin/bash
#
# Setup ERCOT ESR Daily Download Cron Job
# Adds cron job to download ESR data daily at 12:05 AM Central Time
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRON_SCRIPT="${SCRIPT_DIR}/download_esr_daily.sh"

# Ensure cron script is executable
chmod +x "$CRON_SCRIPT"

# Cron schedules: Run twice daily for redundancy
# 12:05 AM - Primary run to capture previous day's complete data
# 12:05 PM - Backup run in case morning run fails
CRON_SCHEDULE_1="5 0 * * *"
CRON_SCHEDULE_2="5 12 * * *"

# Create the cron job entries
CRON_JOB_1="$CRON_SCHEDULE_1 $CRON_SCRIPT"
CRON_JOB_2="$CRON_SCHEDULE_2 $CRON_SCRIPT"

# Check if cron job already exists
if crontab -l 2>/dev/null | grep -q "$CRON_SCRIPT"; then
    echo "⚠️  ESR download cron job already exists"
    echo ""
    echo "Current cron entry:"
    crontab -l | grep "$CRON_SCRIPT"
    echo ""
    read -p "Do you want to remove and re-add it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Remove existing entry
        crontab -l | grep -v "$CRON_SCRIPT" | crontab -
        echo "✅ Removed existing cron job"
    else
        echo "Keeping existing cron job"
        exit 0
    fi
fi

# Add new cron jobs
(crontab -l 2>/dev/null; echo "$CRON_JOB_1"; echo "$CRON_JOB_2") | crontab -

echo "✅ Successfully added ESR download cron jobs"
echo ""
echo "Cron Schedules (twice daily for redundancy):"
echo "  - 12:05 AM Central Time (primary)"
echo "  - 12:05 PM Central Time (backup)"
echo "Script: $CRON_SCRIPT"
echo ""
echo "To verify the cron job was added, run:"
echo "  crontab -l | grep esr"
echo ""
echo "To view download logs:"
echo "  tail -f /pool/ssd8tb/logs/esr_download_latest.log"
echo ""
echo "Data will be saved to:"
echo "  /home/enrico/projects/power_market_pipeline/ercot_battery_storage_data/esr_archive/"
echo ""
echo "To manually test the download:"
echo "  $CRON_SCRIPT"
