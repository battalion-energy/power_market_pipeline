#!/bin/bash
#
# Setup NYISO Data Update Cron Job
# Adds daily cron job to download NYISO market data using auto-resume
#

echo "🔧 NYISO Cron Job Setup"
echo "======================="
echo ""

# Check if logs directory exists
if [ ! -d "/home/enrico/logs" ]; then
    echo "📁 Creating logs directory..."
    mkdir -p /home/enrico/logs
fi

# Check if cron script exists
CRON_SCRIPT="/home/enrico/projects/power_market_pipeline/update_nyiso_cron.sh"
if [ ! -f "$CRON_SCRIPT" ]; then
    echo "❌ Error: $CRON_SCRIPT not found!"
    exit 1
fi

# Make sure it's executable
chmod +x "$CRON_SCRIPT"

# Cron job entry - Run at 4:00 AM daily (after ERCOT and MISO/Weather at 3 AM)
CRON_ENTRY="0 4 * * * $CRON_SCRIPT"
CRON_COMMENT="# NYISO Data Update - Daily at 4:00 AM (auto-resume)"

# Check if cron job already exists
if crontab -l 2>/dev/null | grep -q "$CRON_SCRIPT"; then
    echo "⚠️  NYISO cron job already exists"
    echo ""
    echo "Current NYISO cron job:"
    crontab -l | grep -A1 "NYISO"
    echo ""
    echo "To update, first remove the old entry with:"
    echo "  crontab -e"
    echo "Then re-run this script."
    exit 1
fi

# Add cron job
echo "📋 Adding NYISO cron job..."
(crontab -l 2>/dev/null; echo ""; echo "$CRON_COMMENT"; echo "$CRON_ENTRY") | crontab -

if [ $? -eq 0 ]; then
    echo "✅ NYISO cron job installed successfully!"
    echo ""
    echo "Schedule: Daily at 4:00 AM"
    echo "Downloads: All data types with auto-resume"
    echo "  - LMP Day-Ahead Hourly"
    echo "  - LMP Real-Time 5-min"
    echo "  - Ancillary Services DA & RT"
    echo "  - Load (5-min actual)"
    echo "  - Fuel Mix"
    echo ""
    echo "Log files: /home/enrico/logs/nyiso_update_*.log"
    echo "Latest log: /home/enrico/logs/nyiso_update_latest.log"
    echo ""
    echo "Current crontab:"
    echo "================"
    crontab -l | grep -A1 "NYISO"
    echo ""
    echo "💡 To test manually:"
    echo "   $CRON_SCRIPT"
    echo ""
    echo "💡 To view logs:"
    echo "   tail -f /home/enrico/logs/nyiso_update_latest.log"
    echo ""
    echo "Note: NYISO downloader uses auto-resume - it will automatically"
    echo "      start from the last downloaded date, making it perfect for"
    echo "      cron jobs that might occasionally fail or skip days."
else
    echo "❌ Failed to install cron job"
    exit 1
fi
