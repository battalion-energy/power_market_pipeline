#!/bin/bash
#
# Setup MISO Data Update Cron Job
# Adds daily cron job to download MISO market data
#

echo "🔧 MISO Cron Job Setup"
echo "======================"
echo ""

# Check if logs directory exists
if [ ! -d "/home/enrico/logs" ]; then
    echo "📁 Creating logs directory..."
    mkdir -p /home/enrico/logs
fi

# Check if cron script exists
CRON_SCRIPT="/home/enrico/projects/power_market_pipeline/update_miso_cron.sh"
if [ ! -f "$CRON_SCRIPT" ]; then
    echo "❌ Error: $CRON_SCRIPT not found!"
    exit 1
fi

# Make sure it's executable
chmod +x "$CRON_SCRIPT"

# Cron job entry
CRON_ENTRY="0 3 * * * $CRON_SCRIPT"
CRON_COMMENT="# MISO Data Update - Daily at 3:00 AM"

# Check if cron job already exists
if crontab -l 2>/dev/null | grep -q "$CRON_SCRIPT"; then
    echo "⚠️  MISO cron job already exists"
    echo ""
    echo "Current MISO cron job:"
    crontab -l | grep -A1 "MISO"
    echo ""
    echo "To update, first remove the old entry with:"
    echo "  crontab -e"
    echo "Then re-run this script."
    exit 1
fi

# Add cron job
echo "📋 Adding MISO cron job..."
(crontab -l 2>/dev/null; echo ""; echo "$CRON_COMMENT"; echo "$CRON_ENTRY") | crontab -

if [ $? -eq 0 ]; then
    echo "✅ MISO cron job installed successfully!"
    echo ""
    echo "Schedule: Daily at 3:00 AM"
    echo "Downloads: Yesterday's and today's data"
    echo "Log files: /home/enrico/logs/miso_update_*.log"
    echo "Latest log: /home/enrico/logs/miso_update_latest.log"
    echo ""
    echo "Current crontab:"
    echo "================"
    crontab -l | grep -A1 "MISO"
    echo ""
    echo "💡 To test manually:"
    echo "   $CRON_SCRIPT"
    echo ""
    echo "💡 To view logs:"
    echo "   tail -f /home/enrico/logs/miso_update_latest.log"
else
    echo "❌ Failed to install cron job"
    exit 1
fi
