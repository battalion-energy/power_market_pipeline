#!/bin/bash
#
# Setup SPP Data Update Cron Job
# Adds daily cron job to download SPP market data using auto-resume
#

echo "🔧 SPP Cron Job Setup"
echo "====================="
echo ""

# Check if logs directory exists
if [ ! -d "/home/enrico/logs" ]; then
    echo "📁 Creating logs directory..."
    mkdir -p /home/enrico/logs
fi

# Check if cron script exists
CRON_SCRIPT="/home/enrico/projects/power_market_pipeline/update_spp_cron.sh"
if [ ! -f "$CRON_SCRIPT" ]; then
    echo "❌ Error: $CRON_SCRIPT not found!"
    exit 1
fi

# Make sure it's executable
chmod +x "$CRON_SCRIPT"

# Cron job entry - Run at 5:00 AM daily (after ERCOT, MISO, and NYISO)
CRON_ENTRY="0 5 * * * $CRON_SCRIPT"
CRON_COMMENT="# SPP Data Update - Daily at 5:00 AM (auto-resume)"

# Check if cron job already exists
if crontab -l 2>/dev/null | grep -q "$CRON_SCRIPT"; then
    echo "⚠️  SPP cron job already exists"
    echo ""
    echo "Current SPP cron job:"
    crontab -l | grep -A1 "SPP"
    echo ""
    echo "To update, first remove the old entry with:"
    echo "  crontab -e"
    echo "Then re-run this script."
    exit 1
fi

# Add cron job
echo "📋 Adding SPP cron job..."
(crontab -l 2>/dev/null; echo ""; echo "$CRON_COMMENT"; echo "$CRON_ENTRY") | crontab -

if [ $? -eq 0 ]; then
    echo "✅ SPP cron job installed successfully!"
    echo ""
    echo "Schedule: Daily at 5:00 AM"
    echo "Downloads: All data types with auto-resume"
    echo "  - Day-Ahead LMP"
    echo "  - Real-Time LMP (daily aggregated)"
    echo "  - Ancillary Services (DA & RT MCP)"
    echo ""
    echo "Log files: /home/enrico/logs/spp_update_*.log"
    echo "Latest log: /home/enrico/logs/spp_update_latest.log"
    echo ""
    echo "Current crontab:"
    echo "================"
    crontab -l | grep -A1 "SPP"
    echo ""
    echo "💡 To test manually:"
    echo "   $CRON_SCRIPT"
    echo ""
    echo "💡 To view logs:"
    echo "   tail -f /home/enrico/logs/spp_update_latest.log"
    echo ""
    echo "Note: SPP downloader uses auto-resume - it will automatically"
    echo "      start from the last downloaded date, making it perfect for"
    echo "      cron jobs that might occasionally fail or skip days."
else
    echo "❌ Failed to install cron job"
    exit 1
fi
