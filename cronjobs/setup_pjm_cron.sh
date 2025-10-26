#!/bin/bash
# Setup cron job for PJM daily updates
# Runs once per day at 9 AM CT (data is usually posted by then)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPDATE_SCRIPT="${SCRIPT_DIR}/update_pjm_cron.sh"

echo "================================================================================"
echo "PJM Cron Setup"
echo "================================================================================"
echo "Script directory: $SCRIPT_DIR"
echo "Update script: $UPDATE_SCRIPT"
echo ""

# Make update script executable
chmod +x "$UPDATE_SCRIPT"

# Create the cron entry
# Run at 9:00 AM Central Time daily
# This gives PJM time to post yesterday's data (usually posted by 8 AM CT)
CRON_ENTRY="# PJM Daily Data Updates (9 AM CT)
0 9 * * * ${UPDATE_SCRIPT} >/dev/null 2>&1"

echo "Proposed cron entry:"
echo "$CRON_ENTRY"
echo ""

# Check if entry already exists
if crontab -l 2>/dev/null | grep -q "update_pjm_cron.sh"; then
    echo "⚠️  WARNING: A cron entry for update_pjm_cron.sh already exists!"
    echo ""
    echo "Current PJM crontab entries:"
    crontab -l 2>/dev/null | grep -B1 "PJM\|pjm"
    echo ""
    read -p "Do you want to replace it? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted. No changes made."
        exit 1
    fi
    # Remove old entry
    crontab -l 2>/dev/null | grep -v "update_pjm_cron.sh" | grep -v "PJM Daily Data Updates" | crontab -
fi

# Add new cron entry
(crontab -l 2>/dev/null; echo ""; echo "$CRON_ENTRY") | crontab -

echo "✅ Cron job installed successfully!"
echo ""
echo "The script will run daily at:"
echo "  • 9:00 AM Central Time"
echo ""
echo "What it does:"
echo "  • Auto-resumes from last downloaded date for each data type"
echo "  • Downloads DA nodal LMPs (all ~22K nodes)"
echo "  • Downloads RT hourly nodal LMPs (all ~14K-22K nodes)"
echo "  • Downloads RT 5-minute nodal LMPs (last 6 months, all nodes)"
echo "  • Downloads DA ancillary services"
echo "  • Conservative throttling: 2-second delays + 5 requests/minute"
echo "  • Exponential backoff retry on rate limits (30s, 60s, 120s, 240s, 480s)"
echo "  • Catches up gaps if cron fails for a few days"
echo ""
echo "Logs location: /home/enrico/logs/pjm_update_*.log"
echo "Latest log: /home/enrico/logs/pjm_update_latest.log"
echo ""
echo "To view logs:"
echo "  tail -f /home/enrico/logs/pjm_update_latest.log"
echo ""
echo "To test the update manually:"
echo "  ${UPDATE_SCRIPT}"
echo ""
echo "To remove cron job:"
echo "  crontab -e  (then delete the PJM entry)"
echo ""
echo "Current crontab:"
crontab -l
echo ""
echo "================================================================================"

exit 0
