#!/bin/bash
# Setup cron job for CAISO daily updates
# Runs once per day at 10 AM PT (data is usually posted by then)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPDATE_SCRIPT="${SCRIPT_DIR}/update_caiso_cron.sh"

echo "================================================================================"
echo "CAISO Cron Setup"
echo "================================================================================"
echo "Script directory: $SCRIPT_DIR"
echo "Update script: $UPDATE_SCRIPT"
echo ""

# Make update script executable
chmod +x "$UPDATE_SCRIPT"

# Create the cron entry
# Run at 10:00 AM Pacific Time daily
# This gives CAISO time to post yesterday's data (usually posted by 9 AM PT)
CRON_ENTRY="# CAISO Daily Data Updates (10 AM PT)
0 10 * * * ${UPDATE_SCRIPT} >/dev/null 2>&1"

echo "Proposed cron entry:"
echo "$CRON_ENTRY"
echo ""

# Check if entry already exists
if crontab -l 2>/dev/null | grep -q "update_caiso_cron.sh"; then
    echo "⚠️  WARNING: A cron entry for update_caiso_cron.sh already exists!"
    echo ""
    echo "Current CAISO crontab entries:"
    crontab -l 2>/dev/null | grep -B1 "CAISO\|caiso"
    echo ""
    read -p "Do you want to replace it? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted. No changes made."
        exit 1
    fi
    # Remove old entry
    crontab -l 2>/dev/null | grep -v "update_caiso_cron.sh" | grep -v "CAISO Daily Data Updates" | crontab -
fi

# Add new cron entry
(crontab -l 2>/dev/null; echo ""; echo "$CRON_ENTRY") | crontab -

echo "✅ Cron job installed successfully!"
echo ""
echo "The script will run daily at:"
echo "  • 10:00 AM Pacific Time"
echo ""
echo "What it does:"
echo "  • Auto-resumes from last downloaded date for each data type"
echo "  • Downloads DA nodal LMPs (all nodes, hourly)"
echo "  • Downloads RT 5-minute nodal LMPs (all nodes, 5-min intervals)"
echo "  • Downloads DA ancillary services (RU, RD, SR, NR)"
echo "  • Conservative throttling: 5-second delays between requests"
echo "  • Exponential backoff retry on failures (30s, 60s, 120s, 240s, 480s)"
echo "  • Catches up gaps if cron fails for a few days"
echo ""
echo "Note: RT 5-min downloads are LARGE (~1.8 GB per day)"
echo "      Daily updates may take 1-2 hours depending on catchup needed"
echo ""

# Load LOGS_DIR from .env
if [ -f "${SCRIPT_DIR}/../.env" ]; then
    LOGS_DIR=$(grep "^LOGS_DIR=" "${SCRIPT_DIR}/../.env" | cut -d '=' -f2)
fi
LOGS_DIR="${LOGS_DIR:-/pool/ssd8tb/logs}"

echo "Logs location: ${LOGS_DIR}/caiso_update_*.log"
echo "Latest log: ${LOGS_DIR}/caiso_update_latest.log"
echo ""
echo "To view logs:"
echo "  tail -f ${LOGS_DIR}/caiso_update_latest.log"
echo ""
echo "To test the update manually:"
echo "  ${UPDATE_SCRIPT}"
echo ""
echo "To remove cron job:"
echo "  crontab -e  (then delete the CAISO entry)"
echo ""
echo "Current crontab:"
crontab -l
echo ""
echo "================================================================================"

exit 0
