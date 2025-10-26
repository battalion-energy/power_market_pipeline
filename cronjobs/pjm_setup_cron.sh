#!/bin/bash
# Setup cron jobs for PJM daily updates
# Run 3x daily: 8am, 2pm, 8pm ET

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Path to the daily update script
DAILY_UPDATE_SCRIPT="$SCRIPT_DIR/daily_update_pjm.py"

# Path to Python (using system python3)
PYTHON_BIN=$(which python3)

# Log file
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

echo "PJM Cron Setup"
echo "=============="
echo "Script directory: $SCRIPT_DIR"
echo "Daily update script: $DAILY_UPDATE_SCRIPT"
echo "Python: $PYTHON_BIN"
echo "Log directory: $LOG_DIR"
echo ""

# Create the cron entry
CRON_ENTRY="# PJM Daily Data Updates (3x daily: 8am, 2pm, 8pm ET)
0 8,14,20 * * * cd $SCRIPT_DIR && $PYTHON_BIN $DAILY_UPDATE_SCRIPT >> $LOG_DIR/daily_update.log 2>&1"

echo "Proposed cron entry:"
echo "$CRON_ENTRY"
echo ""

# Check if entry already exists
if crontab -l 2>/dev/null | grep -q "daily_update_pjm.py"; then
    echo "WARNING: A cron entry for daily_update_pjm.py already exists!"
    echo "Current crontab entries:"
    echo ""
    crontab -l | grep -A1 "PJM"
    echo ""
    read -p "Do you want to replace it? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted. No changes made."
        exit 1
    fi
    # Remove old entry
    crontab -l | grep -v "daily_update_pjm.py" | grep -v "PJM Daily Data Updates" | crontab -
fi

# Add new cron entry
(crontab -l 2>/dev/null; echo "$CRON_ENTRY") | crontab -

echo "✓ Cron job installed successfully!"
echo ""
echo "Current crontab:"
crontab -l
echo ""
echo "The script will run at:"
echo "  - 8:00 AM ET"
echo "  - 2:00 PM ET"
echo "  - 8:00 PM ET"
echo ""
echo "Logs will be written to: $LOG_DIR/daily_update.log"
echo ""
echo "To view logs: tail -f $LOG_DIR/daily_update.log"
echo "To remove cron job: crontab -e (then delete the PJM entry)"
