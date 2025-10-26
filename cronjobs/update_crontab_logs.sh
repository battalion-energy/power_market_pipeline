#!/bin/bash
# Update crontab to remove hardcoded log paths and use scripts' internal logging

# Load LOGS_DIR from .env
cd /home/enrico/projects/power_market_pipeline || exit 1
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi
LOGS_DIR="${LOGS_DIR:-/pool/ssd8tb/logs}"

echo "Updating crontab to use LOGS_DIR: $LOGS_DIR"

# Create a temporary file with the updated crontab
TEMP_CRON=$(mktemp)

# Export current crontab and update the log redirections
crontab -l > "$TEMP_CRON"

# Update the weather forecasts line to remove hardcoded path
sed -i "s|>> /pool/ssd8tb/data/weather_data/logs/cron.log 2>&1|2>> ${LOGS_DIR}/cron_errors.log|g" "$TEMP_CRON"

# Update PJM line to log errors instead of suppressing
sed -i 's|update_pjm_cron.sh >/dev/null 2>&1|update_pjm_cron.sh 2>> '"${LOGS_DIR}"'/cron_errors.log|g' "$TEMP_CRON"

# Update CAISO line to log errors instead of suppressing
sed -i 's|update_caiso_cron.sh >/dev/null 2>&1|update_caiso_cron.sh 2>> '"${LOGS_DIR}"'/cron_errors.log|g' "$TEMP_CRON"

echo ""
echo "New crontab (changes highlighted):"
echo "=========================================="
cat "$TEMP_CRON"
echo "=========================================="
echo ""

read -p "Apply these changes to crontab? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    crontab "$TEMP_CRON"
    echo "✅ Crontab updated successfully!"
    echo ""
    echo "All cron jobs now:"
    echo "  • Use scripts' internal logging (via tee)"
    echo "  • Send cron errors to: ${LOGS_DIR}/cron_errors.log"
    echo ""
    echo "Updated crontab:"
    crontab -l
else
    echo "❌ Aborted. No changes made."
fi

rm "$TEMP_CRON"
