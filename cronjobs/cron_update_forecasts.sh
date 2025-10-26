#!/bin/bash
#
# Cronjob script to update weather forecasts
# This script should be run daily (or multiple times per day)
#
# Add to crontab with:
#   crontab -e
# Then add lines like:
#   # Update weather forecasts every 6 hours
#   0 */6 * * * /home/enrico/projects/power_market_pipeline/scripts/cron_update_forecasts.sh >> /pool/ssd8tb/data/weather_data/logs/cron.log 2>&1
#
#   # Or update daily at 6 AM
#   0 6 * * * /home/enrico/projects/power_market_pipeline/scripts/cron_update_forecasts.sh >> /pool/ssd8tb/data/weather_data/logs/cron.log 2>&1

# Load environment
export PATH="/home/enrico/.pyenv/shims:/home/enrico/.pyenv/bin:$PATH"
export PYENV_ROOT="/home/enrico/.pyenv"
eval "$(pyenv init -)"

# Change to project directory
cd /home/enrico/projects/power_market_pipeline || exit 1

# Load .env file
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Run the forecast updater
echo "=========================================="
echo "Weather Forecast Update"
echo "Time: $(date)"
echo "=========================================="

python3 scripts/update_weather_forecasts.py --source openmeteo

EXIT_CODE=$?

echo "Exit code: $EXIT_CODE"
echo ""

exit $EXIT_CODE
