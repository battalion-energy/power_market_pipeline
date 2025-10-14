#!/bin/bash
#
# Weather Data Update Cron Job
#
# Add to crontab to run daily at 2 AM:
#   0 2 * * * /home/enrico/projects/power_market_pipeline/update_weather_data_cron.sh
#
# Or weekly on Sundays at 3 AM:
#   0 3 * * 0 /home/enrico/projects/power_market_pipeline/update_weather_data_cron.sh
#

set -e  # Exit on error

# Configuration
PROJECT_DIR="/home/enrico/projects/power_market_pipeline"
LOG_DIR="/pool/ssd8tb/data/weather_data/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/weather_update_$TIMESTAMP.log"

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Start logging
log "=========================================="
log "Weather Data Update Started"
log "=========================================="

# Change to project directory
cd "$PROJECT_DIR"

# Update NASA POWER data (incremental)
log "Updating NASA POWER satellite data..."
uv run python download_nasa_power_weather_v2.py --incremental >> "$LOG_FILE" 2>&1
if [ $? -eq 0 ]; then
    log "✓ NASA POWER update completed"
else
    log "✗ NASA POWER update failed"
fi

# Wait between downloads to be respectful
sleep 10

# Update Meteostat data (incremental)
log "Updating Meteostat ground station data..."
uv run python download_meteostat_weather_v2.py --incremental >> "$LOG_FILE" 2>&1
if [ $? -eq 0 ]; then
    log "✓ Meteostat update completed"
else
    log "✗ Meteostat update failed"
fi

# Summary
log "=========================================="
log "Weather Data Update Finished"
log "=========================================="
log "Log saved to: $LOG_FILE"

# Keep only last 30 days of logs
find "$LOG_DIR" -name "weather_update_*.log" -mtime +30 -delete

# Send notification (optional - uncomment if you want email notifications)
# echo "Weather data updated. See $LOG_FILE" | mail -s "Weather Data Update Complete" your-email@example.com
