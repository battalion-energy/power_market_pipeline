#!/bin/bash
#
# Setup ENTSO-E / Germany Data Auto-Update Cronjob
# Installs daily cron job to update German electricity market data
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "================================================================================"
echo "ENTSO-E / Germany Data Auto-Update Cronjob Setup"
echo "================================================================================"
echo ""
echo "This will install a daily cronjob to automatically update:"
echo "  • ENTSO-E Day-Ahead Prices (Germany-Luxembourg zone)"
echo "  • ENTSO-E Imbalance Prices (Real-time equivalent)"
echo "  • Regelleistung FCR Capacity (Frequency Containment Reserve)"
echo "  • Regelleistung aFRR Capacity & Energy (automatic FRR)"
echo "  • Regelleistung mFRR Capacity & Energy (manual FRR)"
echo ""
echo "Schedule: Daily at 6:00 AM local time (after German market data published)"
echo "Priority: Low (nice -n 19, ionice -c 3) - minimal system impact"
echo "Auto-resume: Yes - continues from last downloaded date for each product"
echo "Timeout: 2 hours"
echo "Logs: \${LOGS_DIR:-/pool/ssd8tb/logs}/entso_e_update_*.log"
echo ""

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "❌ Error: 'uv' command not found"
    echo "   Please install uv: https://github.com/astral-sh/uv"
    exit 1
fi

# Check if update script exists
UPDATE_SCRIPT="${PROJECT_DIR}/iso_markets/entso_e/update_germany_with_resume.py"
if [ ! -f "$UPDATE_SCRIPT" ]; then
    echo "❌ Error: Update script not found at $UPDATE_SCRIPT"
    exit 1
fi

# Check if cron wrapper script exists
CRON_SCRIPT="${SCRIPT_DIR}/update_entso_e_cron.sh"
if [ ! -f "$CRON_SCRIPT" ]; then
    echo "❌ Error: Cron wrapper script not found at $CRON_SCRIPT"
    exit 1
fi

# Make sure cron script is executable
chmod +x "$CRON_SCRIPT"

echo "✅ Prerequisites check passed"
echo ""

# Check current crontab
if crontab -l 2>/dev/null | grep -q "update_entso_e_cron.sh"; then
    echo "⚠️  ENTSO-E cronjob already exists in crontab"
    echo ""
    echo "Current entry:"
    crontab -l 2>/dev/null | grep "update_entso_e_cron.sh"
    echo ""
    read -p "Remove existing entry and reinstall? [y/N]: " -r
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ Installation cancelled"
        exit 0
    fi
    # Remove existing entry
    crontab -l 2>/dev/null | grep -v "update_entso_e_cron.sh" | crontab -
    echo "✅ Removed existing cronjob"
    echo ""
fi

# Install cronjob
echo "📝 Installing cronjob..."

# Create temporary crontab file
TEMP_CRON=$(mktemp)
crontab -l 2>/dev/null > "$TEMP_CRON" || true

# Add new cronjob
# Run at 6:00 AM daily (German market data typically published by this time)
cat >> "$TEMP_CRON" << EOF

# ENTSO-E / Germany Market Data Auto-Update (DA, RT, FCR, aFRR, mFRR)
# Runs daily at 6:00 AM local time with low priority
# Auto-resumes from last downloaded date for each product
0 6 * * * ${CRON_SCRIPT} >/dev/null 2>&1

EOF

# Install new crontab
crontab "$TEMP_CRON"
rm "$TEMP_CRON"

echo "✅ Cronjob installed successfully!"
echo ""
echo "================================================================================"
echo "Installation Complete"
echo "================================================================================"
echo ""
echo "Cronjob Schedule:"
echo "  📅 Daily at 6:00 AM local time"
echo "  🔄 Auto-resume from last download"
echo "  ⏱️  Timeout: 2 hours"
echo "  📊 Priority: Low (nice -n 19, ionice -c 3)"
echo ""
echo "View installed cronjobs:"
echo "  crontab -l"
echo ""
echo "Monitor latest update:"
echo "  tail -f \${LOGS_DIR:-/pool/ssd8tb/logs}/entso_e_update_latest.log"
echo ""
echo "Manual update (test run):"
echo "  ${CRON_SCRIPT}"
echo ""
echo "Uninstall cronjob:"
echo "  crontab -l | grep -v 'update_entso_e_cron.sh' | crontab -"
echo ""
echo "================================================================================"
