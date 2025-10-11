#!/usr/bin/env bash
#
# Resume Open-Meteo download with rate limiting
# Run this script to continue downloading remaining locations
# Will take ~12 minutes to complete all 48 locations
#

echo "=========================================================================="
echo "RESUMING OPEN-METEO DOWNLOAD"
echo "=========================================================================="
echo "This will download remaining locations at 4 requests/minute"
echo "Estimated time: ~12 minutes for all remaining locations"
echo ""
echo "Press Ctrl+C to stop at any time - script can be re-run to resume"
echo "=========================================================================="
echo ""

cd "$(dirname "$0")"
uv run python download_openmeteo_weather.py
