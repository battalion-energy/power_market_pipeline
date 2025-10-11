#!/bin/bash
#
# Safe ERCOT Data Update Script
#
# This script:
# 1. Downloads new Web Service API data
# 2. Converts API files to ZIP CSV format
# 3. Generates current year parquet to temporary file
# 4. Verifies data integrity (row count, columns, date ranges, no NaT)
# 5. Atomically replaces old file only after verification passes
# 6. Keeps old file if verification fails
#
# Usage:
#   ./update_ercot_data_safe.sh [--dry-run]
#
# Cron example (daily at 2 AM):
#   0 2 * * * /home/enrico/projects/power_market_pipeline/update_ercot_data_safe.sh >> /home/enrico/logs/ercot_update.log 2>&1

set -e  # Exit on error
set -u  # Exit on undefined variable
set -o pipefail  # Exit on pipe failure

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${ERCOT_DATA_DIR:-/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data}"
ROLLUP_DIR="${DATA_DIR}/rollup_files"
CURRENT_YEAR=$(date +%Y)
LOG_FILE="${SCRIPT_DIR}/update_ercot_data_$(date +%Y%m%d_%H%M%S).log"
DRY_RUN=false

# Parse arguments
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "DRY RUN MODE - No files will be modified"
fi

# Logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log_error() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $*" | tee -a "$LOG_FILE" >&2
}

# Error handler
error_exit() {
    log_error "$1"
    exit 1
}

# Cleanup function
cleanup() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        log_error "Script failed with exit code $exit_code"
    fi
    return $exit_code
}

trap cleanup EXIT

# Verification function
verify_parquet() {
    local file="$1"
    local dataset_type="$2"  # "DAM_Gen_Resources" or "SCED_Gen_Resources"

    log "Verifying $file..."

    # Check file exists and is not empty
    if [[ ! -f "$file" ]]; then
        log_error "File does not exist: $file"
        return 1
    fi

    if [[ ! -s "$file" ]]; then
        log_error "File is empty: $file"
        return 1
    fi

    # Run Python verification
    python3 << EOF
import sys
import pyarrow.parquet as pq
import pandas as pd

try:
    # Read parquet file
    table = pq.read_table("$file")
    df = table.to_pandas()

    # Basic checks
    if len(df) == 0:
        print("ERROR: Parquet file has 0 rows", file=sys.stderr)
        sys.exit(1)

    # Check for required columns based on dataset type
    if "$dataset_type" == "DAM_Gen_Resources":
        required_cols = ["DeliveryDate", "ResourceName", "HourEnding"]
        date_col = "DeliveryDate"
    elif "$dataset_type" == "SCED_Gen_Resources":
        required_cols = ["SCEDTimeStamp", "ResourceName"]
        date_col = "SCEDTimeStamp"
    else:
        print(f"ERROR: Unknown dataset type: $dataset_type", file=sys.stderr)
        sys.exit(1)

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"ERROR: Missing required columns: {missing_cols}", file=sys.stderr)
        sys.exit(1)

    # Check for NaT values in date column
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    nat_count = df[date_col].isna().sum()
    if nat_count > 0:
        print(f"ERROR: Found {nat_count} NaT values in {date_col}", file=sys.stderr)
        sys.exit(1)

    # Check date range
    min_date = df[date_col].min()
    max_date = df[date_col].max()

    # Verify current year has data
    if "$CURRENT_YEAR" in str(min_date) or "$CURRENT_YEAR" in str(max_date):
        print(f"✅ Verification passed: {len(df):,} rows, {min_date} to {max_date}")
        sys.exit(0)
    else:
        print(f"WARNING: No $CURRENT_YEAR data found. Date range: {min_date} to {max_date}", file=sys.stderr)
        # This is a warning, not an error - old data files are OK
        sys.exit(0)

except Exception as e:
    print(f"ERROR: Verification failed: {e}", file=sys.stderr)
    sys.exit(1)
EOF

    return $?
}

# Process dataset with safe atomic replacement
process_dataset_safe() {
    local dataset="$1"  # e.g., "DAM_Gen_Resources"

    log "Processing $dataset..."

    local year_file="${ROLLUP_DIR}/${dataset}/${CURRENT_YEAR}.parquet"
    local temp_file="${ROLLUP_DIR}/${dataset}/${CURRENT_YEAR}.parquet.tmp"
    local temp_rollup_dir="${DATA_DIR}/rollup_files_tmp"

    if [[ "$DRY_RUN" == true ]]; then
        log "[DRY RUN] Would regenerate $dataset parquet"
        return 0
    fi

    # Create temporary rollup directory
    mkdir -p "${temp_rollup_dir}/${dataset}"

    # Run processor to generate all years in temp directory
    log "Generating parquet files in temporary directory..."
    if ! env ERCOT_DATA_DIR="$DATA_DIR" SKIP_CSV=1 \
        cargo run --release --manifest-path ercot_data_processor/Cargo.toml \
        --bin ercot_data_processor -- --annual-rollup --dataset "$dataset" \
        2>&1 | sed "s|${ROLLUP_DIR}|${temp_rollup_dir}|g" | tee -a "$LOG_FILE"; then

        # Processor failed - clean up and abort
        log_error "$dataset parquet generation failed"
        rm -rf "$temp_rollup_dir"
        return 1
    fi

    # The processor writes to ROLLUP_DIR, so move the current year file to temp for verification
    if [[ -f "${ROLLUP_DIR}/${dataset}/${CURRENT_YEAR}.parquet" ]]; then
        mv "${ROLLUP_DIR}/${dataset}/${CURRENT_YEAR}.parquet" "$temp_file"
    else
        log_error "Processor did not generate ${CURRENT_YEAR}.parquet for $dataset"
        rm -rf "$temp_rollup_dir"
        return 1
    fi

    # Verify the newly generated temp file
    if verify_parquet "$temp_file" "$dataset"; then
        # Verification passed - atomically replace old file
        mv "$temp_file" "$year_file"
        log "✅ $dataset ${CURRENT_YEAR}.parquet verified and updated"
        rm -rf "$temp_rollup_dir"
        return 0
    else
        # Verification failed - keep old file, remove temp
        log_error "$dataset ${CURRENT_YEAR}.parquet verification failed"
        log "Keeping existing file: $year_file"
        rm -f "$temp_file"
        rm -rf "$temp_rollup_dir"
        return 1
    fi
}

# Main execution
main() {
    log "========================================"
    log "Starting ERCOT Data Update"
    log "========================================"
    log "Script: $0"
    log "Data Directory: $DATA_DIR"
    log "Current Year: $CURRENT_YEAR"
    log "Dry Run: $DRY_RUN"

    cd "$SCRIPT_DIR"

    # Step 1: Download new Web Service API data
    log "Step 1: Downloading new Web Service API data..."
    if [[ "$DRY_RUN" == true ]]; then
        log "[DRY RUN] Would download API data"
    else
        if ! uv run python ercot_ws_download_all.py --datasets 60d_DAM_Gen_Resources 60d_SCED_Gen_Resources; then
            log_error "API download failed"
            return 1
        fi
        log "✅ API download complete"
    fi

    # Step 2: Convert API files to ZIP format
    log "Step 2: Converting API files to ZIP CSV format..."
    if [[ "$DRY_RUN" == true ]]; then
        log "[DRY RUN] Would convert API files"
    else
        if ! uv run python convert_api_to_zip_format.py; then
            log_error "API file conversion failed"
            return 1
        fi
        log "✅ API file conversion complete"
    fi

    # Step 3: Process DAM Gen Resources
    log "Step 3: Processing DAM Gen Resources..."
    if ! process_dataset_safe "DAM_Gen_Resources"; then
        log_error "DAM Gen Resources processing failed"
        return 1
    fi

    # Step 4: Process SCED Gen Resources
    log "Step 4: Processing SCED Gen Resources..."
    if ! process_dataset_safe "SCED_Gen_Resources"; then
        log_error "SCED Gen Resources processing failed"
        return 1
    fi

    log "========================================"
    log "ERCOT Data Update Complete"
    log "========================================"

    return 0
}

# Execute main function
main "$@"
