# ISO Data Status Scripts

Comprehensive data status monitoring scripts for all ISOs in the power market pipeline.

Created: 2025-10-28

## Overview

These scripts provide automated status checks for all ISO datasets, including:
- File counts and date ranges
- Gap detection
- Data completeness metrics
- File size validation
- Days behind current date
- Actionable recommendations

## Available Scripts

### 1. PJM
**Location:** `iso_markets/pjm/check_pjm_data_status.py`

**Datasets monitored:**
- Day-Ahead Nodal LMPs
- Real-Time Hourly Nodal LMPs
- Day-Ahead Ancillary Services
- Real-Time 5-Min Nodal LMPs (6-month retention)

**Usage:**
```bash
python iso_markets/pjm/check_pjm_data_status.py
python iso_markets/pjm/check_pjm_data_status.py --verbose
python iso_markets/pjm/check_pjm_data_status.py --json
python iso_markets/pjm/check_pjm_data_status.py --dataset da_nodal
```

---

### 2. CAISO
**Location:** `iso_markets/check_caiso_data_status.py`

**Datasets monitored:**
- Day-Ahead Nodal LMPs
- Real-Time 5-Min Nodal LMPs
- Day-Ahead Ancillary Services
- LMP Day-Ahead Hourly
- LMP Real-Time 5-Min
- LMP Real-Time 15-Min
- Ancillary Service Prices
- Fuel Mix
- Load Data

**Usage:**
```bash
python iso_markets/check_caiso_data_status.py
python iso_markets/check_caiso_data_status.py --verbose
python iso_markets/check_caiso_data_status.py --json
```

---

### 3. MISO
**Location:** `iso_markets/miso/check_miso_data_status.py`

**Datasets monitored:**
- Day-Ahead Ex-Post LMP
- Day-Ahead Ex-Ante LMP
- Real-Time Final LMP
- Real-Time 5-Min LMP
- Ancillary Services DA Ex-Post
- Ancillary Services RT Final
- Load - Actual
- Load - Forecast
- Generation - Fuel Mix

**Usage:**
```bash
python iso_markets/miso/check_miso_data_status.py
python iso_markets/miso/check_miso_data_status.py --verbose
python iso_markets/miso/check_miso_data_status.py --json
```

---

### 4. NYISO
**Location:** `iso_markets/check_nyiso_data_status.py`

**Datasets monitored:**
- LMP Day-Ahead Hourly
- LMP Real-Time 5-Min
- Ancillary Services Day-Ahead Hourly
- Ancillary Services Real-Time 5-Min
- Load Data
- Fuel Mix

**Usage:**
```bash
python iso_markets/check_nyiso_data_status.py
python iso_markets/check_nyiso_data_status.py --verbose
python iso_markets/check_nyiso_data_status.py --json
```

---

### 5. ISO-NE (ISONE)
**Location:** `iso_markets/check_isone_data_status.py`

**Datasets monitored:**
- Day-Ahead LMP
- Real-Time LMP
- Day-Ahead LMP Nodal
- Real-Time LMP Nodal
- Frequency Regulation Prices
- Reserve Prices

**Usage:**
```bash
python iso_markets/check_isone_data_status.py
python iso_markets/check_isone_data_status.py --verbose
python iso_markets/check_isone_data_status.py --json
```

---

### 6. SPP
**Location:** `iso_markets/spp/check_spp_data_status.py`

**Datasets monitored:**
- Day-Ahead LMP
- Real-Time LMP Daily
- Day-Ahead LMP Nodal
- Real-Time LMP Nodal
- Day-Ahead Market Clearing Prices
- Real-Time Market Clearing Prices

**Usage:**
```bash
python iso_markets/spp/check_spp_data_status.py
python iso_markets/spp/check_spp_data_status.py --verbose
python iso_markets/spp/check_spp_data_status.py --json
```

---

### 7. ERCOT
**Location:** `iso_markets/check_ercot_data_status.py`

**Datasets monitored:**
- Day-Ahead Market Settlement Point Prices
- Real-Time Market Settlement Point Prices
- Ancillary Services Prices
- Fuel Mix
- Actual System Load by Forecast Zone
- Actual System Load by Weather Zone
- Wind Power Production
- Solar Power Production

**Note:** ERCOT uses date-range files rather than daily files. Coverage metrics account for overlapping ranges.

**Usage:**
```bash
python iso_markets/check_ercot_data_status.py
python iso_markets/check_ercot_data_status.py --verbose
python iso_markets/check_ercot_data_status.py --json
```

---

### 8. ENTSO-E
**Location:** `iso_markets/check_entsoe_data_status.py`

**Datasets monitored:**
- Day-Ahead Prices
- Imbalance Prices
- Germany Ancillary Services

**Usage:**
```bash
python iso_markets/check_entsoe_data_status.py
python iso_markets/check_entsoe_data_status.py --verbose
python iso_markets/check_entsoe_data_status.py --json
```

---

## Status Indicators

All scripts use a consistent status classification:

- **✓ GOOD**: Up to date (≤2 days behind), ≥95% complete, no small files
- **⚠ OK**: Moderately current (≤7 days behind), ≥90% complete
- **⚠ NEEDS_UPDATE**: Behind schedule but recoverable
- **✗ CRITICAL**: Severely outdated (>30 days behind) or <50% complete
- **✗ NO_DATA**: No data files found

## Common Options

All scripts support the following command-line options:

- `--verbose` or `-v`: Show detailed gap information and small file details
- `--json`: Output results in JSON format for programmatic processing
- `--dataset <name>`: Check only a specific dataset

## Output Format

### Standard Output
```
================================================================================
[ISO NAME] DATA STATUS REPORT
Generated: 2025-10-28 18:17:38
================================================================================

✓ Dataset Name
  Status: GOOD
  Files: 2,493 (80.4 GB)
  Date Range: 2019-01-01 → 2025-10-28 (2493 days)
  Completeness: 100.0%
  Days Behind: 0
```

### JSON Output
```json
[
  {
    "dataset": "da_nodal",
    "description": "Day-Ahead Nodal LMPs",
    "status": "GOOD",
    "file_count": 2493,
    "date_range": {
      "start": "2019-01-01",
      "end": "2025-10-28",
      "days": 2493
    },
    "completeness_pct": 100.0,
    "days_behind": 0,
    "gaps": [],
    "small_files": [],
    "total_size_gb": 80.4
  }
]
```

## Environment Variables

Each script supports environment variables to override default data directories:

- `PJM_DATA_DIR` - Default: `/pool/ssd8tb/data/iso/PJM_data`
- `CAISO_DATA_DIR` - Default: `/pool/ssd8tb/data/iso/CAISO_data`
- `MISO_DATA_DIR` - Default: `/pool/ssd8tb/data/iso/MISO`
- `NYISO_DATA_DIR` - Default: `/pool/ssd8tb/data/iso/NYISO_data`
- `ISONE_DATA_DIR` - Default: `/pool/ssd8tb/data/iso/ISONE`
- `SPP_DATA_DIR` - Default: `/pool/ssd8tb/data/iso/SPP`
- `ERCOT_DATA_DIR` - Default: `/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data`
- `ENTSOE_DATA_DIR` - Default: `/pool/ssd8tb/data/iso/ENTSO_E`

## Automation

These scripts can be integrated into monitoring workflows:

### Daily Status Check (All ISOs)
```bash
#!/bin/bash
# check_all_isos.sh

echo "=== Daily ISO Data Status Check ==="
date

python iso_markets/pjm/check_pjm_data_status.py
python iso_markets/check_caiso_data_status.py
python iso_markets/miso/check_miso_data_status.py
python iso_markets/check_nyiso_data_status.py
python iso_markets/check_isone_data_status.py
python iso_markets/spp/check_spp_data_status.py
python iso_markets/check_ercot_data_status.py
python iso_markets/check_entsoe_data_status.py
```

### JSON Export for Monitoring Dashboard
```bash
# Export all status data to JSON files
mkdir -p data_status_reports/$(date +%Y-%m-%d)

for iso in pjm caiso miso nyiso isone spp ercot entsoe; do
  python iso_markets/check_${iso}_data_status.py --json > \
    data_status_reports/$(date +%Y-%m-%d)/${iso}_status.json
done
```

## Troubleshooting

### Script not found
Make sure you're running from the project root directory:
```bash
cd /home/enrico/projects/power_market_pipeline
```

### Permission denied
Ensure scripts are executable:
```bash
chmod +x iso_markets/check_*.py
chmod +x iso_markets/*/check_*.py
```

### No data found
Check that environment variables point to correct directories, or verify data exists:
```bash
ls -la /pool/ssd8tb/data/iso/PJM_data/csv_files/
```

## Integration with Update Scripts

Each ISO has corresponding update scripts that can be run based on status check results:

- **PJM**: `iso_markets/pjm/update_pjm_with_resume.py`
- **MISO**: `iso_markets/miso/update_miso_with_resume.py`
- **SPP**: `iso_markets/spp/update_spp_with_resume.py`
- **Others**: Check respective ISO directories

## Notes

1. **File Size Thresholds**: Each dataset has minimum file size thresholds to detect incomplete downloads
2. **Gap Detection**: Daily-based ISOs check for missing dates; ERCOT handles overlapping date ranges
3. **Retention Policies**: Some datasets (e.g., PJM 5-min RT) have limited retention at the source
4. **Time Zones**: All dates are in the respective ISO's local timezone

## Updates and Maintenance

Last updated: 2025-10-28

To update these scripts or add new datasets, modify the `DATASETS` dictionary in each script following the existing pattern.
