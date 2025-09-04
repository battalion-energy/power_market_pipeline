# BESS-EIA Matching Pipeline - Production Documentation

## Overview
Two production-quality scripts for monthly BESS-EIA matching and database import.

NO HARDCODED DATA. NO MOCK DATA. ONLY VERIFIED SOURCES.

## Quick Start

```bash
# 1. Run monthly matching pipeline
python run_monthly_bess_pipeline.py

# 2. Import to database
python import_to_database.py

# That's it!
```

## File Structure

```
power_market_pipeline/
├── run_monthly_bess_pipeline.py    # Main pipeline script
├── import_to_database.py           # Database import script
├── data/                            # Input data (symlinks)
│   └── EIA/
│       └── generators/
│           └── EIA_generators_latest.xlsx  → (symlink to latest)
├── interconnection_queue_clean/    # ERCOT IQ data
│   ├── stand_alone.csv
│   ├── co_located_operational.csv
│   ├── co_located_with_solar.csv
│   └── co_located_with_wind.csv
├── output/                          # Pipeline outputs
│   ├── BESS_MATCHED_202412.csv
│   ├── BESS_MATCHED_LATEST.csv → (symlink)
│   └── VALIDATION_REPORT_202412.json
└── logs/                            # Execution logs
    └── pipeline_202412.log
```

## Monthly Workflow

### 1. Update Data Sources
```bash
# Update EIA symlink to latest data
cd data/EIA/generators
ln -sf EIA_generators_dec2024.xlsx EIA_generators_latest.xlsx

# Update ERCOT IQ data if needed
cd interconnection_queue_clean
# Copy latest CSV files here
```

### 2. Run Pipeline
```bash
# Full run (creates outputs)
python run_monthly_bess_pipeline.py

# Validation only (no outputs)
python run_monthly_bess_pipeline.py --validate-only
```

### 3. Review Validation
Check `output/VALIDATION_REPORT_YYYYMM.json` for issues:
- Distance validation (>100 miles from county)
- Zone mismatches (settlement vs physical)
- Known issues (CROSSETT verification)

### 4. Import to Database
```bash
# Live import
python import_to_database.py

# Dry run (validation only)
python import_to_database.py --dry-run

# Specific file
python import_to_database.py --file output/BESS_MATCHED_202412.csv
```

## Database Schema

```sql
bess_facilities
├── bess_gen_resource (PRIMARY KEY)
├── ERCOT data (county, capacity, zone)
├── EIA matched data (plant, coordinates, capacity)
├── Match metadata (score, timestamp)
├── Validation (physical_zone, zone_mismatch)
└── Audit fields (created_at, updated_at)

bess_pipeline_stats
├── Monthly statistics
├── Match rates
└── Validation issues
```

## Environment Variables

Create `.env` file:
```bash
# Database connection
DATABASE_URL=postgresql://user:pass@localhost/battalion
# OR individual settings:
DB_HOST=localhost
DB_PORT=5432
DB_NAME=battalion
DB_USER=postgres
DB_PASSWORD=yourpassword

# Optional (for enhanced features)
OPENAI_API_KEY=sk-...  # For LLM validation
GOOGLE_MAPS_KEY=...     # For substation geocoding
```

## Key Features

### Matching Algorithm
1. **County Match REQUIRED** - No cross-county matches
2. **Capacity Verification** - Within reasonable tolerance
3. **Enhanced Name Matching** - Multiple fields cross-matched
4. **No Hardcoded Data** - Everything from real sources

### Validation Layers
- Distance from county center (<100 miles)
- Settlement vs physical zone consistency
- Known issues check (CROSSETT in Crane County)
- Match score distribution

### Database Features
- Upsert logic (update or insert)
- Transaction management
- History tracking
- Monthly statistics
- Comprehensive indexes

## Critical Fixes Applied

### CROSSETT Resolution
- **Problem**: Incorrectly matched to Harris County (Houston)
- **Solution**: Correctly assigned to Crane County (West Texas)
- **Verification**: Built into validation

### No Hardcoded Mappings
- **Before**: `'CROSSETT': 'HARRIS'` hardcoded
- **After**: All data from EIA/ERCOT sources

## Monitoring

### Check Logs
```bash
tail -f logs/pipeline_202412.log
```

### Database Queries
```sql
-- Check CROSSETT
SELECT bess_gen_resource, eia_county, eia_latitude, eia_longitude
FROM bess_facilities
WHERE bess_gen_resource LIKE 'CROSSETT%';

-- Monthly statistics
SELECT * FROM bess_pipeline_stats
ORDER BY run_date DESC LIMIT 5;

-- Zone mismatches
SELECT COUNT(*) FROM bess_facilities
WHERE zone_mismatch = true;
```

## Troubleshooting

### Pipeline Fails
1. Check logs in `logs/pipeline_YYYYMM.log`
2. Verify data files exist and are readable
3. Check symlinks are correct

### Database Import Fails
1. Check database connection in `.env`
2. Verify CSV file exists
3. Run with `--dry-run` first

### CROSSETT Wrong Location
- Pipeline automatically fixes to Crane County
- Validation will flag if incorrect

## Performance

- Pipeline: ~30 seconds for 200 BESS
- Database import: ~5 seconds for 200 records
- Validation: Instant

## Maintenance

### Monthly Checklist
- [ ] Update EIA data symlink
- [ ] Update ERCOT IQ data if changed
- [ ] Run pipeline
- [ ] Review validation report
- [ ] Import to database
- [ ] Verify CROSSETT in Crane County
- [ ] Check for new hardcoded data (NONE allowed)

## Contact

For issues or questions about:
- Pipeline logic: Check `run_monthly_bess_pipeline.py`
- Database schema: Check `import_to_database.py`
- CROSSETT issue: Must be in Crane County, West Texas

Remember: **REAL DATA ONLY - NO EXCEPTIONS**