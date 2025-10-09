# TODO: Investigate Missing RT Price Data

**Status**: Deferred for later investigation
**Date**: October 8, 2025
**Priority**: Medium

## Issue

When calculating BESS revenues, we're excluding 4% of SCED intervals (~1,140 out of 28,395) because they don't have matching RT prices after rounding to 15-minute intervals.

## Details

- SCED timestamps are in Central Time, ~15-minute intervals (actual execution varies)
- RT prices are at exact 15-minute intervals in UTC (converted from epoch ms)
- After timezone conversion and truncation, ~4% of SCED intervals don't match

## Possible Causes

1. SCED execution delays cause timestamps that don't round cleanly to 15-min
2. Missing RT price intervals (but only 3 days have gaps: 2023-12-31, 2024-03-10 DST, 2024-12-31)
3. Timezone conversion issues (CST vs CDT handling)
4. Data quality issues in source files

## Current Workaround

Filtering out unmatched intervals and using the 96% that do match. This is acceptable for revenue calculations but may slightly underestimate revenues.

## Future Investigation

- [ ] Check source CSV files for RT prices to see if parquet conversion lost data
- [ ] Investigate SCED timestamp distribution (are they really ~15-min?)
- [ ] Consider using nearest-neighbor matching instead of exact match
- [ ] Compare with other BESS to see if pattern is consistent

## Impact

Low - 4% missing data has minimal impact on annual revenue calculations, and we're not using fake data to fill gaps.
