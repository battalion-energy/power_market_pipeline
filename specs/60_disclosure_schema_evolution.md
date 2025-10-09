Tell me about the schema changes that occurred in the csv SCED files that we had to work around over the years. 

● Looking at the code and handling in bess_revenue_calculator.py, here are the major SCED schema changes we've had to work around:

  1. TelemeteredNetOutput Addition (Mid-2020s)

  Line 346-367 in bess_revenue_calculator.py:
  has_telemetry = "TelemeteredNetOutput" in df_gen_raw.columns
  discharge_col = "TelemeteredNetOutput" if has_telemetry else "BasePoint"

  if has_telemetry:
      logger.info(f"Using TelemeteredNetOutput for {gen_resource} (actual metered)")
  else:
      logger.warning(f"TelemeteredNetOutput not available for {gen_resource}, using BasePoint")

  - Early years: Only had BasePoint (ERCOT's dispatch instruction)
  - Later years: Added TelemeteredNetOutput (actual metered output from the battery)
  - Why it matters: Telemetry shows what the battery ACTUALLY did, BasePoint shows what ERCOT TOLD it to do
  - Revenue impact: Using BasePoint can miss real-world deviations and curtailments

  2. SCED Load Resources ResourceName Column (2023-2024)

  The BIGGEST schema change - mentioned in the comments at line 369:

  # Load charging data (NOW POSSIBLE with fixed ResourceName!)
  df_load_raw = pl.read_parquet(sced_load_file).filter(
      pl.col("ResourceName") == load_resource
  )

  - Before (~2022): SCED_Load_Resources file was MISSING the ResourceName column entirely
  - After (2023+): Added ResourceName column, making it possible to track individual load resources
  - Impact: This was CRITICAL for BESS revenue - we couldn't properly calculate charging costs without being able to identify the specific load resource

  3. Ancillary Services Disaggregation (Post-2020)

  Referenced in the AS calculation fixes:

  Old Schema (2019-2020):
  - RRSAwarded (single column)
  - ECRSAwarded (single column)

  New Schema (2021+):
  - RRS split into:
    - RRSPFRAwarded (Primary Frequency Response)
    - RRSFFRAwarded (Fast Frequency Response)
    - RRSUFRAwarded (Ultra-Fast Frequency Response)
  - ECRS split into:
    - ECRSSDAwarded (Service Deployment)
    - ECRSMDAwarded (Manual Deployment)

  The code handles both (lines 656-664):
  (pl.sum_horizontal([
      pl.col(c).fill_null(0.0) for c in [
          "RRSPFRAwarded","RRSFFRAwarded","RRSUFRAwarded","RRSAwarded"
      ] if c in gen.columns
  ])).alias("rrs_mw")

  4. DSTFlag Column Addition (2011)

  From the Rust processor documentation - mentioned in CLAUDE.md:

  Automatic schema evolution handling (e.g., 2011 DSTFlag column addition)

  - Pre-2011 files don't have DSTFlag column
  - 2011+ files include it to handle Daylight Saving Time transitions
  - Rust processor normalizes schemas automatically

  5. Timestamp Format Variations

  The code shows we handle multiple timestamp formats:

  pl.col("SCEDTimeStamp").str.strptime(pl.Datetime, "%m/%d/%Y %H:%M:%S")

  - Some files use different date formats
  - All are in Central Time (America/Chicago) but parsing varies

  Summary Table

  | Year Range | Schema Change             | Impact                                  | Workaround                     |
  |------------|---------------------------|-----------------------------------------|--------------------------------|
  | 2019-2020  | No TelemeteredNetOutput   | Less accurate actual output             | Fallback to BasePoint          |
  | 2019-2022  | Load ResourceName missing | Couldn't track individual BESS charging | Major limitation!              |
  | 2020-2021  | AS columns disaggregated  | Revenue calculations broke              | Check for both old/new columns |
  | Pre-2011   | No DSTFlag                | DST handling unclear                    | Schema normalization           |
  | All years  | Timestamp format variance | Parsing errors                          | Flexible strptime              |

  The Load ResourceName missing was the most problematic - it made proper BESS revenue tracking nearly impossible for early years since we couldn't distinguish which
  load resource was charging!

