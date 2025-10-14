#!/usr/bin/env python3
"""
Run BESS revenue calculation for past 5 years (2020-2024)
Using Telemetered Net Output for actual metered discharge
"""

import polars as pl
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from bess_revenue_calculator import BESSRevenueCalculator
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
BASE_DIR = Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data")
MAPPING_FILE = Path("bess_mapping/BESS_UNIFIED_MAPPING_V3_CLARIFIED.csv")
YEARS = [2020, 2021, 2022, 2023, 2024, 2025]

def main():
    logger.info("=" * 100)
    logger.info("BESS REVENUE CALCULATION - 5 YEARS (2020-2024)")
    logger.info("Using TelemeteredNetOutput for discharge (actual metered)")
    logger.info("=" * 100)
    logger.info("")

    # Load BESS mapping
    logger.info(f"Loading BESS mapping from {MAPPING_FILE}")
    df_mapping = pl.read_csv(MAPPING_FILE)

    # Filter to operational BESS with both Gen and Load resources
    df_operational = df_mapping.filter(
        (pl.col("True_Operational_Status").str.contains("Operational")) &
        (pl.col("BESS_Load_Resource").is_not_null()) &
        (pl.col("BESS_Load_Resource") != "")
    )

    logger.info(f"Found {len(df_operational)} operational BESS units with Load Resources")
    logger.info("")

    # Process each year
    for year in YEARS:
        logger.info("=" * 100)
        logger.info(f"PROCESSING YEAR: {year}")
        logger.info("=" * 100)
        logger.info("")

        # Check if telemetry data is available
        sced_gen_file = BASE_DIR / "rollup_files" / f"SCED_Gen_Resources/{year}.parquet"
        if not sced_gen_file.exists():
            logger.error(f"SCED Gen Resources file not found for {year}: {sced_gen_file}")
            continue

        # Quick check for telemetry column
        df_check = pl.read_parquet(sced_gen_file, n_rows=0)
        has_telemetry = "TelemeteredNetOutput" in df_check.columns
        logger.info(f"TelemeteredNetOutput available: {has_telemetry}")
        logger.info("")

        # Initialize calculator
        # Limit backend threads to avoid oversubscription
        BESSRevenueCalculator.configure_threads(int(os.getenv('PMP_MAX_THREADS', '0')) or None)

        calculator = BESSRevenueCalculator(
            base_dir=BASE_DIR,
            year=year
        )

        # Process all BESS
        all_results = []

        for row in df_operational.iter_rows(named=True):
            try:
                result = calculator.calculate_bess_revenue(
                    bess_name=row['BESS_Gen_Resource'],  # Use Gen Resource as BESS name
                    gen_resource=row['BESS_Gen_Resource'],
                    load_resource=row['BESS_Load_Resource'],
                    resource_node=row['Settlement_Point'],
                    capacity_mw=row['IQ_Capacity_MW']
                )
                # Skip units with zero active days to avoid noisy errors for pre-COD years
                if result.get('active_days', 0) in (None, 0):
                    logger.info(f"  Skipping {row['BESS_Gen_Resource']} (no active days in {year})")
                    continue
                result['year'] = year
                all_results.append(result)

                # Log progress every 10 units
                if len(all_results) % 10 == 0:
                    logger.info(f"Processed {len(all_results)}/{len(df_operational)} units...")

            except Exception as e:
                # Only log unexpected errors loudly. Expected pre-COD/no-data cases are skipped above.
                logger.warning(f"  Unexpected error for {row['BESS_Gen_Resource']}: {e}")

        # Save results
        output_file = Path(f"bess_revenue_{year}_TELEMETERED.csv")
        if all_results:
            df_results = pl.DataFrame(all_results)
            df_results.write_csv(output_file)

            logger.info("")
            logger.info(f"✅ Saved {len(all_results)} results to {output_file}")

            # Calculate summary statistics (no 'error' column in success-only rows)
            df_valid = df_results
            if 'error' in df_results.columns:
                df_valid = df_results.filter(pl.col("error").is_null())
            if len(df_valid) > 0:
                total_revenue = df_valid.select(pl.col("total_revenue").sum()).item()
                avg_per_mw = df_valid.select(pl.col("revenue_per_mw_month").mean()).item()
                median_per_mw = df_valid.select(pl.col("revenue_per_mw_month").median()).item()

                logger.info("")
                logger.info(f"YEAR {year} SUMMARY:")
                logger.info(f"  Valid units:      {len(df_valid)}")
                logger.info(f"  Total revenue:    ${total_revenue:,.0f}")
                logger.info(f"  Avg $/MW-month:   ${avg_per_mw:,.2f}")
                logger.info(f"  Median $/MW-month: ${median_per_mw:,.2f}")
        else:
            logger.info("No results for this year (all units had zero active days or missing data).")

        logger.info("")

    logger.info("=" * 100)
    logger.info("ALL YEARS COMPLETE!")
    logger.info("=" * 100)

if __name__ == "__main__":
    main()
