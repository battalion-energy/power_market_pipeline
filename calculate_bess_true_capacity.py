#!/usr/bin/env python3
"""
Calculate BESS true capacity by finding max simultaneous commitment across all services.

For each hour, sum all services that could be simultaneously awarded:
  Discharge (DAM Energy Award) + RegUp + RRS + ECRS + NonSpin

This represents the maximum MW the battery committed to provide in any single hour,
which should equal (or be close to) its true nameplate capacity.

Note: We use DAM awards since RT BasePoints can exceed capacity during actual operation.
"""

import polars as pl
import pandas as pd
from pathlib import Path
from typing import Dict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class BESSCapacityCalculator:
    """Calculate true BESS capacity from maximum simultaneous commitments"""

    def __init__(self, rollup_dir: Path):
        self.rollup_dir = rollup_dir

    def calculate_capacity(self, gen_resource: str, year: int) -> Dict:
        """
        Calculate capacity for a single battery-year

        Returns dict with:
            - max_total_commitment: Peak simultaneous commitment across all services
            - max_dam_energy: Peak DAM energy award
            - max_regup: Peak RegUp award
            - max_regdown: Peak RegDown award
            - max_rrs: Peak RRS award
            - max_ecrs: Peak ECRS award
            - max_nonspin: Peak NonSpin award
            - hour_of_max: When peak occurred
        """

        dam_file = self.rollup_dir / f"DAM_Gen_Resources/{year}.parquet"

        if not dam_file.exists():
            logger.warning(f"No DAM data for {gen_resource} in {year}")
            return None

        # Load DAM Gen data for this battery
        df = pl.read_parquet(dam_file).filter(
            pl.col("ResourceName") == gen_resource
        )

        if len(df) == 0:
            logger.warning(f"No awards found for {gen_resource} in {year}")
            return None

        # Calculate total simultaneous commitment per hour
        # DAM Energy (discharge) + all AS products can be awarded simultaneously
        df = df.with_columns([
            # Total commitment = energy + all AS services
            # Note: Battery can provide RegUp, RRS, ECRS, NonSpin simultaneously
            # RegDown is mutually exclusive with other services during discharge
            (
                pl.col("AwardedQuantity").fill_null(0) +  # DAM discharge
                pl.col("RegUpAwarded").fill_null(0) +
                pl.col("RegDownAwarded").fill_null(0) +  # Usually 0 during discharge
                pl.col("RRSAwarded").fill_null(0) +
                pl.col("ECRSAwarded").fill_null(0) +
                pl.col("NonSpinAwarded").fill_null(0)
            ).alias("total_commitment"),

            # Also track discharge-only services (Energy + RegUp + RRS + ECRS + NonSpin)
            # This excludes RegDown since it's typically on Load side
            (
                pl.col("AwardedQuantity").fill_null(0) +
                pl.col("RegUpAwarded").fill_null(0) +
                pl.col("RRSAwarded").fill_null(0) +
                pl.col("ECRSAwarded").fill_null(0) +
                pl.col("NonSpinAwarded").fill_null(0)
            ).alias("discharge_services")
        ])

        # Find hour with maximum total commitment
        max_row = df.sort("total_commitment", descending=True).head(1)

        if len(max_row) == 0:
            return None

        result = {
            'gen_resource': gen_resource,
            'year': year,
            'max_total_commitment': max_row.select(pl.col("total_commitment")).item(),
            'max_discharge_services': max_row.select(pl.col("discharge_services")).item(),
            'max_dam_energy': max_row.select(pl.col("AwardedQuantity").fill_null(0)).item(),
            'max_regup': max_row.select(pl.col("RegUpAwarded").fill_null(0)).item(),
            'max_regdown': max_row.select(pl.col("RegDownAwarded").fill_null(0)).item(),
            'max_rrs': max_row.select(pl.col("RRSAwarded").fill_null(0)).item(),
            'max_ecrs': max_row.select(pl.col("ECRSAwarded").fill_null(0)).item(),
            'max_nonspin': max_row.select(pl.col("NonSpinAwarded").fill_null(0)).item(),
            'hour_of_max': max_row.select(pl.col("datetime")).item(),
        }

        # Also get individual service maxima (not necessarily same hour)
        result['max_regup_ever'] = df.select(pl.col("RegUpAwarded").max()).item() or 0
        result['max_regdown_ever'] = df.select(pl.col("RegDownAwarded").max()).item() or 0
        result['max_rrs_ever'] = df.select(pl.col("RRSAwarded").max()).item() or 0
        result['max_ecrs_ever'] = df.select(pl.col("ECRSAwarded").max()).item() or 0
        result['max_nonspin_ever'] = df.select(pl.col("NonSpinAwarded").max()).item() or 0

        return result


def main():
    logger.info("=" * 100)
    logger.info("BESS TRUE CAPACITY CALCULATION")
    logger.info("Finding max simultaneous commitment: DAM Energy + RegUp + RRS + ECRS + NonSpin")
    logger.info("=" * 100)

    rollup_dir = Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/rollup_files")

    # Load BESS mapping
    mapping_file = Path("bess_mapping/BESS_UNIFIED_MAPPING_V3_CLARIFIED.csv")
    df_mapping = pd.read_csv(mapping_file)

    # Filter to operational BESS
    df_operational = df_mapping[
        df_mapping['True_Operational_Status'] == 'Operational (has Load Resource)'
    ].copy()

    logger.info(f"Analyzing {len(df_operational)} operational BESS units")
    logger.info("")

    calculator = BESSCapacityCalculator(rollup_dir)

    all_results = []

    # Process each battery across all years
    for _, row in df_operational.iterrows():
        gen_resource = row['BESS_Gen_Resource']

        battery_results = []

        for year in range(2019, 2025):
            result = calculator.calculate_capacity(gen_resource, year)
            if result:
                battery_results.append(result)

        if battery_results:
            all_results.extend(battery_results)

            # Find max across all years for this battery
            max_commitment = max(r['max_total_commitment'] for r in battery_results)
            max_discharge = max(r['max_discharge_services'] for r in battery_results)
            max_year = max(battery_results, key=lambda x: x['max_total_commitment'])['year']

            stated_capacity = row['IQ_Capacity_MW']

            logger.info(f"{gen_resource}:")
            logger.info(f"  Stated capacity: {stated_capacity:.1f} MW")
            logger.info(f"  Max total commitment: {max_commitment:.1f} MW (in {max_year})")
            logger.info(f"  Max discharge services: {max_discharge:.1f} MW")
            logger.info(f"  Difference: {max_commitment - stated_capacity:.1f} MW ({(max_commitment/stated_capacity - 1)*100:.1f}%)")
            logger.info("")

    # Save results
    df_results = pd.DataFrame(all_results)
    df_results.to_csv('bess_capacity_analysis.csv', index=False)
    logger.info(f"✅ Saved detailed results: bess_capacity_analysis.csv")

    # Create summary by battery (max across all years)
    df_summary = df_results.groupby('gen_resource').agg({
        'max_total_commitment': 'max',
        'max_discharge_services': 'max',
        'max_dam_energy': 'max',
        'max_regup_ever': 'max',
        'max_regdown_ever': 'max',
        'max_rrs_ever': 'max',
        'max_ecrs_ever': 'max',
        'max_nonspin_ever': 'max',
    }).reset_index()

    # Add stated capacity from mapping
    capacity_map = df_operational.set_index('BESS_Gen_Resource')['IQ_Capacity_MW'].to_dict()
    df_summary['stated_capacity'] = df_summary['gen_resource'].map(capacity_map)

    # Calculate differences
    df_summary['capacity_diff_mw'] = df_summary['max_total_commitment'] - df_summary['stated_capacity']
    df_summary['capacity_diff_pct'] = (df_summary['max_total_commitment'] / df_summary['stated_capacity'] - 1) * 100

    # Sort by total commitment
    df_summary = df_summary.sort_values('max_total_commitment', ascending=False)

    df_summary.to_csv('bess_capacity_summary.csv', index=False)
    logger.info(f"✅ Saved summary: bess_capacity_summary.csv")

    # Print statistics
    logger.info("=" * 100)
    logger.info("SUMMARY STATISTICS")
    logger.info("=" * 100)
    logger.info(f"Batteries analyzed: {len(df_summary)}")
    logger.info(f"Total stated capacity: {df_summary['stated_capacity'].sum():.1f} MW")
    logger.info(f"Total max commitment: {df_summary['max_total_commitment'].sum():.1f} MW")
    logger.info(f"Average max commitment: {df_summary['max_total_commitment'].mean():.1f} MW")
    logger.info(f"Median max commitment: {df_summary['max_total_commitment'].median():.1f} MW")

    # Batteries with significant differences
    significant = df_summary[abs(df_summary['capacity_diff_pct']) > 10]
    if len(significant) > 0:
        logger.info(f"\n⚠️  Batteries with >10% difference ({len(significant)} batteries):")
        for _, row in significant.head(10).iterrows():
            logger.info(f"  {row['gen_resource']}: {row['stated_capacity']:.1f} → {row['max_total_commitment']:.1f} MW ({row['capacity_diff_pct']:.1f}%)")

    logger.info("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
