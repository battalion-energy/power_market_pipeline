#!/usr/bin/env python3
"""
Calculate active trading periods for each BESS by analyzing actual market data.

Determines:
1. First date each battery appears in DAM awards (operational start date)
2. Number of active days per year
3. Whether battery meets minimum activity threshold (60 days = ~2 months)
"""

import polars as pl
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def calculate_active_days_for_year(year: int, rollup_dir: Path) -> pl.DataFrame:
    """Calculate active days per battery for a specific year"""

    dam_file = rollup_dir / f"DAM_Gen_Resources/{year}.parquet"

    if not dam_file.exists():
        logger.warning(f"No DAM data for {year}")
        return None

    logger.info(f"Processing {year}...")

    # Load DAM Gen data
    df = pl.read_parquet(dam_file)

    # Count unique dates per battery where they had any DAM activity
    # (awards > 0 or AS awards > 0)
    df_active = df.filter(
        (pl.col('AwardedQuantity') > 0) |
        (pl.col('RegUpAwarded') > 0) |
        (pl.col('RegDownAwarded') > 0) |
        (pl.col('RRSAwarded') > 0) |
        (pl.col('ECRSAwarded') > 0) |
        (pl.col('NonSpinAwarded') > 0)
    )

    # Get unique dates per battery
    activity = df_active.group_by('ResourceName').agg([
        pl.col('DeliveryDate').n_unique().alias('active_days'),
        pl.col('DeliveryDate').min().alias('first_active_date'),
        pl.col('DeliveryDate').max().alias('last_active_date'),
    ])

    activity = activity.with_columns([
        pl.lit(year).alias('year'),
        # Cast dates to Date type for consistency
        pl.col('first_active_date').cast(pl.Date),
        pl.col('last_active_date').cast(pl.Date)
    ])

    return activity


def main():
    logger.info("=" * 100)
    logger.info("CALCULATING BESS ACTIVE TRADING PERIODS")
    logger.info("=" * 100)

    rollup_dir = Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/rollup_files")

    # Calculate for all years
    all_activity = []

    for year in range(2019, 2025):
        activity = calculate_active_days_for_year(year, rollup_dir)
        if activity is not None:
            all_activity.append(activity)

    # Combine all years
    df_all_activity = pl.concat(all_activity)

    logger.info(f"\n✅ Calculated activity for {df_all_activity.select(pl.col('ResourceName').n_unique()).item()} batteries")

    # Convert to pandas for easier manipulation
    df_activity_pd = df_all_activity.to_pandas()

    # Save detailed activity by year
    df_activity_pd.to_csv('bess_activity_by_year.csv', index=False)
    logger.info(f"✅ Saved: bess_activity_by_year.csv")

    # Calculate overall operational start date (first date across all years)
    df_start_dates = df_activity_pd.groupby('ResourceName').agg({
        'first_active_date': 'min',
        'last_active_date': 'max',
        'active_days': 'sum'
    }).reset_index()

    df_start_dates = df_start_dates.rename(columns={
        'ResourceName': 'gen_resource',
        'first_active_date': 'operational_start_date',
        'last_active_date': 'last_active_date',
        'active_days': 'total_active_days'
    })

    df_start_dates.to_csv('bess_operational_dates.csv', index=False)
    logger.info(f"✅ Saved: bess_operational_dates.csv")

    # Print statistics
    logger.info("\n" + "=" * 100)
    logger.info("ACTIVITY STATISTICS")
    logger.info("=" * 100)

    # Batteries with < 60 days activity in any year
    df_low_activity = df_activity_pd[df_activity_pd['active_days'] < 60]

    if len(df_low_activity) > 0:
        logger.info(f"\n⚠️  Batteries with <60 days activity (will be excluded from that year):")
        logger.info(f"   Count: {len(df_low_activity)} battery-years")

        # Show examples
        for _, row in df_low_activity.head(10).iterrows():
            logger.info(f"   {row['ResourceName']} ({row['year']}): {row['active_days']} days " +
                       f"({row['first_active_date']} to {row['last_active_date']})")

    # Show batteries that started mid-year
    logger.info(f"\n📅 Batteries that started mid-year (first 10):")
    df_mid_year = df_activity_pd[
        (df_activity_pd['first_active_date'].dt.month > 1) |
        (df_activity_pd['first_active_date'].dt.day > 15)
    ].sort_values('first_active_date')

    for _, row in df_mid_year.head(10).iterrows():
        logger.info(f"   {row['ResourceName']} ({row['year']}): Started {row['first_active_date']}, " +
                   f"{row['active_days']} active days")

    logger.info("\n✅ Activity calculation complete!")

    return df_activity_pd, df_start_dates


if __name__ == "__main__":
    main()
