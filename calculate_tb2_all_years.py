#!/usr/bin/env python3
"""
Calculate TB2 (Total Bi-directional price spread) for all settlement points, all years.

TB2 = max(DA price) - min(DA price) for each day
This represents the theoretical maximum arbitrage opportunity available.

For BESS analysis:
  % DA TB2 Captured = (Actual DA Revenue per MW) / (TB2 per MW) * 100
"""

import polars as pl
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_tb2_for_year(year: int, da_prices_dir: Path) -> pl.DataFrame:
    """Calculate daily TB2 for all settlement points in a given year"""

    da_file = da_prices_dir / f"{year}.parquet"

    if not da_file.exists():
        logger.warning(f"No DA prices for {year}")
        return None

    logger.info(f"Processing {year}...")

    # Load DA prices
    df = pl.read_parquet(da_file)

    # Calculate TB2 per settlement point per day
    tb2_daily = df.group_by(['SettlementPoint', 'DeliveryDate']).agg([
        pl.col('SettlementPointPrice').max().alias('max_price'),
        pl.col('SettlementPointPrice').min().alias('min_price'),
        pl.col('SettlementPointPrice').mean().alias('avg_price'),
        pl.col('SettlementPointPrice').count().alias('hours')
    ])

    # TB2 = max - min (price spread)
    tb2_daily = tb2_daily.with_columns([
        (pl.col('max_price') - pl.col('min_price')).alias('TB2'),
        pl.lit(year).alias('year')
    ])

    return tb2_daily


def main():
    logger.info("=" * 100)
    logger.info("TB2 CALCULATION FOR ALL YEARS (2019-2025)")
    logger.info("TB2 = Daily max price - min price (arbitrage opportunity)")
    logger.info("=" * 100)

    da_prices_dir = Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/rollup_files/DA_prices")

    all_tb2 = []

    for year in range(2019, 2026):
        tb2_year = calculate_tb2_for_year(year, da_prices_dir)
        if tb2_year is not None:
            all_tb2.append(tb2_year)

    # Combine all years
    df_all_tb2 = pl.concat(all_tb2)

    logger.info(f"✅ Calculated TB2 for {len(all_tb2)} years")
    logger.info(f"   Total records: {len(df_all_tb2):,}")
    logger.info(f"   Unique settlement points: {df_all_tb2.select(pl.col('SettlementPoint').n_unique()).item():,}")

    # Save daily TB2 data
    df_all_tb2.write_parquet('tb2_daily_2019_2025.parquet')
    logger.info(f"✅ Saved: tb2_daily_2019_2025.parquet")

    # Calculate average TB2 by settlement point (across all years)
    avg_tb2 = df_all_tb2.group_by('SettlementPoint').agg([
        pl.col('TB2').mean().alias('avg_TB2'),
        pl.col('TB2').median().alias('median_TB2'),
        pl.col('TB2').std().alias('std_TB2'),
        pl.col('TB2').max().alias('max_TB2'),
        pl.col('TB2').min().alias('min_TB2'),
        pl.col('TB2').count().alias('days'),
    ]).sort('avg_TB2', descending=True)

    avg_tb2.write_parquet('tb2_by_settlement_point_avg.parquet')
    logger.info(f"✅ Saved: tb2_by_settlement_point_avg.parquet")

    # Calculate yearly averages by settlement point
    yearly_tb2 = df_all_tb2.group_by(['SettlementPoint', 'year']).agg([
        pl.col('TB2').mean().alias('avg_TB2'),
        pl.col('TB2').median().alias('median_TB2'),
        pl.col('TB2').count().alias('days'),
    ]).sort(['SettlementPoint', 'year'])

    yearly_tb2.write_parquet('tb2_by_settlement_point_yearly.parquet')
    logger.info(f"✅ Saved: tb2_by_settlement_point_yearly.parquet")

    # Print top settlement points by TB2
    logger.info("\n" + "=" * 100)
    logger.info("TOP 20 SETTLEMENT POINTS BY AVERAGE TB2 (2019-2025)")
    logger.info("=" * 100)
    print(avg_tb2.head(20))

    # Show statistics
    logger.info("\n" + "=" * 100)
    logger.info("TB2 STATISTICS (FLEET-WIDE)")
    logger.info("=" * 100)
    logger.info(f"Average TB2 across all nodes: ${df_all_tb2.select(pl.col('TB2').mean()).item():.2f}/MWh")
    logger.info(f"Median TB2 across all nodes: ${df_all_tb2.select(pl.col('TB2').median()).item():.2f}/MWh")
    logger.info(f"Max TB2 ever observed: ${df_all_tb2.select(pl.col('TB2').max()).item():.2f}/MWh")

    logger.info("\n✅ TB2 calculation complete!")


if __name__ == "__main__":
    main()
