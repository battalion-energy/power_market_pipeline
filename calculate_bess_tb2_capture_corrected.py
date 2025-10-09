#!/usr/bin/env python3
"""
Calculate TOTAL Revenue vs TB2 Benchmark for each BESS.

TB2 = Theoretical max revenue from perfect DA arbitrage (max price - min price daily)
% Total Revenue vs TB2 = (Total Actual Revenue / Capacity) / (TB2 / 1 MW) * 100

This measures:
- >100%: Battery beat the DA arbitrage benchmark (AS/RT added value beyond simple arbitrage)
- <100%: Battery underperformed benchmark (inefficiencies, poor timing, or opportunity cost)

The TB2 represents what a battery would make if it:
  1. Perfectly predicted daily price highs and lows
  2. Charged at minimum price, discharged at maximum price
  3. Had perfect efficiency (100%)
  4. Did ONLY DA energy arbitrage (no AS, no RT)
"""

import pandas as pd
import polars as pl
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("=" * 100)
    logger.info("BESS TOTAL REVENUE vs TB2 BENCHMARK")
    logger.info("TB2 = Theoretical perfect DA arbitrage opportunity")
    logger.info("% vs TB2 = Total Actual Revenue / TB2 Benchmark")
    logger.info("=" * 100)

    # Load BESS mapping
    mapping_file = Path("bess_mapping/BESS_UNIFIED_MAPPING_V3_CLARIFIED.csv")
    df_mapping = pd.read_csv(mapping_file)

    df_bess = df_mapping[
        df_mapping['True_Operational_Status'] == 'Operational (has Load Resource)'
    ][['BESS_Gen_Resource', 'Settlement_Point', 'IQ_Capacity_MW']].copy()

    df_bess = df_bess.rename(columns={
        'BESS_Gen_Resource': 'gen_resource',
        'Settlement_Point': 'settlement_point',
        'IQ_Capacity_MW': 'capacity_mw'
    })

    logger.info(f"Loaded {len(df_bess)} operational BESS units")

    # Load TB2 data
    tb2_yearly = pl.read_parquet('tb2_by_settlement_point_yearly.parquet')
    logger.info(f"Loaded TB2 data: {len(tb2_yearly)} settlement point-years")

    # Load BESS revenue data
    all_revenue = []
    for year in range(2019, 2025):
        file = f"bess_revenue_{year}.csv"
        if Path(file).exists():
            df_year = pd.read_csv(file)
            all_revenue.append(df_year)

    df_revenue = pd.concat(all_revenue, ignore_index=True)
    logger.info(f"Loaded revenue data: {len(df_revenue)} battery-years")

    # Merge with mapping to get settlement points and corrected capacity
    df_analysis = df_revenue.merge(
        df_bess[['gen_resource', 'settlement_point', 'capacity_mw']],
        on='gen_resource',
        how='left',
        suffixes=('_revenue', '_mapping')
    )

    # Use capacity from mapping file (corrected nameplate)
    df_analysis['capacity_mw'] = df_analysis['capacity_mw_mapping'].fillna(df_analysis['capacity_mw_revenue'])

    # Use settlement point from mapping, fallback to resource_node
    df_analysis['settlement_point_final'] = df_analysis['settlement_point'].fillna(df_analysis['resource_node'])

    logger.info(f"Merged data: {len(df_analysis)} records")
    logger.info(f"Records with settlement points: {df_analysis['settlement_point_final'].notna().sum()}")

    # Convert to polars
    df_analysis_pl = pl.from_pandas(df_analysis)

    # Cast year to int32
    df_analysis_pl = df_analysis_pl.with_columns([
        pl.col('year').cast(pl.Int32)
    ])

    # Join with TB2 data
    df_with_tb2 = df_analysis_pl.join(
        tb2_yearly,
        left_on=['settlement_point_final', 'year'],
        right_on=['SettlementPoint', 'year'],
        how='left'
    )

    # Calculate % Total Revenue vs TB2
    # TB2 = average daily price spread ($/MWh)
    # For full year comparison: multiply by number of days
    # Total Revenue per MW = total_revenue / capacity_mw
    # TB2 Benchmark per MW = avg_TB2 * days (total opportunity for the year)
    # % vs TB2 = (Total Revenue / MW) / (TB2 Benchmark / MW) * 100

    df_with_tb2 = df_with_tb2.with_columns([
        # Total revenue per MW (all revenue streams)
        (pl.col('total_revenue') / pl.col('capacity_mw')).alias('total_rev_per_mw'),

        # TB2 benchmark per MW (theoretical perfect arbitrage)
        (pl.col('avg_TB2') * pl.col('days')).alias('tb2_benchmark_per_mw'),

        # % Total Revenue vs TB2
        ((pl.col('total_revenue') / pl.col('capacity_mw')) / (pl.col('avg_TB2') * pl.col('days')) * 100).alias('pct_total_vs_tb2'),

        # Also calculate individual components vs TB2 for analysis
        (pl.col('dam_discharge_revenue') / pl.col('capacity_mw') / (pl.col('avg_TB2') * pl.col('days')) * 100).alias('pct_da_energy_vs_tb2'),
        (pl.col('rt_net_revenue') / pl.col('capacity_mw') / (pl.col('avg_TB2') * pl.col('days')) * 100).alias('pct_rt_vs_tb2'),
        ((pl.col('dam_as_gen_regup') + pl.col('dam_as_load_regup')) / pl.col('capacity_mw') / (pl.col('avg_TB2') * pl.col('days')) * 100).alias('pct_regup_vs_tb2'),
    ])

    # Convert to pandas
    df_final = df_with_tb2.to_pandas()

    # Filter to valid records
    df_valid = df_final[df_final['avg_TB2'].notna()].copy()

    logger.info(f"\n✅ Total Revenue vs TB2 calculated for {len(df_valid)} battery-years")

    # Summary by battery (average across years)
    summary = df_valid.groupby('bess_name').agg({
        'capacity_mw': 'first',
        'settlement_point_final': 'first',
        'pct_total_vs_tb2': 'mean',
        'pct_da_energy_vs_tb2': 'mean',
        'pct_rt_vs_tb2': 'mean',
        'pct_regup_vs_tb2': 'mean',
        'total_rev_per_mw': 'mean',
        'tb2_benchmark_per_mw': 'mean',
        'avg_TB2': 'mean',
        'year': 'count'
    }).rename(columns={'year': 'years_operating'}).reset_index()

    summary = summary.sort_values('pct_total_vs_tb2', ascending=False)

    # Save results
    df_valid.to_csv('bess_total_vs_tb2_detailed.csv', index=False)
    logger.info("✅ Saved: bess_total_vs_tb2_detailed.csv")

    summary.to_csv('bess_total_vs_tb2_summary.csv', index=False)
    logger.info("✅ Saved: bess_total_vs_tb2_summary.csv")

    # Print results
    logger.info("\n" + "=" * 100)
    logger.info("TOP 20 BESS BY TOTAL REVENUE vs TB2 BENCHMARK")
    logger.info("=" * 100)
    print(summary.head(20)[['bess_name', 'settlement_point_final', 'capacity_mw', 'avg_TB2', 'pct_total_vs_tb2', 'years_operating']])

    logger.info("\n" + "=" * 100)
    logger.info("FLEET-WIDE STATISTICS")
    logger.info("=" * 100)
    logger.info(f"Average % Total Revenue vs TB2: {summary['pct_total_vs_tb2'].mean():.2f}%")
    logger.info(f"Median % Total Revenue vs TB2: {summary['pct_total_vs_tb2'].median():.2f}%")
    logger.info(f"Max: {summary['pct_total_vs_tb2'].max():.2f}% ({summary.iloc[0]['bess_name']})")
    logger.info(f"Min: {summary['pct_total_vs_tb2'].min():.2f}% ({summary.iloc[-1]['bess_name']})")

    # How many beat TB2?
    beat_tb2 = summary[summary['pct_total_vs_tb2'] > 100]
    logger.info(f"\n🏆 Batteries beating TB2 benchmark: {len(beat_tb2)}/{len(summary)} ({len(beat_tb2)/len(summary)*100:.1f}%)")

    # Revenue breakdown vs TB2
    logger.info("\n" + "=" * 100)
    logger.info("REVENUE COMPONENTS vs TB2 (Fleet Average)")
    logger.info("=" * 100)
    logger.info(f"  DA Energy: {summary['pct_da_energy_vs_tb2'].mean():.2f}% of TB2")
    logger.info(f"  RT Net: {summary['pct_rt_vs_tb2'].mean():.2f}% of TB2")
    logger.info(f"  Reg Up: {summary['pct_regup_vs_tb2'].mean():.2f}% of TB2")
    logger.info(f"  TOTAL: {summary['pct_total_vs_tb2'].mean():.2f}% of TB2")

    logger.info("\n✅ Analysis complete!")

    return summary


if __name__ == "__main__":
    main()
