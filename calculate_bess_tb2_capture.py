#!/usr/bin/env python3
"""
Calculate % DA TB2 Captured for each BESS.

% DA TB2 Captured = (Actual DA Energy Revenue / Capacity) / (TB2 Opportunity / 1 MW) * 100

This measures how much of the theoretical arbitrage opportunity the battery actually captured.
"""

import pandas as pd
import polars as pl
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("=" * 100)
    logger.info("BESS TB2 CAPTURE ANALYSIS")
    logger.info("=" * 100)

    # Load BESS mapping to get settlement points
    mapping_file = Path("bess_mapping/BESS_UNIFIED_MAPPING_V3_CLARIFIED.csv")
    df_mapping = pd.read_csv(mapping_file)

    # Filter to operational BESS
    df_bess = df_mapping[
        df_mapping['True_Operational_Status'] == 'Operational (has Load Resource)'
    ][['BESS_Gen_Resource', 'Settlement_Point', 'IQ_Capacity_MW']].copy()

    df_bess = df_bess.rename(columns={
        'BESS_Gen_Resource': 'gen_resource',
        'Settlement_Point': 'settlement_point',
        'IQ_Capacity_MW': 'capacity_mw'
    })

    logger.info(f"Loaded {len(df_bess)} operational BESS units")

    # Load TB2 data (yearly averages by settlement point)
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

    # Merge with mapping to get settlement points
    # Revenue file already has 'resource_node', just need to update capacity from mapping
    df_analysis = df_revenue.merge(
        df_bess[['gen_resource', 'settlement_point', 'capacity_mw']],
        on='gen_resource',
        how='left',
        suffixes=('_revenue', '_mapping')
    )

    # Use capacity from mapping file (corrected nameplate) if available
    df_analysis['capacity_mw'] = df_analysis['capacity_mw_mapping'].fillna(df_analysis['capacity_mw_revenue'])

    # Use settlement_point from mapping if different from resource_node
    df_analysis['settlement_point_final'] = df_analysis['settlement_point'].fillna(df_analysis['resource_node'])

    logger.info(f"Merged data: {len(df_analysis)} records")
    logger.info(f"Records with settlement points: {df_analysis['settlement_point_final'].notna().sum()}")

    # Convert to polars for joining with TB2
    df_analysis_pl = pl.from_pandas(df_analysis)

    # Cast year to int32 to match TB2 data
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

    # Calculate % DA TB2 Captured
    # TB2 is $/MWh opportunity per day
    # DA Revenue is total $ for the year
    # Need to normalize both per MW and per day

    df_with_tb2 = df_with_tb2.with_columns([
        # DA energy revenue per MW per day
        (pl.col('dam_discharge_revenue') / pl.col('capacity_mw') / pl.col('days')).alias('da_rev_per_mw_day'),

        # TB2 is already per MWh per day (average)
        pl.col('avg_TB2').alias('tb2_per_mw_day'),

        # % TB2 Captured = (actual revenue per MW day) / (TB2 opportunity) * 100
        ((pl.col('dam_discharge_revenue') / pl.col('capacity_mw') / pl.col('days')) / pl.col('avg_TB2') * 100).alias('pct_da_tb2_captured')
    ])

    # Convert back to pandas for easier manipulation
    df_final = df_with_tb2.to_pandas()

    # Filter to valid records (have TB2 data)
    df_valid = df_final[df_final['avg_TB2'].notna()].copy()

    logger.info(f"\n✅ TB2 Capture calculated for {len(df_valid)} battery-years")

    # Summary by battery (average across years)
    summary = df_valid.groupby('bess_name').agg({
        'capacity_mw': 'first',
        'settlement_point_final': 'first',
        'pct_da_tb2_captured': 'mean',
        'tb2_per_mw_day': 'mean',
        'da_rev_per_mw_day': 'mean',
        'year': 'count'
    }).rename(columns={'year': 'years_operating'}).reset_index()

    summary = summary.sort_values('pct_da_tb2_captured', ascending=False)

    # Save detailed results
    df_valid.to_csv('bess_tb2_capture_detailed.csv', index=False)
    logger.info("✅ Saved: bess_tb2_capture_detailed.csv")

    # Save summary
    summary.to_csv('bess_tb2_capture_summary.csv', index=False)
    logger.info("✅ Saved: bess_tb2_capture_summary.csv")

    # Print top performers
    logger.info("\n" + "=" * 100)
    logger.info("TOP 20 BESS BY % DA TB2 CAPTURED (Average 2019-2024)")
    logger.info("=" * 100)
    print(summary.head(20)[['bess_name', 'settlement_point_final', 'capacity_mw', 'tb2_per_mw_day', 'pct_da_tb2_captured', 'years_operating']])

    # Statistics
    logger.info("\n" + "=" * 100)
    logger.info("FLEET-WIDE TB2 CAPTURE STATISTICS")
    logger.info("=" * 100)
    logger.info(f"Average % DA TB2 Captured: {summary['pct_da_tb2_captured'].mean():.2f}%")
    logger.info(f"Median % DA TB2 Captured: {summary['pct_da_tb2_captured'].median():.2f}%")
    logger.info(f"Max % DA TB2 Captured: {summary['pct_da_tb2_captured'].max():.2f}% ({summary.iloc[0]['bess_name']})")
    logger.info(f"Min % DA TB2 Captured: {summary['pct_da_tb2_captured'].min():.2f}% ({summary.iloc[-1]['bess_name']})")

    logger.info("\n✅ TB2 capture analysis complete!")

    return summary


if __name__ == "__main__":
    main()
