#!/usr/bin/env python3
"""
Create BESS revenue stacked bar chart with % TB2 line overlay.
Matches the Gridmatic market report format.

Generates charts for 2024, 2023, and 2022.

NORMALIZED VERSION: Uses normalized revenue data that accounts for partial-year operations.
Excludes batteries with <60 active days (<2 months).
"""

import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def create_chart_for_year(year: int, df_revenue_all: pd.DataFrame, tb2_yearly: pl.DataFrame):
    """Create revenue chart with TB2 overlay for a specific year"""

    logger.info(f"\n{'='*100}")
    logger.info(f"Creating chart for {year}")
    logger.info(f"{'='*100}")

    # Filter to year
    df_year = df_revenue_all[df_revenue_all['year'] == year].copy()

    if len(df_year) == 0:
        logger.warning(f"No data for {year}")
        return

    logger.info(f"Found {len(df_year)} batteries for {year}")

    # Load BESS mapping for settlement points
    df_mapping = pd.read_csv('bess_mapping/BESS_UNIFIED_MAPPING_V3_CLARIFIED.csv')
    df_mapping = df_mapping[['BESS_Gen_Resource', 'Settlement_Point', 'IQ_Capacity_MW']].copy()
    df_mapping = df_mapping.rename(columns={
        'BESS_Gen_Resource': 'gen_resource',
        'Settlement_Point': 'settlement_point',
        'IQ_Capacity_MW': 'capacity_mw_mapping'
    })

    # Merge to get settlement points
    df_year = df_year.merge(df_mapping, on='gen_resource', how='left')

    # Use mapping capacity if available
    df_year['capacity_mw'] = df_year['capacity_mw_mapping'].fillna(df_year['capacity_mw'])
    df_year['settlement_point_final'] = df_year['settlement_point'].fillna(df_year['resource_node'])

    # Join with TB2 data
    df_year_pl = pl.from_pandas(df_year)
    df_year_pl = df_year_pl.with_columns([pl.col('year').cast(pl.Int32)])

    df_with_tb2 = df_year_pl.join(
        tb2_yearly,
        left_on=['settlement_point_final', 'year'],
        right_on=['SettlementPoint', 'year'],
        how='left'
    )

    # Use normalized revenue per kW-year (already calculated in normalization script)
    # For batteries with <365 days, these values are annualized
    df_with_tb2 = df_with_tb2.with_columns([
        # Use pre-calculated normalized values (already in $/kW-year)
        pl.col('dam_discharge_revenue_per_kw_year_normalized').alias('da_per_kw'),
        pl.col('rt_net_revenue_per_kw_year_normalized').alias('rt_per_kw'),
        # Combine gen + load for AS products (already normalized)
        (pl.col('dam_as_gen_regup_per_kw_year_normalized') +
         (pl.col('dam_as_load_regup_per_kw_year_normalized').fill_null(0))).alias('regup_per_kw'),
        (pl.col('dam_as_gen_regdown_per_kw_year_normalized') +
         (pl.col('dam_as_load_regdown_per_kw_year_normalized').fill_null(0))).alias('regdown_per_kw'),
        (pl.col('dam_as_gen_rrs_per_kw_year_normalized') +
         (pl.col('dam_as_load_rrs_per_kw_year_normalized').fill_null(0))).alias('reserves_per_kw'),
        (pl.col('dam_as_gen_ecrs_per_kw_year_normalized') +
         (pl.col('dam_as_load_ecrs_per_kw_year_normalized').fill_null(0))).alias('ecrs_per_kw'),
        (pl.col('dam_as_gen_nonspin_per_kw_year_normalized') +
         (pl.col('dam_as_load_nonspin_per_kw_year_normalized').fill_null(0))).alias('nonspin_per_kw'),
        pl.col('normalized_revenue_per_kw_year').alias('total_per_kw'),

        # % vs TB2 calculation:
        # - normalized_revenue_per_kw_year is in $/kW-year
        # - avg_TB2 is average daily spread in $/MWh
        # - Full year TB2 per kW = (avg_TB2 * days) / 1000
        (pl.col('normalized_revenue_per_kw_year') / ((pl.col('avg_TB2') * pl.col('days')) / 1000) * 100).alias('pct_total_vs_tb2')
    ])

    df_chart = df_with_tb2.to_pandas()

    # Filter to batteries with TB2 data and sort by total revenue per kW
    df_chart = df_chart[df_chart['avg_TB2'].notna()].copy()
    df_chart = df_chart.sort_values('total_per_kw', ascending=False)

    logger.info(f"Chart will show {len(df_chart)} batteries with TB2 data")

    if len(df_chart) == 0:
        logger.warning(f"No batteries with TB2 data for {year}")
        return

    # Create figure with dual y-axes
    fig, ax1 = plt.subplots(figsize=(24, 10))

    x = np.arange(len(df_chart))
    width = 0.8

    # Colors matching market report
    colors = {
        'da': '#8B7BC8',      # Purple
        'rt': '#5FD4AF',      # Teal
        'regup': '#F4E76E',   # Yellow
        'regdown': '#F4A6A6', # Pink
        'reserves': '#5DADE2', # Blue
        'ecrs': '#D7BDE2',    # Light purple
        'nonspin': '#34495E'  # Dark
    }

    # Build stacked bars (left y-axis: $/kW)
    bottom = np.zeros(len(df_chart))

    for component, color in colors.items():
        col = f'{component}_per_kw'
        ax1.bar(x, df_chart[col], width, bottom=bottom, color=color,
               label=component.upper().replace('_', ' '))
        bottom += df_chart[col].values

    # Configure left y-axis ($/kW-year)
    ax1.set_ylabel('Revenue ($/kW-year)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_chart['bess_name'], rotation=90, ha='right', fontsize=8)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'${int(y)}'))
    ax1.grid(axis='y', alpha=0.3, linestyle='--', zorder=0)
    ax1.axhline(y=0, color='black', linewidth=0.8, zorder=1)

    # Create second y-axis for % TB2
    ax2 = ax1.twinx()

    # Plot % TB2 as line
    line = ax2.plot(x, df_chart['pct_total_vs_tb2'],
                    color='black', linewidth=2, marker='o', markersize=3,
                    label='% DA TB2 Captured', zorder=10)

    # Configure right y-axis (% TB2)
    ax2.set_ylabel('% DA TB2 Captured', fontsize=14, fontweight='bold')
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y)}%'))

    # Add 100% reference line
    ax2.axhline(y=100, color='red', linewidth=1, linestyle='--',
               alpha=0.5, label='100% TB2 (Benchmark)', zorder=9)

    # Title
    ax1.set_title(f'{year} ERCOT BESS Fleet by Total Revenue\n' +
                 'Stacked Revenue Components ($/kW-year) with % DA TB2 Captured',
                 fontsize=16, fontweight='bold', pad=20)

    # Combine legends in upper right
    # Get handles and labels from both axes
    bars_handles, bars_labels = ax1.get_legend_handles_labels()
    line_handles, line_labels = ax2.get_legend_handles_labels()

    # Combine all handles and labels
    all_handles = bars_handles + line_handles
    all_labels = bars_labels + line_labels

    # Create single legend in upper right
    ax1.legend(all_handles, all_labels, loc='upper right', fontsize=10,
              framealpha=0.9, ncol=2)

    plt.tight_layout()

    # Save
    filename = f'bess_revenue_chart_{year}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    logger.info(f"✅ Saved: {filename}")

    plt.close()

    # Print statistics
    logger.info(f"\n{year} Statistics:")
    logger.info(f"  Batteries: {len(df_chart)}")
    logger.info(f"  Avg revenue/kW: ${df_chart['total_per_kw'].mean():.2f}")
    logger.info(f"  Avg % TB2: {df_chart['pct_total_vs_tb2'].mean():.2f}%")
    logger.info(f"  Batteries >100% TB2: {(df_chart['pct_total_vs_tb2'] > 100).sum()}")
    logger.info(f"  Top performer: {df_chart.iloc[0]['bess_name']} " +
               f"(${df_chart.iloc[0]['total_per_kw']:.2f}/kW, " +
               f"{df_chart.iloc[0]['pct_total_vs_tb2']:.1f}% TB2)")


def main():
    logger.info("=" * 100)
    logger.info("BESS REVENUE CHARTS WITH TB2 OVERLAY (2022, 2023, 2024)")
    logger.info("=" * 100)

    # Load normalized revenue data (accounts for partial-year operations)
    all_revenue = []
    for year in range(2019, 2025):
        file = f"bess_revenue_{year}_normalized.csv"
        if Path(file).exists():
            df_year = pd.read_csv(file)
            all_revenue.append(df_year)
            logger.info(f"  Loaded {year}: {len(df_year)} batteries (after <60 day filter)")

    df_revenue_all = pd.concat(all_revenue, ignore_index=True)
    logger.info(f"Loaded {len(df_revenue_all)} battery-year records (normalized)")

    # Load TB2 data
    tb2_yearly = pl.read_parquet('tb2_by_settlement_point_yearly.parquet')
    logger.info(f"Loaded TB2 data for {len(tb2_yearly)} settlement point-years")

    # Create charts for each year
    for year in [2024, 2023, 2022]:
        create_chart_for_year(year, df_revenue_all, tb2_yearly)

    logger.info("\n" + "=" * 100)
    logger.info("✅ All charts generated!")
    logger.info("=" * 100)


if __name__ == "__main__":
    main()
