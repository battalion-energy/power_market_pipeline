#!/usr/bin/env python3
"""
Create BESS revenue stacked bar charts with % TB2 line overlay.
Generates quarterly and monthly charts for 2022, 2023, and 2024.
"""

import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def create_chart(df_period: pd.DataFrame, period_name: str, year: int, output_dir: Path):
    """Create revenue chart with TB2 overlay for a specific period"""

    if len(df_period) == 0:
        logger.warning(f"No data for {period_name}")
        return

    # Sort by total revenue per kW
    df_period = df_period.sort_values('total_per_kw', ascending=False)

    logger.info(f"  {period_name}: {len(df_period)} batteries")

    # Create figure with dual y-axes
    fig, ax1 = plt.subplots(figsize=(24, 10))

    x = np.arange(len(df_period))
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

    # Build stacked bars
    bottom = np.zeros(len(df_period))

    for component, color in colors.items():
        col = f'{component}_per_kw'
        ax1.bar(x, df_period[col], width, bottom=bottom, color=color,
               label=component.upper().replace('_', ' '))
        bottom += df_period[col].values

    # Configure left y-axis ($/kW-year equivalent)
    ax1.set_ylabel('Revenue ($/kW-year equivalent)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_period['bess_name'], rotation=90, ha='right', fontsize=8)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'${int(y)}'))
    ax1.grid(axis='y', alpha=0.3, linestyle='--', zorder=0)
    ax1.axhline(y=0, color='black', linewidth=0.8, zorder=1)

    # Create second y-axis for % TB2
    ax2 = ax1.twinx()

    # Plot % TB2 as line
    line = ax2.plot(x, df_period['pct_total_vs_tb2'],
                    color='black', linewidth=2, marker='o', markersize=3,
                    label='% DA TB2 Captured', zorder=10)

    # Configure right y-axis
    ax2.set_ylabel('% DA TB2 Captured', fontsize=14, fontweight='bold')
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y)}%'))

    # Add 100% reference line
    ax2.axhline(y=100, color='red', linewidth=1, linestyle='--',
               alpha=0.5, label='100% TB2 (Benchmark)', zorder=9)

    # Title
    ax1.set_title(f'{period_name} ERCOT BESS Fleet by Total Revenue\n' +
                 'Stacked Revenue Components ($/kW-year equivalent) with % DA TB2 Captured',
                 fontsize=16, fontweight='bold', pad=20)

    # Combine legends in upper right
    bars_handles, bars_labels = ax1.get_legend_handles_labels()
    line_handles, line_labels = ax2.get_legend_handles_labels()
    all_handles = bars_handles + line_handles
    all_labels = bars_labels + line_labels
    ax1.legend(all_handles, all_labels, loc='upper right', fontsize=10,
              framealpha=0.9, ncol=2)

    plt.tight_layout()

    # Save
    filename = output_dir / f"bess_revenue_{period_name.lower().replace(' ', '_').replace('q', 'q')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    logger.info(f"  ✅ Saved: {filename.name}")

    plt.close()

    # Print statistics
    logger.info(f"  Stats: Avg ${df_period['total_per_kw'].mean():.2f}/kW-year, " +
               f"Avg {df_period['pct_total_vs_tb2'].mean():.2f}% TB2, " +
               f"{(df_period['pct_total_vs_tb2'] > 100).sum()} batteries >100% TB2")


def calculate_revenue_data_with_tb2(year: int, df_revenue_all: pd.DataFrame,
                                     tb2_daily: pl.DataFrame, df_mapping: pd.DataFrame):
    """Calculate revenue data with TB2 for a specific year, return for aggregation"""

    # Filter to year
    df_year = df_revenue_all[df_revenue_all['year'] == year].copy()

    if len(df_year) == 0:
        return None

    # Merge to get settlement points and corrected capacity
    df_year = df_year.merge(df_mapping, on='gen_resource', how='left')
    df_year['capacity_mw'] = df_year['capacity_mw_mapping'].fillna(df_year['capacity_mw_revenue'])
    df_year['settlement_point_final'] = df_year['settlement_point'].fillna(df_year['resource_node'])

    # Load detailed revenue records to get timestamps
    revenue_file = f"bess_revenue_{year}.csv"
    if not Path(revenue_file).exists():
        return None

    # For quarterly/monthly, we need to aggregate from daily TB2 and match to revenue periods
    # We'll return the yearly data with settlement points for period aggregation
    return df_year


def _period_revenue_exact(year: int, month_start: int, month_end: int, df_map: pd.DataFrame, base_dir: Path) -> pd.DataFrame:
    """Compute exact period revenues by summing hourly parquet components per unit.
    Returns a DataFrame with columns needed for plotting ($, capacity, settlement_point).
    """
    rows = []
    for _, r in df_map.iterrows():
        bess = r['gen_resource']
        cap = r['capacity_mw_mapping']
        sp = r['settlement_point']
        dp = base_dir / 'bess_analysis' / 'hourly' / 'dispatch' / f'{bess}_{year}_dispatch.parquet'
        aw = base_dir / 'bess_analysis' / 'hourly' / 'awards' / f'{bess}_{year}_awards.parquet'
        if not dp.exists() or not aw.exists() or pd.isna(cap) or cap == 0:
            continue
        df_dp = pl.read_parquet(dp)
        df_aw = pl.read_parquet(aw)
        # filter months
        df_dp = df_dp.filter((pl.col('local_date').dt.month() >= month_start) & (pl.col('local_date').dt.month() <= month_end))
        df_aw = df_aw.filter((pl.col('local_date').dt.month() >= month_start) & (pl.col('local_date').dt.month() <= month_end))
        if len(df_dp) == 0 and len(df_aw) == 0:
            continue
        # Exact components
        rt_net = float(df_dp.select(pl.col('rt_net_revenue_hour').sum()).item() or 0.0)
        da_energy = float(df_dp.select(pl.col('da_energy_revenue_hour').sum()).item() or 0.0)
        # AS revenues (hourly awards × MCPC)
        for c_mw, c_p in [('regup_mw','regup_mcpc'),('regdown_mw','regdown_mcpc'),('rrs_mw','rrs_mcpc'),('ecrs_mw','ecrs_mcpc'),('nonspin_mw','nonspin_mcpc')]:
            if c_mw in df_aw.columns and c_p in df_aw.columns:
                df_aw = df_aw.with_columns((pl.col(c_mw).fill_null(0.0) * pl.col(c_p).fill_null(0.0)).alias(f'{c_mw}_rev'))
        as_total = 0.0
        for c in ['regup_mw_rev','regdown_mw_rev','rrs_mw_rev','ecrs_mw_rev','nonspin_mw_rev']:
            if c in df_aw.columns:
                as_total += float(df_aw.select(pl.col(c).sum()).item() or 0.0)
        total = da_energy + rt_net + as_total
        rows.append({'bess_name': bess,
                     'capacity_mw': float(cap),
                     'settlement_point_final': sp,
                     'da_revenue': da_energy,
                     'rt_revenue': rt_net,
                     'as_revenue': as_total,
                     'total_revenue': total})
    return pd.DataFrame(rows)


def create_period_charts(year: int, df_revenue_all: pd.DataFrame, tb2_daily: pl.DataFrame,
                        df_mapping: pd.DataFrame, output_dir: Path, base_dir: Path):
    """Create quarterly and monthly charts for a specific year"""

    logger.info(f"\n{'='*100}")
    logger.info(f"Creating period charts for {year}")
    logger.info(f"{'='*100}")

    # Load raw revenue file to get monthly/quarterly breakdowns
    # Try TELEMETERED first, fallback to regular
    revenue_file_telem = f"bess_revenue_{year}_TELEMETERED.csv"
    revenue_file = f"bess_revenue_{year}.csv"

    if Path(revenue_file_telem).exists():
        revenue_file = revenue_file_telem

    df_year_detail = pd.read_csv(revenue_file)

    # Merge with mapping
    df_year_detail = df_year_detail.merge(
        df_mapping[['gen_resource', 'settlement_point', 'capacity_mw_mapping']],
        on='gen_resource', how='left', suffixes=('_orig', '_from_mapping')
    )
    # Use mapping capacity if available, otherwise use existing
    if 'capacity_mw' in df_year_detail.columns:
        df_year_detail['capacity_mw'] = df_year_detail['capacity_mw_mapping'].fillna(df_year_detail['capacity_mw'])
    else:
        df_year_detail['capacity_mw'] = df_year_detail['capacity_mw_mapping']

    # Use settlement_point from mapping if available
    if 'settlement_point_from_mapping' in df_year_detail.columns:
        df_year_detail['settlement_point_final'] = df_year_detail['settlement_point_from_mapping'].fillna(df_year_detail['resource_node'])
    else:
        df_year_detail['settlement_point_final'] = df_year_detail['resource_node']

    # For simplicity, divide annual revenue into periods and use proportional TB2
    # This is approximate - ideally we'd recalculate from monthly data

    # Quarterly breakdowns (exact aggregation from hourly parquets)
    quarters = {
        'Q1': (1, 91),   # Jan-Mar (91 days in non-leap year)
        'Q2': (2, 91),   # Apr-Jun
        'Q3': (3, 92),   # Jul-Sep
        'Q4': (4, 91),   # Oct-Dec
    }

    # Get TB2 data for this year
    tb2_year = tb2_daily.filter(pl.col('year') == year)

    logger.info(f"\nGenerating Quarterly Charts for {year}:")
    for q_name, (q_num, days_in_q) in quarters.items():
        # Calculate quarter months
        start_month = (q_num - 1) * 3 + 1
        end_month = start_month + 2

        # Filter TB2 to quarter
        tb2_q = tb2_year.filter(
            (pl.col('DeliveryDate').dt.month() >= start_month) &
            (pl.col('DeliveryDate').dt.month() <= end_month)
        )

        # Calculate average TB2 for quarter
        tb2_q_avg = tb2_q.group_by('SettlementPoint').agg([
            pl.col('TB2').mean().alias('avg_TB2'),
            pl.col('TB2').count().alias('days')
        ])

        # Exact quarterly revenue from hourly parquets
        df_q = _period_revenue_exact(year, start_month, end_month, df_mapping, base_dir)

        # Join with TB2
        df_q_pl = pl.from_pandas(df_q)
        df_q_with_tb2 = df_q_pl.join(
            tb2_q_avg,
            left_on='settlement_point_final',
            right_on='SettlementPoint',
            how='left'
        )

        # Calculate per kW and % TB2 (annualized)
        df_q_with_tb2 = df_q_with_tb2.with_columns([
            # Annualized revenue per kW (multiply quarterly by 4 to get annual rate)
            (pl.col('da_revenue') / (pl.col('capacity_mw') * 1000) * (365/days_in_q)).alias('da_per_kw'),
            (pl.col('rt_revenue') / (pl.col('capacity_mw') * 1000) * (365/days_in_q)).alias('rt_per_kw'),
            pl.lit(0.0).alias('regup_per_kw'),
            pl.lit(0.0).alias('regdown_per_kw'),
            (pl.col('as_revenue') / (pl.col('capacity_mw') * 1000) * (365/days_in_q)).alias('reserves_per_kw'),
            pl.lit(0.0).alias('ecrs_per_kw'),
            pl.lit(0.0).alias('nonspin_per_kw'),
            (pl.col('total_revenue') / (pl.col('capacity_mw') * 1000) * (365/days_in_q)).alias('total_per_kw'),

            # % vs TB2
            ((pl.col('total_revenue') / pl.col('capacity_mw')) / (pl.col('avg_TB2') * pl.col('days')) * 100).alias('pct_total_vs_tb2')
        ])

        df_q_chart = df_q_with_tb2.to_pandas()
        df_q_chart = df_q_chart[df_q_chart['avg_TB2'].notna()].copy()

        create_chart(df_q_chart, f"{year} {q_name}", year, output_dir)

    # Monthly breakdowns
    logger.info(f"\nGenerating Monthly Charts for {year}:")
    months = [
        ('Jan', 1, 31), ('Feb', 2, 28), ('Mar', 3, 31),
        ('Apr', 4, 30), ('May', 5, 31), ('Jun', 6, 30),
        ('Jul', 7, 31), ('Aug', 8, 31), ('Sep', 9, 30),
        ('Oct', 10, 31), ('Nov', 11, 30), ('Dec', 12, 31)
    ]

    # Adjust Feb for leap year
    if year == 2024:
        months[1] = ('Feb', 2, 29)

    for month_name, month_num, days_in_month in months:
        # Filter TB2 to month
        tb2_m = tb2_year.filter(pl.col('DeliveryDate').dt.month() == month_num)

        # Calculate average TB2 for month
        tb2_m_avg = tb2_m.group_by('SettlementPoint').agg([
            pl.col('TB2').mean().alias('avg_TB2'),
            pl.col('TB2').count().alias('days')
        ])

        # Exact monthly revenue from hourly parquets
        df_m = _period_revenue_exact(year, month_num, month_num, df_mapping, base_dir)

        # Join with TB2
        df_m_pl = pl.from_pandas(df_m)
        df_m_with_tb2 = df_m_pl.join(
            tb2_m_avg,
            left_on='settlement_point_final',
            right_on='SettlementPoint',
            how='left'
        )

        # Calculate per kW and % TB2 (annualized)
        df_m_with_tb2 = df_m_with_tb2.with_columns([
            (pl.col('da_revenue') / (pl.col('capacity_mw') * 1000) * (365/days_in_month)).alias('da_per_kw'),
            (pl.col('rt_revenue') / (pl.col('capacity_mw') * 1000) * (365/days_in_month)).alias('rt_per_kw'),
            pl.lit(0.0).alias('regup_per_kw'),
            pl.lit(0.0).alias('regdown_per_kw'),
            (pl.col('as_revenue') / (pl.col('capacity_mw') * 1000) * (365/days_in_month)).alias('reserves_per_kw'),
            pl.lit(0.0).alias('ecrs_per_kw'),
            pl.lit(0.0).alias('nonspin_per_kw'),
            (pl.col('total_revenue') / (pl.col('capacity_mw') * 1000) * (365/days_in_month)).alias('total_per_kw'),

            ((pl.col('total_revenue') / pl.col('capacity_mw')) / (pl.col('avg_TB2') * pl.col('days')) * 100).alias('pct_total_vs_tb2')
        ])

        df_m_chart = df_m_with_tb2.to_pandas()
        df_m_chart = df_m_chart[df_m_chart['avg_TB2'].notna()].copy()

        create_chart(df_m_chart, f"{year} {month_name}", year, output_dir)


def main():
    logger.info("=" * 100)
    logger.info("BESS QUARTERLY & MONTHLY REVENUE CHARTS (2022-2024)")
    logger.info("=" * 100)

    # Create output directory
    output_dir = Path("bess_revenue_charts_periods")
    output_dir.mkdir(exist_ok=True)

    # Load all revenue data (prefer TELEMETERED)
    all_revenue = []
    for year in range(2019, 2025):
        file_telem = f"bess_revenue_{year}_TELEMETERED.csv"
        file_regular = f"bess_revenue_{year}.csv"
        p_t = Path(file_telem)
        p_r = Path(file_regular)
        try:
            if p_t.exists() and p_t.stat().st_size > 0:
                all_revenue.append(pd.read_csv(p_t))
            elif p_r.exists() and p_r.stat().st_size > 0:
                all_revenue.append(pd.read_csv(p_r))
        except Exception:
            pass

    if not all_revenue:
        logger.warning("No annual revenue CSVs found; skipping charts")
        return
    df_revenue_all = pd.concat(all_revenue, ignore_index=True)
    logger.info(f"Loaded {len(df_revenue_all)} battery-year records")

    # Load TB2 daily data
    tb2_daily = pl.read_parquet('tb2_daily_2019_2024.parquet')
    logger.info(f"Loaded daily TB2 data")

    # Load mapping
    df_mapping = pd.read_csv('bess_mapping/BESS_UNIFIED_MAPPING_V3_CLARIFIED.csv')
    df_mapping = df_mapping[['BESS_Gen_Resource', 'Settlement_Point', 'IQ_Capacity_MW']].copy()
    df_mapping = df_mapping.rename(columns={
        'BESS_Gen_Resource': 'gen_resource',
        'Settlement_Point': 'settlement_point',
        'IQ_Capacity_MW': 'capacity_mw_mapping'
    })

    # Create charts for each year (exact aggregation)
    for year in [2024, 2023, 2022]:
        create_period_charts(year, df_revenue_all, tb2_daily, df_mapping, output_dir, Path('/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data'))

    logger.info("\n" + "=" * 100)
    logger.info(f"✅ All quarterly and monthly charts generated in {output_dir}/")
    logger.info("=" * 100)


if __name__ == "__main__":
    main()
