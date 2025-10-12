#!/usr/bin/env python3
"""
Create BESS revenue stacked bar charts with % TB2 line overlay.
Generates quarterly, monthly, and YTD charts. Now supports 2025.
"""

import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import logging
from datetime import datetime, date

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
        # filter target year and months
        df_dp = df_dp.filter(
            (pl.col('local_date').dt.year() == year) &
            (pl.col('local_date').dt.month() >= month_start) &
            (pl.col('local_date').dt.month() <= month_end)
        )
        df_aw = df_aw.filter(
            (pl.col('local_date').dt.year() == year) &
            (pl.col('local_date').dt.month() >= month_start) &
            (pl.col('local_date').dt.month() <= month_end)
        )
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
    # Always return a DataFrame with the expected schema so downstream joins don't fail
    expected_cols = [
        'bess_name',
        'capacity_mw',
        'settlement_point_final',
        'da_revenue',
        'rt_revenue',
        'as_revenue',
        'total_revenue',
    ]
    if not rows:
        return pd.DataFrame(columns=expected_cols)
    # Ensure column order
    df = pd.DataFrame(rows)
    # Add any missing expected columns (robust to future additions)
    for c in expected_cols:
        if c not in df.columns:
            df[c] = np.nan
    return df[expected_cols]


def _compute_cod_map(year: int, df_map: pd.DataFrame, base_dir: Path) -> pd.DataFrame:
    """Infer COD/start date per BESS for the target year using first dispatch date.
    Returns columns: bess_name, settlement_point, cod_date (date)
    """
    rows: list[dict] = []
    ddir = base_dir / 'bess_analysis' / 'hourly' / 'dispatch'
    for _, r in df_map.iterrows():
        bess = r['gen_resource']
        sp = r['settlement_point']
        dp = ddir / f'{bess}_{year}_dispatch.parquet'
        if not dp.exists():
            continue
        try:
            df_dp = pl.read_parquet(dp)
            # restrict to target year and find earliest local_date
            df_dp_y = df_dp.filter(pl.col('local_date').dt.year() == year)
            if len(df_dp_y) == 0:
                continue
            first = df_dp_y.select(pl.col('local_date').min()).item()
            if isinstance(first, datetime):
                first = first.date()
            rows.append({'bess_name': bess, 'settlement_point': sp, 'cod_date': first})
        except Exception:
            continue
    if not rows:
        return pd.DataFrame(columns=['bess_name', 'settlement_point', 'cod_date'])
    return pd.DataFrame(rows)


def create_period_charts(year: int, df_revenue_all: pd.DataFrame, tb2_daily: pl.DataFrame,
                        df_mapping: pd.DataFrame, output_dir: Path, base_dir: Path):
    """Create quarterly and monthly charts for a specific year"""

    logger.info(f"\n{'='*100}")
    logger.info(f"Creating period charts for {year}")
    logger.info(f"{'='*100}")

    # Load raw revenue file for the year if present (not strictly required for exact parquets aggregation)
    revenue_file_telem = f"bess_revenue_{year}_TELEMETERED.csv"
    revenue_file_regular = f"bess_revenue_{year}.csv"
    df_year_detail = pd.DataFrame()
    try:
        if Path(revenue_file_telem).exists():
            df_year_detail = pd.read_csv(revenue_file_telem)
        elif Path(revenue_file_regular).exists():
            df_year_detail = pd.read_csv(revenue_file_regular)
        else:
            logger.warning(f"No annual revenue CSV found for {year}; proceeding with parquet aggregation only")
    except Exception as e:
        logger.warning(f"Could not load revenue CSV for {year}: {e}")

    # Merge with mapping
    # If revenue detail is available, enrich with mapping; otherwise skip (we aggregate from parquets below)
    if not df_year_detail.empty and 'gen_resource' in df_year_detail.columns:
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
        'Q4': (4, 92),   # Oct-Dec
    }

    # Compute COD map for this year and prepare TB2 view
    cod_df_pd = _compute_cod_map(year, df_mapping, base_dir)
    cod_pl = (
        pl.from_pandas(cod_df_pd)
        .with_columns(pl.col('cod_date').cast(pl.Date, strict=False))
        if len(cod_df_pd)
        else pl.DataFrame({'bess_name': [], 'settlement_point': [], 'cod_date': []},
                          schema={'bess_name': pl.Utf8, 'settlement_point': pl.Utf8, 'cod_date': pl.Date})
    )

    # Get TB2 data for this year (by DeliveryDate)
    tb2_year = tb2_daily.filter(pl.col('DeliveryDate').dt.year() == year)

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

        # Effective TB2 per unit after COD, summed over the quarter
        tb2_q_by_unit = (
            tb2_q.join(cod_pl, left_on='SettlementPoint', right_on='settlement_point', how='inner')
                 .filter(pl.col('DeliveryDate') >= pl.col('cod_date'))
                 .group_by('bess_name')
                 .agg([
                     pl.col('TB2').sum().alias('tb2_sum'),
                     pl.len().alias('days_eff')
                 ])
        )

        # Exact quarterly revenue from hourly parquets
        df_q = _period_revenue_exact(year, start_month, end_month, df_mapping, base_dir)

        # Join with TB2
        df_q_pl = pl.from_pandas(df_q).with_columns([
            pl.col('capacity_mw').cast(pl.Float64, strict=False),
            pl.col('da_revenue').cast(pl.Float64, strict=False),
            pl.col('rt_revenue').cast(pl.Float64, strict=False),
            pl.col('as_revenue').cast(pl.Float64, strict=False),
            pl.col('total_revenue').cast(pl.Float64, strict=False),
        ])
        df_q_with_tb2 = df_q_pl.join(tb2_q_by_unit, on='bess_name', how='left')

        # Calculate per kW and % TB2 (annualized)
        # Annualization scale based on effective days with TB2 after COD per unit
        _scale_q_expr = (
            pl.when(pl.col('days_eff').fill_null(0) > 0)
              .then(365 / pl.col('days_eff'))
              .otherwise(0.0)
        )
        df_q_with_tb2 = df_q_with_tb2.with_columns([
            pl.when(pl.col('capacity_mw') > 0)
              .then(pl.col('da_revenue') / (pl.col('capacity_mw') * 1000.0) * _scale_q_expr)
              .otherwise(0.0)
              .alias('da_per_kw'),
            pl.when(pl.col('capacity_mw') > 0)
              .then(pl.col('rt_revenue') / (pl.col('capacity_mw') * 1000.0) * _scale_q_expr)
              .otherwise(0.0)
              .alias('rt_per_kw'),
            pl.lit(0.0).alias('regup_per_kw'),
            pl.lit(0.0).alias('regdown_per_kw'),
            pl.when(pl.col('capacity_mw') > 0)
              .then(pl.col('as_revenue') / (pl.col('capacity_mw') * 1000.0) * _scale_q_expr)
              .otherwise(0.0)
              .alias('reserves_per_kw'),
            pl.lit(0.0).alias('ecrs_per_kw'),
            pl.lit(0.0).alias('nonspin_per_kw'),
            pl.when(pl.col('capacity_mw') > 0)
              .then(pl.col('total_revenue') / (pl.col('capacity_mw') * 1000.0) * _scale_q_expr)
              .otherwise(0.0)
              .alias('total_per_kw'),

            # % vs TB2 using effective TB2 sum after COD
            pl.when((pl.col('capacity_mw') > 0) & (pl.col('tb2_sum').fill_null(0.0) > 0))
              .then(((pl.col('total_revenue') / pl.col('capacity_mw')) / pl.col('tb2_sum')) * 100.0)
              .otherwise(None)
              .alias('pct_total_vs_tb2')
        ])

        df_q_chart = df_q_with_tb2.to_pandas()
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

        # Effective TB2 per unit after COD, summed over the month
        tb2_m_by_unit = (
            tb2_m.join(cod_pl, left_on='SettlementPoint', right_on='settlement_point', how='inner')
                 .filter(pl.col('DeliveryDate') >= pl.col('cod_date'))
                 .group_by('bess_name')
                 .agg([
                     pl.col('TB2').sum().alias('tb2_sum'),
                     pl.len().alias('days_eff')
                 ])
        )

        # Exact monthly revenue from hourly parquets
        df_m = _period_revenue_exact(year, month_num, month_num, df_mapping, base_dir)

        # Join with TB2
        df_m_pl = pl.from_pandas(df_m).with_columns([
            pl.col('capacity_mw').cast(pl.Float64, strict=False),
            pl.col('da_revenue').cast(pl.Float64, strict=False),
            pl.col('rt_revenue').cast(pl.Float64, strict=False),
            pl.col('as_revenue').cast(pl.Float64, strict=False),
            pl.col('total_revenue').cast(pl.Float64, strict=False),
        ])
        df_m_with_tb2 = df_m_pl.join(tb2_m_by_unit, on='bess_name', how='left')

        # Calculate per kW and % TB2 (annualized)
        _scale_m_expr = (
            pl.when(pl.col('days_eff').fill_null(0) > 0)
              .then(365 / pl.col('days_eff'))
              .otherwise(0.0)
        )
        df_m_with_tb2 = df_m_with_tb2.with_columns([
            pl.when(pl.col('capacity_mw') > 0)
              .then(pl.col('da_revenue') / (pl.col('capacity_mw') * 1000.0) * _scale_m_expr)
              .otherwise(0.0)
              .alias('da_per_kw'),
            pl.when(pl.col('capacity_mw') > 0)
              .then(pl.col('rt_revenue') / (pl.col('capacity_mw') * 1000.0) * _scale_m_expr)
              .otherwise(0.0)
              .alias('rt_per_kw'),
            pl.lit(0.0).alias('regup_per_kw'),
            pl.lit(0.0).alias('regdown_per_kw'),
            pl.when(pl.col('capacity_mw') > 0)
              .then(pl.col('as_revenue') / (pl.col('capacity_mw') * 1000.0) * _scale_m_expr)
              .otherwise(0.0)
              .alias('reserves_per_kw'),
            pl.lit(0.0).alias('ecrs_per_kw'),
            pl.lit(0.0).alias('nonspin_per_kw'),
            pl.when(pl.col('capacity_mw') > 0)
              .then(pl.col('total_revenue') / (pl.col('capacity_mw') * 1000.0) * _scale_m_expr)
              .otherwise(0.0)
              .alias('total_per_kw'),

            pl.when((pl.col('capacity_mw') > 0) & (pl.col('tb2_sum').fill_null(0.0) > 0))
              .then(((pl.col('total_revenue') / pl.col('capacity_mw')) / pl.col('tb2_sum')) * 100.0)
              .otherwise(None)
              .alias('pct_total_vs_tb2')
        ])

        df_m_chart = df_m_with_tb2.to_pandas()
        create_chart(df_m_chart, f"{year} {month_name}", year, output_dir)

    # YTD breakdown (through last available date for the year)
    # Prefer TB2 last date; if unavailable, fall back to max local_date from any dispatch parquet.
    try:
        last_dt = tb2_year.select(pl.col('DeliveryDate').max()).item()
    except Exception:
        last_dt = None
    if last_dt is None:
        try:
            ddir = base_dir / 'bess_analysis' / 'hourly' / 'dispatch'
            # pick any matching file for the year
            candidates = list(ddir.glob(f'*_{year}_dispatch.parquet'))
            if candidates:
                df_any = pl.read_parquet(candidates[0])
                last_dt = df_any.select(pl.col('local_date').max()).item()
        except Exception:
            last_dt = None
    if last_dt is not None:
        if isinstance(last_dt, datetime):
            last_date = last_dt.date()
        else:
            last_date = last_dt
        if isinstance(last_date, date) and last_date.year == year:
            # TB2 over YTD (sum per unit after COD)
            tb2_ytd = tb2_year.filter(pl.col('DeliveryDate') <= pl.lit(last_date))
            tb2_ytd_by_unit = (
                tb2_ytd.join(cod_pl, left_on='SettlementPoint', right_on='settlement_point', how='inner')
                        .filter(pl.col('DeliveryDate') >= pl.col('cod_date'))
                        .group_by('bess_name')
                        .agg([
                            pl.col('TB2').sum().alias('tb2_sum'),
                            pl.len().alias('days_eff')
                        ])
            )

            # Exact YTD revenue (aggregate months 1..last_date.month)
            df_ytd = _period_revenue_exact(year, 1, int(last_date.month), df_mapping, base_dir)

            df_ytd_pl = pl.from_pandas(df_ytd).with_columns([
                pl.col('capacity_mw').cast(pl.Float64, strict=False),
                pl.col('da_revenue').cast(pl.Float64, strict=False),
                pl.col('rt_revenue').cast(pl.Float64, strict=False),
                pl.col('as_revenue').cast(pl.Float64, strict=False),
                pl.col('total_revenue').cast(pl.Float64, strict=False),
            ])
            df_ytd_with_tb2 = df_ytd_pl.join(tb2_ytd_by_unit, on='bess_name', how='left')

            # Determine number of days in YTD from the last_date
            start_of_year = date(year, 1, 1)
            days_in_ytd = (last_date - start_of_year).days + 1
            _scale_ytd = 365 / days_in_ytd if days_in_ytd > 0 else 0.0

            df_ytd_with_tb2 = df_ytd_with_tb2.with_columns([
                pl.when(pl.col('capacity_mw') > 0)
                  .then(pl.col('da_revenue') / (pl.col('capacity_mw') * 1000.0) * pl.lit(_scale_ytd))
                  .otherwise(0.0)
                  .alias('da_per_kw'),
                pl.when(pl.col('capacity_mw') > 0)
                  .then(pl.col('rt_revenue') / (pl.col('capacity_mw') * 1000.0) * pl.lit(_scale_ytd))
                  .otherwise(0.0)
                  .alias('rt_per_kw'),
                pl.lit(0.0).alias('regup_per_kw'),
                pl.lit(0.0).alias('regdown_per_kw'),
                pl.when(pl.col('capacity_mw') > 0)
                  .then(pl.col('as_revenue') / (pl.col('capacity_mw') * 1000.0) * pl.lit(_scale_ytd))
                  .otherwise(0.0)
                  .alias('reserves_per_kw'),
                pl.lit(0.0).alias('ecrs_per_kw'),
                pl.lit(0.0).alias('nonspin_per_kw'),
                pl.when(pl.col('capacity_mw') > 0)
                  .then(pl.col('total_revenue') / (pl.col('capacity_mw') * 1000.0) * pl.lit(_scale_ytd))
                  .otherwise(0.0)
                  .alias('total_per_kw'),
                pl.when((pl.col('capacity_mw') > 0) & (pl.col('tb2_sum').fill_null(0.0) > 0))
                  .then(((pl.col('total_revenue') / pl.col('capacity_mw')) / pl.col('tb2_sum')) * 100.0)
                  .otherwise(None)
                  .alias('pct_total_vs_tb2')
            ])

            df_ytd_chart = df_ytd_with_tb2.to_pandas()
            create_chart(df_ytd_chart, f"{year} YTD (thru {last_date:%b %d})", year, output_dir)


def main():
    logger.info("=" * 100)
    logger.info("BESS QUARTERLY, MONTHLY & YTD REVENUE CHARTS (2022-2025)")
    logger.info("=" * 100)

    # Optional CLI for year selection and base-dir override
    try:
        import argparse
        ap = argparse.ArgumentParser(add_help=False)
        ap.add_argument('--years', nargs='*', type=int, default=None)
        ap.add_argument('--base-dir', default=None)
        known, _ = ap.parse_known_args()
    except Exception:
        known = type('K', (), {'years': None, 'base_dir': None})()

    # Create output directory
    output_dir = Path("bess_revenue_charts_periods")
    output_dir.mkdir(exist_ok=True)

    # Load all revenue data (prefer TELEMETERED)
    all_revenue = []
    for year in range(2019, 2026):
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

    # Load TB2 daily data (try 2019-2025, fallback to 2019-2024)
    tb2_path_candidates = [
        'tb2_daily_2019_2025.parquet',
        'tb2_daily_2019_2024.parquet',
        'tb2_daily.parquet',
    ]
    tb2_daily = None
    for p in tb2_path_candidates:
        if Path(p).exists():
            tb2_daily = pl.read_parquet(p)
            logger.info(f"Loaded daily TB2 data from {p}")
            break
    if tb2_daily is None:
        logger.warning("No TB2 daily parquet found; skipping period charts")
        return

    # Load mapping
    df_mapping = pd.read_csv('bess_mapping/BESS_UNIFIED_MAPPING_V3_CLARIFIED.csv')
    df_mapping = df_mapping[['BESS_Gen_Resource', 'Settlement_Point', 'IQ_Capacity_MW']].copy()
    df_mapping = df_mapping.rename(columns={
        'BESS_Gen_Resource': 'gen_resource',
        'Settlement_Point': 'settlement_point',
        'IQ_Capacity_MW': 'capacity_mw_mapping'
    })

    # Create charts for each year (exact aggregation)
    years = known.years if known.years else [2025, 2024, 2023, 2022]
    base_dir = Path(known.base_dir) if known.base_dir else Path('/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data')
    for year in years:
        create_period_charts(year, df_revenue_all, tb2_daily, df_mapping, output_dir, base_dir)

    logger.info("\n" + "=" * 100)
    logger.info(f"✅ All quarterly and monthly charts generated in {output_dir}/")
    logger.info("=" * 100)


if __name__ == "__main__":
    main()
