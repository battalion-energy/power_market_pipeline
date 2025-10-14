#!/usr/bin/env python3
"""
Normalize BESS revenue by active trading days.

Takes existing revenue calculations and normalizes $/kW-year by actual trading activity:
- Excludes batteries with <60 active days (<2 months) from each year
- Normalizes $/kW-year = (revenue / capacity) * (365 / active_days) for partial-year batteries
- Adds operational_start_date to output

This allows fair comparison of batteries that started mid-year.
"""

import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def normalize_revenue_by_activity(year: int, min_days: int = 60):
    """
    Normalize revenue for a specific year by active trading days

    Args:
        year: Year to process
        min_days: Minimum active days to include (default 60 = ~2 months)
    """
    logger.info(f"Processing {year}...")

    # Load revenue data (try TELEMETERED first, then fallback to regular)
    revenue_file_telem = f"bess_revenue_{year}_TELEMETERED.csv"
    revenue_file = f"bess_revenue_{year}.csv"

    if Path(revenue_file_telem).exists():
        revenue_file = revenue_file_telem
        logger.info(f"  Using TELEMETERED file: {revenue_file}")
    elif not Path(revenue_file).exists():
        logger.warning(f"  No revenue file: {revenue_file}")
        return None

    df_revenue = pd.read_csv(revenue_file)
    logger.info(f"  Loaded {len(df_revenue)} revenue records")

    # Load activity data
    df_activity = pd.read_csv('bess_activity_by_year.csv')
    df_activity_year = df_activity[df_activity['year'] == year].copy()
    logger.info(f"  Loaded {len(df_activity_year)} activity records")

    # Merge on gen_resource (ResourceName in activity data)
    df_merged = df_revenue.merge(
        df_activity_year[['ResourceName', 'active_days', 'first_active_date', 'last_active_date']],
        left_on='gen_resource',
        right_on='ResourceName',
        how='left'
    )

    # Count batteries with missing activity data
    missing_activity = df_merged['active_days'].isna().sum()
    if missing_activity > 0:
        logger.warning(f"  ⚠️  {missing_activity} batteries missing activity data")

    # Filter out batteries with <min_days activity
    df_filtered = df_merged[df_merged['active_days'] >= min_days].copy()
    excluded = len(df_merged) - len(df_filtered)
    logger.info(f"  Excluded {excluded} batteries with <{min_days} days activity")

    # Calculate normalized revenue
    # Formula: (revenue / capacity_kW) * (365 / active_days)
    # This annualizes partial-year revenue to full-year equivalent
    # NOTE: Must divide by (capacity_mw * 1000) to get $/kW-year!
    df_filtered['normalized_revenue_per_kw_year'] = (
        (df_filtered['total_revenue'] / (df_filtered['capacity_mw'] * 1000)) *
        (365.0 / df_filtered['active_days'])
    )

    # Also calculate normalized components (gen + load AS products)
    for col in ['dam_discharge_revenue', 'dam_as_gen_revenue', 'dam_as_load_revenue',
                'rt_net_revenue', 'dam_as_gen_regup', 'dam_as_gen_regdown',
                'dam_as_gen_rrs', 'dam_as_gen_ecrs', 'dam_as_gen_nonspin',
                'dam_as_load_regup', 'dam_as_load_regdown', 'dam_as_load_rrs',
                'dam_as_load_ecrs', 'dam_as_load_nonspin']:
        if col in df_filtered.columns:
            df_filtered[f'{col}_per_kw_year_normalized'] = (
                (df_filtered[col] / (df_filtered['capacity_mw'] * 1000)) *
                (365.0 / df_filtered['active_days'])
            )

    # Add summary columns
    df_filtered['days_in_year'] = 365 if year != 2024 else 334  # 2024 through Nov 28
    df_filtered['fraction_of_year'] = df_filtered['active_days'] / df_filtered['days_in_year']
    df_filtered['is_full_year'] = df_filtered['active_days'] >= (df_filtered['days_in_year'] - 10)  # Allow 10-day gap

    # Reorder columns to put activity data first
    cols_order = [
        'bess_name', 'gen_resource', 'load_resource', 'resource_node', 'capacity_mw', 'year',
        'active_days', 'first_active_date', 'last_active_date', 'fraction_of_year', 'is_full_year',
        'total_revenue', 'revenue_per_mw_year', 'normalized_revenue_per_kw_year'
    ]
    # Add remaining columns
    remaining_cols = [c for c in df_filtered.columns if c not in cols_order]
    df_filtered = df_filtered[cols_order + remaining_cols]

    # Save normalized data
    output_file = f"bess_revenue_{year}_normalized.csv"
    df_filtered.to_csv(output_file, index=False)
    logger.info(f"  ✅ Saved: {output_file}")

    # Print statistics
    logger.info(f"  Statistics:")
    logger.info(f"    Full-year batteries: {df_filtered['is_full_year'].sum()}")
    logger.info(f"    Partial-year batteries: {(~df_filtered['is_full_year']).sum()}")
    logger.info(f"    Avg active days: {df_filtered['active_days'].mean():.1f}")
    logger.info(f"    Avg normalization factor: {(365.0 / df_filtered['active_days']).mean():.2f}x")

    return df_filtered


def main():
    logger.info("=" * 100)
    logger.info("NORMALIZING BESS REVENUE BY ACTIVE TRADING DAYS")
    logger.info("=" * 100)

    # Process all years
    all_normalized = []

    for year in range(2019, 2025):
        df_norm = normalize_revenue_by_activity(year, min_days=60)
        if df_norm is not None:
            all_normalized.append(df_norm)

    # Combine all years
    if all_normalized:
        df_all = pd.concat(all_normalized, ignore_index=True)
        df_all.to_csv('bess_revenue_all_years_normalized.csv', index=False)
        logger.info(f"\n✅ Saved combined: bess_revenue_all_years_normalized.csv")
        logger.info(f"   Total records: {len(df_all)}")

        # Overall statistics
        logger.info("\n" + "=" * 100)
        logger.info("OVERALL STATISTICS")
        logger.info("=" * 100)
        logger.info(f"Total battery-years: {len(df_all)}")
        logger.info(f"Full-year batteries: {df_all['is_full_year'].sum()} ({100*df_all['is_full_year'].mean():.1f}%)")
        logger.info(f"Partial-year batteries: {(~df_all['is_full_year']).sum()} ({100*(~df_all['is_full_year']).mean():.1f}%)")

        # Show impact of normalization
        avg_actual = df_all['revenue_per_mw_year'].mean()
        avg_normalized = df_all['normalized_revenue_per_kw_year'].mean()
        logger.info(f"\nAverage $/kW-year:")
        logger.info(f"  Actual (as-reported): ${avg_actual:,.2f}")
        logger.info(f"  Normalized (annualized): ${avg_normalized:,.2f}")
        logger.info(f"  Difference: ${avg_normalized - avg_actual:,.2f} ({100*(avg_normalized/avg_actual - 1):.1f}%)")

        # Show batteries with biggest normalization impact
        df_all['normalization_impact'] = df_all['normalized_revenue_per_kw_year'] - df_all['revenue_per_mw_year']
        df_top_impact = df_all.nlargest(10, 'normalization_impact')

        logger.info(f"\nTop 10 batteries with largest normalization impact:")
        for _, row in df_top_impact.iterrows():
            logger.info(f"  {row['bess_name']:30s} ({row['year']}): " +
                       f"{row['active_days']:3.0f} days, " +
                       f"${row['revenue_per_mw_year']:6,.0f} → ${row['normalized_revenue_per_kw_year']:6,.0f}/kW-year " +
                       f"(+${row['normalization_impact']:,.0f})")

    logger.info("\n✅ Normalization complete!")


if __name__ == "__main__":
    main()
