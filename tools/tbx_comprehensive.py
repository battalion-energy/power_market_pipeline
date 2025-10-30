#!/usr/bin/env python3
"""
Comprehensive TBX Calculator: TB1-TB12 for Day Ahead and Real-Time

Calculates theoretical battery arbitrage revenue for 1-12 hour duration batteries.
Outputs daily, monthly, quarterly, annual, and YTD aggregations.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv

warnings.filterwarnings('ignore')


def calc_tbx_daily(prices: np.ndarray, hours: int, efficiency: float = 0.90) -> float:
    """
    Calculate TBX revenue for a single day.

    Args:
        prices: Array of hourly prices (length 24 for DA, or 96 for 15-min RT)
        hours: Duration in hours (1-12)
        efficiency: Round-trip efficiency (default 0.9)

    Returns:
        Daily revenue in $/MW-day
    """
    if len(prices) < 24 or np.any(np.isnan(prices)):
        return 0.0

    # For 15-minute data, convert hours to number of intervals
    if len(prices) == 96:  # 15-minute intervals
        intervals = hours * 4
        prices_to_use = prices[:96]
    else:  # Hourly data
        intervals = hours
        prices_to_use = prices[:24]

    if len(prices_to_use) < 2 * intervals:
        return 0.0

    # Sort prices and find charge/discharge periods
    idx = np.argsort(prices_to_use)
    charge_idx = idx[:intervals]
    discharge_idx = idx[-intervals:]

    # Calculate costs and revenues
    # Using standard approach: efficiency loss on round-trip
    charge_cost = prices_to_use[charge_idx].sum()
    discharge_revenue = prices_to_use[discharge_idx].sum()

    # Apply efficiency: you only get efficiency * energy out
    net_revenue = discharge_revenue * efficiency - charge_cost

    return float(net_revenue)


def process_da_year(
    da_file: Path,
    year: int,
    efficiency: float,
    tb_hours: List[int]
) -> pd.DataFrame:
    """Process Day Ahead prices for one year."""
    print(f"  [DA] Loading {da_file.name}...")

    df = pd.read_parquet(da_file)

    # Normalize to have 'date' and 'hour' columns
    if 'DeliveryDate' in df.columns:
        df['date'] = pd.to_datetime(df['DeliveryDate']).dt.date

    if 'hour' in df.columns:
        if df['hour'].dtype == object:
            hr = pd.to_datetime(df['hour'].str.replace('24:00', '00:00'), format='%H:%M').dt.hour
            hr = hr.replace(0, 24)
        else:
            hr = df['hour'].astype(int)
    elif 'HourEnding' in df.columns:
        he = df['HourEnding']
        if he.dtype == object:
            hr = pd.to_datetime(he.str.replace('24:00', '00:00'), format='%H:%M').dt.hour
            hr = hr.replace(0, 24)
        else:
            hr = he.astype(int)
    else:
        raise ValueError("DA parquet must contain 'hour' or 'HourEnding'")

    df['hour'] = hr

    # Get settlement point column name
    sp_col = 'SettlementPoint' if 'SettlementPoint' in df.columns else 'settlement_point'
    price_col = 'SettlementPointPrice' if 'SettlementPointPrice' in df.columns else 'price'

    # Get settlement point types if available
    type_map = {}
    if 'SettlementPointType' in df.columns:
        type_df = df[[sp_col, 'SettlementPointType']].drop_duplicates()
        type_map = dict(zip(type_df[sp_col], type_df['SettlementPointType']))

    # Process each day
    daily_results = []

    for date, day_df in df.groupby('date'):
        # Pivot to get 24 hourly prices per settlement point
        pivot = day_df.pivot_table(
            index=sp_col,
            columns='hour',
            values=price_col,
            aggfunc='first'
        )

        # Ensure all 24 hours exist
        for h in range(1, 25):
            if h not in pivot.columns:
                pivot[h] = np.nan
        pivot = pivot[[h for h in range(1, 25)]]

        # Calculate all TBX values
        for settlement_point in pivot.index:
            prices_24 = pivot.loc[settlement_point].values

            if np.all(np.isnan(prices_24)):
                continue

            row = {
                'settlement_point': settlement_point,
                'settlement_point_type': type_map.get(settlement_point, 'UNKNOWN'),
                'delivery_date': date,
                'year': year,
            }

            # Calculate TB1-TB12
            for hours in tb_hours:
                revenue = calc_tbx_daily(prices_24, hours, efficiency)
                row[f'tb{hours}_da'] = revenue

            daily_results.append(row)

    print(f"  [DA] Processed {len(daily_results)} daily records")
    return pd.DataFrame(daily_results)


def process_rt_year(
    rt_file: Path,
    year: int,
    efficiency: float,
    tb_hours: List[int]
) -> pd.DataFrame:
    """Process Real-Time 15-minute prices for one year."""
    print(f"  [RT] Loading {rt_file.name}...")

    df = pd.read_parquet(rt_file)

    # Normalize columns
    if 'DeliveryDate' in df.columns:
        df['date'] = pd.to_datetime(df['DeliveryDate'], format='%m/%d/%Y', errors='coerce').dt.date

    # Determine intervals per hour
    if 'DeliveryInterval' in df.columns:
        intervals_per_hour = int(df['DeliveryInterval'].max())
    else:
        intervals_per_hour = 4  # Default to 15-minute

    if intervals_per_hour not in (4, 12):
        print(f"  [RT] Warning: Unexpected intervals_per_hour={intervals_per_hour}, defaulting to 4")
        intervals_per_hour = 4

    # Create interval index
    if 'DeliveryHour' in df.columns and 'DeliveryInterval' in df.columns:
        df['interval_index'] = (df['DeliveryHour'].astype(int) - 1) * intervals_per_hour + df['DeliveryInterval'].astype(int)
    else:
        print(f"  [RT] Missing DeliveryHour/DeliveryInterval columns, skipping RT")
        return pd.DataFrame()

    sp_col = 'SettlementPointName' if 'SettlementPointName' in df.columns else 'SettlementPoint'
    price_col = 'SettlementPointPrice' if 'SettlementPointPrice' in df.columns else 'price'

    # Get settlement point types
    type_map = {}
    if 'SettlementPointType' in df.columns:
        type_df = df[[sp_col, 'SettlementPointType']].drop_duplicates()
        type_map = dict(zip(type_df[sp_col], type_df['SettlementPointType']))

    # Process each day
    daily_results = []
    total_intervals = intervals_per_hour * 24

    for date, day_df in df.groupby('date'):
        # Pivot to get all intervals per settlement point
        pivot = day_df.pivot_table(
            index=sp_col,
            columns='interval_index',
            values=price_col,
            aggfunc='first'
        )

        # Ensure all intervals exist
        for i in range(1, total_intervals + 1):
            if i not in pivot.columns:
                pivot[i] = np.nan
        pivot = pivot[[i for i in range(1, total_intervals + 1)]]

        # Calculate all TBX values
        for settlement_point in pivot.index:
            prices_intervals = pivot.loc[settlement_point].values

            if np.all(np.isnan(prices_intervals)):
                continue

            row = {
                'settlement_point': settlement_point,
                'settlement_point_type': type_map.get(settlement_point, 'UNKNOWN'),
                'delivery_date': date,
                'year': year,
            }

            # Calculate TB1-TB12 for RT
            for hours in tb_hours:
                # For 15-minute data, need hours * 4 intervals
                intervals_needed = hours * intervals_per_hour

                if len(prices_intervals) < 2 * intervals_needed:
                    row[f'tb{hours}_rt'] = 0.0
                    continue

                # Filter out NaNs
                valid_prices = prices_intervals[~np.isnan(prices_intervals)]
                if len(valid_prices) < 2 * intervals_needed:
                    row[f'tb{hours}_rt'] = 0.0
                    continue

                # Sort and select
                idx = np.argsort(valid_prices)
                charge_idx = idx[:intervals_needed]
                discharge_idx = idx[-intervals_needed:]

                charge_cost = valid_prices[charge_idx].sum()
                discharge_revenue = valid_prices[discharge_idx].sum()

                # Apply efficiency and convert to hourly equivalent
                # Each interval is 1/intervals_per_hour hours
                hours_per_interval = 1.0 / intervals_per_hour
                net_revenue = (discharge_revenue * efficiency - charge_cost) * hours_per_interval

                row[f'tb{hours}_rt'] = net_revenue

            daily_results.append(row)

    print(f"  [RT] Processed {len(daily_results)} daily records")
    return pd.DataFrame(daily_results)


def create_aggregations(daily_df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Create monthly, quarterly, annual, and YTD aggregations."""
    if daily_df.empty:
        return daily_df

    # Ensure delivery_date is datetime and derive time fields from actual dates
    daily_df['delivery_date'] = pd.to_datetime(daily_df['delivery_date'])
    daily_df['year'] = daily_df['delivery_date'].dt.year  # FIX: Derive year from actual date
    daily_df['month'] = daily_df['delivery_date'].dt.month
    daily_df['quarter'] = daily_df['delivery_date'].dt.quarter

    # Get TB columns
    tb_cols_da = [col for col in daily_df.columns if col.startswith('tb') and col.endswith('_da')]
    tb_cols_rt = [col for col in daily_df.columns if col.startswith('tb') and col.endswith('_rt')]
    tb_cols = tb_cols_da + tb_cols_rt

    all_results = []

    # Daily records (mark as daily)
    daily_records = daily_df.copy()
    daily_records['period_type'] = 'daily'
    daily_records['period_start'] = daily_records['delivery_date']
    daily_records['period_end'] = daily_records['delivery_date']
    daily_records['days_in_period'] = 1
    all_results.append(daily_records)

    # Monthly aggregations
    monthly = daily_df.groupby(['settlement_point', 'settlement_point_type', 'month', 'year']).agg(
        {**{col: 'sum' for col in tb_cols}, 'delivery_date': ['min', 'max', 'count']}
    ).reset_index()
    monthly.columns = ['settlement_point', 'settlement_point_type', 'month', 'year'] + \
                      tb_cols + ['period_start', 'period_end', 'days_in_period']
    monthly['period_type'] = 'monthly'
    all_results.append(monthly)

    # Quarterly aggregations
    quarterly = daily_df.groupby(['settlement_point', 'settlement_point_type', 'quarter', 'year']).agg(
        {**{col: 'sum' for col in tb_cols}, 'delivery_date': ['min', 'max', 'count']}
    ).reset_index()
    quarterly.columns = ['settlement_point', 'settlement_point_type', 'quarter', 'year'] + \
                        tb_cols + ['period_start', 'period_end', 'days_in_period']
    quarterly['period_type'] = 'quarterly'
    all_results.append(quarterly)

    # Annual aggregation
    annual = daily_df.groupby(['settlement_point', 'settlement_point_type', 'year']).agg(
        {**{col: 'sum' for col in tb_cols}, 'delivery_date': ['min', 'max', 'count']}
    ).reset_index()
    annual.columns = ['settlement_point', 'settlement_point_type', 'year'] + \
                     tb_cols + ['period_start', 'period_end', 'days_in_period']
    annual['period_type'] = 'annual'
    all_results.append(annual)

    # YTD aggregation (for current year only)
    current_year = datetime.now().year
    if year == current_year:
        ytd = daily_df.groupby(['settlement_point', 'settlement_point_type', 'year']).agg(
            {**{col: 'sum' for col in tb_cols}, 'delivery_date': ['min', 'max', 'count']}
        ).reset_index()
        ytd.columns = ['settlement_point', 'settlement_point_type', 'year'] + \
                      tb_cols + ['period_start', 'period_end', 'days_in_period']
        ytd['period_type'] = 'ytd'
        all_results.append(ytd)

    # Combine all
    result = pd.concat(all_results, ignore_index=True)

    # Set representative delivery_date for aggregated records
    # Daily records already have delivery_date, but aggregations need it set
    if result['delivery_date'].isna().any():
        # For monthly: first day of month
        mask_monthly = (result['period_type'] == 'monthly') & result['delivery_date'].isna()
        if mask_monthly.any():
            result.loc[mask_monthly, 'delivery_date'] = pd.to_datetime(
                result.loc[mask_monthly, 'year'].astype(str) + '-' +
                result.loc[mask_monthly, 'month'].astype(int).astype(str).str.zfill(2) + '-01'
            )

        # For quarterly: first day of quarter
        mask_quarterly = (result['period_type'] == 'quarterly') & result['delivery_date'].isna()
        if mask_quarterly.any():
            quarter_month = ((result.loc[mask_quarterly, 'quarter'] - 1) * 3 + 1)
            result.loc[mask_quarterly, 'delivery_date'] = pd.to_datetime(
                result.loc[mask_quarterly, 'year'].astype(str) + '-' +
                quarter_month.astype(int).astype(str).str.zfill(2) + '-01'
            )

        # For annual/ytd: first day of year
        mask_annual = (result['period_type'].isin(['annual', 'ytd'])) & result['delivery_date'].isna()
        if mask_annual.any():
            result.loc[mask_annual, 'delivery_date'] = pd.to_datetime(
                result.loc[mask_annual, 'year'].astype(str) + '-01-01'
            )

    return result


def add_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Add UTC and local (Central) timestamp columns."""
    if df.empty:
        return df

    # Convert period_start and period_end to timestamps
    if 'period_start' in df.columns:
        # Assume dates are in Central Time
        df['period_start_local'] = pd.to_datetime(df['period_start']).dt.tz_localize('US/Central', ambiguous='NaT', nonexistent='NaT')
        df['period_start_utc'] = df['period_start_local'].dt.tz_convert('UTC')

        df['period_end_local'] = pd.to_datetime(df['period_end']).dt.tz_localize('US/Central', ambiguous='NaT', nonexistent='NaT')
        df['period_end_utc'] = df['period_end_local'].dt.tz_convert('UTC')

    return df


def main() -> int:
    load_dotenv()

    ap = argparse.ArgumentParser(description='Comprehensive TBX Calculator')
    ap.add_argument('--years', nargs='*', type=int, default=list(range(2010, 2026)))
    ap.add_argument('--tb-hours', nargs='*', type=int, default=list(range(1, 13)),
                    help='TB durations to calculate (default: 1-12)')
    ap.add_argument('--efficiency', type=float, default=0.90)
    ap.add_argument('--skip-da', action='store_true', help='Skip Day Ahead calculations')
    ap.add_argument('--skip-rt', action='store_true', help='Skip Real-Time calculations')

    default_data = os.getenv('ERCOT_DATA_DIR', '/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data')
    ap.add_argument('--da-dir', default=str(Path(default_data) / 'rollup_files/DA_prices'))
    ap.add_argument('--rt-dir', default=str(Path(default_data) / 'rollup_files/RT_prices'))
    ap.add_argument('--out-dir', default=str(Path(default_data) / 'tbx_comprehensive'))

    args = ap.parse_args()

    da_dir = Path(args.da_dir)
    rt_dir = Path(args.rt_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"TBX Comprehensive Calculator")
    print(f"{'='*80}")
    print(f"Years: {args.years}")
    print(f"TB Hours: {args.tb_hours}")
    print(f"Efficiency: {args.efficiency}")
    print(f"Output: {out_dir}")
    print(f"{'='*80}\n")

    for year in args.years:
        print(f"\n📅 Processing Year {year}")
        print(f"{'='*80}")

        da_file = da_dir / f"{year}.parquet"
        rt_file = rt_dir / f"{year}.parquet"

        # Process DA
        da_df = pd.DataFrame()
        if not args.skip_da and da_file.exists():
            da_df = process_da_year(da_file, year, args.efficiency, args.tb_hours)
        elif not args.skip_da:
            print(f"  [DA] File not found: {da_file}")

        # Process RT
        rt_df = pd.DataFrame()
        if not args.skip_rt and rt_file.exists():
            rt_df = process_rt_year(rt_file, year, args.efficiency, args.tb_hours)
        elif not args.skip_rt:
            print(f"  [RT] File not found: {rt_file}")

        # Merge DA and RT
        if not da_df.empty and not rt_df.empty:
            # Normalize settlement_point_type before merge (prefer DA type if available)
            # Create a type map from DA data
            da_type_map = da_df[['settlement_point', 'settlement_point_type']].drop_duplicates().set_index('settlement_point')['settlement_point_type'].to_dict()

            # Update RT types to match DA where they exist
            rt_df['settlement_point_type'] = rt_df.apply(
                lambda row: da_type_map.get(row['settlement_point'], row['settlement_point_type']),
                axis=1
            )

            # Merge on common columns with suffixes to handle duplicates
            daily_df = pd.merge(
                da_df,
                rt_df,
                on=['settlement_point', 'settlement_point_type', 'delivery_date', 'year'],
                how='outer',
                suffixes=('', '_rt_dup')
            )
            # Fill NaN values with 0
            tb_cols = [col for col in daily_df.columns if col.startswith('tb') and ('_da' in col or '_rt' in col) and '_dup' not in col]
            for col in tb_cols:
                daily_df[col] = daily_df[col].fillna(0.0)

            # Drop any duplicate columns
            daily_df = daily_df[[col for col in daily_df.columns if '_dup' not in col]]
        elif not da_df.empty:
            daily_df = da_df
            # Add missing RT columns with zeros
            for i in range(1, 13):
                daily_df[f'tb{i}_rt'] = 0.0
        elif not rt_df.empty:
            daily_df = rt_df
            # Add missing DA columns with zeros
            for i in range(1, 13):
                daily_df[f'tb{i}_da'] = 0.0
        else:
            print(f"  ⚠️  No data for year {year}")
            continue

        # Create aggregations
        print(f"  [AGG] Creating aggregations...")
        agg_df = create_aggregations(daily_df, year)

        # Add timestamps
        print(f"  [TIME] Adding timestamps...")
        agg_df = add_timestamps(agg_df)

        # Write to Parquet
        out_file = out_dir / f"tbx_all_{year}.parquet"
        agg_df.to_parquet(out_file, engine='pyarrow', compression='snappy', index=False)

        print(f"  ✅ Written {len(agg_df):,} records to {out_file.name}")
        print(f"     - {len(agg_df[agg_df['period_type']=='daily']):,} daily records")
        print(f"     - {len(agg_df[agg_df['period_type']=='monthly']):,} monthly records")
        print(f"     - {len(agg_df[agg_df['period_type']=='quarterly']):,} quarterly records")
        print(f"     - {len(agg_df[agg_df['period_type']=='annual']):,} annual records")
        if year == datetime.now().year:
            print(f"     - {len(agg_df[agg_df['period_type']=='ytd']):,} YTD records")

        # Print sample stats
        if not agg_df.empty:
            annual_data = agg_df[agg_df['period_type'] == 'annual']
            if not annual_data.empty and 'tb2_da' in annual_data.columns:
                top_node = annual_data.nlargest(1, 'tb2_da')
                if not top_node.empty:
                    node_name = top_node.iloc[0]['settlement_point']
                    tb2_value = top_node.iloc[0]['tb2_da']
                    print(f"     Top TB2 DA: {node_name} = ${tb2_value:,.2f}/MW-year")

    print(f"\n{'='*80}")
    print(f"✅ All years processed successfully!")
    print(f"{'='*80}\n")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
