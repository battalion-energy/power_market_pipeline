#!/usr/bin/env python3
"""
ERCOT-Wide BESS Market Rollup
Aggregates all BESS operations across ERCOT for each time period
Author: Generated for BESS market analysis
"""

import pandas as pd
import polars as pl
from pathlib import Path
from datetime import datetime, timedelta
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ERCOTWideBessAggregator:
    """
    Aggregates BESS operations across all ERCOT BESS systems.
    Creates market-wide rollups by time period.
    """

    def __init__(self, data_dir: str = '/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/rollup_files'):
        self.data_dir = Path(data_dir)
        self.output_dir = Path('/home/enrico/projects/power_market_pipeline/output')
        self.output_dir.mkdir(exist_ok=True)

        # Load BESS mapping to filter only BESS resources
        self.load_bess_mapping()

    def load_bess_mapping(self):
        """Load BESS resource mapping to identify all BESS Gen and Load resources"""
        mapping_file = Path('/home/enrico/projects/power_market_pipeline/bess_mapping/BESS_UNIFIED_MAPPING_V3_CLARIFIED.csv')

        if not mapping_file.exists():
            logger.warning("BESS mapping file not found. Will use ResourceType filter only.")
            self.bess_gen_resources = set()
            self.bess_load_resources = set()
            return

        mapping_df = pd.read_csv(mapping_file)

        # Extract all Gen resources (single resource per row)
        self.bess_gen_resources = set()
        if 'BESS_Gen_Resource' in mapping_df.columns:
            for gen_res in mapping_df['BESS_Gen_Resource'].dropna():
                if isinstance(gen_res, str) and gen_res.strip():
                    self.bess_gen_resources.add(gen_res.strip())

        # Extract all Load resources (single resource per row)
        self.bess_load_resources = set()
        if 'BESS_Load_Resource' in mapping_df.columns:
            for load_res in mapping_df['BESS_Load_Resource'].dropna():
                if isinstance(load_res, str) and load_res.strip():
                    self.bess_load_resources.add(load_res.strip())

        logger.info(f"Loaded {len(self.bess_gen_resources)} BESS Gen resources")
        logger.info(f"Loaded {len(self.bess_load_resources)} BESS Load resources")

    def aggregate_dam_hourly(self, year: int) -> pl.DataFrame:
        """
        Aggregate DAM (Day-Ahead Market) operations by hour across all BESS.
        Includes energy awards and ancillary services.
        """
        logger.info(f"  Aggregating DAM data for {year}...")

        # Load DAM Gen Resources (discharge + AS)
        dam_gen_file = self.data_dir / f'DAM_Gen_Resources/{year}.parquet'
        if not dam_gen_file.exists():
            logger.warning(f"DAM Gen file not found for {year}")
            return None

        logger.info(f"    Loading DAM Gen Resources...")
        dam_gen = pl.read_parquet(dam_gen_file)

        # Filter for BESS only (ResourceType = PWRSTR)
        dam_gen = dam_gen.filter(pl.col('ResourceType') == 'PWRSTR')

        # Further filter by known BESS resources if mapping available
        if self.bess_gen_resources:
            dam_gen = dam_gen.filter(pl.col('ResourceName').is_in(list(self.bess_gen_resources)))

        # Convert hour from hour-ending (1-24) to hour 0-23 for consistency
        # Both DAM Gen and Load use hour 1-24 format
        dam_gen = dam_gen.with_columns([
            (pl.col('hour') - 1).alias('hour_0based')
        ])

        # Load AS prices for revenue calculation
        as_prices_file = self.data_dir / f'flattened/AS_prices_{year}.parquet'
        if as_prices_file.exists():
            as_prices = pl.read_parquet(as_prices_file)

            # Add missing AS price columns with 0 values for backward compatibility
            # ECRS was introduced in later years
            expected_as_cols = ['REGUP', 'REGDN', 'RRS', 'ECRS', 'NSPIN']
            for col in expected_as_cols:
                if col not in as_prices.columns:
                    as_prices = as_prices.with_columns([
                        pl.lit(0.0).alias(col)
                    ])

            # Join AS prices to DAM gen data
            # Need to align on DeliveryDate and hour
            dam_gen = dam_gen.join(
                as_prices,
                left_on='DeliveryDate',
                right_on='DeliveryDate',
                how='left'
            )
        else:
            logger.warning(f"AS prices file not found for {year}")

        # Ensure DeliveryDate is Date type for consistent joining
        dam_gen = dam_gen.with_columns([
            pl.col('DeliveryDate').cast(pl.Date).alias('delivery_date')
        ])

        # Aggregate DAM Gen by hour
        dam_gen_agg = dam_gen.group_by(['delivery_date', 'hour_0based']).agg([
            # Count of BESS units participating
            pl.col('ResourceName').n_unique().alias('bess_count_gen'),

            # DAM Energy (Discharge)
            pl.col('AwardedQuantity').sum().alias('dam_discharge_mwh'),
            (pl.col('AwardedQuantity') * pl.col('EnergySettlementPointPrice')).sum().alias('dam_discharge_revenue'),

            # Ancillary Services - Gen Resource
            pl.col('RegUpAwarded').sum().alias('dam_gen_regup_mw'),
            pl.col('RegDownAwarded').sum().alias('dam_gen_regdown_mw'),
            (pl.col('RRSPFRAwarded').fill_null(0) +
             pl.col('RRSFFRAwarded').fill_null(0) +
             pl.col('RRSUFRAwarded').fill_null(0)).sum().alias('dam_gen_rrs_mw'),
            (pl.col('ECRSAwarded').fill_null(0) +
             pl.col('ECRSSDAwarded').fill_null(0)).sum().alias('dam_gen_ecrs_mw'),
            pl.col('NonSpinAwarded').sum().alias('dam_gen_nonspin_mw'),

            # AS Revenue (prices added with 0 default if missing in earlier years)
            (pl.col('RegUpAwarded') * pl.col('REGUP').fill_null(0)).sum().alias('dam_gen_regup_revenue'),
            (pl.col('RegDownAwarded') * pl.col('REGDN').fill_null(0)).sum().alias('dam_gen_regdown_revenue'),
            ((pl.col('RRSPFRAwarded').fill_null(0) +
              pl.col('RRSFFRAwarded').fill_null(0) +
              pl.col('RRSUFRAwarded').fill_null(0)) * pl.col('RRS').fill_null(0)).sum().alias('dam_gen_rrs_revenue'),
            ((pl.col('ECRSAwarded').fill_null(0) +
              pl.col('ECRSSDAwarded').fill_null(0)) * pl.col('ECRS').fill_null(0)).sum().alias('dam_gen_ecrs_revenue'),
            (pl.col('NonSpinAwarded') * pl.col('NSPIN').fill_null(0)).sum().alias('dam_gen_nonspin_revenue'),
        ])

        # Load DAM Load Resources (AS only, no energy)
        dam_load_file = self.data_dir / f'DAM_Load_Resources/{year}.parquet'
        if dam_load_file.exists():
            logger.info(f"    Loading DAM Load Resources...")
            dam_load = pl.read_parquet(dam_load_file)

            # Filter for BESS Load resources
            if self.bess_load_resources:
                # Handle column name variations
                load_col = 'Load Resource Name' if 'Load Resource Name' in dam_load.columns else 'LoadResourceName'
                dam_load = dam_load.filter(pl.col(load_col).is_in(list(self.bess_load_resources)))

            # Standardize date column - convert to Date type for joining
            if 'DeliveryDate' in dam_load.columns:
                dam_load = dam_load.with_columns([
                    pl.col('DeliveryDate').cast(pl.Date).alias('delivery_date')
                ])
            elif 'Delivery Date' in dam_load.columns:
                dam_load = dam_load.with_columns([
                    pl.col('Delivery Date').str.to_date().alias('delivery_date')
                ])

            # Hour column handling: hour is already present as Int32 (hour ending 1-24)
            # Convert to hour 0-23 to align with DAM Gen
            if 'hour' in dam_load.columns:
                dam_load = dam_load.with_columns([
                    (pl.col('hour') - 1).alias('hour_0based')
                ])
            else:
                # Fallback if hour column missing
                logger.warning("Hour column not found in DAM Load, using Hour Ending")
                hour_col = 'Hour Ending' if 'Hour Ending' in dam_load.columns else 'HourEnding'
                dam_load = dam_load.with_columns([
                    (pl.col(hour_col) - 1).alias('hour_0based')
                ])

            # Aggregate DAM Load by hour
            load_col_name = 'Load Resource Name' if 'Load Resource Name' in dam_load.columns else 'LoadResourceName'

            # Build aggregation expressions dynamically based on available columns
            agg_exprs = [
                pl.col(load_col_name).n_unique().alias('bess_count_load'),
            ]

            # RegUp/RegDown (usually available)
            regup_col = 'RegUp Awarded' if 'RegUp Awarded' in dam_load.columns else 'RegUpAwarded'
            regdown_col = 'RegDown Awarded' if 'RegDown Awarded' in dam_load.columns else 'RegDownAwarded'
            regup_mcpc = 'RegUp MCPC' if 'RegUp MCPC' in dam_load.columns else 'RegUpMCPC'
            regdown_mcpc = 'RegDown MCPC' if 'RegDown MCPC' in dam_load.columns else 'RegDownMCPC'

            if regup_col in dam_load.columns:
                agg_exprs.append(pl.col(regup_col).sum().alias('dam_load_regup_mw'))
                if regup_mcpc in dam_load.columns:
                    agg_exprs.append((pl.col(regup_col) * pl.col(regup_mcpc).fill_null(0)).sum().alias('dam_load_regup_revenue'))
                else:
                    agg_exprs.append(pl.lit(0.0).alias('dam_load_regup_revenue'))
            else:
                agg_exprs.extend([pl.lit(0.0).alias('dam_load_regup_mw'), pl.lit(0.0).alias('dam_load_regup_revenue')])

            if regdown_col in dam_load.columns:
                agg_exprs.append(pl.col(regdown_col).sum().alias('dam_load_regdown_mw'))
                if regdown_mcpc in dam_load.columns:
                    agg_exprs.append((pl.col(regdown_col) * pl.col(regdown_mcpc).fill_null(0)).sum().alias('dam_load_regdown_revenue'))
                else:
                    agg_exprs.append(pl.lit(0.0).alias('dam_load_regdown_revenue'))
            else:
                agg_exprs.extend([pl.lit(0.0).alias('dam_load_regdown_mw'), pl.lit(0.0).alias('dam_load_regdown_revenue')])

            # RRS (check for aggregate or separate columns)
            rrs_cols = ['RRSPFR Awarded', 'RRSFFR Awarded', 'RRSUFR Awarded']
            has_rrs_breakdown = all(col in dam_load.columns for col in rrs_cols)
            rrs_mcpc = 'RRS MCPC' if 'RRS MCPC' in dam_load.columns else 'RRSMCPC'

            if has_rrs_breakdown:
                rrs_sum = (pl.col('RRSPFR Awarded').fill_null(0) +
                           pl.col('RRSFFR Awarded').fill_null(0) +
                           pl.col('RRSUFR Awarded').fill_null(0))
                agg_exprs.append(rrs_sum.sum().alias('dam_load_rrs_mw'))
                if rrs_mcpc in dam_load.columns:
                    agg_exprs.append((rrs_sum * pl.col(rrs_mcpc).fill_null(0)).sum().alias('dam_load_rrs_revenue'))
                else:
                    agg_exprs.append(pl.lit(0.0).alias('dam_load_rrs_revenue'))
            elif 'RRS Awarded' in dam_load.columns:
                agg_exprs.append(pl.col('RRS Awarded').sum().alias('dam_load_rrs_mw'))
                if rrs_mcpc in dam_load.columns:
                    agg_exprs.append((pl.col('RRS Awarded') * pl.col(rrs_mcpc).fill_null(0)).sum().alias('dam_load_rrs_revenue'))
                else:
                    agg_exprs.append(pl.lit(0.0).alias('dam_load_rrs_revenue'))
            else:
                agg_exprs.extend([pl.lit(0.0).alias('dam_load_rrs_mw'), pl.lit(0.0).alias('dam_load_rrs_revenue')])

            # ECRS (may not exist in earlier years)
            ecrs_cols = ['ECRSSD Awarded', 'ECRSMD Awarded']
            has_ecrs = any(col in dam_load.columns for col in ecrs_cols)
            ecrs_mcpc = 'ECRS MCPC' if 'ECRS MCPC' in dam_load.columns else 'ECRSMCPC'

            if has_ecrs:
                ecrssd = pl.col('ECRSSD Awarded').fill_null(0) if 'ECRSSD Awarded' in dam_load.columns else pl.lit(0.0)
                ecrsmd = pl.col('ECRSMD Awarded').fill_null(0) if 'ECRSMD Awarded' in dam_load.columns else pl.lit(0.0)
                ecrs_sum = ecrssd + ecrsmd
                agg_exprs.append(ecrs_sum.sum().alias('dam_load_ecrs_mw'))
                if ecrs_mcpc in dam_load.columns:
                    agg_exprs.append((ecrs_sum * pl.col(ecrs_mcpc).fill_null(0)).sum().alias('dam_load_ecrs_revenue'))
                else:
                    agg_exprs.append(pl.lit(0.0).alias('dam_load_ecrs_revenue'))
            else:
                agg_exprs.extend([pl.lit(0.0).alias('dam_load_ecrs_mw'), pl.lit(0.0).alias('dam_load_ecrs_revenue')])

            # NonSpin
            nonspin_col = 'NonSpin Awarded' if 'NonSpin Awarded' in dam_load.columns else 'NonSpinAwarded'
            nonspin_mcpc = 'NonSpin MCPC' if 'NonSpin MCPC' in dam_load.columns else 'NonSpinMCPC'

            if nonspin_col in dam_load.columns:
                agg_exprs.append(pl.col(nonspin_col).sum().alias('dam_load_nonspin_mw'))
                if nonspin_mcpc in dam_load.columns:
                    agg_exprs.append((pl.col(nonspin_col) * pl.col(nonspin_mcpc).fill_null(0)).sum().alias('dam_load_nonspin_revenue'))
                else:
                    agg_exprs.append(pl.lit(0.0).alias('dam_load_nonspin_revenue'))
            else:
                agg_exprs.extend([pl.lit(0.0).alias('dam_load_nonspin_mw'), pl.lit(0.0).alias('dam_load_nonspin_revenue')])

            dam_load_agg = dam_load.group_by(['delivery_date', 'hour_0based']).agg(agg_exprs)

            # Join with Gen aggregation
            dam_agg = dam_gen_agg.join(
                dam_load_agg,
                on=['delivery_date', 'hour_0based'],
                how='outer'
            )
        else:
            logger.warning(f"DAM Load file not found for {year}")
            dam_agg = dam_gen_agg

        # Add timestamp column (hour in Central Time)
        dam_agg = dam_agg.with_columns([
            (pl.col('delivery_date').cast(pl.Datetime) +
             pl.duration(hours=pl.col('hour_0based'))).alias('timestamp_ct')
        ])

        logger.info(f"    DAM aggregation complete: {len(dam_agg)} hours")
        return dam_agg

    def aggregate_rt_15min(self, year: int) -> pl.DataFrame:
        """
        Aggregate Real-Time (RT) operations by 15-minute interval.
        Aggregates 3 x 5-minute SCED intervals into 15-minute periods.
        """
        logger.info(f"  Aggregating RT data for {year}...")

        # Load SCED Gen Resources (discharge)
        sced_gen_file = self.data_dir / f'SCED_Gen_Resources/{year}.parquet'
        if not sced_gen_file.exists():
            logger.warning(f"SCED Gen file not found for {year}")
            return None

        logger.info(f"    Loading SCED Gen Resources...")
        sced_gen = pl.scan_parquet(sced_gen_file)

        # Filter for BESS only
        if self.bess_gen_resources:
            sced_gen = sced_gen.filter(pl.col('ResourceName').is_in(list(self.bess_gen_resources)))

        # Load SCED Load Resources (charging)
        sced_load_file = self.data_dir / f'SCED_Load_Resources/{year}.parquet'
        if not sced_load_file.exists():
            logger.warning(f"SCED Load file not found for {year}")
            sced_load = None
        else:
            logger.info(f"    Loading SCED Load Resources...")
            sced_load = pl.scan_parquet(sced_load_file)

            # Filter for BESS Load resources
            if self.bess_load_resources:
                sced_load = sced_load.filter(pl.col('ResourceName').is_in(list(self.bess_load_resources)))

        # Parse timestamp string and create 15-minute interval
        # SCEDTimeStamp format: "MM/DD/YYYY HH:MM:SS"
        sced_gen = sced_gen.with_columns([
            pl.col('SCEDTimeStamp').str.to_datetime('%m/%d/%Y %H:%M:%S').alias('sced_timestamp_parsed')
        ]).with_columns([
            pl.col('sced_timestamp_parsed').dt.truncate('15m').alias('timestamp_15min')
        ])

        # Aggregate Gen (discharge) by 15-minute interval
        sced_gen_agg = sced_gen.group_by('timestamp_15min').agg([
            pl.col('ResourceName').n_unique().alias('bess_count_gen_rt'),
            pl.col('BasePoint').sum().alias('rt_discharge_mw_total'),
            (pl.col('BasePoint') * 5/60).sum().alias('rt_discharge_mwh'),  # Convert MW to MWh
            pl.col('BasePoint').filter(pl.col('BasePoint') > 0).count().alias('rt_dispatch_intervals_gen'),
        ]).collect()

        if sced_load is not None:
            # Parse timestamp for Load resources
            sced_load = sced_load.with_columns([
                pl.col('SCEDTimeStamp').str.to_datetime('%m/%d/%Y %H:%M:%S').alias('sced_timestamp_parsed')
            ]).with_columns([
                pl.col('sced_timestamp_parsed').dt.truncate('15m').alias('timestamp_15min')
            ])

            # Aggregate Load (charging) by 15-minute interval
            sced_load_agg = sced_load.group_by('timestamp_15min').agg([
                pl.col('ResourceName').n_unique().alias('bess_count_load_rt'),
                pl.col('BasePoint').sum().alias('rt_charge_mw_total'),
                (pl.col('BasePoint') * 5/60).sum().alias('rt_charge_mwh'),  # Convert MW to MWh
                pl.col('BasePoint').filter(pl.col('BasePoint') > 0).count().alias('rt_dispatch_intervals_load'),
            ]).collect()

            # Join Gen and Load aggregations
            rt_agg = sced_gen_agg.join(
                sced_load_agg,
                on='timestamp_15min',
                how='outer'
            )
        else:
            rt_agg = sced_gen_agg

        # Load RT prices for revenue calculation
        rt_prices_file = self.data_dir / f'RT_prices/{year}.parquet'
        if rt_prices_file.exists():
            logger.info(f"    Loading RT prices...")
            rt_prices = pl.scan_parquet(rt_prices_file)

            # Filter for relevant settlement points (BESS resource nodes)
            # For now, we'll use hub average as proxy
            rt_hub_prices = rt_prices.filter(
                pl.col('SettlementPointName') == 'HB_BUSAVG'
            ).select([
                'DeliveryDate',
                'DeliveryHour',
                'DeliveryInterval',
                pl.col('SettlementPointPrice').alias('rt_price_hub')
            ])

            # Create timestamp for joining
            # DeliveryHour is hour-ending (1-24), DeliveryInterval is 1-12 (5-min intervals)
            # Parse date string first: "MM/DD/YYYY" format
            rt_hub_prices = rt_hub_prices.with_columns([
                pl.col('DeliveryDate').str.to_date('%m/%d/%Y').alias('delivery_date_parsed')
            ]).with_columns([
                (pl.col('delivery_date_parsed').cast(pl.Datetime) +
                 pl.duration(hours=pl.col('DeliveryHour') - 1) +
                 pl.duration(minutes=(pl.col('DeliveryInterval') - 1) * 5)).alias('timestamp_5min')
            ])

            # Aggregate to 15-minute intervals (average of 3x5-min prices)
            rt_hub_prices = rt_hub_prices.with_columns([
                (pl.col('timestamp_5min').dt.truncate('15m')).alias('timestamp_15min')
            ]).group_by('timestamp_15min').agg([
                pl.col('rt_price_hub').mean().alias('rt_price_hub_avg')
            ]).collect()

            # Join prices to RT aggregation
            rt_agg = rt_agg.join(
                rt_hub_prices,
                on='timestamp_15min',
                how='left'
            )

            # Calculate revenue
            rt_agg = rt_agg.with_columns([
                (pl.col('rt_discharge_mwh') * pl.col('rt_price_hub_avg')).alias('rt_discharge_revenue'),
                (pl.col('rt_charge_mwh') * pl.col('rt_price_hub_avg')).alias('rt_charge_cost'),
                ((pl.col('rt_discharge_mwh') - pl.col('rt_charge_mwh')) *
                 pl.col('rt_price_hub_avg')).alias('rt_net_revenue'),
            ])
        else:
            logger.warning(f"RT prices file not found for {year}")

        logger.info(f"    RT aggregation complete: {len(rt_agg)} 15-min intervals")
        return rt_agg

    def create_unified_timeseries(self, dam_hourly: pl.DataFrame, rt_15min: pl.DataFrame, year: int) -> pl.DataFrame:
        """
        Create unified time series with both hourly (DAM) and 15-minute (RT) data.
        Returns separate files or combined with appropriate granularity markers.
        """
        logger.info(f"  Creating unified timeseries for {year}...")

        # Add granularity marker
        if dam_hourly is not None:
            dam_hourly = dam_hourly.with_columns([
                pl.lit('hourly').alias('granularity'),
                pl.lit(year).alias('year')
            ])

        if rt_15min is not None:
            rt_15min = rt_15min.with_columns([
                pl.lit('15min').alias('granularity'),
                pl.lit(year).alias('year')
            ])

        return dam_hourly, rt_15min

    def aggregate_year(self, year: int):
        """Aggregate all BESS operations for a given year"""
        logger.info(f"Processing year {year}...")

        # Aggregate DAM (hourly)
        dam_hourly = self.aggregate_dam_hourly(year)

        # Aggregate RT (15-minute)
        rt_15min = self.aggregate_rt_15min(year)

        # Create unified timeseries
        dam_final, rt_final = self.create_unified_timeseries(dam_hourly, rt_15min, year)

        # Save to parquet
        output_dam = self.output_dir / f'ercot_wide_bess_dam_hourly_{year}.parquet'
        output_rt = self.output_dir / f'ercot_wide_bess_rt_15min_{year}.parquet'

        if dam_final is not None:
            dam_final.write_parquet(output_dam)
            logger.info(f"  Saved DAM hourly data to: {output_dam}")
            logger.info(f"    Rows: {len(dam_final)}, Columns: {len(dam_final.columns)}")

        if rt_final is not None:
            rt_final.write_parquet(output_rt)
            logger.info(f"  Saved RT 15-min data to: {output_rt}")
            logger.info(f"    Rows: {len(rt_final)}, Columns: {len(rt_final.columns)}")

        return dam_final, rt_final

    def run_full_aggregation(self, years: list = None):
        """Run aggregation for all specified years"""
        if years is None:
            years = [2022, 2023, 2024]

        logger.info("="*80)
        logger.info("ERCOT-Wide BESS Market Rollup")
        logger.info("Aggregating all BESS operations by time period")
        logger.info("="*80)

        all_dam = []
        all_rt = []

        for year in years:
            dam, rt = self.aggregate_year(year)
            if dam is not None:
                all_dam.append(dam)
            if rt is not None:
                all_rt.append(rt)

        # Combine all years into single files
        if all_dam:
            logger.info("\nCombining all DAM data...")
            combined_dam = pl.concat(all_dam)
            output_dam_all = self.output_dir / 'ercot_wide_bess_dam_hourly_all_years.parquet'
            combined_dam.write_parquet(output_dam_all)
            logger.info(f"Saved combined DAM data: {output_dam_all}")
            logger.info(f"  Total rows: {len(combined_dam)}, Years: {years}")

            # Print summary statistics
            self.print_summary(combined_dam, "DAM Hourly")

        if all_rt:
            logger.info("\nCombining all RT data...")
            combined_rt = pl.concat(all_rt)
            output_rt_all = self.output_dir / 'ercot_wide_bess_rt_15min_all_years.parquet'
            combined_rt.write_parquet(output_rt_all)
            logger.info(f"Saved combined RT data: {output_rt_all}")
            logger.info(f"  Total rows: {len(combined_rt)}, Years: {years}")

            # Print summary statistics
            self.print_summary(combined_rt, "RT 15-Minute")

        logger.info("\n" + "="*80)
        logger.info("AGGREGATION COMPLETE")
        logger.info("="*80)

    def print_summary(self, df: pl.DataFrame, data_type: str):
        """Print summary statistics for the aggregated data"""
        logger.info(f"\n{data_type} Summary Statistics:")
        logger.info("-" * 60)

        if data_type == "DAM Hourly":
            stats = {
                'Total Hours': len(df),
                'Date Range': f"{df['delivery_date'].min()} to {df['delivery_date'].max()}",
                'Total DAM Discharge (MWh)': f"{df['dam_discharge_mwh'].sum():,.0f}",
                'Total DAM Discharge Revenue': f"${df['dam_discharge_revenue'].sum():,.2f}",
                'Total DAM AS Gen Revenue': f"${(df['dam_gen_regup_revenue'].sum() + df['dam_gen_regdown_revenue'].sum() + df['dam_gen_rrs_revenue'].sum() + df['dam_gen_ecrs_revenue'].sum() + df['dam_gen_nonspin_revenue'].sum()):,.2f}",
                'Total DAM AS Load Revenue': f"${(df['dam_load_regup_revenue'].sum() + df['dam_load_regdown_revenue'].sum() + df['dam_load_rrs_revenue'].sum() + df['dam_load_ecrs_revenue'].sum() + df['dam_load_nonspin_revenue'].sum()):,.2f}",
                'Avg BESS Count (Gen)': f"{df['bess_count_gen'].mean():.1f}",
            }
        else:  # RT 15-Minute
            stats = {
                'Total 15-min Intervals': len(df),
                'Date Range': f"{df['timestamp_15min'].min()} to {df['timestamp_15min'].max()}",
                'Total RT Discharge (MWh)': f"{df['rt_discharge_mwh'].sum():,.0f}",
                'Total RT Charge (MWh)': f"{df['rt_charge_mwh'].sum():,.0f}",
                'Total RT Net Revenue': f"${df['rt_net_revenue'].sum():,.2f}",
                'RT Efficiency': f"{(df['rt_discharge_mwh'].sum() / df['rt_charge_mwh'].sum() * 100):.2f}%",
                'Avg BESS Count (Gen)': f"{df['bess_count_gen_rt'].mean():.1f}",
            }

        for key, value in stats.items():
            logger.info(f"  {key}: {value}")


def main():
    """Main execution function"""
    import argparse

    parser = argparse.ArgumentParser(description='Aggregate ERCOT-wide BESS operations')
    parser.add_argument('--years', nargs='+', type=int, default=[2022, 2023, 2024],
                        help='Years to process (default: 2022 2023 2024)')
    parser.add_argument('--data-dir', type=str,
                        default='/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/rollup_files',
                        help='Path to ERCOT data directory')

    args = parser.parse_args()

    # Create aggregator and run
    aggregator = ERCOTWideBessAggregator(data_dir=args.data_dir)
    aggregator.run_full_aggregation(years=args.years)


if __name__ == '__main__':
    main()
