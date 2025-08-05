#!/usr/bin/env python3
"""
ERCOT Battery Analysis - TB2 and TB4 arbitrage analysis for ERCOT load zones.
Analyzes day-ahead and real-time market opportunities for battery energy storage.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pyarrow.parquet as pq

from battery_optimizer import BatteryConfig, BatteryOptimizer
from report_generator import ReportGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ERCOTBatteryAnalyzer:
    """Main class for ERCOT battery analysis."""
    
    # Target load zones for analysis
    LOAD_ZONES = {
        'HOUSTON': 'LZ_HOUSTON',
        'NORTH': 'LZ_NORTH', 
        'WEST': 'LZ_WEST',
        'AUSTIN': 'LZ_AEN',  # Austin Energy
        'SAN_ANTONIO': 'LZ_CPS'  # CPS Energy
    }
    
    def __init__(self, data_path: Path = Path("rt_rust_processor/annual_output")):
        self.data_path = data_path
        self.rt_data = None
        self.da_data = None
        
    def load_data(self, year: int = 2023) -> None:
        """Load ERCOT price data for the specified year."""
        logger.info(f"Loading ERCOT data for year {year}")
        
        # Load real-time 5-minute data
        rt_file = self.data_path / f"LMPs_by_Resource_Nodes__Load_Zones_and_Trading_Hubs/LMPs_by_Resource_Nodes__Load_Zones_and_Trading_Hubs_{year}.parquet"
        if rt_file.exists():
            logger.info(f"Loading RT data from {rt_file}")
            self.rt_data = pd.read_parquet(rt_file)
            # Convert timestamp to datetime
            self.rt_data['timestamp'] = pd.to_datetime(self.rt_data['SCEDTimestamp'])
            self.rt_data = self.rt_data.sort_values('timestamp')
            logger.info(f"Loaded {len(self.rt_data):,} RT records")
        else:
            raise FileNotFoundError(f"RT data file not found: {rt_file}")
            
        # Load day-ahead hourly data
        da_file = self.data_path / f"Settlement_Point_Prices_at_Resource_Nodes__Hubs_and_Load_Zones/Settlement_Point_Prices_at_Resource_Nodes__Hubs_and_Load_Zones_{year}.parquet"
        if da_file.exists():
            logger.info(f"Loading DA data from {da_file}")
            self.da_data = pd.read_parquet(da_file)
            # Create proper datetime - handle hour 24 (midnight of next day)
            def create_timestamp(row):
                hour = row['DeliveryHour']
                date_str = row['DeliveryDate']
                if hour == 24:
                    # Hour 24 is midnight of the next day
                    date = pd.to_datetime(date_str) + timedelta(days=1)
                    return date
                else:
                    return pd.to_datetime(f"{date_str} {hour:02d}:00:00")
            
            self.da_data['timestamp'] = self.da_data.apply(create_timestamp, axis=1)
            logger.info(f"Loaded {len(self.da_data):,} DA records")
        else:
            raise FileNotFoundError(f"DA data file not found: {da_file}")
            
    def get_zone_prices(self, zone_name: str, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Get price data for a specific zone and date range.
        
        Returns dict with 'da' and 'rt' DataFrames.
        """
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        # Get DA prices for the zone
        da_mask = (
            (self.da_data['SettlementPointName'] == zone_name) &
            (self.da_data['timestamp'] >= start) &
            (self.da_data['timestamp'] <= end)
        )
        da_prices = self.da_data[da_mask].copy()
        da_prices = da_prices.set_index('timestamp')['SettlementPointPrice']
        
        # Get RT prices - need to map zone names
        # RT data uses different naming convention, need to find matching settlement point
        rt_zone_map = {
            'LZ_HOUSTON': 'LZ_HOUSTON',
            'LZ_NORTH': 'LZ_NORTH',
            'LZ_WEST': 'LZ_WEST',
            'LZ_AEN': 'LZ_AEN',
            'LZ_CPS': 'LZ_CPS'
        }
        
        rt_zone = rt_zone_map.get(zone_name, zone_name)
        rt_mask = (
            (self.rt_data['SettlementPoint'] == rt_zone) &
            (self.rt_data['timestamp'] >= start) &
            (self.rt_data['timestamp'] <= end)
        )
        rt_prices = self.rt_data[rt_mask].copy()
        rt_prices = rt_prices.set_index('timestamp')['LMP']
        
        return {
            'da': da_prices,
            'rt': rt_prices
        }
    
    def analyze_battery(
        self,
        zone_key: str,
        battery_config: BatteryConfig,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Analyze battery performance for a specific zone.
        
        Returns DataFrame with daily results.
        """
        zone_name = self.LOAD_ZONES[zone_key]
        logger.info(f"Analyzing {zone_key} ({zone_name}) with {battery_config.duration_hours}h battery")
        
        # Get price data
        prices = self.get_zone_prices(zone_name, start_date, end_date)
        
        if len(prices['da']) == 0:
            logger.warning(f"No DA data found for {zone_name}")
            return pd.DataFrame()
            
        if len(prices['rt']) == 0:
            logger.warning(f"No RT data found for {zone_name}")
            return pd.DataFrame()
            
        # Initialize optimizer
        optimizer = BatteryOptimizer(battery_config)
        
        # Process each day
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        current_date = start
        
        daily_results = []
        
        total_days = (end - start).days + 1
        day_count = 0
        
        while current_date <= end:
            try:
                # Get day's data
                day_start = current_date
                day_end = current_date + timedelta(hours=23, minutes=59)
                
                # Extract DA prices for the day (24 hours)
                da_day = prices['da'][
                    (prices['da'].index >= day_start) & 
                    (prices['da'].index <= day_end)
                ]
                
                # Extract RT prices for the day (288 5-minute intervals)
                rt_day = prices['rt'][
                    (prices['rt'].index >= day_start) & 
                    (prices['rt'].index <= day_end)
                ]
                
                # Skip if incomplete data
                if len(da_day) != 24 or len(rt_day) < 280:  # Allow some missing RT intervals
                    logger.debug(f"Incomplete data for {current_date.date()}: DA={len(da_day)}, RT={len(rt_day)}")
                    current_date += timedelta(days=1)
                    day_count += 1
                    if day_count % 10 == 0:
                        logger.info(f"Processing {zone_key}: {day_count}/{total_days} days")
                    continue
                
                # Resample RT to ensure exactly 288 intervals
                rt_resampled = rt_day.resample('5min').mean()
                rt_resampled = rt_resampled.fillna(method='ffill').fillna(method='bfill')
                
                if len(rt_resampled) > 288:
                    rt_resampled = rt_resampled.iloc[:288]
                elif len(rt_resampled) < 288:
                    # Pad with last value
                    last_val = rt_resampled.iloc[-1] if len(rt_resampled) > 0 else 0
                    padding = pd.Series(
                        [last_val] * (288 - len(rt_resampled)),
                        index=pd.date_range(
                            rt_resampled.index[-1] + timedelta(minutes=5),
                            periods=288 - len(rt_resampled),
                            freq='5min'
                        )
                    )
                    rt_resampled = pd.concat([rt_resampled, padding])
                
                # Optimize battery operation
                results = optimizer.optimize_combined_markets(
                    da_day,
                    pd.DataFrame({'price': rt_resampled.values}),
                    current_date
                )
                
                # Store daily summary
                summary = results['summary'].iloc[0]
                daily_results.append({
                    'date': current_date,
                    'zone': zone_key,
                    'da_revenue': summary['da_revenue'],
                    'rt_revenue': summary['rt_revenue'],
                    'total_revenue': summary['total_revenue'],
                    'cycles': summary['cycles'],
                    'avg_da_price': da_day.mean(),
                    'avg_rt_price': rt_resampled.mean(),
                    'da_price_spread': da_day.max() - da_day.min(),
                    'rt_price_spread': rt_resampled.max() - rt_resampled.min()
                })
                
            except Exception as e:
                logger.error(f"Error processing {current_date.date()}: {str(e)}")
                
            current_date += timedelta(days=1)
            day_count += 1
            if day_count % 10 == 0:
                logger.info(f"Processing {zone_key}: {day_count}/{total_days} days")
        
        return pd.DataFrame(daily_results)
    
    def run_full_analysis(self, year: int = 2023) -> Dict[str, pd.DataFrame]:
        """
        Run full analysis for all zones and battery configurations.
        """
        # Load data
        self.load_data(year)
        
        # Battery configurations
        tb2_config = BatteryConfig(power_mw=1.0, duration_hours=2.0)
        tb4_config = BatteryConfig(power_mw=1.0, duration_hours=4.0)
        
        results = {}
        
        # Analyze each zone
        for zone_key in self.LOAD_ZONES:
            logger.info(f"\nAnalyzing {zone_key}")
            
            # TB2 analysis
            tb2_results = self.analyze_battery(
                zone_key,
                tb2_config,
                f"{year}-01-01",
                f"{year}-12-31"
            )
            results[f"{zone_key}_TB2"] = tb2_results
            
            # TB4 analysis
            tb4_results = self.analyze_battery(
                zone_key,
                tb4_config,
                f"{year}-01-01",
                f"{year}-12-31"
            )
            results[f"{zone_key}_TB4"] = tb4_results
            
        return results


def main():
    """Main execution function."""
    analyzer = ERCOTBatteryAnalyzer()
    
    # Run analysis
    results = analyzer.run_full_analysis(2023)
    
    # Generate reports
    report_gen = ReportGenerator()
    
    # Create output directory
    output_dir = Path("battery_analysis_output")
    output_dir.mkdir(exist_ok=True)
    
    # Generate Excel report
    excel_file = output_dir / f"ERCOT_Battery_Analysis_{datetime.now().strftime('%Y%m%d')}.xlsx"
    report_gen.generate_excel_report(results, excel_file)
    logger.info(f"Excel report saved to {excel_file}")
    
    # Generate HTML report
    html_file = output_dir / f"ERCOT_Battery_Analysis_{datetime.now().strftime('%Y%m%d')}.html"
    report_gen.generate_html_report(results, html_file)
    logger.info(f"HTML report saved to {html_file}")
    
    # Print summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    for key, df in results.items():
        if len(df) > 0:
            total_revenue = df['total_revenue'].sum()
            avg_daily_revenue = df['total_revenue'].mean()
            total_cycles = df['cycles'].sum()
            print(f"\n{key}:")
            print(f"  Total Annual Revenue: ${total_revenue:,.2f}")
            print(f"  Average Daily Revenue: ${avg_daily_revenue:,.2f}")
            print(f"  Total Cycles: {total_cycles:,.1f}")


if __name__ == "__main__":
    main()