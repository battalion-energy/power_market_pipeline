#!/usr/bin/env python3
"""
ERCOT Real-Time Battery Analysis - TB2 and TB4 arbitrage analysis using RT 5-minute prices.
Since DA data is sparse, this version focuses on RT-only arbitrage opportunities.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass

from battery_optimizer import BatteryConfig
from report_generator import ReportGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RTBatteryOptimizer:
    """Optimizes battery operation using only real-time 5-minute prices."""
    
    def __init__(self, battery_config: BatteryConfig):
        self.battery = battery_config
        
    def optimize_daily(self, rt_prices: pd.Series, date: pd.Timestamp) -> Dict:
        """
        Optimize battery operation for a single day using RT prices only.
        
        Args:
            rt_prices: 5-minute prices for the day (288 values)
            date: Date for optimization
            
        Returns:
            Dictionary with optimization results
        """
        if len(rt_prices) != 288:
            logger.warning(f"Expected 288 prices, got {len(rt_prices)}")
            
        # Convert to numpy array for easier manipulation
        prices = rt_prices.values
        
        # Initialize schedules
        charge = np.zeros(288)
        discharge = np.zeros(288)
        soc = np.zeros(289)  # State of charge (0 to 288)
        soc[0] = self.battery.energy_mwh * 0.5  # Start at 50% SOC
        
        # Simple greedy algorithm
        # 1. Find the cheapest intervals to charge
        # 2. Find the most expensive intervals to discharge
        
        # Number of 5-minute intervals for full charge/discharge
        intervals_per_hour = 12
        charge_intervals = int(self.battery.duration_hours * intervals_per_hour)
        
        # Sort intervals by price
        sorted_indices = np.argsort(prices)
        
        # Select charging intervals (lowest prices)
        charge_indices = sorted_indices[:charge_intervals]
        # Select discharging intervals (highest prices)
        discharge_indices = sorted_indices[-charge_intervals:]
        
        # Make sure we don't charge and discharge at the same time
        discharge_indices = [idx for idx in discharge_indices if idx not in charge_indices]
        
        # Apply charge/discharge schedule
        for idx in charge_indices:
            if soc[idx] + self.battery.power_mw / 12 <= self.battery.energy_mwh:
                charge[idx] = self.battery.power_mw
                soc[idx + 1] = soc[idx] + self.battery.power_mw / 12 * self.battery.charge_efficiency
            else:
                soc[idx + 1] = soc[idx]
                
        for idx in discharge_indices:
            if soc[idx] - self.battery.power_mw / 12 >= 0:
                discharge[idx] = self.battery.power_mw
                soc[idx + 1] = soc[idx] - self.battery.power_mw / 12 / self.battery.discharge_efficiency
            else:
                soc[idx + 1] = soc[idx]
                
        # Fill in remaining SOC values
        for i in range(288):
            if soc[i + 1] == 0:
                soc[i + 1] = soc[i]
                
        # Calculate revenue
        charge_cost = (charge / 12) * prices  # Convert MW to MWh
        discharge_revenue = (discharge / 12) * prices * self.battery.discharge_efficiency
        degradation_cost = (charge + discharge) / 12 * self.battery.degradation_cost
        
        net_revenue = discharge_revenue - charge_cost - degradation_cost
        total_revenue = net_revenue.sum()
        
        # Calculate cycles
        total_energy = (charge.sum() + discharge.sum()) / 12  # MWh
        cycles = total_energy / (2 * self.battery.energy_mwh)
        
        return {
            'date': date,
            'charge_mw': charge,
            'discharge_mw': discharge,
            'soc_mwh': soc[:-1],  # Remove last element
            'prices': prices,
            'net_revenue': net_revenue,
            'total_revenue': total_revenue,
            'cycles': cycles,
            'avg_price': prices.mean(),
            'price_spread': prices.max() - prices.min(),
            'charge_intervals': len(charge[charge > 0]),
            'discharge_intervals': len(discharge[discharge > 0])
        }


class ERCOTRTBatteryAnalyzer:
    """Analyzes battery performance using RT prices only."""
    
    # Target load zones for analysis
    LOAD_ZONES = {
        'HOUSTON': 'LZ_HOUSTON',
        'NORTH': 'LZ_NORTH',
        'WEST': 'LZ_WEST',
        'AUSTIN': 'LZ_AEN',
        'SAN_ANTONIO': 'LZ_CPS'
    }
    
    def __init__(self, data_path: Path = Path("ercot_data_processor/annual_output")):
        self.data_path = data_path
        self.rt_data = None
        
    def load_data(self, year: int = 2022) -> None:
        """Load ERCOT RT price data for the specified year."""
        logger.info(f"Loading ERCOT RT data for year {year}")
        
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
            
    def analyze_battery(
        self,
        zone_key: str,
        battery_config: BatteryConfig,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Analyze battery performance for a specific zone using RT prices only.
        
        Returns DataFrame with daily results.
        """
        zone_name = self.LOAD_ZONES[zone_key]
        logger.info(f"Analyzing {zone_key} ({zone_name}) with {battery_config.duration_hours}h battery")
        
        # Get RT price data for the zone
        mask = (
            (self.rt_data['SettlementPoint'] == zone_name) &
            (self.rt_data['timestamp'] >= pd.to_datetime(start_date)) &
            (self.rt_data['timestamp'] <= pd.to_datetime(end_date))
        )
        zone_data = self.rt_data[mask].copy()
        zone_data = zone_data.set_index('timestamp')['LMP']
        
        if len(zone_data) == 0:
            logger.warning(f"No RT data found for {zone_name}")
            return pd.DataFrame()
            
        # Initialize optimizer
        optimizer = RTBatteryOptimizer(battery_config)
        
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
                
                # Extract RT prices for the day
                rt_day = zone_data[
                    (zone_data.index >= day_start) & 
                    (zone_data.index <= day_end)
                ]
                
                # Skip if incomplete data
                if len(rt_day) < 280:  # Allow some missing intervals
                    current_date += timedelta(days=1)
                    day_count += 1
                    continue
                
                # Resample to ensure exactly 288 intervals
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
                results = optimizer.optimize_daily(rt_resampled, current_date)
                
                # Store daily summary
                daily_results.append({
                    'date': current_date,
                    'zone': zone_key,
                    'total_revenue': results['total_revenue'],
                    'cycles': results['cycles'],
                    'avg_price': results['avg_price'],
                    'price_spread': results['price_spread'],
                    'charge_intervals': results['charge_intervals'],
                    'discharge_intervals': results['discharge_intervals']
                })
                
            except Exception as e:
                logger.error(f"Error processing {current_date.date()}: {str(e)}")
                
            current_date += timedelta(days=1)
            day_count += 1
            if day_count % 30 == 0:
                logger.info(f"Processing {zone_key}: {day_count}/{total_days} days")
                
        return pd.DataFrame(daily_results)
    
    def run_full_analysis(self, year: int = 2022) -> Dict[str, pd.DataFrame]:
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
    analyzer = ERCOTRTBatteryAnalyzer()
    
    # Run analysis for 2022
    results = analyzer.run_full_analysis(2022)
    
    # Generate reports
    report_gen = ReportGenerator()
    
    # Create output directory
    output_dir = Path("battery_analysis_output")
    output_dir.mkdir(exist_ok=True)
    
    # Generate Excel report
    excel_file = output_dir / f"ERCOT_RT_Battery_Analysis_{datetime.now().strftime('%Y%m%d')}.xlsx"
    report_gen.generate_excel_report(results, excel_file)
    logger.info(f"Excel report saved to {excel_file}")
    
    # Generate HTML report
    html_file = output_dir / f"ERCOT_RT_Battery_Analysis_{datetime.now().strftime('%Y%m%d')}.html"
    report_gen.generate_html_report(results, html_file)
    logger.info(f"HTML report saved to {html_file}")
    
    # Print summary statistics
    print("\n=== SUMMARY STATISTICS (RT-Only Analysis) ===")
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