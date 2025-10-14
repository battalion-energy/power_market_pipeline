#!/usr/bin/env python3
"""
Comprehensive BESS Revenue Calculator for ERCOT
Handles RT (15-min settlement), DA (hourly), and AS revenues
With proper aggregation to hourly, daily, monthly, and annual levels
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import glob
from pathlib import Path
import warnings
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import json

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Base data directories
DISCLOSURE_DIR = "/Users/enrico/data/ERCOT_data"
PRICE_DATA_DIR = "/Users/enrico/data/ERCOT_data"

@dataclass
class BessRevenue15Min:
    """Revenue data for 15-minute settlement interval"""
    resource_name: str
    settlement_point: str
    interval_start: datetime
    interval_end: datetime
    rt_position_mw: float  # Average of 3 SCED intervals
    rt_price: float
    rt_revenue: float  # Can be negative if charging
    as_deployment: Dict[str, float]  # MW deployed by service

@dataclass
class BessRevenueHourly:
    """Hourly aggregated revenue data"""
    resource_name: str
    settlement_point: str
    hour_start: datetime
    dam_award_mw: float
    dam_price: float
    dam_revenue: float
    rt_revenue_net: float  # Sum of 4 settlement intervals
    # AS capacity revenues (hourly)
    regup_mw: float
    regup_mcpc: float
    regup_revenue: float
    regdown_mw: float
    regdown_mcpc: float
    regdown_revenue: float
    rrs_mw: float
    rrs_mcpc: float
    rrs_revenue: float
    ecrs_mw: float
    ecrs_mcpc: float
    ecrs_revenue: float
    nonspin_mw: float
    nonspin_mcpc: float
    nonspin_revenue: float
    total_revenue: float

class ComprehensiveBessCalculator:
    def __init__(self, start_date: datetime, end_date: datetime):
        self.start_date = start_date
        self.end_date = end_date
        self.bess_resources = {}  # name -> info mapping
        self.results_15min = []
        self.results_hourly = []
        self.results_daily = []
        self.results_monthly = []
        self.results_annual = []
        
    def identify_all_bess_resources(self):
        """Identify all BESS resources across historical data"""
        logger.info("Identifying all BESS resources in historical data...")
        
        # Look through all DAM disclosure files to find BESS
        pattern = f"{DISCLOSURE_DIR}/60-Day_DAM_Disclosure_Reports/csv/60d_DAM_Gen_Resource_Data-*.csv"
        dam_files = glob.glob(pattern)
        
        total_resources = set()
        earliest_date = None
        
        for file in dam_files:
            try:
                # Extract date from filename
                date_str = Path(file).stem.split('-')[-1]
                file_date = datetime.strptime(date_str, "%d-%b-%y")
                
                if earliest_date is None or file_date < earliest_date:
                    earliest_date = file_date
                
                # Read file and identify BESS
                df = pd.read_csv(file)
                
                if 'Resource Type' in df.columns:
                    # Primary method: Resource Type = PWRSTR
                    bess_mask = df['Resource Type'] == 'PWRSTR'
                    bess_df = df[bess_mask]
                    
                    for _, row in bess_df.iterrows():
                        resource_name = row['Resource Name']
                        if resource_name not in self.bess_resources:
                            self.bess_resources[resource_name] = {
                                'settlement_point': row.get('Settlement Point Name', resource_name),
                                'qse': row.get('QSE', ''),
                                'first_seen': file_date,
                                'capacity_mw': 0  # Will update from HSL
                            }
                            total_resources.add(resource_name)
                        
                        # Update capacity from HSL if available
                        if 'HSL' in row and pd.notna(row['HSL']):
                            self.bess_resources[resource_name]['capacity_mw'] = max(
                                self.bess_resources[resource_name]['capacity_mw'],
                                row['HSL']
                            )
                            
            except Exception as e:
                logger.warning(f"Error processing {file}: {e}")
                continue
        
        logger.info(f"Found {len(total_resources)} unique BESS resources")
        logger.info(f"Earliest BESS data found: {earliest_date}")
        
        # Save resource list for reference
        with open('bess_resources_historical.json', 'w') as f:
            json.dump(self.bess_resources, f, indent=2, default=str)
            
        return earliest_date
    
    def process_rt_15min_intervals(self, date: datetime) -> List[BessRevenue15Min]:
        """Process real-time data at 15-minute settlement intervals"""
        results = []
        
        # Load SCED data (5-minute intervals)
        date_str = date.strftime("%d-%b-%y").upper()
        sced_file = f"{DISCLOSURE_DIR}/60-Day_SCED_Disclosure_Reports/csv/60d_SCED_Gen_Resource_Data-{date_str}.csv"
        
        if not os.path.exists(sced_file):
            return results
            
        try:
            sced_df = pd.read_csv(sced_file)
            
            # Filter for BESS resources
            # In SCED data, BESS may have different resource type
            bess_names = list(self.bess_resources.keys())
            if bess_names:
                sced_df = sced_df[sced_df['Resource Name'].isin(bess_names)]
            
            # Convert SCED timestamp
            sced_df['SCED Time'] = pd.to_datetime(sced_df['SCED Time Stamp'])
            
            # Create 15-minute intervals
            sced_df['Interval_15min'] = sced_df['SCED Time'].dt.floor('15min')
            
            # Load RT prices for the date (15-minute settlement files)
            rt_prices = self.load_rt_prices_for_date(date)
            
            # Get DAM awards for baseline
            dam_awards = self.load_dam_awards_for_date(date)
            
            # Process each resource and 15-min interval
            for resource in sced_df['Resource Name'].unique():
                resource_data = sced_df[sced_df['Resource Name'] == resource]
                settlement_point = self.bess_resources.get(resource, {}).get('settlement_point', resource)
                
                # Group by 15-minute intervals
                for interval, interval_data in resource_data.groupby('Interval_15min'):
                    # Average base points over the 3 SCED runs
                    avg_base_point = interval_data['Base Point'].mean()
                    
                    # Get DAM award for this hour
                    hour = interval.hour
                    dam_award = dam_awards.get((resource, hour), 0)
                    
                    # Calculate RT position
                    rt_position = avg_base_point - dam_award
                    
                    # Get RT price
                    rt_price = rt_prices.get((settlement_point, interval), 0)
                    
                    # Calculate revenue (positive for generation, negative for charging)
                    rt_revenue = rt_position * rt_price * 0.25  # 15 min = 0.25 hour
                    
                    # Track AS deployment if available
                    as_deployment = {}
                    if 'RegUp Deployed MW' in interval_data.columns:
                        as_deployment['RegUp'] = interval_data['RegUp Deployed MW'].mean()
                    if 'RegDown Deployed MW' in interval_data.columns:
                        as_deployment['RegDown'] = interval_data['RegDown Deployed MW'].mean()
                    
                    results.append(BessRevenue15Min(
                        resource_name=resource,
                        settlement_point=settlement_point,
                        interval_start=interval,
                        interval_end=interval + timedelta(minutes=15),
                        rt_position_mw=rt_position,
                        rt_price=rt_price,
                        rt_revenue=rt_revenue,
                        as_deployment=as_deployment
                    ))
                    
        except Exception as e:
            logger.error(f"Error processing RT data for {date}: {e}")
            
        return results
    
    def process_hourly_revenues(self, date: datetime) -> List[BessRevenueHourly]:
        """Process and aggregate to hourly revenues"""
        results = []
        
        # Get 15-minute RT revenues
        rt_15min = self.process_rt_15min_intervals(date)
        
        # Load DAM data
        dam_data = self.load_dam_data_for_date(date)
        
        # Load AS clearing prices
        as_prices = self.load_as_prices_for_date(date)
        
        # Process each hour
        for hour in range(24):
            hour_start = date.replace(hour=hour, minute=0, second=0, microsecond=0)
            
            # Get RT revenues for this hour (sum of 4 intervals)
            hour_rt = [r for r in rt_15min if r.interval_start.hour == hour]
            
            for resource in self.bess_resources:
                # Get DAM data
                dam_row = dam_data.get((resource, hour), {})
                
                # Sum RT revenues for the hour
                resource_rt = [r for r in hour_rt if r.resource_name == resource]
                rt_revenue_net = sum(r.rt_revenue for r in resource_rt)
                
                # Calculate AS revenues
                hourly_revenue = BessRevenueHourly(
                    resource_name=resource,
                    settlement_point=self.bess_resources[resource]['settlement_point'],
                    hour_start=hour_start,
                    dam_award_mw=dam_row.get('Awarded Quantity', 0),
                    dam_price=dam_row.get('Energy Settlement Point Price', 0),
                    dam_revenue=dam_row.get('Awarded Quantity', 0) * dam_row.get('Energy Settlement Point Price', 0),
                    rt_revenue_net=rt_revenue_net,
                    regup_mw=dam_row.get('RegUp Awarded', 0),
                    regup_mcpc=as_prices.get(('REGUP', hour), 0),
                    regup_revenue=dam_row.get('RegUp Awarded', 0) * as_prices.get(('REGUP', hour), 0),
                    regdown_mw=dam_row.get('RegDown Awarded', 0),
                    regdown_mcpc=as_prices.get(('REGDN', hour), 0),
                    regdown_revenue=dam_row.get('RegDown Awarded', 0) * as_prices.get(('REGDN', hour), 0),
                    rrs_mw=dam_row.get('RRSFFR Awarded', 0) + dam_row.get('RRSPFR Awarded', 0) + dam_row.get('RRSUFR Awarded', 0),
                    rrs_mcpc=as_prices.get(('RRS', hour), 0),
                    rrs_revenue=(dam_row.get('RRSFFR Awarded', 0) + dam_row.get('RRSPFR Awarded', 0) + dam_row.get('RRSUFR Awarded', 0)) * as_prices.get(('RRS', hour), 0),
                    ecrs_mw=dam_row.get('ECRSSD Awarded', 0),
                    ecrs_mcpc=as_prices.get(('ECRS', hour), 0),
                    ecrs_revenue=dam_row.get('ECRSSD Awarded', 0) * as_prices.get(('ECRS', hour), 0),
                    nonspin_mw=dam_row.get('NonSpin Awarded', 0),
                    nonspin_mcpc=as_prices.get(('NONSPIN', hour), 0),
                    nonspin_revenue=dam_row.get('NonSpin Awarded', 0) * as_prices.get(('NONSPIN', hour), 0),
                    total_revenue=0  # Will calculate after
                )
                
                # Calculate total revenue
                hourly_revenue.total_revenue = (
                    hourly_revenue.dam_revenue +
                    hourly_revenue.rt_revenue_net +
                    hourly_revenue.regup_revenue +
                    hourly_revenue.regdown_revenue +
                    hourly_revenue.rrs_revenue +
                    hourly_revenue.ecrs_revenue +
                    hourly_revenue.nonspin_revenue
                )
                
                results.append(hourly_revenue)
                
        return results
    
    def load_rt_prices_for_date(self, date: datetime) -> Dict[Tuple[str, datetime], float]:
        """Load 15-minute RT settlement prices"""
        prices = {}
        
        # RT files are organized by 15-minute intervals
        # Look 60 days ahead due to disclosure lag
        price_date = date + timedelta(days=60)
        date_str = price_date.strftime("%Y%m%d")
        pattern = f"{PRICE_DATA_DIR}/Settlement_Point_Prices_at_Resource_Nodes/csv/*{date_str}*.csv"
        
        for file in glob.glob(pattern):
            try:
                df = pd.read_csv(file)
                
                # Extract interval from filename
                filename = Path(file).stem
                parts = filename.split('_')
                if len(parts) >= 7:
                    start_time = parts[-2]  # HHMM
                    hour = int(start_time[:2])
                    minute = int(start_time[2:])
                    interval_start = price_date.replace(hour=hour, minute=minute)
                    
                    for _, row in df.iterrows():
                        settlement_point = row['Settlement Point Name']
                        price = row['Settlement Point Price']
                        prices[(settlement_point, interval_start)] = price
                        
            except Exception as e:
                logger.warning(f"Error loading RT prices from {file}: {e}")
                
        return prices
    
    def load_dam_data_for_date(self, date: datetime) -> Dict[Tuple[str, int], dict]:
        """Load DAM awards and prices"""
        dam_data = {}
        
        date_str = date.strftime("%d-%b-%y").upper()
        dam_file = f"{DISCLOSURE_DIR}/60-Day_DAM_Disclosure_Reports/csv/60d_DAM_Gen_Resource_Data-{date_str}.csv"
        
        if os.path.exists(dam_file):
            try:
                df = pd.read_csv(dam_file)
                
                # Filter for BESS
                if 'Resource Type' in df.columns:
                    df = df[df['Resource Type'] == 'PWRSTR']
                
                for _, row in df.iterrows():
                    resource = row['Resource Name']
                    hour = int(row['Hour Ending'])
                    dam_data[(resource, hour)] = row.to_dict()
                    
            except Exception as e:
                logger.error(f"Error loading DAM data: {e}")
                
        return dam_data
    
    def load_as_prices_for_date(self, date: datetime) -> Dict[Tuple[str, int], float]:
        """Load AS clearing prices"""
        as_prices = {}
        
        # Look for AS price file 60 days in the future (due to disclosure lag)
        price_date = date + timedelta(days=60)
        date_str = price_date.strftime("%Y%m%d")
        pattern = f"{PRICE_DATA_DIR}/DAM_Clearing_Prices_for_Capacity/csv/*{date_str}*.csv"
        
        for file in glob.glob(pattern):
            try:
                df = pd.read_csv(file)
                
                for _, row in df.iterrows():
                    hour = int(row['Hour Ending'])
                    
                    # Map AS types
                    if 'MCPC RegUp' in row:
                        as_prices[('REGUP', hour)] = row['MCPC RegUp']
                    if 'MCPC RegDn' in row:
                        as_prices[('REGDN', hour)] = row['MCPC RegDn']
                    if 'MCPC RRS' in row:
                        as_prices[('RRS', hour)] = row['MCPC RRS']
                    if 'MCPC ECRS' in row:
                        as_prices[('ECRS', hour)] = row['MCPC ECRS']
                    if 'MCPC Non-Spin' in row:
                        as_prices[('NONSPIN', hour)] = row['MCPC Non-Spin']
                        
            except Exception as e:
                logger.warning(f"Error loading AS prices: {e}")
                
        return as_prices
    
    def aggregate_to_daily(self, hourly_data: List[BessRevenueHourly]) -> pd.DataFrame:
        """Aggregate hourly to daily"""
        if not hourly_data:
            return pd.DataFrame()
            
        # Convert to DataFrame for easier aggregation
        df = pd.DataFrame([{
            'resource_name': h.resource_name,
            'date': h.hour_start.date(),
            'dam_revenue': h.dam_revenue,
            'rt_revenue': h.rt_revenue_net,
            'regup_revenue': h.regup_revenue,
            'regdown_revenue': h.regdown_revenue,
            'rrs_revenue': h.rrs_revenue,
            'ecrs_revenue': h.ecrs_revenue,
            'nonspin_revenue': h.nonspin_revenue,
            'total_revenue': h.total_revenue,
            'dam_mwh': h.dam_award_mw,
            'regup_mw': h.regup_mw,
            'regdown_mw': h.regdown_mw,
            'rrs_mw': h.rrs_mw,
            'ecrs_mw': h.ecrs_mw,
            'nonspin_mw': h.nonspin_mw
        } for h in hourly_data])
        
        # Aggregate by resource and date
        daily = df.groupby(['resource_name', 'date']).agg({
            'dam_revenue': 'sum',
            'rt_revenue': 'sum',
            'regup_revenue': 'sum',
            'regdown_revenue': 'sum',
            'rrs_revenue': 'sum',
            'ecrs_revenue': 'sum',
            'nonspin_revenue': 'sum',
            'total_revenue': 'sum',
            'dam_mwh': 'sum',
            'regup_mw': 'sum',
            'regdown_mw': 'sum',
            'rrs_mw': 'sum',
            'ecrs_mw': 'sum',
            'nonspin_mw': 'sum'
        }).reset_index()
        
        # Add calculated fields
        daily['energy_revenue'] = daily['dam_revenue'] + daily['rt_revenue']
        daily['as_revenue'] = (daily['regup_revenue'] + daily['regdown_revenue'] + 
                               daily['rrs_revenue'] + daily['ecrs_revenue'] + 
                               daily['nonspin_revenue'])
        
        return daily
    
    def aggregate_to_monthly(self, daily_data: pd.DataFrame) -> pd.DataFrame:
        """Aggregate daily to monthly"""
        if daily_data.empty:
            return pd.DataFrame()
            
        # Add year and month columns
        daily_data['year'] = pd.to_datetime(daily_data['date']).dt.year
        daily_data['month'] = pd.to_datetime(daily_data['date']).dt.month
        
        # Aggregate
        monthly = daily_data.groupby(['resource_name', 'year', 'month']).agg({
            'dam_revenue': 'sum',
            'rt_revenue': 'sum',
            'regup_revenue': 'sum',
            'regdown_revenue': 'sum',
            'rrs_revenue': 'sum',
            'ecrs_revenue': 'sum',
            'nonspin_revenue': 'sum',
            'total_revenue': 'sum',
            'energy_revenue': 'sum',
            'as_revenue': 'sum',
            'dam_mwh': 'sum',
            'regup_mw': 'mean',  # Average daily capacity
            'regdown_mw': 'mean',
            'rrs_mw': 'mean',
            'ecrs_mw': 'mean',
            'nonspin_mw': 'mean'
        }).reset_index()
        
        # Add days in month for daily average
        monthly['days_in_month'] = daily_data.groupby(['resource_name', 'year', 'month']).size().values
        monthly['avg_daily_revenue'] = monthly['total_revenue'] / monthly['days_in_month']
        
        return monthly
    
    def aggregate_to_annual(self, monthly_data: pd.DataFrame) -> pd.DataFrame:
        """Aggregate monthly to annual"""
        if monthly_data.empty:
            return pd.DataFrame()
            
        annual = monthly_data.groupby(['resource_name', 'year']).agg({
            'dam_revenue': 'sum',
            'rt_revenue': 'sum',
            'regup_revenue': 'sum',
            'regdown_revenue': 'sum',
            'rrs_revenue': 'sum',
            'ecrs_revenue': 'sum',
            'nonspin_revenue': 'sum',
            'total_revenue': 'sum',
            'energy_revenue': 'sum',
            'as_revenue': 'sum',
            'dam_mwh': 'sum',
            'days_in_month': 'sum'  # Total days
        }).reset_index()
        
        annual.rename(columns={'days_in_month': 'days_in_year'}, inplace=True)
        annual['avg_daily_revenue'] = annual['total_revenue'] / annual['days_in_year']
        
        # Calculate revenue split
        annual['energy_revenue_pct'] = annual['energy_revenue'] / annual['total_revenue'] * 100
        annual['as_revenue_pct'] = annual['as_revenue'] / annual['total_revenue'] * 100
        
        return annual
    
    def process_historical_data(self):
        """Process all historical data"""
        # First identify all BESS resources
        earliest_date = self.identify_all_bess_resources()
        
        # Determine date range
        if earliest_date:
            self.start_date = max(earliest_date, datetime(2019, 1, 1))  # Reasonable start
        
        logger.info(f"Processing data from {self.start_date} to {self.end_date}")
        
        # Process in chunks (monthly)
        current_date = self.start_date
        all_hourly = []
        
        while current_date <= self.end_date:
            month_end = (current_date.replace(day=1) + timedelta(days=32)).replace(day=1) - timedelta(days=1)
            month_end = min(month_end, self.end_date)
            
            logger.info(f"Processing {current_date.strftime('%Y-%m')}")
            
            # Process each day in the month
            day = current_date
            while day <= month_end:
                hourly_data = self.process_hourly_revenues(day)
                all_hourly.extend(hourly_data)
                day += timedelta(days=1)
            
            # Aggregate monthly data
            if all_hourly:
                # Convert to daily
                daily_df = self.aggregate_to_daily(all_hourly)
                self.results_daily.append(daily_df)
                
                # Monthly aggregation
                monthly_df = self.aggregate_to_monthly(daily_df)
                self.results_monthly.append(monthly_df)
                
            current_date = month_end + timedelta(days=1)
        
        # Combine all results
        if self.results_daily:
            self.results_daily = pd.concat(self.results_daily, ignore_index=True)
            self.results_monthly = pd.concat(self.results_monthly, ignore_index=True)
            self.results_annual = self.aggregate_to_annual(self.results_monthly)
            
            # Save results
            self.save_results()
    
    def save_results(self):
        """Save all results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Daily results
        if isinstance(self.results_daily, pd.DataFrame) and not self.results_daily.empty:
            daily_file = f"bess_revenue_daily_{timestamp}.csv"
            self.results_daily.to_csv(daily_file, index=False)
            logger.info(f"Saved daily results to {daily_file}")
        
        # Monthly results
        if isinstance(self.results_monthly, pd.DataFrame) and not self.results_monthly.empty:
            monthly_file = f"bess_revenue_monthly_{timestamp}.csv"
            self.results_monthly.to_csv(monthly_file, index=False)
            logger.info(f"Saved monthly results to {monthly_file}")
        
        # Annual results
        if isinstance(self.results_annual, pd.DataFrame) and not self.results_annual.empty:
            annual_file = f"bess_revenue_annual_{timestamp}.csv"
            self.results_annual.to_csv(annual_file, index=False)
            logger.info(f"Saved annual results to {annual_file}")
            
            # Print summary
            print("\n" + "="*80)
            print("BESS REVENUE SUMMARY - ALL HISTORICAL DATA")
            print("="*80)
            
            # Top performers by year
            for year in sorted(self.results_annual['year'].unique()):
                year_data = self.results_annual[self.results_annual['year'] == year]
                print(f"\n{year} Top BESS by Total Revenue:")
                top_10 = year_data.nlargest(10, 'total_revenue')
                for _, row in top_10.iterrows():
                    print(f"  {row['resource_name']}: ${row['total_revenue']:,.0f} "
                          f"(Energy: {row['energy_revenue_pct']:.1f}%, AS: {row['as_revenue_pct']:.1f}%)")


def main():
    """Run comprehensive BESS revenue calculation"""
    # Calculate for all available historical data
    end_date = datetime.now() - timedelta(days=60)  # Account for 60-day lag
    start_date = datetime(2019, 1, 1)  # Reasonable start for BESS in ERCOT
    
    calculator = ComprehensiveBessCalculator(start_date, end_date)
    calculator.process_historical_data()


if __name__ == "__main__":
    main()