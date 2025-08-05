#!/usr/bin/env python3
"""
Comprehensive BESS Revenue Calculator for ERCOT - Version 2
Correctly uses Settlement Point Prices (SPP) for all revenue calculations
Handles RT (15-min settlement), DA (hourly), and AS revenues
With proper aggregation to hourly, daily, monthly, and annual levels

Key Assumptions:
1. Use Settlement Point Prices (SPP) not LMPs - SPPs include scarcity adders
2. 60-day disclosure lag only affects operations data, not price data
3. RT settlement at 15-minute intervals (average of 3 SCED 5-minute runs)
4. BESS identified by Resource Type = 'PWRSTR'
5. Settlement points embedded in DAM Gen Resource Data
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
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import re

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Base data directory
BASE_DIR = "/Users/enrico/data/ERCOT_data"

# Clear directory paths for each data type
DISCLOSURE_DIRS = {
    'DAM': f"{BASE_DIR}/60-Day_DAM_Disclosure_Reports/csv",
    'SCED': f"{BASE_DIR}/60-Day_SCED_Disclosure_Reports/csv",
    'COP': f"{BASE_DIR}/60-Day_COP_Adjustment_Period_Snapshot/csv"
}

PRICE_DIRS = {
    'DAM_SPP': f"{BASE_DIR}/DAM_Settlement_Point_Prices/csv",
    'RT_SPP': f"{BASE_DIR}/Settlement_Point_Prices_at_Resource_Nodes,_Hubs_and_Load_Zones/csv",
    'AS_MCPC': f"{BASE_DIR}/DAM_Clearing_Prices_for_Capacity/csv"
}

@dataclass
class BessRevenue15Min:
    """Revenue data for 15-minute settlement interval"""
    resource_name: str
    settlement_point: str
    interval_start: datetime
    interval_end: datetime
    rt_position_mw: float  # Average of 3 SCED intervals
    rt_spp: float
    rt_revenue: float  # Can be negative if charging
    as_deployment: Dict[str, float]  # MW deployed by service

@dataclass
class BessRevenueHourly:
    """Hourly aggregated revenue data"""
    resource_name: str
    settlement_point: str
    hour_start: datetime
    # DAM values
    dam_award_mw: float
    dam_spp: float
    dam_revenue: float
    # RT values (sum of 4 intervals)
    rt_revenue_net: float
    rt_generation_mwh: float
    rt_charging_mwh: float
    # AS capacity (hourly)
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
    # Totals
    total_as_revenue: float
    total_revenue: float

class ComprehensiveBessCalculator:
    def __init__(self, start_date: datetime, end_date: datetime):
        self.start_date = start_date
        self.end_date = end_date
        self.bess_resources = {}  # name -> info mapping
        self.results_15min = []
        self.results_hourly = []
        self.results_daily = pd.DataFrame()
        self.results_monthly = pd.DataFrame()
        self.results_annual = pd.DataFrame()
        
    def identify_all_bess_resources(self):
        """Identify all BESS resources across historical data"""
        logger.info("Identifying all BESS resources in historical data...")
        
        # Look through all DAM disclosure files to find BESS
        pattern = f"{DISCLOSURE_DIRS['DAM']}/60d_DAM_Gen_Resource_Data-*.csv"
        dam_files = sorted(glob.glob(pattern))
        
        if not dam_files:
            logger.error(f"No DAM files found at {pattern}")
            return None
            
        total_resources = set()
        earliest_date = None
        latest_date = None
        
        logger.info(f"Found {len(dam_files)} DAM disclosure files")
        
        for i, file in enumerate(dam_files):
            if i % 50 == 0:
                logger.info(f"Processing file {i+1}/{len(dam_files)}")
                
            try:
                # Extract date from filename: 60d_DAM_Gen_Resource_Data-DD-MMM-YY.csv
                filename = Path(file).stem
                date_part = filename.split('-', 1)[1]  # Get "DD-MMM-YY"
                file_date = datetime.strptime(date_part, "%d-%b-%y")
                
                if earliest_date is None or file_date < earliest_date:
                    earliest_date = file_date
                if latest_date is None or file_date > latest_date:
                    latest_date = file_date
                
                # Read file and identify BESS
                df = pd.read_csv(file, nrows=10000)  # Sample for speed
                
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
        logger.info(f"Date range: {earliest_date} to {latest_date}")
        
        # Save resource list for reference
        with open('bess_resources_historical.json', 'w') as f:
            json.dump(self.bess_resources, f, indent=2, default=str)
            
        return earliest_date
    
    def load_dam_spp_prices(self, date: datetime) -> Dict[Tuple[str, int], float]:
        """Load DAM Settlement Point Prices for a given date"""
        prices = {}
        
        # Find DAM SPP file for this date
        date_str = date.strftime("%Y%m%d")
        pattern = f"{PRICE_DIRS['DAM_SPP']}/cdr.*.{date_str}.*.DAMSPNP4190.csv"
        files = glob.glob(pattern)
        
        if not files:
            logger.warning(f"No DAM SPP file found for {date}")
            return prices
            
        try:
            df = pd.read_csv(files[0])
            
            # Convert DeliveryDate to datetime if needed
            if 'DeliveryDate' in df.columns:
                df['DeliveryDate'] = pd.to_datetime(df['DeliveryDate'])
                
            # Create price lookup
            for _, row in df.iterrows():
                settlement_point = row['SettlementPoint']
                hour = int(row['HourEnding'].split(':')[0])
                price = row['SettlementPointPrice']
                prices[(settlement_point, hour)] = price
                
        except Exception as e:
            logger.error(f"Error loading DAM SPP prices for {date}: {e}")
            
        return prices
    
    def load_rt_spp_prices(self, date: datetime) -> Dict[Tuple[str, datetime], float]:
        """Load RT Settlement Point Prices for all 15-minute intervals"""
        prices = {}
        
        # RT files have complex naming: cdr.*.YYYYMMDD.*.SPPHLZNP6905_YYYYMMDD_HHMM.csv
        date_str = date.strftime("%Y%m%d")
        pattern = f"{PRICE_DIRS['RT_SPP']}/cdr.*.{date_str}.*.SPPHLZNP6905_*.csv"
        files = glob.glob(pattern)
        
        if not files:
            logger.warning(f"No RT SPP files found for {date}")
            return prices
            
        logger.info(f"Found {len(files)} RT SPP files for {date}")
        
        for file in files:
            try:
                # Extract interval time from filename
                filename = Path(file).stem
                # Look for pattern SPPHLZNP6905_YYYYMMDD_HHMM
                match = re.search(r'SPPHLZNP6905_\d{8}_(\d{4})', filename)
                if not match:
                    continue
                    
                time_str = match.group(1)
                hour = int(time_str[:2])
                minute = int(time_str[2:])
                
                df = pd.read_csv(file)
                
                # Create interval timestamp
                interval_time = date.replace(hour=hour, minute=minute)
                
                # Extract prices
                for _, row in df.iterrows():
                    settlement_point = row['SettlementPointName']
                    # DeliveryInterval: 1-4 within the hour
                    interval = int(row['DeliveryInterval'])
                    # Calculate actual 15-min interval time
                    actual_time = interval_time + timedelta(minutes=(interval-1)*15)
                    price = row['SettlementPointPrice']
                    prices[(settlement_point, actual_time)] = price
                    
            except Exception as e:
                logger.warning(f"Error loading RT SPP file {file}: {e}")
                
        return prices
    
    def load_as_mcpc_prices(self, date: datetime) -> Dict[Tuple[str, int], float]:
        """Load AS Market Clearing Prices for Capacity"""
        prices = {}
        
        date_str = date.strftime("%Y%m%d")
        pattern = f"{PRICE_DIRS['AS_MCPC']}/cdr.*.{date_str}.*.DAMCPCNP4188.csv"
        files = glob.glob(pattern)
        
        if not files:
            logger.warning(f"No AS MCPC file found for {date}")
            return prices
            
        try:
            df = pd.read_csv(files[0])
            
            # Create price lookup by service and hour
            for _, row in df.iterrows():
                hour = int(row['HourEnding'].split(':')[0])
                ancillary_type = row['AncillaryType']
                mcpc = row['MCPC']
                prices[(ancillary_type, hour)] = mcpc
                
        except Exception as e:
            logger.error(f"Error loading AS MCPC prices for {date}: {e}")
            
        return prices
    
    def process_rt_15min_intervals(self, date: datetime) -> List[BessRevenue15Min]:
        """Process real-time data at 15-minute settlement intervals"""
        results = []
        
        # Load SCED data (5-minute intervals)
        date_str = date.strftime("%d-%b-%y").upper()
        sced_file = f"{DISCLOSURE_DIRS['SCED']}/60d_SCED_Gen_Resource_Data-{date_str}.csv"
        
        if not os.path.exists(sced_file):
            return results
            
        try:
            logger.info(f"Processing RT data for {date}")
            sced_df = pd.read_csv(sced_file)
            
            # Filter for BESS resources
            bess_names = list(self.bess_resources.keys())
            if bess_names:
                sced_df = sced_df[sced_df['Resource Name'].isin(bess_names)]
            
            if sced_df.empty:
                return results
                
            # Convert SCED timestamp
            sced_df['SCED Time'] = pd.to_datetime(sced_df['SCED Time Stamp'])
            
            # Create 15-minute intervals
            sced_df['Interval_15min'] = sced_df['SCED Time'].dt.floor('15min')
            
            # Load RT prices for the date
            rt_prices = self.load_rt_spp_prices(date)
            
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
                    
                    # Get RT SPP price
                    rt_spp = rt_prices.get((settlement_point, interval), 0)
                    
                    # Calculate revenue (positive for generation, negative for charging)
                    rt_revenue = rt_position * rt_spp * 0.25  # 15 min = 0.25 hour
                    
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
                        rt_spp=rt_spp,
                        rt_revenue=rt_revenue,
                        as_deployment=as_deployment
                    ))
                    
        except Exception as e:
            logger.error(f"Error processing RT data for {date}: {e}")
            
        return results
    
    def load_dam_awards_for_date(self, date: datetime) -> Dict[Tuple[str, int], float]:
        """Load DAM energy awards for all BESS resources"""
        awards = {}
        
        date_str = date.strftime("%d-%b-%y").upper()
        dam_file = f"{DISCLOSURE_DIRS['DAM']}/60d_DAM_Gen_Resource_Data-{date_str}.csv"
        
        if os.path.exists(dam_file):
            try:
                df = pd.read_csv(dam_file)
                
                # Filter for BESS
                bess_names = list(self.bess_resources.keys())
                if bess_names:
                    df = df[df['Resource Name'].isin(bess_names)]
                
                for _, row in df.iterrows():
                    resource = row['Resource Name']
                    hour = int(row['Hour Ending'])
                    award = row.get('Awarded Quantity', 0)
                    awards[(resource, hour)] = award
                    
            except Exception as e:
                logger.error(f"Error loading DAM awards: {e}")
                
        return awards
    
    def process_hourly_revenues(self, date: datetime) -> List[BessRevenueHourly]:
        """Process and aggregate to hourly revenues"""
        results = []
        
        # Get 15-minute RT revenues
        rt_15min = self.process_rt_15min_intervals(date)
        
        # Load all price data
        dam_spp_prices = self.load_dam_spp_prices(date)
        as_mcpc_prices = self.load_as_mcpc_prices(date)
        
        # Load DAM disclosure data
        date_str = date.strftime("%d-%b-%y").upper()
        dam_file = f"{DISCLOSURE_DIRS['DAM']}/60d_DAM_Gen_Resource_Data-{date_str}.csv"
        
        dam_data = {}
        if os.path.exists(dam_file):
            try:
                df = pd.read_csv(dam_file)
                # Filter for BESS
                bess_names = list(self.bess_resources.keys())
                if bess_names:
                    df = df[df['Resource Name'].isin(bess_names)]
                    
                for _, row in df.iterrows():
                    resource = row['Resource Name']
                    hour = int(row['Hour Ending'])
                    dam_data[(resource, hour)] = row.to_dict()
            except Exception as e:
                logger.error(f"Error loading DAM data: {e}")
        
        # Process each hour
        for hour in range(24):
            hour_start = date.replace(hour=hour, minute=0, second=0, microsecond=0)
            
            # Get RT revenues for this hour (sum of 4 intervals)
            hour_rt = [r for r in rt_15min if r.interval_start.hour == hour]
            
            for resource in self.bess_resources:
                settlement_point = self.bess_resources[resource]['settlement_point']
                
                # Get DAM data
                dam_row = dam_data.get((resource, hour + 1), {})  # Hour ending convention
                
                # Calculate RT revenues for the hour
                resource_rt = [r for r in hour_rt if r.resource_name == resource]
                rt_revenue_net = sum(r.rt_revenue for r in resource_rt)
                rt_generation_mwh = sum(r.rt_position_mw * 0.25 for r in resource_rt if r.rt_position_mw > 0)
                rt_charging_mwh = sum(-r.rt_position_mw * 0.25 for r in resource_rt if r.rt_position_mw < 0)
                
                # Get prices
                dam_spp = dam_spp_prices.get((settlement_point, hour + 1), 0)
                
                # Calculate revenues
                hourly_revenue = BessRevenueHourly(
                    resource_name=resource,
                    settlement_point=settlement_point,
                    hour_start=hour_start,
                    # DAM
                    dam_award_mw=dam_row.get('Awarded Quantity', 0),
                    dam_spp=dam_spp,
                    dam_revenue=dam_row.get('Awarded Quantity', 0) * dam_spp,
                    # RT
                    rt_revenue_net=rt_revenue_net,
                    rt_generation_mwh=rt_generation_mwh,
                    rt_charging_mwh=rt_charging_mwh,
                    # AS - RegUp
                    regup_mw=dam_row.get('RegUp Awarded', 0),
                    regup_mcpc=as_mcpc_prices.get(('REGUP', hour + 1), 0),
                    regup_revenue=dam_row.get('RegUp Awarded', 0) * as_mcpc_prices.get(('REGUP', hour + 1), 0),
                    # AS - RegDown
                    regdown_mw=dam_row.get('RegDown Awarded', 0),
                    regdown_mcpc=as_mcpc_prices.get(('REGDN', hour + 1), 0),
                    regdown_revenue=dam_row.get('RegDown Awarded', 0) * as_mcpc_prices.get(('REGDN', hour + 1), 0),
                    # AS - RRS (sum of all types)
                    rrs_mw=(dam_row.get('RRSFFR Awarded', 0) + 
                            dam_row.get('RRSPFR Awarded', 0) + 
                            dam_row.get('RRSUFR Awarded', 0)),
                    rrs_mcpc=as_mcpc_prices.get(('RRS', hour + 1), 0),
                    rrs_revenue=(dam_row.get('RRSFFR Awarded', 0) + 
                                dam_row.get('RRSPFR Awarded', 0) + 
                                dam_row.get('RRSUFR Awarded', 0)) * as_mcpc_prices.get(('RRS', hour + 1), 0),
                    # AS - ECRS
                    ecrs_mw=dam_row.get('ECRSSD Awarded', 0),
                    ecrs_mcpc=as_mcpc_prices.get(('ECRS', hour + 1), 0),
                    ecrs_revenue=dam_row.get('ECRSSD Awarded', 0) * as_mcpc_prices.get(('ECRS', hour + 1), 0),
                    # AS - NonSpin
                    nonspin_mw=dam_row.get('NonSpin Awarded', 0),
                    nonspin_mcpc=as_mcpc_prices.get(('NSPIN', hour + 1), 0),
                    nonspin_revenue=dam_row.get('NonSpin Awarded', 0) * as_mcpc_prices.get(('NSPIN', hour + 1), 0),
                    # Initialize totals
                    total_as_revenue=0,
                    total_revenue=0
                )
                
                # Calculate total AS revenue
                hourly_revenue.total_as_revenue = (
                    hourly_revenue.regup_revenue +
                    hourly_revenue.regdown_revenue +
                    hourly_revenue.rrs_revenue +
                    hourly_revenue.ecrs_revenue +
                    hourly_revenue.nonspin_revenue
                )
                
                # Calculate total revenue
                hourly_revenue.total_revenue = (
                    hourly_revenue.dam_revenue +
                    hourly_revenue.rt_revenue_net +
                    hourly_revenue.total_as_revenue
                )
                
                results.append(hourly_revenue)
                
        return results
    
    def aggregate_to_daily(self, hourly_data: List[BessRevenueHourly]) -> pd.DataFrame:
        """Aggregate hourly to daily"""
        if not hourly_data:
            return pd.DataFrame()
            
        # Convert to DataFrame for easier aggregation
        df = pd.DataFrame([asdict(h) for h in hourly_data])
        df['date'] = df['hour_start'].dt.date
        
        # Aggregate by resource and date
        daily = df.groupby(['resource_name', 'date']).agg({
            'settlement_point': 'first',
            # Energy
            'dam_award_mw': 'sum',
            'dam_revenue': 'sum',
            'rt_revenue_net': 'sum',
            'rt_generation_mwh': 'sum',
            'rt_charging_mwh': 'sum',
            # AS MW (daily average)
            'regup_mw': 'mean',
            'regdown_mw': 'mean',
            'rrs_mw': 'mean',
            'ecrs_mw': 'mean',
            'nonspin_mw': 'mean',
            # AS Revenue
            'regup_revenue': 'sum',
            'regdown_revenue': 'sum',
            'rrs_revenue': 'sum',
            'ecrs_revenue': 'sum',
            'nonspin_revenue': 'sum',
            'total_as_revenue': 'sum',
            'total_revenue': 'sum'
        }).reset_index()
        
        # Add calculated fields
        daily['energy_revenue'] = daily['dam_revenue'] + daily['rt_revenue_net']
        daily['net_mwh'] = daily['rt_generation_mwh'] - daily['rt_charging_mwh']
        
        # Add capacity info
        daily['capacity_mw'] = daily['resource_name'].map(
            lambda x: self.bess_resources.get(x, {}).get('capacity_mw', 0)
        )
        daily['qse'] = daily['resource_name'].map(
            lambda x: self.bess_resources.get(x, {}).get('qse', '')
        )
        
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
            'settlement_point': 'first',
            'capacity_mw': 'first',
            'qse': 'first',
            # Energy
            'dam_revenue': 'sum',
            'rt_revenue_net': 'sum',
            'energy_revenue': 'sum',
            # AS Revenue
            'regup_revenue': 'sum',
            'regdown_revenue': 'sum',
            'rrs_revenue': 'sum',
            'ecrs_revenue': 'sum',
            'nonspin_revenue': 'sum',
            'total_as_revenue': 'sum',
            'total_revenue': 'sum',
            # Operations
            'dam_award_mw': 'sum',
            'rt_generation_mwh': 'sum',
            'rt_charging_mwh': 'sum',
            'net_mwh': 'sum',
            # AS MW (monthly average)
            'regup_mw': 'mean',
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
            'settlement_point': 'first',
            'capacity_mw': 'first',
            'qse': 'first',
            # Revenue
            'dam_revenue': 'sum',
            'rt_revenue_net': 'sum',
            'energy_revenue': 'sum',
            'regup_revenue': 'sum',
            'regdown_revenue': 'sum',
            'rrs_revenue': 'sum',
            'ecrs_revenue': 'sum',
            'nonspin_revenue': 'sum',
            'total_as_revenue': 'sum',
            'total_revenue': 'sum',
            # Operations
            'dam_award_mw': 'sum',
            'rt_generation_mwh': 'sum',
            'rt_charging_mwh': 'sum',
            'net_mwh': 'sum',
            'days_in_month': 'sum'  # Total days
        }).reset_index()
        
        annual.rename(columns={'days_in_month': 'days_in_year'}, inplace=True)
        annual['avg_daily_revenue'] = annual['total_revenue'] / annual['days_in_year']
        
        # Calculate revenue split
        annual['energy_revenue_pct'] = np.where(
            annual['total_revenue'] > 0,
            annual['energy_revenue'] / annual['total_revenue'] * 100,
            0
        )
        annual['as_revenue_pct'] = np.where(
            annual['total_revenue'] > 0,
            annual['total_as_revenue'] / annual['total_revenue'] * 100,
            0
        )
        
        # Calculate capacity factor
        annual['capacity_factor'] = np.where(
            annual['capacity_mw'] > 0,
            annual['dam_award_mw'] / (annual['capacity_mw'] * annual['days_in_year'] * 24) * 100,
            0
        )
        
        return annual
    
    def process_historical_data(self):
        """Process all historical data"""
        # First identify all BESS resources
        earliest_date = self.identify_all_bess_resources()
        
        if not self.bess_resources:
            logger.error("No BESS resources found!")
            return
        
        # Determine date range
        if earliest_date:
            self.start_date = max(earliest_date, datetime(2019, 1, 1))  # Reasonable start
        
        logger.info(f"Processing data from {self.start_date} to {self.end_date}")
        logger.info(f"Total BESS resources to process: {len(self.bess_resources)}")
        
        # Process in chunks (monthly)
        current_date = self.start_date
        all_daily = []
        
        while current_date <= self.end_date:
            month_end = (current_date.replace(day=1) + timedelta(days=32)).replace(day=1) - timedelta(days=1)
            month_end = min(month_end, self.end_date)
            
            logger.info(f"Processing {current_date.strftime('%Y-%m')}")
            
            # Process each day in the month
            month_hourly = []
            day = current_date
            while day <= month_end:
                hourly_data = self.process_hourly_revenues(day)
                month_hourly.extend(hourly_data)
                day += timedelta(days=1)
            
            # Aggregate monthly data
            if month_hourly:
                # Convert to daily
                daily_df = self.aggregate_to_daily(month_hourly)
                all_daily.append(daily_df)
                
            current_date = month_end + timedelta(days=1)
        
        # Combine all results
        if all_daily:
            self.results_daily = pd.concat(all_daily, ignore_index=True)
            self.results_monthly = self.aggregate_to_monthly(self.results_daily)
            self.results_annual = self.aggregate_to_annual(self.results_monthly)
            
            # Save results
            self.save_results()
            
            # Print summary
            self.print_summary()
    
    def save_results(self):
        """Save all results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Daily results
        if not self.results_daily.empty:
            daily_file = f"bess_revenue_daily_{timestamp}.csv"
            self.results_daily.to_csv(daily_file, index=False)
            logger.info(f"Saved daily results to {daily_file}")
        
        # Monthly results
        if not self.results_monthly.empty:
            monthly_file = f"bess_revenue_monthly_{timestamp}.csv"
            self.results_monthly.to_csv(monthly_file, index=False)
            logger.info(f"Saved monthly results to {monthly_file}")
        
        # Annual results
        if not self.results_annual.empty:
            annual_file = f"bess_revenue_annual_{timestamp}.csv"
            self.results_annual.to_csv(annual_file, index=False)
            logger.info(f"Saved annual results to {annual_file}")
    
    def print_summary(self):
        """Print comprehensive summary of results"""
        if self.results_annual.empty:
            logger.warning("No results to summarize")
            return
            
        print("\n" + "="*100)
        print("BESS REVENUE SUMMARY - ALL HISTORICAL DATA")
        print("="*100)
        
        # Overall statistics
        total_bess = self.results_annual['resource_name'].nunique()
        years = sorted(self.results_annual['year'].unique())
        
        print(f"\nAnalysis Period: {years[0]} - {years[-1]}")
        print(f"Total BESS Resources Analyzed: {total_bess}")
        
        # Top performers by year
        for year in years:
            year_data = self.results_annual[self.results_annual['year'] == year]
            if year_data.empty:
                continue
                
            print(f"\n{year} Summary:")
            print(f"  Active BESS: {len(year_data)}")
            print(f"  Total Revenue: ${year_data['total_revenue'].sum():,.0f}")
            
            print(f"\n  Top 10 BESS by Total Revenue:")
            top_10 = year_data.nlargest(10, 'total_revenue')
            for _, row in top_10.iterrows():
                print(f"    {row['resource_name']:30s} ${row['total_revenue']:12,.0f} "
                      f"(Energy: {row['energy_revenue_pct']:5.1f}%, AS: {row['as_revenue_pct']:5.1f}%) "
                      f"Capacity: {row['capacity_mw']:6.1f} MW")
        
        # Market evolution
        print("\n" + "-"*80)
        print("MARKET EVOLUTION")
        print("-"*80)
        
        yearly_summary = self.results_annual.groupby('year').agg({
            'resource_name': 'count',
            'total_revenue': 'sum',
            'energy_revenue': 'sum',
            'total_as_revenue': 'sum',
            'capacity_mw': 'sum'
        }).rename(columns={'resource_name': 'bess_count'})
        
        yearly_summary['avg_revenue_per_bess'] = yearly_summary['total_revenue'] / yearly_summary['bess_count']
        yearly_summary['energy_pct'] = yearly_summary['energy_revenue'] / yearly_summary['total_revenue'] * 100
        yearly_summary['as_pct'] = yearly_summary['total_as_revenue'] / yearly_summary['total_revenue'] * 100
        
        print("\nYearly Market Summary:")
        print(f"{'Year':>6} {'BESS':>6} {'Total MW':>10} {'Total Revenue':>15} {'Avg Rev/BESS':>15} {'Energy %':>10} {'AS %':>8}")
        print("-" * 90)
        
        for year, row in yearly_summary.iterrows():
            print(f"{year:>6} {row['bess_count']:>6} {row['capacity_mw']:>10.0f} "
                  f"${row['total_revenue']:>14,.0f} ${row['avg_revenue_per_bess']:>14,.0f} "
                  f"{row['energy_pct']:>9.1f}% {row['as_pct']:>7.1f}%")
        
        # Revenue distribution
        print("\n" + "-"*80)
        print("REVENUE DISTRIBUTION (Latest Year)")
        print("-"*80)
        
        latest_year = years[-1]
        latest_data = self.results_annual[self.results_annual['year'] == latest_year]
        
        # Group by revenue ranges
        bins = [0, 100000, 500000, 1000000, 5000000, 10000000, float('inf')]
        labels = ['<$100k', '$100k-500k', '$500k-1M', '$1M-5M', '$5M-10M', '>$10M']
        latest_data['revenue_range'] = pd.cut(latest_data['total_revenue'], bins=bins, labels=labels)
        
        revenue_dist = latest_data.groupby('revenue_range').size()
        
        print("\nBESS Count by Annual Revenue Range:")
        for range_label, count in revenue_dist.items():
            print(f"  {range_label:>12s}: {count:>4} BESS")


def main():
    """Run comprehensive BESS revenue calculation"""
    # Calculate for all available historical data
    end_date = datetime.now() - timedelta(days=60)  # Account for 60-day lag
    start_date = datetime(2019, 1, 1)  # Reasonable start for BESS in ERCOT
    
    print("Starting comprehensive BESS revenue calculation...")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Using Settlement Point Prices (SPP) for all calculations")
    print(f"Data sources:")
    print(f"  - Disclosure: {DISCLOSURE_DIRS}")
    print(f"  - Prices: {PRICE_DIRS}")
    
    calculator = ComprehensiveBessCalculator(start_date, end_date)
    calculator.process_historical_data()


if __name__ == "__main__":
    main()