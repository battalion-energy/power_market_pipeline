#!/usr/bin/env python3
"""
Extract BESS resource mapping from ERCOT 60-day disclosure data.
Creates a CSV mapping battery base names to their generation and load resources,
along with power capacity and operational data.

Key insights about discharge data columns:
- DAM (Day-Ahead Market):
  - "Base Point" = The scheduled/awarded discharge amount for the hour (MW)
  - This is what the battery was awarded to discharge in the day-ahead market
  
- SCED (Real-Time):
  - "Telemetered Net Output" = Actual measured discharge output (MW)
  - This is what the battery actually discharged in real-time (5-min intervals)
  - "Output Schedule" = What SCED told it to discharge
  - "Base Point" = The dispatch instruction

The revenue calculation uses:
- DAM: Awards (Base Point) × DAM Price = Day-ahead revenue commitment
- RT: Actual output (Telemetered Net Output) × RT Price = Real-time revenue
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
from datetime import datetime
import logging
from typing import Dict, List, Set, Tuple
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_ercot_data_dir():
    """Get ERCOT data directory from environment or default."""
    data_dir = os.getenv("ERCOT_DATA_DIR")
    if data_dir:
        return Path(data_dir)
    return Path("/home/enrico/data/ERCOT_data")

class BESSResourceMapper:
    """Extract and map BESS resources from ERCOT disclosure data."""
    
    def __init__(self, base_dir: Path = None):
        self.base_dir = base_dir or get_ercot_data_dir()
        self.dam_gen_dir = self.base_dir / "60-Day_DAM_Disclosure_Reports" / "csv"
        self.dam_load_dir = self.base_dir / "60-Day_DAM_Disclosure_Reports" / "csv"
        self.sced_gen_dir = self.base_dir / "60-Day_SCED_Disclosure_Reports" / "csv"
        self.sced_load_dir = self.base_dir / "60-Day_SCED_Disclosure_Reports" / "csv"
        self.dam_price_dir = self.base_dir / "DAM_Settlement_Point_Prices" / "csv"
        self.rt_price_dir = self.base_dir / "Settlement_Point_Prices_at_Resource_Nodes,_Hubs_and_Load_Zones" / "csv"
        
        # Storage for discovered resources
        self.gen_resources = {}  # resource_name -> {details}
        self.load_resources = {}  # resource_name -> {details}
        self.battery_mapping = {}  # base_name -> {gen_resources, load_resources, etc}
        self.settlement_points = set()  # All discovered settlement points
        
    def extract_base_name(self, resource_name: str) -> Tuple[str, str]:
        """
        Extract base name and type from resource name.
        
        Returns:
            (base_name, resource_type)
        """
        # Handle generation resources
        if '_UNIT' in resource_name:
            base = resource_name.split('_UNIT')[0]
            return base, 'generation'
        
        # Handle load resources
        if '_LD' in resource_name:
            base = resource_name.split('_LD')[0]
            return base, 'load'
        
        # Handle other patterns (some batteries might not follow standard naming)
        if resource_name.endswith('_RN') or resource_name.endswith('_ALL'):
            base = resource_name.rsplit('_', 1)[0]
            return base, 'settlement'
        
        return resource_name, 'unknown'
    
    def is_battery_resource(self, resource_type: str, hsl: float = None, lsl: float = None) -> bool:
        """
        Determine if a resource is likely a battery based on type and characteristics.
        
        BESS indicators:
        - Resource Type = 'PWRSTR' (Power Storage)
        - Has both positive HSL (discharge) and negative LSL (charge capability)
        - Load resources with matching base names to PWRSTR resources
        """
        if pd.isna(resource_type):
            return False
            
        # Direct battery type
        if resource_type == 'PWRSTR':
            return True
        
        # Some batteries might be listed as other types but have storage characteristics
        if hsl and lsl:
            try:
                if float(hsl) > 0 and float(lsl) < 0:
                    return True
            except (ValueError, TypeError):
                pass
        
        return False
    
    def scan_dam_gen_resources(self, sample_file: str = None) -> Dict:
        """Scan DAM Generation Resource files for battery resources."""
        logger.info("Scanning DAM Generation Resources...")
        
        # Get a recent file
        if not sample_file:
            files = sorted(self.dam_gen_dir.glob("60d_DAM_Gen_Resource_Data-*.csv"))
            if not files:
                logger.error("No DAM Gen Resource files found")
                return {}
            sample_file = files[-1]
        else:
            sample_file = Path(sample_file)
        
        logger.info(f"Reading: {sample_file}")
        
        try:
            # Read the file
            df = pd.read_csv(sample_file)
            
            # Filter for battery resources
            if 'Resource Type' in df.columns:
                # Filter for PWRSTR (Power Storage) type
                batteries = df[df['Resource Type'] == 'PWRSTR'].copy()
            else:
                # If no type column, try to identify by HSL/LSL characteristics
                batteries = df.copy()
            
            # Process each unique resource
            for _, row in batteries.iterrows():
                resource_name = row['Resource Name']
                base_name, res_type = self.extract_base_name(resource_name)
                
                if resource_name not in self.gen_resources:
                    self.gen_resources[resource_name] = {
                        'base_name': base_name,
                        'resource_type': row.get('Resource Type', 'UNKNOWN'),
                        'hsl': row.get('HSL', np.nan),  # High Sustained Limit (Max discharge)
                        'lsl': row.get('LSL', np.nan),  # Low Sustained Limit (charge capability)
                        'qse': row.get('QSE Name', ''),
                        'dme': row.get('DME', ''),
                        'source_file': sample_file.name
                    }
                
                # Track in battery mapping
                if base_name not in self.battery_mapping:
                    self.battery_mapping[base_name] = {
                        'gen_resources': set(),
                        'load_resources': set(),
                        'max_power_mw': 0,
                        'resource_type': row.get('Resource Type', 'UNKNOWN'),
                        'settlement_points': set()
                    }
                
                self.battery_mapping[base_name]['gen_resources'].add(resource_name)
                
                # Update max power from HSL
                try:
                    hsl = float(row.get('HSL', 0))
                    if hsl > self.battery_mapping[base_name]['max_power_mw']:
                        self.battery_mapping[base_name]['max_power_mw'] = hsl
                except (ValueError, TypeError):
                    pass
                    
        except Exception as e:
            logger.error(f"Error reading {sample_file}: {e}")
        
        logger.info(f"Found {len(self.gen_resources)} generation resources")
        return self.gen_resources
    
    def scan_dam_load_resources(self, sample_file: str = None) -> Dict:
        """Scan DAM Load Resource files."""
        logger.info("Scanning DAM Load Resources...")
        
        # Get a recent file
        if not sample_file:
            files = sorted(self.dam_load_dir.glob("60d_DAM_Load_Resource_Data-*.csv"))
            if not files:
                logger.error("No DAM Load Resource files found")
                return {}
            sample_file = files[-1]
        else:
            sample_file = Path(sample_file)
        
        logger.info(f"Reading: {sample_file}")
        
        try:
            df = pd.read_csv(sample_file)
            
            # Process each unique resource
            for resource_name in df['Load Resource Name'].unique():
                base_name, res_type = self.extract_base_name(resource_name)
                
                # Only include if we have a matching battery base name
                if base_name in self.battery_mapping or '_LD' in resource_name:
                    row = df[df['Load Resource Name'] == resource_name].iloc[0]
                    
                    self.load_resources[resource_name] = {
                        'base_name': base_name,
                        'max_consumption': row.get('Max Power Consumption for Load Resource', np.nan),
                        'low_consumption': row.get('Low Power Consumption for Load Resource', np.nan),
                        'source_file': sample_file.name
                    }
                    
                    # Track in battery mapping
                    if base_name not in self.battery_mapping:
                        self.battery_mapping[base_name] = {
                            'gen_resources': set(),
                            'load_resources': set(),
                            'max_power_mw': 0,
                            'resource_type': 'INFERRED_STORAGE',
                            'settlement_points': set()
                        }
                    
                    self.battery_mapping[base_name]['load_resources'].add(resource_name)
                    
                    # Update max power from consumption
                    try:
                        max_consumption = float(row.get('Max Power Consumption for Load Resource', 0))
                        if max_consumption > self.battery_mapping[base_name]['max_power_mw']:
                            self.battery_mapping[base_name]['max_power_mw'] = max_consumption
                    except (ValueError, TypeError):
                        pass
                        
        except Exception as e:
            logger.error(f"Error reading {sample_file}: {e}")
        
        logger.info(f"Found {len(self.load_resources)} load resources")
        return self.load_resources
    
    def scan_settlement_points(self):
        """Scan price files to find actual settlement point names for batteries."""
        logger.info("Scanning for settlement points in price files...")
        
        # Scan DAM price files
        dam_files = sorted(self.dam_price_dir.glob("*.csv"))
        if dam_files:
            sample_file = dam_files[-1]
            logger.info(f"Scanning DAM prices: {sample_file.name}")
            try:
                # Read a sample to get settlement points
                df = pd.read_csv(sample_file, nrows=50000)
                if 'SettlementPoint' in df.columns:
                    all_points = df['SettlementPoint'].unique()
                    
                    # Match settlement points to battery base names
                    for point in all_points:
                        # Try to match to a battery base name
                        for base_name in self.battery_mapping.keys():
                            if base_name in point:
                                self.battery_mapping[base_name]['settlement_points'].add(point)
                                self.settlement_points.add(point)
                    
                    logger.info(f"Found {len(self.settlement_points)} settlement points")
            except Exception as e:
                logger.error(f"Error reading DAM prices: {e}")
        
        # Scan RT price files for additional points
        rt_files = sorted(self.rt_price_dir.glob("*.csv"))
        if rt_files:
            sample_file = rt_files[-1]
            logger.info(f"Scanning RT prices: {sample_file.name}")
            try:
                # Read a sample
                df = pd.read_csv(sample_file, nrows=50000)
                if 'SettlementPointName' in df.columns:
                    all_points = df['SettlementPointName'].unique()
                    
                    # Match settlement points to battery base names
                    for point in all_points:
                        for base_name in self.battery_mapping.keys():
                            if base_name in point:
                                self.battery_mapping[base_name]['settlement_points'].add(point)
                                self.settlement_points.add(point)
                                
            except Exception as e:
                logger.error(f"Error reading RT prices: {e}")
    
    def scan_sced_resources(self):
        """Scan SCED files for additional resource information."""
        logger.info("Scanning SCED Resources for additional data...")
        
        # SCED Gen Resources
        files = sorted(self.sced_gen_dir.glob("60d_SCED_Gen_Resource_Data-*.csv"))
        if files:
            sample_file = files[-1]
            logger.info(f"Reading SCED Gen: {sample_file}")
            try:
                df = pd.read_csv(sample_file, nrows=10000)  # Sample for speed
                
                # Look for PWRSTR resources
                if 'Resource Type' in df.columns:
                    batteries = df[df['Resource Type'] == 'PWRSTR']['Resource Name'].unique()
                    for resource_name in batteries:
                        base_name, _ = self.extract_base_name(resource_name)
                        if base_name not in self.battery_mapping:
                            self.battery_mapping[base_name] = {
                                'gen_resources': {resource_name},
                                'load_resources': set(),
                                'max_power_mw': 0,
                                'resource_type': 'PWRSTR',
                                'settlement_points': set()
                            }
                        else:
                            self.battery_mapping[base_name]['gen_resources'].add(resource_name)
            except Exception as e:
                logger.error(f"Error reading SCED Gen: {e}")
        
        # SCED Load Resources
        files = sorted(self.sced_load_dir.glob("60d_Load_Resource_Data_in_SCED-*.csv"))
        if files:
            sample_file = files[-1]
            logger.info(f"Reading SCED Load: {sample_file}")
            try:
                df = pd.read_csv(sample_file, nrows=10000)  # Sample for speed
                
                for resource_name in df['Resource Name'].unique():
                    if '_LD' in resource_name:
                        base_name, _ = self.extract_base_name(resource_name)
                        if base_name in self.battery_mapping:
                            self.battery_mapping[base_name]['load_resources'].add(resource_name)
            except Exception as e:
                logger.error(f"Error reading SCED Load: {e}")
    
    def get_duration_if_available(self, base_name: str) -> float:
        """
        Get battery duration ONLY if we have actual data.
        Returns None if duration cannot be determined from actual data.
        
        Note: Duration (MWh capacity) is not directly available in ERCOT disclosure data.
        This would need to come from:
        - ERCOT resource registration data
        - Interconnection agreements
        - Public filings
        - Actual SOC tracking over time
        """
        # We DO NOT have duration data in the disclosure files
        # Return None to indicate this data is not available
        return None
    
    def create_mapping_dataframe(self) -> pd.DataFrame:
        """Create a DataFrame with the complete battery mapping."""
        rows = []
        
        for base_name, info in self.battery_mapping.items():
            # Only include entries that have both gen and load resources (true batteries)
            if info['gen_resources'] or info['load_resources']:
                # Get actual settlement points found in price files
                settlement_points_str = '|'.join(sorted(info.get('settlement_points', set())))
                if not settlement_points_str:
                    # If no actual settlement points found, mark as UNKNOWN
                    settlement_points_str = 'NOT_FOUND_IN_PRICE_FILES'
                
                row = {
                    'battery_name': base_name,
                    'gen_resources': '|'.join(sorted(info['gen_resources'])),
                    'load_resources': '|'.join(sorted(info['load_resources'])),
                    'num_gen_units': len(info['gen_resources']),
                    'num_load_units': len(info['load_resources']),
                    'max_power_mw': info['max_power_mw'] if info['max_power_mw'] > 0 else None,
                    'duration_hours': None,  # We don't have this data
                    'capacity_mwh': None,  # We don't have this data
                    'resource_type': info['resource_type'],
                    'settlement_points': settlement_points_str,
                    'has_gen_resource': len(info['gen_resources']) > 0,
                    'has_load_resource': len(info['load_resources']) > 0,
                    'is_complete': len(info['gen_resources']) > 0 and len(info['load_resources']) > 0
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Sort by battery name
        df = df.sort_values('battery_name')
        
        return df
    
    def run(self, output_file: str = 'bess_resource_mapping.csv'):
        """Run the complete extraction process."""
        logger.info("="*60)
        logger.info("BESS Resource Mapping Extraction")
        logger.info("="*60)
        
        # Scan all resource types
        self.scan_dam_gen_resources()
        self.scan_dam_load_resources()
        self.scan_sced_resources()
        self.scan_settlement_points()  # Add settlement point scanning
        
        # Create mapping DataFrame
        df = self.create_mapping_dataframe()
        
        # Save to CSV
        output_path = Path(output_file)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved mapping to: {output_path}")
        
        # Print summary statistics
        logger.info("\n" + "="*60)
        logger.info("Summary Statistics")
        logger.info("="*60)
        logger.info(f"Total batteries found: {len(df)}")
        logger.info(f"Batteries with complete mapping: {df['is_complete'].sum()}")
        logger.info(f"Batteries missing load resources: {(~df['has_load_resource']).sum()}")
        logger.info(f"Batteries missing gen resources: {(~df['has_gen_resource']).sum()}")
        logger.info(f"Total known power capacity (MW): {df['max_power_mw'].sum():.1f}")
        logger.info(f"Batteries with settlement points found: {(df['settlement_points'] != 'NOT_FOUND_IN_PRICE_FILES').sum()}")
        logger.info("Note: Duration/MWh capacity data not available in disclosure files")
        
        # Show sample of mapping
        logger.info("\n" + "="*60)
        logger.info("Sample Battery Mappings")
        logger.info("="*60)
        
        # Show a few complete examples
        complete_batteries = df[df['is_complete']].head(5)
        for _, battery in complete_batteries.iterrows():
            logger.info(f"\nBattery: {battery['battery_name']}")
            logger.info(f"  Power: {battery['max_power_mw']} MW")
            logger.info(f"  Resource Type: {battery['resource_type']}")
            logger.info(f"  Gen Resources: {battery['gen_resources']}")
            logger.info(f"  Load Resources: {battery['load_resources']}")
            logger.info(f"  Settlement Points: {battery['settlement_points']}")
        
        return df


def main():
    """Main execution function."""
    mapper = BESSResourceMapper()
    df = mapper.run('bess_resource_mapping.csv')
    
    # Also create a simplified version for quick reference
    simple_df = df[['battery_name', 'max_power_mw', 'resource_type', 'settlement_points', 'is_complete']].copy()
    simple_df.to_csv('bess_resource_summary.csv', index=False)
    logger.info(f"\nAlso saved summary to: bess_resource_summary.csv")
    
    # Note about missing data
    logger.info("\n" + "="*60)
    logger.info("IMPORTANT NOTE ON DATA AVAILABILITY")
    logger.info("="*60)
    logger.info("Duration (hours) and capacity (MWh) are NOT available in ERCOT disclosure data.")
    logger.info("To get this information, you would need:")
    logger.info("  - ERCOT resource registration database")
    logger.info("  - Interconnection agreements")
    logger.info("  - Public filings or press releases")
    logger.info("  - Calculate from actual SOC patterns over time")


if __name__ == "__main__":
    main()