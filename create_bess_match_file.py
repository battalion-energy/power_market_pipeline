#!/usr/bin/env python3
"""
Create a curated BESS resource matching file.

This script generates a CSV file that maps battery resources based on their naming patterns.
The matching algorithm uses the prefix before the first underscore to group related resources.

The output file can be manually edited to correct any mismatches or add missing relationships.
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
import json

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
    if os.name == 'posix':  # Linux/Mac
        return Path("/home/enrico/data/ERCOT_data")
    else:
        return Path("/Users/enrico/data/ERCOT_data")

class BESSMatchFileCreator:
    """Create a curated match file for BESS resources."""
    
    def __init__(self, base_dir: Path = None):
        self.base_dir = base_dir or get_ercot_data_dir()
        self.dam_gen_dir = self.base_dir / "60-Day_DAM_Disclosure_Reports" / "csv"
        self.dam_load_dir = self.base_dir / "60-Day_DAM_Disclosure_Reports" / "csv"
        self.sced_gen_dir = self.base_dir / "60-Day_SCED_Disclosure_Reports" / "csv"
        self.sced_load_dir = self.base_dir / "60-Day_SCED_Disclosure_Reports" / "csv"
        
        # Storage for discovered resources
        self.all_gen_resources = set()
        self.all_load_resources = set()
        self.pwrstr_resources = set()  # Known battery (PWRSTR) resources
        self.resource_details = {}  # resource_name -> details
        
    def extract_prefix(self, resource_name: str) -> str:
        """
        Extract the prefix of a resource name up to the first underscore.
        
        Examples:
        - MADERO_UNIT1 -> MADERO
        - MADERO_LD1 -> MADERO
        - BRP_PBL1_UNIT1 -> BRP (first part only)
        - ANCHOR_BESS1 -> ANCHOR
        
        Special cases:
        - If no underscore, return the full name
        - Remove common suffixes like _BESS, _ESS, _BAT
        """
        if not resource_name:
            return ""
        
        # First, get the part before the first underscore
        parts = resource_name.split('_')
        if len(parts) == 1:
            # No underscore, return full name
            return resource_name
        
        prefix = parts[0]
        
        # Special handling for compound names
        # If the second part looks like it's part of the name (not UNIT, LD, BESS, etc.)
        if len(parts) > 1:
            second_part = parts[1]
            # Keep second part if it's likely part of the facility name
            if second_part not in ['UNIT', 'LD', 'BESS', 'ESS', 'BAT', 'BES', 'BATTERY']:
                # Check if it's a number or common suffix
                if not second_part.isdigit() and not second_part.startswith('LD'):
                    # Could be part of name like BRP_PBL1
                    # But we still want just the first part for matching
                    pass
        
        return prefix
    
    def scan_all_resources(self):
        """Scan all disclosure files to find all resources."""
        logger.info("Scanning all resource files...")
        
        # Scan DAM Gen Resources
        dam_gen_files = sorted(self.dam_gen_dir.glob("60d_DAM_Gen_Resource_Data-*.csv"))
        if dam_gen_files:
            sample_file = dam_gen_files[-1]
            logger.info(f"Scanning DAM Gen: {sample_file.name}")
            try:
                df = pd.read_csv(sample_file)
                if 'Resource Name' in df.columns:
                    self.all_gen_resources.update(df['Resource Name'].unique())
                    
                    # Track PWRSTR resources
                    if 'Resource Type' in df.columns:
                        pwrstr_df = df[df['Resource Type'] == 'PWRSTR']
                        self.pwrstr_resources.update(pwrstr_df['Resource Name'].unique())
                        
                        # Store details
                        for _, row in pwrstr_df.iterrows():
                            name = row['Resource Name']
                            self.resource_details[name] = {
                                'type': 'gen',
                                'resource_type': 'PWRSTR',
                                'hsl': row.get('HSL', np.nan),
                                'lsl': row.get('LSL', np.nan)
                            }
            except Exception as e:
                logger.error(f"Error reading DAM Gen: {e}")
        
        # Scan DAM Load Resources
        dam_load_files = sorted(self.dam_load_dir.glob("60d_DAM_Load_Resource_Data-*.csv"))
        if dam_load_files:
            sample_file = dam_load_files[-1]
            logger.info(f"Scanning DAM Load: {sample_file.name}")
            try:
                df = pd.read_csv(sample_file)
                if 'Load Resource Name' in df.columns:
                    self.all_load_resources.update(df['Load Resource Name'].unique())
                    
                    # Store details
                    for name in df['Load Resource Name'].unique():
                        if '_LD' in name:  # Likely a battery load resource
                            row = df[df['Load Resource Name'] == name].iloc[0]
                            self.resource_details[name] = {
                                'type': 'load',
                                'max_consumption': row.get('Max Power Consumption for Load Resource', np.nan)
                            }
            except Exception as e:
                logger.error(f"Error reading DAM Load: {e}")
        
        # Scan SCED resources for additional data
        sced_gen_files = sorted(self.sced_gen_dir.glob("60d_SCED_Gen_Resource_Data-*.csv"))
        if sced_gen_files:
            sample_file = sced_gen_files[-1]
            logger.info(f"Scanning SCED Gen: {sample_file.name}")
            try:
                df = pd.read_csv(sample_file, nrows=50000)
                if 'Resource Name' in df.columns:
                    self.all_gen_resources.update(df['Resource Name'].unique())
                    if 'Resource Type' in df.columns:
                        pwrstr_df = df[df['Resource Type'] == 'PWRSTR']
                        self.pwrstr_resources.update(pwrstr_df['Resource Name'].unique())
            except Exception as e:
                logger.error(f"Error reading SCED Gen: {e}")
        
        sced_load_files = sorted(self.sced_load_dir.glob("60d_Load_Resource_Data_in_SCED-*.csv"))
        if sced_load_files:
            sample_file = sced_load_files[-1]
            logger.info(f"Scanning SCED Load: {sample_file.name}")
            try:
                df = pd.read_csv(sample_file, nrows=50000)
                if 'Resource Name' in df.columns:
                    load_names = df['Resource Name'].unique()
                    self.all_load_resources.update(load_names)
            except Exception as e:
                logger.error(f"Error reading SCED Load: {e}")
        
        logger.info(f"Found {len(self.all_gen_resources)} generation resources")
        logger.info(f"Found {len(self.all_load_resources)} load resources")
        logger.info(f"Found {len(self.pwrstr_resources)} PWRSTR (battery) resources")
    
    def create_match_groups(self) -> Dict[str, Dict]:
        """
        Create match groups based on prefix matching.
        
        Returns a dictionary where keys are prefixes and values contain:
        - gen_resources: list of generation resources with this prefix
        - load_resources: list of load resources with this prefix
        - is_confirmed_battery: True if any gen resource is PWRSTR type
        - match_confidence: HIGH, MEDIUM, or LOW
        """
        match_groups = {}
        
        # Process generation resources
        for resource in self.all_gen_resources:
            prefix = self.extract_prefix(resource)
            if prefix not in match_groups:
                match_groups[prefix] = {
                    'prefix': prefix,
                    'gen_resources': [],
                    'load_resources': [],
                    'is_confirmed_battery': False,
                    'match_confidence': 'LOW',
                    'max_power_mw': 0
                }
            match_groups[prefix]['gen_resources'].append(resource)
            
            # Check if it's a confirmed battery
            if resource in self.pwrstr_resources:
                match_groups[prefix]['is_confirmed_battery'] = True
                
                # Get power capacity
                if resource in self.resource_details:
                    hsl = self.resource_details[resource].get('hsl', 0)
                    try:
                        hsl_val = float(hsl) if not pd.isna(hsl) else 0
                        if hsl_val > match_groups[prefix]['max_power_mw']:
                            match_groups[prefix]['max_power_mw'] = hsl_val
                    except:
                        pass
        
        # Process load resources
        for resource in self.all_load_resources:
            prefix = self.extract_prefix(resource)
            if prefix not in match_groups:
                match_groups[prefix] = {
                    'prefix': prefix,
                    'gen_resources': [],
                    'load_resources': [],
                    'is_confirmed_battery': False,
                    'match_confidence': 'LOW',
                    'max_power_mw': 0
                }
            match_groups[prefix]['load_resources'].append(resource)
            
            # Get power capacity from load side
            if resource in self.resource_details:
                max_cons = self.resource_details[resource].get('max_consumption', 0)
                try:
                    cons_val = float(max_cons) if not pd.isna(max_cons) else 0
                    if cons_val > match_groups[prefix]['max_power_mw']:
                        match_groups[prefix]['max_power_mw'] = cons_val
                except:
                    pass
        
        # Assess match confidence
        for prefix, group in match_groups.items():
            has_gen = len(group['gen_resources']) > 0
            has_load = len(group['load_resources']) > 0
            
            # HIGH confidence: Has both gen and load, and is PWRSTR
            if has_gen and has_load and group['is_confirmed_battery']:
                group['match_confidence'] = 'HIGH'
            # MEDIUM confidence: Has both gen and load, or is PWRSTR
            elif (has_gen and has_load) or group['is_confirmed_battery']:
                group['match_confidence'] = 'MEDIUM'
            # Check for battery-like load resources
            elif any('_LD' in r for r in group['load_resources']):
                group['match_confidence'] = 'MEDIUM'
            else:
                group['match_confidence'] = 'LOW'
        
        return match_groups
    
    def create_match_dataframe(self, match_groups: Dict) -> pd.DataFrame:
        """Convert match groups to a DataFrame for export."""
        rows = []
        
        for prefix, group in match_groups.items():
            # Only include groups that are likely batteries
            if group['match_confidence'] in ['HIGH', 'MEDIUM'] or group['is_confirmed_battery']:
                row = {
                    'battery_prefix': prefix,
                    'gen_resources': '|'.join(sorted(group['gen_resources'])),
                    'load_resources': '|'.join(sorted(group['load_resources'])),
                    'num_gen_units': len(group['gen_resources']),
                    'num_load_units': len(group['load_resources']),
                    'is_confirmed_battery': group['is_confirmed_battery'],
                    'match_confidence': group['match_confidence'],
                    'max_power_mw': group['max_power_mw'] if group['max_power_mw'] > 0 else None,
                    'has_complete_pair': len(group['gen_resources']) > 0 and len(group['load_resources']) > 0,
                    'manual_review_needed': group['match_confidence'] == 'MEDIUM',
                    'notes': ''  # Field for manual notes
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Sort by confidence and then by name
        df['sort_order'] = df['match_confidence'].map({'HIGH': 1, 'MEDIUM': 2, 'LOW': 3})
        df = df.sort_values(['sort_order', 'battery_prefix'])
        df = df.drop('sort_order', axis=1)
        
        return df
    
    def create_json_match_file(self, match_groups: Dict, filename: str = 'bess_match_rules.json'):
        """Create a JSON file with matching rules that can be edited."""
        match_rules = {
            'metadata': {
                'created': datetime.now().isoformat(),
                'description': 'BESS resource matching rules - can be manually edited',
                'matching_algorithm': 'prefix_before_first_underscore'
            },
            'auto_matches': {},
            'manual_overrides': {},
            'exclusions': [],
            'notes': {}
        }
        
        # Create auto matches
        for prefix, group in match_groups.items():
            if group['match_confidence'] in ['HIGH', 'MEDIUM'] or group['is_confirmed_battery']:
                match_rules['auto_matches'][prefix] = {
                    'gen_patterns': list(set([self.extract_prefix(r) + '_UNIT*' for r in group['gen_resources']])),
                    'load_patterns': list(set([self.extract_prefix(r) + '_LD*' for r in group['load_resources']])),
                    'confidence': group['match_confidence'],
                    'is_battery': group['is_confirmed_battery'],
                    'actual_gen_resources': sorted(group['gen_resources']),
                    'actual_load_resources': sorted(group['load_resources'])
                }
        
        # Add template for manual overrides
        match_rules['manual_overrides']['EXAMPLE'] = {
            'comment': 'Example override - delete or modify',
            'gen_resources': ['EXAMPLE_UNIT1', 'EXAMPLE_UNIT2'],
            'load_resources': ['EXAMPLE_LD1'],
            'settlement_points': ['EXAMPLE_RN', 'EXAMPLE_ALL']
        }
        
        # Save to JSON
        with open(filename, 'w') as f:
            json.dump(match_rules, f, indent=2)
        
        logger.info(f"Created editable match rules file: {filename}")
        return match_rules
    
    def run(self):
        """Run the complete match file creation process."""
        logger.info("="*60)
        logger.info("BESS Resource Match File Creator")
        logger.info("="*60)
        
        # Scan all resources
        self.scan_all_resources()
        
        # Create match groups
        match_groups = self.create_match_groups()
        
        # Create DataFrame
        df = self.create_match_dataframe(match_groups)
        
        # Save to CSV
        csv_filename = 'bess_match_file.csv'
        df.to_csv(csv_filename, index=False)
        logger.info(f"Created match file: {csv_filename}")
        
        # Create JSON match rules
        self.create_json_match_file(match_groups)
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("Summary")
        logger.info("="*60)
        logger.info(f"Total battery groups identified: {len(df)}")
        logger.info(f"HIGH confidence matches: {(df['match_confidence'] == 'HIGH').sum()}")
        logger.info(f"MEDIUM confidence matches: {(df['match_confidence'] == 'MEDIUM').sum()}")
        logger.info(f"Complete pairs (gen + load): {df['has_complete_pair'].sum()}")
        logger.info(f"Confirmed PWRSTR batteries: {df['is_confirmed_battery'].sum()}")
        logger.info(f"Requiring manual review: {df['manual_review_needed'].sum()}")
        
        # Show examples of each confidence level
        logger.info("\n" + "="*60)
        logger.info("Sample Matches by Confidence Level")
        logger.info("="*60)
        
        for confidence in ['HIGH', 'MEDIUM']:
            conf_df = df[df['match_confidence'] == confidence].head(3)
            if not conf_df.empty:
                logger.info(f"\n{confidence} Confidence Examples:")
                for _, row in conf_df.iterrows():
                    logger.info(f"  {row['battery_prefix']}:")
                    logger.info(f"    Gen: {row['gen_resources'][:50]}...")
                    logger.info(f"    Load: {row['load_resources'][:50]}...")
                    logger.info(f"    Power: {row['max_power_mw']} MW")
        
        logger.info("\n" + "="*60)
        logger.info("Next Steps")
        logger.info("="*60)
        logger.info("1. Review bess_match_file.csv for accuracy")
        logger.info("2. Edit bess_match_rules.json to add manual overrides")
        logger.info("3. Mark any false positives in the 'notes' column")
        logger.info("4. Add missing relationships for split facilities")
        
        return df


def main():
    """Main execution function."""
    creator = BESSMatchFileCreator()
    df = creator.run()
    
    # Also create a simplified review file
    review_df = df[df['manual_review_needed'] == True][
        ['battery_prefix', 'gen_resources', 'load_resources', 'match_confidence', 'notes']
    ].copy()
    
    if not review_df.empty:
        review_df.to_csv('bess_matches_for_review.csv', index=False)
        logger.info(f"\nCreated review file: bess_matches_for_review.csv")
        logger.info(f"Contains {len(review_df)} matches requiring manual review")


if __name__ == "__main__":
    main()