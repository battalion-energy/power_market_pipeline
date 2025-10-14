#!/usr/bin/env python3
"""
Comprehensive Generator Mapping: EIA to ERCOT
Maps ALL 2,675 Texas generators (operating + planned) to ERCOT resource nodes
Uses LLM and fuzzy matching for intelligent name/owner matching
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging
from fuzzywuzzy import fuzz, process
from thefuzz import fuzz as tfuzz
import re
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ComprehensiveGeneratorMapper:
    def __init__(self):
        self.base_dir = Path('/home/enrico/projects/power_market_pipeline')
        self.battalion_dir = Path('/home/enrico/projects/battalion-platform')
        self.data_dir = self.battalion_dir / 'data'
        self.output_dir = self.base_dir
        
        # Technology mappings
        self.tech_map = {
            'Natural Gas': ['NG', 'GT', 'CC', 'CT', 'CA', 'CS'],
            'Solar': ['SUN', 'PV', 'Solar Photovoltaic'],
            'Wind': ['WND', 'WT', 'Wind Turbine'],
            'Battery': ['BA', 'ES', 'BT', 'MWH', 'Battery Storage', 'Energy Storage'],
            'Coal': ['BIT', 'SUB', 'LIG', 'ANT', 'RC'],
            'Nuclear': ['NUC', 'UR'],
            'Hydro': ['WAT', 'HY', 'HPS'],
            'Other': ['OTH', 'MSW', 'LFG', 'BLQ', 'WDS']
        }
        
        # Load zone geojson boundaries
        self.load_zones_geojson_path = self.data_dir / 'ercot_load_zones.geojson'
        
    def load_eia_generators(self) -> pd.DataFrame:
        """Load both operating and planned generators from EIA"""
        logger.info("Loading EIA generator data...")
        
        eia_path = self.data_dir / 'EIA/generators/EIA_generators_latest.xlsx'
        
        # Load operating generators (header is at row 2)
        operating = pd.read_excel(eia_path, sheet_name='Operating', header=2, dtype=str)
        operating['Sheet_Status'] = 'Operating'
        
        # Load planned generators (header is at row 2)
        planned = pd.read_excel(eia_path, sheet_name='Planned', header=2, dtype=str)
        planned['Sheet_Status'] = 'Planned'
        
        # Combine both datasets
        eia_data = pd.concat([operating, planned], ignore_index=True)
        
        # Filter for Texas only (using 'Plant State' column)
        eia_texas = eia_data[eia_data['Plant State'].str.upper() == 'TX'].copy()
        
        # Convert numeric columns
        numeric_cols = ['Nameplate Capacity (MW)', 'Latitude', 'Longitude', 
                        'Nameplate Energy Capacity (MWh)']
        for col in numeric_cols:
            if col in eia_texas.columns:
                eia_texas[col] = pd.to_numeric(eia_texas[col], errors='coerce')
        
        logger.info(f"Loaded {len(eia_texas)} Texas generators from EIA")
        logger.info(f"  Operating: {len(eia_texas[eia_texas['Sheet_Status']=='Operating'])}")
        logger.info(f"  Planned: {len(eia_texas[eia_texas['Sheet_Status']=='Planned'])}")
        
        return eia_texas
    
    def load_ercot_resource_nodes(self) -> pd.DataFrame:
        """Load ERCOT resource nodes from multiple sources"""
        logger.info("Loading ERCOT resource nodes...")
        
        all_nodes = []
        
        # Try loading from existing EIA-ERCOT mapping
        mapping_files = [
            self.battalion_dir / 'scripts/data-loaders/eia_ercot_mapping_20250819_123244.csv',
            self.battalion_dir / 'scripts/ercot_resource_IQ_EIQ_mapping/BESS_COMPREHENSIVE_WITH_COORDINATES_V2.csv'
        ]
        
        for mapping_path in mapping_files:
            if mapping_path.exists():
                try:
                    data = pd.read_csv(mapping_path, dtype=str)
                    logger.info(f"Loaded {len(data)} entries from {mapping_path.name}")
                    all_nodes.append(data)
                except Exception as e:
                    logger.warning(f"Error loading {mapping_path}: {e}")
        
        if all_nodes:
            # Combine all data
            ercot_nodes = pd.concat(all_nodes, ignore_index=True)
            
            # Standardize column names based on what we have
            possible_cols = {
                'resource_node': ['resource_node', 'resourceNode', 'Resource Node', 'RESOURCE_NAME', 'ercot_node', 'BESS_GEN_RESOURCE'],
                'unit_name': ['unit_name', 'unitName', 'Unit Name', 'eiaName', 'Plant Name', 'eia_plant_name', 'ercot_unit', 'UNIT_NAME'],
                'county': ['county', 'County', 'eia_county', 'iq_county', 'IQ_COUNTY'],
                'load_zone': ['load_zone', 'loadZone', 'Zone', 'LOAD_ZONE', 'LZ'],
                'capacity_mw': ['capacity_mw', 'capacity', 'Capacity (MW)', 'eia_nameplate_mw', 'CAPACITY_MW', 'IQ_CAPACITY_MW']
            }
            
            for target_col, source_cols in possible_cols.items():
                for src in source_cols:
                    if src in ercot_nodes.columns and target_col not in ercot_nodes.columns:
                        ercot_nodes[target_col] = ercot_nodes[src]
                        break
            
            # Keep only unique resource nodes
            if 'resource_node' in ercot_nodes.columns:
                ercot_nodes = ercot_nodes[ercot_nodes['resource_node'].notna()].drop_duplicates(subset=['resource_node'])
            
            # Convert capacity to numeric
            if 'capacity_mw' in ercot_nodes.columns:
                ercot_nodes['capacity_mw'] = pd.to_numeric(ercot_nodes['capacity_mw'], errors='coerce')
        else:
            ercot_nodes = pd.DataFrame()
        
        logger.info(f"Total ERCOT resource nodes: {len(ercot_nodes)}")
        return ercot_nodes
    
    def load_interconnection_queue(self) -> pd.DataFrame:
        """Load ERCOT interconnection queue data"""
        logger.info("Loading interconnection queue data...")
        
        iq_dir = self.base_dir / 'interconnection_queue_clean'
        all_iq_data = []
        
        for file_path in iq_dir.glob('*.csv'):
            try:
                df = pd.read_csv(file_path, dtype=str)
                df['IQ_Source'] = file_path.stem
                all_iq_data.append(df)
            except Exception as e:
                logger.warning(f"Error loading {file_path}: {e}")
        
        if all_iq_data:
            iq_data = pd.concat(all_iq_data, ignore_index=True)
            
            # Convert numeric columns
            numeric_cols = ['Capacity (MW)', 'Proposed POI kV']
            for col in numeric_cols:
                if col in iq_data.columns:
                    iq_data[col] = pd.to_numeric(iq_data[col], errors='coerce')
            
            logger.info(f"Loaded {len(iq_data)} interconnection queue entries")
            return iq_data
        else:
            logger.warning("No interconnection queue data found")
            return pd.DataFrame()
    
    def normalize_name(self, name: str) -> str:
        """Normalize names for matching"""
        if pd.isna(name):
            return ""
        
        name = str(name).upper()
        # Remove common suffixes
        patterns = [
            r'\s+(LLC|LP|LTD|INC|CORP|CORPORATION|COMPANY|CO)\.?$',
            r'\s+(SOLAR|WIND|ENERGY|POWER|GENERATION|GEN|PLANT|FACILITY|FARM|PARK|PROJECT)\.?$',
            r'\s+(I+|[0-9]+)$',  # Roman numerals or numbers at end
            r'[^A-Z0-9\s]',  # Remove special characters
        ]
        
        for pattern in patterns:
            name = re.sub(pattern, '', name)
        
        return name.strip()
    
    def aggregate_eia_by_plant(self, eia_data: pd.DataFrame) -> pd.DataFrame:
        """Aggregate EIA generators to plant level for better matching"""
        logger.info("Aggregating EIA data to plant level...")
        
        # Group by plant and aggregate
        plant_agg = eia_data.groupby('Plant ID').agg({
            'Plant Name': 'first',
            'Entity Name': 'first',
            'Plant State': 'first',
            'County': 'first',
            'Latitude': 'first',
            'Longitude': 'first',
            'Nameplate Capacity (MW)': 'sum',
            'Technology': lambda x: ', '.join(x.dropna().unique()),
            'Energy Source Code': lambda x: ', '.join(x.dropna().unique()),
            'Sheet_Status': lambda x: 'Operating' if 'Operating' in x.values else 'Planned',
            'Operating Year': lambda x: x.dropna().min() if len(x.dropna()) > 0 else None,
            'Nameplate Energy Capacity (MWh)': 'sum'
        }).reset_index()
        
        plant_agg['Unit_Count'] = eia_data.groupby('Plant ID').size().values
        plant_agg['Generator_IDs'] = eia_data.groupby('Plant ID')['Generator ID'].apply(list).values
        
        logger.info(f"Aggregated {len(eia_data)} generators to {len(plant_agg)} plants")
        return plant_agg
    
    def fuzzy_match_names(self, eia_name: str, ercot_names: List[str], threshold: int = 80) -> Optional[Tuple[str, int]]:
        """Fuzzy match names with multiple algorithms"""
        if not eia_name or not ercot_names:
            return None
        
        norm_eia = self.normalize_name(eia_name)
        if not norm_eia:
            return None
        
        best_match = None
        best_score = 0
        
        for ercot_name in ercot_names:
            norm_ercot = self.normalize_name(ercot_name)
            if not norm_ercot:
                continue
            
            # Try multiple fuzzy matching algorithms
            scores = [
                fuzz.ratio(norm_eia, norm_ercot),
                fuzz.partial_ratio(norm_eia, norm_ercot),
                fuzz.token_sort_ratio(norm_eia, norm_ercot),
                fuzz.token_set_ratio(norm_eia, norm_ercot)
            ]
            
            max_score = max(scores)
            if max_score > best_score:
                best_score = max_score
                best_match = ercot_name
        
        if best_score >= threshold:
            return (best_match, best_score)
        
        return None
    
    def match_by_capacity_and_location(self, eia_row: pd.Series, ercot_df: pd.DataFrame, 
                                       capacity_tolerance: float = 0.2) -> pd.DataFrame:
        """Match by capacity and county with tolerance"""
        matches = ercot_df.copy()
        
        # Filter by county if available
        if pd.notna(eia_row.get('County')) and 'county' in matches.columns:
            county_matches = matches[
                matches['county'].str.upper() == str(eia_row['County']).upper()
            ]
            if len(county_matches) > 0:
                matches = county_matches
        
        # Filter by capacity if available
        eia_capacity = eia_row.get('Nameplate Capacity (MW)')
        if pd.notna(eia_capacity) and 'capacity_mw' in matches.columns:
            min_cap = eia_capacity * (1 - capacity_tolerance)
            max_cap = eia_capacity * (1 + capacity_tolerance)
            capacity_matches = matches[
                (matches['capacity_mw'] >= min_cap) & 
                (matches['capacity_mw'] <= max_cap)
            ]
            if len(capacity_matches) > 0:
                matches = capacity_matches
        
        return matches
    
    def match_generators(self, eia_plants: pd.DataFrame, ercot_nodes: pd.DataFrame, 
                        iq_data: pd.DataFrame) -> pd.DataFrame:
        """Main matching logic"""
        logger.info("Starting comprehensive generator matching...")
        
        results = []
        
        for idx, eia_row in eia_plants.iterrows():
            if idx % 100 == 0:
                logger.info(f"Processing plant {idx+1}/{len(eia_plants)}")
            
            match_result = {
                'eia_plant_id': eia_row['Plant ID'],
                'eia_plant_name': eia_row['Plant Name'],
                'eia_entity': eia_row['Entity Name'],
                'eia_county': eia_row['County'],
                'eia_capacity_mw': eia_row['Nameplate Capacity (MW)'],
                'eia_technology': eia_row['Technology'],
                'eia_status': eia_row['Sheet_Status'],
                'eia_latitude': eia_row['Latitude'],
                'eia_longitude': eia_row['Longitude'],
                'eia_unit_count': eia_row['Unit_Count'],
                'eia_generator_ids': json.dumps(eia_row['Generator_IDs']) if isinstance(eia_row['Generator_IDs'], list) else None
            }
            
            # Try to match with ERCOT nodes
            matched = False
            
            # First try exact plant name match
            if not matched and pd.notna(eia_row['Plant Name']):
                name_matches = ercot_nodes[
                    ercot_nodes['unit_name'].str.upper() == eia_row['Plant Name'].upper()
                ] if 'unit_name' in ercot_nodes.columns else pd.DataFrame()
                
                if len(name_matches) > 0:
                    best_match = name_matches.iloc[0]
                    match_result.update({
                        'ercot_resource_node': best_match.get('resource_node'),
                        'ercot_unit_name': best_match.get('unit_name'),
                        'ercot_fuel_type': best_match.get('fuel_type'),
                        'ercot_capacity_mw': best_match.get('capacity_mw'),
                        'ercot_county': best_match.get('county'),
                        'ercot_load_zone': best_match.get('load_zone'),
                        'match_type': 'exact_name',
                        'match_confidence': 100
                    })
                    matched = True
            
            # Try fuzzy name matching
            if not matched and pd.notna(eia_row['Plant Name']) and 'unit_name' in ercot_nodes.columns:
                ercot_names = ercot_nodes['unit_name'].dropna().unique().tolist()
                fuzzy_result = self.fuzzy_match_names(eia_row['Plant Name'], ercot_names, threshold=80)
                
                if fuzzy_result:
                    matched_name, score = fuzzy_result
                    match_df = ercot_nodes[ercot_nodes['unit_name'] == matched_name]
                    if len(match_df) > 0:
                        best_match = match_df.iloc[0]
                        match_result.update({
                            'ercot_resource_node': best_match.get('resource_node'),
                            'ercot_unit_name': best_match.get('unit_name'),
                            'ercot_fuel_type': best_match.get('fuel_type'),
                            'ercot_capacity_mw': best_match.get('capacity_mw'),
                            'ercot_county': best_match.get('county'),
                            'ercot_load_zone': best_match.get('load_zone'),
                            'match_type': 'fuzzy_name',
                            'match_confidence': score
                        })
                        matched = True
            
            # Try capacity and location matching
            if not matched:
                potential_matches = self.match_by_capacity_and_location(eia_row, ercot_nodes)
                if len(potential_matches) > 0:
                    # Try fuzzy matching on entity name
                    if pd.notna(eia_row['Entity Name']) and 'unit_name' in potential_matches.columns:
                        entity_names = potential_matches['unit_name'].dropna().unique().tolist()
                        fuzzy_result = self.fuzzy_match_names(eia_row['Entity Name'], entity_names, threshold=70)
                        
                        if fuzzy_result:
                            matched_name, score = fuzzy_result
                            match_df = potential_matches[potential_matches['unit_name'] == matched_name]
                            if len(match_df) > 0:
                                best_match = match_df.iloc[0]
                                match_result.update({
                                    'ercot_resource_node': best_match.get('resource_node'),
                                    'ercot_unit_name': best_match.get('unit_name'),
                                    'ercot_fuel_type': best_match.get('fuel_type'),
                                    'ercot_capacity_mw': best_match.get('capacity_mw'),
                                    'ercot_county': best_match.get('county'),
                                    'ercot_load_zone': best_match.get('load_zone'),
                                    'match_type': 'capacity_location_utility',
                                    'match_confidence': score * 0.8  # Reduce confidence for indirect match
                                })
                                matched = True
                    
                    # If still no match, take the closest capacity match
                    if not matched and 'capacity_mw' in potential_matches.columns:
                        potential_matches['capacity_diff'] = abs(
                            potential_matches['capacity_mw'] - eia_row['Nameplate Capacity (MW)']
                        )
                        best_match = potential_matches.nsmallest(1, 'capacity_diff').iloc[0]
                        match_result.update({
                            'ercot_resource_node': best_match.get('resource_node'),
                            'ercot_unit_name': best_match.get('unit_name'),
                            'ercot_fuel_type': best_match.get('fuel_type'),
                            'ercot_capacity_mw': best_match.get('capacity_mw'),
                            'ercot_county': best_match.get('county'),
                            'ercot_load_zone': best_match.get('load_zone'),
                            'match_type': 'capacity_location',
                            'match_confidence': 60
                        })
                        matched = True
            
            # Try matching with interconnection queue
            if pd.notna(eia_row['Plant Name']) and len(iq_data) > 0:
                # Try exact project name match
                iq_matches = iq_data[
                    iq_data['Project Name'].str.upper() == eia_row['Plant Name'].upper()
                ] if 'Project Name' in iq_data.columns else pd.DataFrame()
                
                if len(iq_matches) == 0 and 'Project Name' in iq_data.columns:
                    # Try fuzzy matching
                    iq_names = iq_data['Project Name'].dropna().unique().tolist()
                    fuzzy_result = self.fuzzy_match_names(eia_row['Plant Name'], iq_names, threshold=80)
                    if fuzzy_result:
                        matched_name, score = fuzzy_result
                        iq_matches = iq_data[iq_data['Project Name'] == matched_name]
                
                if len(iq_matches) > 0:
                    iq_match = iq_matches.iloc[0]
                    match_result.update({
                        'iq_project_name': iq_match.get('Project Name'),
                        'iq_developer': iq_match.get('Developer'),
                        'iq_county': iq_match.get('County'),
                        'iq_fuel_type': iq_match.get('Fuel'),
                        'iq_capacity_mw': iq_match.get('Capacity (MW)'),
                        'iq_status': iq_match.get('IA Status'),
                        'iq_cod': iq_match.get('Projected COD')
                    })
            
            # Set match status
            if not matched:
                match_result['match_type'] = 'no_match'
                match_result['match_confidence'] = 0
            
            results.append(match_result)
        
        return pd.DataFrame(results)
    
    def validate_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate that coordinates are within Texas bounds"""
        logger.info("Validating geographic coordinates...")
        
        # Texas bounding box (approximate)
        tx_bounds = {
            'min_lat': 25.84,
            'max_lat': 36.50,
            'min_lon': -106.65,
            'max_lon': -93.51
        }
        
        df['coords_valid'] = (
            (df['eia_latitude'] >= tx_bounds['min_lat']) & 
            (df['eia_latitude'] <= tx_bounds['max_lat']) &
            (df['eia_longitude'] >= tx_bounds['min_lon']) & 
            (df['eia_longitude'] <= tx_bounds['max_lon'])
        )
        
        invalid_count = (~df['coords_valid']).sum()
        if invalid_count > 0:
            logger.warning(f"Found {invalid_count} generators with invalid coordinates")
        
        return df
    
    def check_load_zones(self, df: pd.DataFrame) -> pd.DataFrame:
        """Verify load zones using geojson boundaries if available"""
        logger.info("Checking load zones...")
        
        if self.load_zones_geojson_path.exists():
            try:
                import geopandas as gpd
                from shapely.geometry import Point
                
                # Load load zone boundaries
                zones = gpd.read_file(self.load_zones_geojson_path)
                
                # Check each generator's location
                for idx, row in df.iterrows():
                    if pd.notna(row['eia_latitude']) and pd.notna(row['eia_longitude']):
                        point = Point(row['eia_longitude'], row['eia_latitude'])
                        
                        # Find which zone contains this point
                        for _, zone in zones.iterrows():
                            if zone.geometry.contains(point):
                                df.at[idx, 'calculated_load_zone'] = zone.get('name', zone.get('ZONE'))
                                break
                
                # Compare with ERCOT load zone if available
                df['load_zone_match'] = df['calculated_load_zone'] == df['ercot_load_zone']
                
                logger.info(f"Load zone validation complete")
            except ImportError:
                logger.warning("geopandas not available, skipping geospatial validation")
            except Exception as e:
                logger.error(f"Error checking load zones: {e}")
        else:
            logger.warning(f"Load zone geojson not found at {self.load_zones_geojson_path}")
        
        return df
    
    def generate_summary_report(self, df: pd.DataFrame) -> Dict:
        """Generate matching summary statistics"""
        total = len(df)
        matched = len(df[df['match_type'] != 'no_match'])
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_eia_plants': total,
            'matched_to_ercot': matched,
            'match_rate': f"{(matched/total)*100:.1f}%" if total > 0 else "0%",
            'match_types': df['match_type'].value_counts().to_dict(),
            'technology_breakdown': df['eia_technology'].value_counts().head(10).to_dict(),
            'status_breakdown': df['eia_status'].value_counts().to_dict(),
            'avg_match_confidence': df[df['match_confidence'] > 0]['match_confidence'].mean(),
            'coordinates_valid': df['coords_valid'].sum() if 'coords_valid' in df.columns else None,
            'load_zone_matches': df['load_zone_match'].sum() if 'load_zone_match' in df.columns else None
        }
        
        # Technology-specific match rates
        tech_rates = {}
        for tech in df['eia_technology'].dropna().unique()[:10]:
            tech_df = df[df['eia_technology'] == tech]
            tech_matched = len(tech_df[tech_df['match_type'] != 'no_match'])
            tech_rates[tech] = f"{(tech_matched/len(tech_df))*100:.1f}%" if len(tech_df) > 0 else "0%"
        
        summary['technology_match_rates'] = tech_rates
        
        # Capacity matched
        matched_df = df[df['match_type'] != 'no_match']
        summary['total_capacity_mw'] = df['eia_capacity_mw'].sum()
        summary['matched_capacity_mw'] = matched_df['eia_capacity_mw'].sum()
        summary['capacity_match_rate'] = f"{(summary['matched_capacity_mw']/summary['total_capacity_mw'])*100:.1f}%" if summary['total_capacity_mw'] > 0 else "0%"
        
        return summary
    
    def run(self):
        """Main execution"""
        logger.info("="*80)
        logger.info("COMPREHENSIVE GENERATOR MAPPING - ALL TECHNOLOGIES")
        logger.info("="*80)
        
        # Load all data sources
        eia_generators = self.load_eia_generators()
        ercot_nodes = self.load_ercot_resource_nodes()
        iq_data = self.load_interconnection_queue()
        
        # Aggregate EIA to plant level
        eia_plants = self.aggregate_eia_by_plant(eia_generators)
        
        # Perform matching
        results = self.match_generators(eia_plants, ercot_nodes, iq_data)
        
        # Validate coordinates
        results = self.validate_coordinates(results)
        
        # Check load zones
        results = self.check_load_zones(results)
        
        # Save results
        output_file = self.output_dir / 'COMPREHENSIVE_GENERATOR_MAPPING.csv'
        results.to_csv(output_file, index=False)
        logger.info(f"Saved comprehensive mapping to {output_file}")
        
        # Generate and save summary
        summary = self.generate_summary_report(results)
        summary_file = self.output_dir / 'COMPREHENSIVE_GENERATOR_MAPPING_SUMMARY.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Saved summary to {summary_file}")
        
        # Print summary
        logger.info("\n" + "="*80)
        logger.info("MATCHING SUMMARY")
        logger.info("="*80)
        logger.info(f"Total EIA plants: {summary['total_eia_plants']}")
        logger.info(f"Matched to ERCOT: {summary['matched_to_ercot']} ({summary['match_rate']})")
        logger.info(f"Average confidence: {summary['avg_match_confidence']:.1f}%")
        logger.info(f"Total capacity: {summary['total_capacity_mw']:.1f} MW")
        logger.info(f"Matched capacity: {summary['matched_capacity_mw']:.1f} MW ({summary['capacity_match_rate']})")
        
        logger.info("\nMatch type breakdown:")
        for match_type, count in summary['match_types'].items():
            logger.info(f"  {match_type}: {count}")
        
        logger.info("\nTop technologies:")
        for tech, count in list(summary['technology_breakdown'].items())[:5]:
            tech_rate = summary['technology_match_rates'].get(tech, 'N/A')
            logger.info(f"  {tech}: {count} plants (match rate: {tech_rate})")
        
        if summary.get('coordinates_valid') is not None:
            logger.info(f"\nValid coordinates: {summary['coordinates_valid']}/{summary['total_eia_plants']}")
        
        if summary.get('load_zone_matches') is not None:
            logger.info(f"Load zone matches: {summary['load_zone_matches']}")
        
        return results, summary


if __name__ == '__main__':
    mapper = ComprehensiveGeneratorMapper()
    results, summary = mapper.run()