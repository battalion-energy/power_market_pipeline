#!/usr/bin/env python3
"""
Match BESS resources with ERCOT interconnection queue data
Uses multiple data sources and fuzzy matching with county information
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
from difflib import SequenceMatcher
import warnings
warnings.filterwarnings('ignore')

def normalize_name(name):
    """Normalize resource names for comparison"""
    if pd.isna(name):
        return ""
    # Convert to uppercase, remove special chars
    name = str(name).upper()
    # Remove underscores for better matching
    name = name.replace('_', '')
    # Remove numbers at the end for base comparison
    name = re.sub(r'\d+$', '', name)
    return name

def calculate_similarity(str1, str2):
    """Calculate similarity score between two strings"""
    if not str1 or not str2:
        return 0
    return SequenceMatcher(None, str1, str2).ratio()

def load_all_interconnection_data():
    """Load all interconnection queue data"""
    data_dir = Path('/home/enrico/projects/power_market_pipeline/interconnection_queue_clean')
    
    all_data = {}
    
    # Load co-located operational (has Unit Code)
    df_op = pd.read_csv(data_dir / 'co_located_operational.csv')
    # Filter for batteries
    battery_mask = (df_op['Fuel'].str.upper().str.contains('OTH|BATT|STOR|ESS', na=False) | 
                   df_op['Technology'].str.upper().str.contains('BA|BATT|STOR|ESS', na=False))
    all_data['operational'] = df_op[battery_mask].copy()
    
    # Load stand-alone batteries (has INR and more details)
    df_standalone = pd.read_csv(data_dir / 'stand_alone.csv')
    # Remove empty rows
    df_standalone = df_standalone[df_standalone['Project Name'].notna()]
    all_data['standalone'] = df_standalone
    
    # Load co-located with solar
    df_solar = pd.read_csv(data_dir / 'co_located_with_solar.csv')
    df_solar = df_solar[df_solar['Project Name'].notna()]
    # Filter for battery component
    battery_mask = (df_solar['Fuel'].str.upper().str.contains('OTH|BATT|STOR|ESS', na=False) | 
                   df_solar['Technology'].str.upper().str.contains('BA|BATT|STOR|ESS', na=False))
    all_data['solar_colocated'] = df_solar[battery_mask].copy()
    
    # Load co-located with wind
    df_wind = pd.read_csv(data_dir / 'co_located_with_wind.csv')
    df_wind = df_wind[df_wind['Project Name'].notna()]
    battery_mask = (df_wind['Fuel'].str.upper().str.contains('OTH|BATT|STOR|ESS', na=False) | 
                   df_wind['Technology'].str.upper().str.contains('BA|BATT|STOR|ESS', na=False))
    all_data['wind_colocated'] = df_wind[battery_mask].copy()
    
    print('Loaded interconnection data:')
    for key, df in all_data.items():
        print(f'  {key}: {len(df)} battery projects')
    
    return all_data

def match_with_operational(bess_resource, substation, county, operational_df):
    """Match BESS resource with operational data using Unit Code"""
    best_match = None
    best_score = 0
    
    bess_norm = normalize_name(bess_resource)
    
    for _, row in operational_df.iterrows():
        score = 0
        reasons = []
        
        # Direct Unit Code match
        if pd.notna(row.get('Unit Code')):
            unit_code = str(row['Unit Code']).upper()
            # Check exact match
            if bess_resource == unit_code:
                score = 100
                reasons.append('exact Unit Code match')
            else:
                # Check similarity
                unit_norm = normalize_name(unit_code)
                similarity = calculate_similarity(bess_norm, unit_norm)
                if similarity > 0.8:
                    score = similarity * 80
                    reasons.append(f'Unit Code similarity: {similarity:.2f}')
        
        # County match
        if pd.notna(row.get('County')) and county:
            if str(row['County']).upper() == county.upper():
                score += 10
                reasons.append('county match')
        
        if score > best_score:
            best_score = score
            best_match = row
            match_reasons = ', '.join(reasons)
    
    if best_match is not None and best_score > 50:
        return {
            'match_score': best_score,
            'match_reason': match_reasons,
            'Unit_Name': best_match.get('Unit Name'),
            'Unit_Code': best_match.get('Unit Code'),
            'IQ_County': best_match.get('County'),
            'IQ_Zone': best_match.get('CDR Reporting Zone'),
            'IQ_Capacity_MW': best_match.get('Capacity (MW)*'),
            'IQ_In_Service': best_match.get('In Service'),
            'IQ_Fuel': best_match.get('Fuel'),
            'IQ_Technology': best_match.get('Technology'),
            'Source': 'Operational'
        }
    
    return None

def match_with_queue_data(bess_resource, substation, county, queue_df, source_name):
    """Match BESS resource with queue data (standalone, solar, wind)"""
    best_match = None
    best_score = 0
    
    bess_norm = normalize_name(bess_resource)
    sub_norm = normalize_name(substation) if substation else ""
    
    for _, row in queue_df.iterrows():
        score = 0
        reasons = []
        
        # Project Name match
        if pd.notna(row.get('Project Name')):
            project_norm = normalize_name(row['Project Name'])
            similarity = calculate_similarity(bess_norm, project_norm)
            if similarity > 0.6:
                score = similarity * 60
                reasons.append(f'project name similarity: {similarity:.2f}')
        
        # POI Location match (similar to substation)
        if pd.notna(row.get('POI Location')) and sub_norm:
            poi_norm = normalize_name(row['POI Location'])
            similarity = calculate_similarity(sub_norm, poi_norm)
            if similarity > 0.7:
                score += similarity * 30
                reasons.append(f'POI similarity: {similarity:.2f}')
        
        # County match
        if pd.notna(row.get('County')) and county:
            if str(row['County']).upper() == county.upper():
                score += 10
                reasons.append('county match')
        
        if score > best_score:
            best_score = score
            best_match = row
            match_reasons = ', '.join(reasons)
    
    if best_match is not None and best_score > 40:
        return {
            'match_score': best_score,
            'match_reason': match_reasons,
            'IQ_INR': best_match.get('INR'),
            'IQ_Project_Name': best_match.get('Project Name'),
            'IQ_Status': best_match.get('Project Status'),
            'IQ_Entity': best_match.get('Interconnecting Entity'),
            'IQ_POI': best_match.get('POI Location'),
            'IQ_County': best_match.get('County'),
            'IQ_Zone': best_match.get('CDR Reporting Zone'),
            'IQ_Capacity_MW': best_match.get('Capacity (MW)'),
            'IQ_COD': best_match.get('Projected COD'),
            'IQ_IA_Signed': best_match.get('IA Signed'),
            'IQ_Fuel': best_match.get('Fuel'),
            'IQ_Technology': best_match.get('Technology'),
            'Source': source_name
        }
    
    return None

def get_county_from_substation(substation):
    """Try to map substation to county based on known mappings"""
    # This is a simplified mapping - could be enhanced with more data
    county_mappings = {
        'ALVIN': 'BRAZORIA',
        'ANCHOR': 'EASTLAND',
        'ANGLETON': 'BRAZORIA',
        'BATCAVE': 'MEDINA',
        'BLUE_BONNET': 'TRAVIS',
        'CAMERON': 'CAMERON',
        'COMAL': 'COMAL',
        'DCSES': 'HOOD',
        'EAGLE_PASS': 'MAVERICK',
        'FORT_STOCKTON': 'PECOS',
        'GRAND_VIEW': 'JOHNSON',
        'HOUSTON': 'HARRIS',
        'JACKSBORO': 'JACK',
        'LOBO': 'CULBERSON',
        'MIDLAND': 'MIDLAND',
        'NOTREES': 'ECTOR',
        'ODESSA': 'ECTOR',
        'PARIS': 'LAMAR',
        'PECOS': 'REEVES',
        'PHARR': 'HIDALGO',
        'PORT_LAVACA': 'CALHOUN',
        'RAYMONDVILLE': 'WILLACY',
        'RIO_HONDO': 'CAMERON',
        'SAN_ANGELO': 'TOM GREEN',
        'SWEETWATER': 'NOLAN',
        'TYLER': 'SMITH',
        'VICTORIA': 'VICTORIA',
        'WACO': 'MCLENNAN',
        'WHARTON': 'WHARTON'
    }
    
    if not substation:
        return None
    
    sub_upper = str(substation).upper()
    for key, county in county_mappings.items():
        if key in sub_upper:
            return county
    
    return None

def main():
    """Main function to match BESS resources with interconnection queue"""
    
    # Load BESS mapping
    print('Loading BESS resource mapping...')
    bess_mapping = pd.read_csv('/home/enrico/projects/power_market_pipeline/BESS_RESOURCE_MAPPING_REAL_ONLY.csv')
    print(f'Loaded {len(bess_mapping)} BESS resources\n')
    
    # Load all interconnection data
    iq_data = load_all_interconnection_data()
    
    print('\n=== Matching BESS Resources ===\n')
    
    matches = []
    
    for idx, bess_row in bess_mapping.iterrows():
        gen_resource = bess_row['BESS_Gen_Resource']
        substation = bess_row.get('Substation')
        
        # Try to get county from substation
        county = get_county_from_substation(substation)
        
        match_result = {
            'BESS_Gen_Resource': gen_resource,
            'BESS_Load_Resource': bess_row.get('BESS_Load_Resource'),
            'Settlement_Point': bess_row.get('Settlement_Point'),
            'Substation': substation,
            'Load_Zone': bess_row.get('Load_Zone'),
            'Estimated_County': county
        }
        
        # Try operational data first (most reliable)
        op_match = match_with_operational(gen_resource, substation, county, iq_data['operational'])
        
        if op_match:
            match_result.update(op_match)
        else:
            # Try other sources
            best_alternative = None
            best_alt_score = 0
            
            # Try standalone
            standalone_match = match_with_queue_data(gen_resource, substation, county, 
                                                    iq_data['standalone'], 'Standalone')
            if standalone_match and standalone_match['match_score'] > best_alt_score:
                best_alternative = standalone_match
                best_alt_score = standalone_match['match_score']
            
            # Try solar co-located
            solar_match = match_with_queue_data(gen_resource, substation, county,
                                               iq_data['solar_colocated'], 'Solar Co-located')
            if solar_match and solar_match['match_score'] > best_alt_score:
                best_alternative = solar_match
                best_alt_score = solar_match['match_score']
            
            # Try wind co-located
            wind_match = match_with_queue_data(gen_resource, substation, county,
                                              iq_data['wind_colocated'], 'Wind Co-located')
            if wind_match and wind_match['match_score'] > best_alt_score:
                best_alternative = wind_match
                best_alt_score = wind_match['match_score']
            
            if best_alternative:
                match_result.update(best_alternative)
            else:
                match_result['match_score'] = 0
                match_result['match_reason'] = 'No match found'
        
        matches.append(match_result)
        
        # Progress indicator
        if (idx + 1) % 20 == 0:
            print(f'  Processed {idx + 1}/{len(bess_mapping)} resources...')
    
    # Create DataFrame with results
    results_df = pd.DataFrame(matches)
    
    # Sort by match score
    results_df = results_df.sort_values('match_score', ascending=False)
    
    # Save results
    output_file = '/home/enrico/projects/power_market_pipeline/BESS_INTERCONNECTION_MATCHED.csv'
    results_df.to_csv(output_file, index=False)
    print(f'\n✅ Saved matched results to: {output_file}')
    
    # Show statistics
    print('\n=== Matching Statistics ===')
    
    excellent_matches = results_df[results_df['match_score'] >= 90]
    good_matches = results_df[(results_df['match_score'] >= 70) & (results_df['match_score'] < 90)]
    fair_matches = results_df[(results_df['match_score'] >= 50) & (results_df['match_score'] < 70)]
    poor_matches = results_df[(results_df['match_score'] > 0) & (results_df['match_score'] < 50)]
    no_matches = results_df[results_df['match_score'] == 0]
    
    print(f'Excellent matches (≥90): {len(excellent_matches)} ({100*len(excellent_matches)/len(results_df):.1f}%)')
    print(f'Good matches (70-90): {len(good_matches)} ({100*len(good_matches)/len(results_df):.1f}%)')
    print(f'Fair matches (50-70): {len(fair_matches)} ({100*len(fair_matches)/len(results_df):.1f}%)')
    print(f'Poor matches (<50): {len(poor_matches)} ({100*len(poor_matches)/len(results_df):.1f}%)')
    print(f'No match: {len(no_matches)} ({100*len(no_matches)/len(results_df):.1f}%)')
    
    # Show sample excellent matches
    if len(excellent_matches) > 0:
        print('\n=== Sample Excellent Matches ===')
        sample_cols = ['BESS_Gen_Resource', 'Unit_Code', 'IQ_County', 'IQ_Capacity_MW', 'match_score', 'match_reason']
        available_cols = [col for col in sample_cols if col in excellent_matches.columns]
        print(excellent_matches[available_cols].head(10).to_string(index=False))
    
    # Show unmatched resources
    if len(no_matches) > 0:
        print(f'\n=== Unmatched BESS Resources ({len(no_matches)} total) ===')
        print('First 20:')
        print(no_matches[['BESS_Gen_Resource', 'Substation', 'Load_Zone']].head(20).to_string(index=False))
    
    return results_df

if __name__ == '__main__':
    matched_data = main()