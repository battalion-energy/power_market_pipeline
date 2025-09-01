#!/usr/bin/env python3
"""
Cross-reference BESS resources with ERCOT interconnection queue data
Matches BESS Generation resources to their official project information
Uses location data (county, substation) to help with fuzzy matching
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
    # Convert to uppercase, remove special chars except underscore
    name = str(name).upper()
    # Remove common suffixes
    name = re.sub(r'(_BESS\d*|_BES\d*|_UNIT\d*|_ESS\d*|_LD\d*)$', '', name)
    # Replace underscores with spaces for better matching
    name = name.replace('_', ' ')
    # Remove extra spaces
    name = ' '.join(name.split())
    return name

def calculate_similarity(str1, str2):
    """Calculate similarity score between two strings"""
    if not str1 or not str2:
        return 0
    return SequenceMatcher(None, str1, str2).ratio()

def load_interconnection_queue():
    """Load and filter ERCOT interconnection queue for operating BESS projects"""
    print('Loading ERCOT interconnection queue data...')
    
    # Load the Excel file
    excel_path = '/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/ERCOT_InterconnectionQueue/interconnection_queue.xlsx'
    
    try:
        # This file contains battery-specific sheets
        # Let's combine Stand-Alone and Co-located Operational batteries
        all_bess_projects = []
        
        # Load Stand-Alone batteries
        try:
            df_standalone = pd.read_excel(excel_path, sheet_name='Stand-Alone')
            df_standalone['Source_Sheet'] = 'Stand-Alone'
            all_bess_projects.append(df_standalone)
            print(f'Loaded {len(df_standalone)} stand-alone BESS projects')
        except Exception as e:
            print(f'Could not load Stand-Alone sheet: {e}')
        
        # Load Co-located Operational batteries
        try:
            df_colocated = pd.read_excel(excel_path, sheet_name='Co-located Operational')
            df_colocated['Source_Sheet'] = 'Co-located Operational'
            all_bess_projects.append(df_colocated)
            print(f'Loaded {len(df_colocated)} co-located operational BESS projects')
        except Exception as e:
            print(f'Could not load Co-located Operational sheet: {e}')
        
        # Load Co-located with Solar
        try:
            df_solar = pd.read_excel(excel_path, sheet_name='Co-located with Solar')
            df_solar['Source_Sheet'] = 'Co-located with Solar'
            all_bess_projects.append(df_solar)
            print(f'Loaded {len(df_solar)} BESS co-located with solar')
        except Exception as e:
            print(f'Could not load Co-located with Solar sheet: {e}')
        
        # Load Co-located with Wind
        try:
            df_wind = pd.read_excel(excel_path, sheet_name='Co-located with Wind')
            df_wind['Source_Sheet'] = 'Co-located with Wind'
            all_bess_projects.append(df_wind)
            print(f'Loaded {len(df_wind)} BESS co-located with wind')
        except Exception as e:
            print(f'Could not load Co-located with Wind sheet: {e}')
        
        if all_bess_projects:
            # Combine all dataframes
            bess_operating = pd.concat(all_bess_projects, ignore_index=True)
            print(f'\nTotal BESS projects loaded: {len(bess_operating)}')
            
            # Filter for operational status if status column exists
            status_col = None
            for col in bess_operating.columns:
                if 'STATUS' in col.upper():
                    status_col = col
                    break
            
            if status_col:
                # Filter for operational projects
                operational_mask = bess_operating[status_col].astype(str).str.upper().str.contains('OPERAT|COMMERCIAL|ONLINE', na=False)
                operational_bess = bess_operating[operational_mask].copy()
                print(f'Filtered to {len(operational_bess)} operational BESS projects')
            else:
                operational_bess = bess_operating
            
            # Display columns for debugging
            print('\nAvailable columns in interconnection queue:')
            for col in operational_bess.columns[:30]:  # Show first 30 columns
                print(f'  - {col}')
            if len(operational_bess.columns) > 30:
                print(f'  ... and {len(operational_bess.columns) - 30} more columns')
            
            return operational_bess
        else:
            print('No BESS data could be loaded')
            return pd.DataFrame()
        
    except Exception as e:
        print(f'Error loading interconnection queue: {e}')
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def match_bess_resources(bess_mapping_df, interconnection_df):
    """Match BESS resources to interconnection queue projects"""
    print('\n=== Matching BESS Resources to Interconnection Queue ===\n')
    
    matches = []
    
    # Normalize columns for matching
    # Find relevant columns in interconnection data
    project_name_col = None
    county_col = None
    capacity_col = None
    poi_col = None  # Point of Interconnection
    
    for col in interconnection_df.columns:
        col_upper = col.upper()
        if 'PROJECT' in col_upper and 'NAME' in col_upper:
            project_name_col = col
        elif 'COUNTY' in col_upper:
            county_col = col
        elif 'CAPACITY' in col_upper or 'MW' in col_upper:
            capacity_col = col
        elif 'POI' in col_upper or 'INTERCONNECTION' in col_upper:
            poi_col = col
    
    print(f'Using columns:')
    print(f'  Project Name: {project_name_col}')
    print(f'  County: {county_col}')
    print(f'  Capacity: {capacity_col}')
    print(f'  POI: {poi_col}')
    
    # For each BESS resource, try to find a match
    for _, bess_row in bess_mapping_df.iterrows():
        gen_resource = bess_row['BESS_Gen_Resource']
        substation = bess_row['Substation']
        
        # Normalize names for matching
        gen_norm = normalize_name(gen_resource)
        sub_norm = normalize_name(substation)
        
        best_match = None
        best_score = 0
        match_reason = ""
        
        # Try to match each interconnection project
        for _, iq_row in interconnection_df.iterrows():
            score = 0
            reasons = []
            
            # Match by project name
            if project_name_col and pd.notna(iq_row[project_name_col]):
                project_norm = normalize_name(iq_row[project_name_col])
                name_similarity = calculate_similarity(gen_norm, project_norm)
                if name_similarity > 0.6:  # Threshold for name matching
                    score += name_similarity * 50  # Weight name matching heavily
                    reasons.append(f"name similarity: {name_similarity:.2f}")
            
            # Match by POI/Substation
            if poi_col and pd.notna(iq_row[poi_col]):
                poi_norm = normalize_name(iq_row[poi_col])
                poi_similarity = calculate_similarity(sub_norm, poi_norm)
                if poi_similarity > 0.7:  # Higher threshold for substation
                    score += poi_similarity * 30
                    reasons.append(f"substation match: {poi_similarity:.2f}")
            
            # Check if county is mentioned in resource name
            if county_col and pd.notna(iq_row[county_col]):
                county_norm = normalize_name(iq_row[county_col])
                if county_norm and (county_norm in gen_norm or county_norm in sub_norm):
                    score += 20
                    reasons.append("county match")
            
            if score > best_score:
                best_score = score
                best_match = iq_row
                match_reason = ', '.join(reasons)
        
        # Record the match
        match_data = {
            'BESS_Gen_Resource': gen_resource,
            'BESS_Load_Resource': bess_row['BESS_Load_Resource'],
            'Substation': substation,
            'Load_Zone': bess_row['Load_Zone'],
            'Match_Score': best_score,
            'Match_Reason': match_reason
        }
        
        if best_match is not None and best_score > 30:  # Minimum score threshold
            if project_name_col:
                match_data['IQ_Project_Name'] = best_match[project_name_col]
            if county_col:
                match_data['IQ_County'] = best_match[county_col]
            if capacity_col:
                match_data['IQ_Capacity_MW'] = best_match[capacity_col]
            if poi_col:
                match_data['IQ_POI'] = best_match[poi_col]
            
            # Add other useful columns if they exist
            for col in ['INR', 'Company', 'Status', 'Signed IA Date', 'Commercial Operation Date']:
                if col in interconnection_df.columns:
                    match_data[f'IQ_{col.replace(" ", "_")}'] = best_match[col]
        else:
            match_data['IQ_Project_Name'] = None
            match_data['Match_Status'] = 'No match found'
        
        matches.append(match_data)
    
    return pd.DataFrame(matches)

def main():
    """Main function to cross-reference BESS resources with interconnection queue"""
    
    # Load BESS mapping
    print('Loading BESS resource mapping...')
    bess_mapping = pd.read_csv('/home/enrico/projects/power_market_pipeline/BESS_RESOURCE_MAPPING_REAL_ONLY.csv')
    print(f'Loaded {len(bess_mapping)} BESS resources')
    
    # Load interconnection queue
    iq_operating = load_interconnection_queue()
    
    if iq_operating.empty:
        print('Error: Could not load interconnection queue data')
        return
    
    # Match resources
    matched_df = match_bess_resources(bess_mapping, iq_operating)
    
    # Save results
    output_file = '/home/enrico/projects/power_market_pipeline/BESS_INTERCONNECTION_CROSS_REFERENCE.csv'
    matched_df.to_csv(output_file, index=False)
    print(f'\n✅ Saved cross-reference to: {output_file}')
    
    # Show statistics
    print('\n=== Matching Statistics ===')
    high_confidence = matched_df[matched_df['Match_Score'] > 50]
    medium_confidence = matched_df[(matched_df['Match_Score'] > 30) & (matched_df['Match_Score'] <= 50)]
    no_match = matched_df[matched_df['Match_Score'] <= 30]
    
    print(f'High confidence matches (>50): {len(high_confidence)} ({100*len(high_confidence)/len(matched_df):.1f}%)')
    print(f'Medium confidence matches (30-50): {len(medium_confidence)} ({100*len(medium_confidence)/len(matched_df):.1f}%)')
    print(f'No match found (<30): {len(no_match)} ({100*len(no_match)/len(matched_df):.1f}%)')
    
    # Show sample matches
    if len(high_confidence) > 0:
        print('\n=== Sample High Confidence Matches ===')
        sample_cols = ['BESS_Gen_Resource', 'Substation', 'IQ_Project_Name', 'IQ_County', 'Match_Score', 'Match_Reason']
        available_cols = [col for col in sample_cols if col in high_confidence.columns]
        print(high_confidence[available_cols].head(10).to_string(index=False))
    
    # Show unmatched resources for manual review
    if len(no_match) > 0:
        print('\n=== BESS Resources Needing Manual Review ===')
        print(no_match[['BESS_Gen_Resource', 'Substation', 'Load_Zone']].head(20).to_string(index=False))
    
    return matched_df

if __name__ == '__main__':
    matched_data = main()