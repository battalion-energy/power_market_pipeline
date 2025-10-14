#!/usr/bin/env python3
"""
Enhanced BESS matching with two passes:
Pass 1: Heuristic matching logic
Pass 2: LLM-based matching via OpenRouter for difficult cases
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
from difflib import SequenceMatcher
import os
import requests
import json
from dotenv import load_dotenv
import time
import warnings
warnings.filterwarnings('ignore')

# Load environment variables from battalion-platform
load_dotenv('/home/enrico/projects/battalion-platform/.env')

def normalize_name(name):
    """Normalize resource names for comparison"""
    if pd.isna(name):
        return ""
    name = str(name).upper()
    name = name.replace('_', '')
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
    
    # Load co-located operational
    df_op = pd.read_csv(data_dir / 'co_located_operational.csv')
    battery_mask = (df_op['Fuel'].str.upper().str.contains('OTH|BATT|STOR|ESS', na=False) | 
                   df_op['Technology'].str.upper().str.contains('BA|BATT|STOR|ESS', na=False))
    all_data['operational'] = df_op[battery_mask].copy()
    
    # Load stand-alone batteries
    df_standalone = pd.read_csv(data_dir / 'stand_alone.csv')
    df_standalone = df_standalone[df_standalone['Project Name'].notna()]
    all_data['standalone'] = df_standalone
    
    # Load co-located with solar
    df_solar = pd.read_csv(data_dir / 'co_located_with_solar.csv')
    df_solar = df_solar[df_solar['Project Name'].notna()]
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

def match_with_heuristics(bess_resource, substation, county, iq_data):
    """Pass 1: Heuristic matching"""
    best_match = None
    best_score = 0
    best_source = None
    
    bess_norm = normalize_name(bess_resource)
    sub_norm = normalize_name(substation) if substation else ""
    
    # Check operational data first
    for _, row in iq_data['operational'].iterrows():
        score = 0
        reasons = []
        
        if pd.notna(row.get('Unit Code')):
            unit_code = str(row['Unit Code']).upper()
            if bess_resource == unit_code:
                score = 100
                reasons.append('exact Unit Code match')
            else:
                unit_norm = normalize_name(unit_code)
                similarity = calculate_similarity(bess_norm, unit_norm)
                if similarity > 0.8:
                    score = similarity * 80
                    reasons.append(f'Unit Code similarity: {similarity:.2f}')
        
        if pd.notna(row.get('County')) and county:
            if str(row['County']).upper() == county.upper():
                score += 10
                reasons.append('county match')
        
        if score > best_score:
            best_score = score
            best_match = row.to_dict()
            best_match['match_reason'] = ', '.join(reasons)
            best_source = 'Operational'
    
    # Check other sources if no good operational match
    if best_score < 50:
        for source_name, df in [('Standalone', iq_data['standalone']),
                                ('Solar Co-located', iq_data['solar_colocated']),
                                ('Wind Co-located', iq_data['wind_colocated'])]:
            
            for _, row in df.iterrows():
                score = 0
                reasons = []
                
                if pd.notna(row.get('Project Name')):
                    project_norm = normalize_name(row['Project Name'])
                    similarity = calculate_similarity(bess_norm, project_norm)
                    if similarity > 0.6:
                        score = similarity * 60
                        reasons.append(f'project name similarity: {similarity:.2f}')
                
                if pd.notna(row.get('POI Location')) and sub_norm:
                    poi_norm = normalize_name(row['POI Location'])
                    similarity = calculate_similarity(sub_norm, poi_norm)
                    if similarity > 0.7:
                        score += similarity * 30
                        reasons.append(f'POI similarity: {similarity:.2f}')
                
                if pd.notna(row.get('County')) and county:
                    if str(row['County']).upper() == county.upper():
                        score += 10
                        reasons.append('county match')
                
                if score > best_score:
                    best_score = score
                    best_match = row.to_dict()
                    best_match['match_reason'] = ', '.join(reasons)
                    best_source = source_name
    
    if best_match and best_score > 40:
        best_match['match_score'] = best_score
        best_match['Source'] = best_source
        return best_match
    
    return None

def call_llm_for_matching(bess_info, candidate_projects, api_key):
    """Use LLM via OpenRouter to match difficult cases"""
    
    # Prepare enhanced prompt with more context
    prompt = f"""You are an expert at matching ERCOT battery energy storage system (BESS) resources with interconnection queue projects. You understand Texas power grid naming conventions and abbreviations.

TASK: Match this BESS resource to the most likely interconnection queue project.

BESS RESOURCE TO MATCH:
- Resource Name: {bess_info['resource']}
- Substation: {bess_info.get('substation', 'Unknown')}
- Load Zone: {bess_info.get('load_zone', 'Unknown')} (Texas ERCOT zone)
- Estimated County: {bess_info.get('county', 'Unknown')}

Key patterns to know:
- BESS resource names often use abbreviations (e.g., BRP_PBL1 might be "Broad Reach Power Battery Line 1")
- "_UNIT1", "_BES1", "_BESS1" suffixes indicate battery unit numbers
- Substations often match or abbreviate project locations
- Projects may use developer names (e.g., BRP = Broad Reach Power)

CANDIDATE INTERCONNECTION QUEUE PROJECTS:
"""
    
    for i, proj in enumerate(candidate_projects[:15], 1):  # Increased to 15 candidates
        proj_info = f"\n{i}. "
        details = []
        
        if 'Unit Code' in proj and pd.notna(proj.get('Unit Code')):
            details.append(f"Unit Code: {proj.get('Unit Code')}")
        if 'Unit Name' in proj and pd.notna(proj.get('Unit Name')):
            details.append(f"Unit Name: {proj.get('Unit Name')}")
        if 'Project Name' in proj and pd.notna(proj.get('Project Name')):
            details.append(f"Project: {proj.get('Project Name')}")
        if 'IQ_Entity' in proj and pd.notna(proj.get('IQ_Entity')):
            details.append(f"Entity: {proj.get('IQ_Entity')}")
        if 'County' in proj and pd.notna(proj.get('County')):
            details.append(f"County: {proj.get('County')}")
        if 'POI Location' in proj and pd.notna(proj.get('POI Location')):
            details.append(f"POI: {proj.get('POI Location')}")
        if 'IQ_POI' in proj and pd.notna(proj.get('IQ_POI')):
            details.append(f"POI: {proj.get('IQ_POI')}")
            
        capacity = proj.get('Capacity (MW)', proj.get('Capacity (MW)*', proj.get('IQ_Capacity_MW')))
        if capacity and pd.notna(capacity):
            details.append(f"Capacity: {capacity} MW")
            
        proj_info += " | ".join(details)
        prompt += proj_info
    
    prompt += """

MATCHING CRITERIA (in order of importance):
1. Direct name matches or clear abbreviation expansions
2. County must match or be adjacent if location data is available
3. Substation/POI alignment (consider abbreviations and variations)
4. Developer/Entity patterns (e.g., BRP projects, Key Capture projects)
5. Capacity and technical characteristics if relevant

Consider these common Texas BESS naming patterns:
- Geographic names often abbreviated (SAN_ANG = San Angelo, BRAZ = Brazoria)
- Developer prefixes (BRP = Broad Reach Power, KEY = Key Capture Energy)
- Technical suffixes (_ESS = Energy Storage System, _BESS = Battery ESS, _BES = Battery Energy Storage)

RESPONSE FORMAT:
Return ONLY a JSON object with your analysis:
{"match_index": <1-15 or 0>, "confidence": <0-100>, "reasoning": "<concise explanation of match logic>"}

Example good match: {"match_index": 3, "confidence": 85, "reasoning": "BRP_PBL1 matches Broad Reach Power Pablo 1 project - developer prefix and location align"}
Example no match: {"match_index": 0, "confidence": 0, "reasoning": "No projects match location or naming pattern"}"""

    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "openai/gpt-4o",  # Most advanced available model for complex matching
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2,
                "max_tokens": 300
            },
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            # Clean up response - remove markdown code blocks if present
            if '```json' in content:
                content = content.split('```json')[-1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].strip()
            
            # Parse JSON response
            try:
                match_data = json.loads(content)
                if match_data['match_index'] > 0 and match_data['match_index'] <= len(candidate_projects):
                    matched_project = candidate_projects[match_data['match_index'] - 1]
                    matched_project['match_score'] = match_data['confidence']
                    matched_project['match_reason'] = f"LLM: {match_data['reasoning']}"
                    matched_project['Source'] = 'LLM Match'
                    return matched_project
            except json.JSONDecodeError as e:
                print(f"  Failed to parse LLM response: {e}")
                
        else:
            print(f"  LLM API error: {response.status_code}")
            
    except Exception as e:
        print(f"  LLM matching error: {e}")
    
    return None

def get_county_from_substation(substation):
    """Map substation to county"""
    county_mappings = {
        'ALVIN': 'BRAZORIA',
        'ANCHOR': 'EASTLAND',
        'ANGLETON': 'BRAZORIA',
        'BATCAVE': 'MEDINA',
        'BRAZORIA': 'BRAZORIA',
        'CAMERON': 'CAMERON',
        'COMAL': 'COMAL',
        'DCSES': 'HOOD',
        'DICKNSON': 'GALVESTON',
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
    """Main function with two-pass matching"""
    
    # Get API key
    api_key = os.getenv('OPENROUTER_API_KEY')
    use_llm = api_key is not None
    
    if use_llm:
        print('✅ OpenRouter API key found - will use LLM for difficult matches')
    else:
        print('⚠️  No OpenRouter API key found - using heuristic matching only')
        print('   To enable LLM matching, add OPENROUTER_API_KEY to .env file')
    
    # Load BESS mapping
    print('\nLoading BESS resource mapping...')
    bess_mapping = pd.read_csv('/home/enrico/projects/power_market_pipeline/BESS_RESOURCE_MAPPING_REAL_ONLY.csv')
    print(f'Loaded {len(bess_mapping)} BESS resources')
    
    # Load interconnection data
    iq_data = load_all_interconnection_data()
    
    print('\n=== Pass 1: Heuristic Matching ===\n')
    
    matches = []
    unmatched_resources = []
    
    for idx, bess_row in bess_mapping.iterrows():
        gen_resource = bess_row['BESS_Gen_Resource']
        substation = bess_row.get('Substation')
        county = get_county_from_substation(substation)
        
        # Pass 1: Heuristic matching
        match = match_with_heuristics(gen_resource, substation, county, iq_data)
        
        if match:
            match['BESS_Gen_Resource'] = gen_resource
            match['BESS_Load_Resource'] = bess_row.get('BESS_Load_Resource')
            match['Settlement_Point'] = bess_row.get('Settlement_Point')
            match['Substation'] = substation
            match['Load_Zone'] = bess_row.get('Load_Zone')
            match['Estimated_County'] = county
            matches.append(match)
        else:
            unmatched_resources.append({
                'resource': gen_resource,
                'substation': substation,
                'load_zone': bess_row.get('Load_Zone'),
                'county': county,
                'bess_row': bess_row
            })
        
        if (idx + 1) % 20 == 0:
            print(f'  Processed {idx + 1}/{len(bess_mapping)} resources...')
    
    print(f'\nPass 1 Results:')
    print(f'  Matched: {len(matches)}')
    print(f'  Unmatched: {len(unmatched_resources)}')
    
    # Pass 2: LLM matching for unmatched resources
    if use_llm and len(unmatched_resources) > 0:
        print(f'\n=== Pass 2: LLM Matching for {len(unmatched_resources)} unmatched resources ===\n')
        
        llm_matches = 0
        llm_attempts = 0
        
        # Combine all potential candidates
        all_candidates = []
        for source, df in iq_data.items():
            candidates = df.to_dict('records')
            for c in candidates:
                c['_source'] = source
            all_candidates.extend(candidates)
        
        # Limit to 10 for testing and cost control
        max_llm_attempts = min(10, len(unmatched_resources))
        for unmatched in unmatched_resources[:max_llm_attempts]:
            llm_attempts += 1
            print(f"  Attempting LLM match for {unmatched['resource']}...")
            
            # Get potential candidates (those with some similarity)
            bess_norm = normalize_name(unmatched['resource'])
            candidates = []
            
            for candidate in all_candidates:
                score = 0
                if 'Unit Code' in candidate and pd.notna(candidate['Unit Code']):
                    score = max(score, calculate_similarity(bess_norm, normalize_name(candidate['Unit Code'])))
                if 'Project Name' in candidate and pd.notna(candidate['Project Name']):
                    score = max(score, calculate_similarity(bess_norm, normalize_name(candidate['Project Name'])))
                
                if score > 0.3:  # Low threshold to get more candidates
                    candidate['_similarity'] = score
                    candidates.append(candidate)
            
            # Sort by similarity and take top candidates
            candidates = sorted(candidates, key=lambda x: x['_similarity'], reverse=True)
            
            if candidates:
                llm_match = call_llm_for_matching(unmatched, candidates[:10], api_key)
                
                if llm_match and llm_match.get('match_score', 0) > 50:
                    llm_match['BESS_Gen_Resource'] = unmatched['resource']
                    llm_match['BESS_Load_Resource'] = unmatched['bess_row'].get('BESS_Load_Resource')
                    llm_match['Settlement_Point'] = unmatched['bess_row'].get('Settlement_Point')
                    llm_match['Substation'] = unmatched['substation']
                    llm_match['Load_Zone'] = unmatched['load_zone']
                    llm_match['Estimated_County'] = unmatched['county']
                    matches.append(llm_match)
                    llm_matches += 1
                    print(f"    ✓ Found match with {llm_match['match_score']}% confidence")
                else:
                    print(f"    ✗ No suitable match found")
            
            # Rate limiting for API
            time.sleep(0.5)
        
        print(f'\nPass 2 Results:')
        print(f'  LLM attempts: {llm_attempts}')
        print(f'  New matches found: {llm_matches}')
    
    # Create final results DataFrame
    if matches:
        results_df = pd.DataFrame(matches)
        
        # Clean up columns
        standard_cols = ['BESS_Gen_Resource', 'BESS_Load_Resource', 'Settlement_Point', 
                        'Substation', 'Load_Zone', 'Estimated_County', 'match_score', 
                        'match_reason', 'Source']
        
        # Add IQ columns
        iq_cols = []
        for col in results_df.columns:
            if col not in standard_cols:
                iq_cols.append(col)
        
        # Rename IQ columns
        rename_map = {}
        for col in iq_cols:
            if not col.startswith('IQ_') and col not in standard_cols:
                rename_map[col] = f'IQ_{col}'
        
        results_df = results_df.rename(columns=rename_map)
        
        # Sort by match score
        results_df = results_df.sort_values('match_score', ascending=False)
    else:
        # Create empty DataFrame with expected columns
        results_df = pd.DataFrame(columns=['BESS_Gen_Resource', 'match_score'])
    
    # Add unmatched resources with zero score
    for unmatched in unmatched_resources[30 if use_llm else 0:]:
        results_df = pd.concat([results_df, pd.DataFrame([{
            'BESS_Gen_Resource': unmatched['resource'],
            'BESS_Load_Resource': unmatched['bess_row'].get('BESS_Load_Resource'),
            'Settlement_Point': unmatched['bess_row'].get('Settlement_Point'),
            'Substation': unmatched['substation'],
            'Load_Zone': unmatched['load_zone'],
            'Estimated_County': unmatched['county'],
            'match_score': 0,
            'match_reason': 'No match found',
            'Source': None
        }])], ignore_index=True)
    
    # Save results
    output_file = '/home/enrico/projects/power_market_pipeline/BESS_INTERCONNECTION_MATCHED_WITH_LLM.csv'
    results_df.to_csv(output_file, index=False)
    print(f'\n✅ Saved matched results to: {output_file}')
    
    # Show final statistics
    print('\n=== Final Matching Statistics ===')
    
    excellent = results_df[results_df['match_score'] >= 90]
    good = results_df[(results_df['match_score'] >= 70) & (results_df['match_score'] < 90)]
    fair = results_df[(results_df['match_score'] >= 50) & (results_df['match_score'] < 70)]
    poor = results_df[(results_df['match_score'] > 0) & (results_df['match_score'] < 50)]
    no_match = results_df[results_df['match_score'] == 0]
    
    total = len(results_df)
    print(f'Excellent matches (≥90): {len(excellent)} ({100*len(excellent)/total:.1f}%)')
    print(f'Good matches (70-90): {len(good)} ({100*len(good)/total:.1f}%)')
    print(f'Fair matches (50-70): {len(fair)} ({100*len(fair)/total:.1f}%)')
    print(f'Poor matches (<50): {len(poor)} ({100*len(poor)/total:.1f}%)')
    print(f'No match: {len(no_match)} ({100*len(no_match)/total:.1f}%)')
    
    if use_llm:
        llm_matches = results_df[results_df['Source'] == 'LLM Match']
        print(f'\nLLM contributions: {len(llm_matches)} matches')
    
    return results_df

if __name__ == '__main__':
    results = main()