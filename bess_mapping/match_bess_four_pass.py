#!/usr/bin/env python3
"""
Enhanced BESS matching with four passes:
Pass 1: Heuristic matching with operational data
Pass 2: LLM matching for difficult cases with operational data
Pass 3: Heuristic matching with Planned projects
Pass 4: LLM matching for remaining with Planned projects
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
    
    # Load stand-alone batteries (mostly Planned)
    df_standalone = pd.read_csv(data_dir / 'stand_alone.csv')
    df_standalone = df_standalone[df_standalone['Project Name'].notna()]
    all_data['standalone'] = df_standalone
    
    # Separate Planned projects for Pass 3/4
    all_data['planned'] = df_standalone[df_standalone['Project Status'] == 'Planned'].copy()
    
    # Load co-located with solar
    df_solar = pd.read_csv(data_dir / 'co_located_with_solar.csv')
    df_solar = df_solar[df_solar['Project Name'].notna()]
    battery_mask = (df_solar['Fuel'].str.upper().str.contains('OTH|BATT|STOR|ESS', na=False) | 
                   df_solar['Technology'].str.upper().str.contains('BA|BATT|STOR|ESS', na=False))
    all_data['solar_colocated'] = df_solar[battery_mask].copy()
    
    # Add planned solar co-located
    planned_solar = df_solar[(df_solar['Project Status'] == 'Planned') & battery_mask].copy()
    all_data['planned_solar'] = planned_solar
    
    # Load co-located with wind
    df_wind = pd.read_csv(data_dir / 'co_located_with_wind.csv')
    df_wind = df_wind[df_wind['Project Name'].notna()]
    battery_mask = (df_wind['Fuel'].str.upper().str.contains('OTH|BATT|STOR|ESS', na=False) | 
                   df_wind['Technology'].str.upper().str.contains('BA|BATT|STOR|ESS', na=False))
    all_data['wind_colocated'] = df_wind[battery_mask].copy()
    
    # Add planned wind co-located
    planned_wind = df_wind[(df_wind['Project Status'] == 'Planned') & battery_mask].copy()
    all_data['planned_wind'] = planned_wind
    
    print('Loaded interconnection data:')
    for key, df in all_data.items():
        print(f'  {key}: {len(df)} projects')
    
    return all_data

def match_with_heuristics(bess_resource, substation, county, iq_data, data_sources):
    """Heuristic matching with specified data sources"""
    best_match = None
    best_score = 0
    best_source = None
    
    bess_norm = normalize_name(bess_resource)
    sub_norm = normalize_name(substation) if substation else ""
    
    for source_name in data_sources:
        if source_name not in iq_data:
            continue
            
        df = iq_data[source_name]
        
        # Check if it's operational data (has Unit Code)
        if 'Unit Code' in df.columns:
            for _, row in df.iterrows():
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
                    best_source = source_name
        
        # For planned/other data (has Project Name)
        else:
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
        best_match['Source'] = best_source.title()
        return best_match
    
    return None

def call_llm_for_matching(bess_info, candidate_projects, api_key, focus_planned=False):
    """Enhanced LLM matching with planned project awareness"""
    
    # Enhanced prompt for planned projects
    context_addition = ""
    if focus_planned:
        context_addition = """
Note: These are PLANNED projects (not yet operational), so:
- Project names may be preliminary or working titles
- Developer names may differ from final operator names
- Consider development timelines (2025-2028 COD typical)
- Location matching is especially important for new projects
"""
    
    prompt = f"""You are an expert at matching ERCOT battery energy storage system (BESS) resources with interconnection queue projects.

TASK: Match this BESS resource to the most likely interconnection queue project.

BESS RESOURCE TO MATCH:
- Resource Name: {bess_info['resource']}
- Substation: {bess_info.get('substation', 'Unknown')}
- Load Zone: {bess_info.get('load_zone', 'Unknown')} (Texas ERCOT zone)
- Estimated County: {bess_info.get('county', 'Unknown')}

{context_addition}

Key patterns to know:
- BESS resource names often use abbreviations
- "_UNIT1", "_BES1", "_BESS1" suffixes indicate battery unit numbers
- Substations often match or abbreviate project locations
- Common developer prefixes: BRP (Broad Reach Power), KEY (Key Capture), SMT, PP, etc.

CANDIDATE INTERCONNECTION QUEUE PROJECTS:
"""
    
    for i, proj in enumerate(candidate_projects[:15], 1):
        proj_info = f"\n{i}. "
        details = []
        
        # Include all relevant fields
        for field in ['Unit Code', 'Unit Name', 'Project Name', 'Interconnecting Entity', 
                      'County', 'POI Location', 'Project Status', 'Projected COD']:
            if field in proj and pd.notna(proj.get(field)):
                details.append(f"{field}: {proj.get(field)}")
        
        capacity = proj.get('Capacity (MW)', proj.get('Capacity (MW)*', proj.get('IQ_Capacity_MW')))
        if capacity and pd.notna(capacity):
            details.append(f"Capacity: {capacity} MW")
            
        proj_info += " | ".join(details)
        prompt += proj_info
    
    prompt += """

MATCHING CRITERIA (in order of importance):
1. Direct name matches or clear abbreviation expansions
2. County must match or be adjacent if location data available
3. Substation/POI alignment (consider variations)
4. Developer/Entity patterns
5. For planned projects, consider timeline alignment

Common Texas BESS naming patterns:
- Geographic abbreviations (SAN_ANG = San Angelo, BRAZ = Brazoria, etc.)
- Developer codes (BRP projects often use coded names)
- Technical suffixes (_ESS, _BESS, _BES)
- Military/aviation themes (SWOOSE, etc.)

RESPONSE FORMAT:
Return ONLY a JSON object:
{"match_index": <1-15 or 0>, "confidence": <0-100>, "reasoning": "<concise explanation>"}"""

    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "openai/gpt-4o",
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
            
            # Clean up response
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
                    matched_project['Source'] = 'LLM Match (Planned)' if focus_planned else 'LLM Match'
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
        'DICKNSON': 'GALVESTON',
        'HEIGHTTN': 'HARRIS',  # Heights/Houston area
        'LOPENO': 'STARR',     # Lopeño
        'MADERO': 'HIDALGO',   # Madero area
        'SWEENY': 'BRAZORIA',
        'SWOOSE': 'WARD',      # Military theme
        'BRP': 'VARIOUS',      # Broad Reach Power - multiple locations
    }
    
    if not substation:
        return None
    
    sub_upper = str(substation).upper()
    for key, county in county_mappings.items():
        if key in sub_upper:
            return county
    
    return None

def main():
    """Main function with four-pass matching"""
    
    # Get API key
    api_key = os.getenv('OPENROUTER_API_KEY')
    use_llm = api_key is not None
    
    if use_llm:
        print('✅ OpenRouter API key found - will use LLM for difficult matches')
    else:
        print('⚠️  No OpenRouter API key found - using heuristic matching only')
    
    # Load BESS mapping
    print('\nLoading BESS resource mapping...')
    bess_mapping = pd.read_csv('/home/enrico/projects/power_market_pipeline/BESS_RESOURCE_MAPPING_REAL_ONLY.csv')
    print(f'Loaded {len(bess_mapping)} BESS resources')
    
    # Load interconnection data
    iq_data = load_all_interconnection_data()
    
    matches = []
    unmatched_resources = []
    
    # PASS 1: Heuristic matching with operational data
    print('\n=== Pass 1: Heuristic Matching (Operational) ===\n')
    
    for idx, bess_row in bess_mapping.iterrows():
        gen_resource = bess_row['BESS_Gen_Resource']
        substation = bess_row.get('Substation')
        county = get_county_from_substation(substation)
        
        match = match_with_heuristics(gen_resource, substation, county, iq_data, 
                                     ['operational', 'standalone', 'solar_colocated', 'wind_colocated'])
        
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
    
    # PASS 2: LLM matching for unmatched (operational focus)
    if use_llm and len(unmatched_resources) > 0:
        print(f'\n=== Pass 2: LLM Matching (Operational) ===\n')
        
        # Combine operational candidates
        operational_candidates = []
        for source in ['operational', 'standalone', 'solar_colocated', 'wind_colocated']:
            if source in iq_data:
                candidates = iq_data[source].to_dict('records')
                for c in candidates:
                    c['_source'] = source
                operational_candidates.extend(candidates)
        
        llm_matches_p2 = 0
        max_attempts = min(10, len(unmatched_resources))
        
        for i, unmatched in enumerate(unmatched_resources[:max_attempts]):
            print(f"  Attempting LLM match for {unmatched['resource']}...")
            
            # Get candidates with similarity
            bess_norm = normalize_name(unmatched['resource'])
            candidates = []
            
            for candidate in operational_candidates:
                score = 0
                if 'Unit Code' in candidate and pd.notna(candidate['Unit Code']):
                    score = max(score, calculate_similarity(bess_norm, normalize_name(candidate['Unit Code'])))
                if 'Project Name' in candidate and pd.notna(candidate['Project Name']):
                    score = max(score, calculate_similarity(bess_norm, normalize_name(candidate['Project Name'])))
                
                if score > 0.3:
                    candidate['_similarity'] = score
                    candidates.append(candidate)
            
            candidates = sorted(candidates, key=lambda x: x['_similarity'], reverse=True)
            
            if candidates:
                llm_match = call_llm_for_matching(unmatched, candidates[:15], api_key, focus_planned=False)
                
                if llm_match and llm_match.get('match_score', 0) > 50:
                    llm_match['BESS_Gen_Resource'] = unmatched['resource']
                    llm_match['BESS_Load_Resource'] = unmatched['bess_row'].get('BESS_Load_Resource')
                    llm_match['Settlement_Point'] = unmatched['bess_row'].get('Settlement_Point')
                    llm_match['Substation'] = unmatched['substation']
                    llm_match['Load_Zone'] = unmatched['load_zone']
                    llm_match['Estimated_County'] = unmatched['county']
                    matches.append(llm_match)
                    unmatched_resources.remove(unmatched)
                    llm_matches_p2 += 1
                    print(f"    ✓ Found match with {llm_match['match_score']}% confidence")
                else:
                    print(f"    ✗ No suitable match found")
            
            time.sleep(0.5)  # Rate limiting
        
        print(f'\nPass 2 Results:')
        print(f'  New matches found: {llm_matches_p2}')
        print(f'  Remaining unmatched: {len(unmatched_resources)}')
    
    # PASS 3: Heuristic matching with Planned projects
    print(f'\n=== Pass 3: Heuristic Matching (Planned Projects) ===\n')
    
    still_unmatched = []
    heuristic_planned_matches = 0
    
    for unmatched in unmatched_resources:
        match = match_with_heuristics(unmatched['resource'], unmatched['substation'], 
                                     unmatched['county'], iq_data, 
                                     ['planned', 'planned_solar', 'planned_wind'])
        
        if match and match.get('match_score', 0) > 50:
            match['BESS_Gen_Resource'] = unmatched['resource']
            match['BESS_Load_Resource'] = unmatched['bess_row'].get('BESS_Load_Resource')
            match['Settlement_Point'] = unmatched['bess_row'].get('Settlement_Point')
            match['Substation'] = unmatched['substation']
            match['Load_Zone'] = unmatched['load_zone']
            match['Estimated_County'] = unmatched['county']
            match['Source'] = f"Planned ({match['Source']})"
            matches.append(match)
            heuristic_planned_matches += 1
        else:
            still_unmatched.append(unmatched)
    
    print(f'Pass 3 Results:')
    print(f'  New matches found: {heuristic_planned_matches}')
    print(f'  Remaining unmatched: {len(still_unmatched)}')
    
    # PASS 4: LLM matching with Planned projects
    if use_llm and len(still_unmatched) > 0:
        print(f'\n=== Pass 4: LLM Matching (Planned Projects) ===\n')
        
        # Get planned candidates
        planned_candidates = []
        for source in ['planned', 'planned_solar', 'planned_wind']:
            if source in iq_data:
                candidates = iq_data[source].to_dict('records')
                for c in candidates:
                    c['_source'] = source
                planned_candidates.extend(candidates)
        
        llm_matches_p4 = 0
        max_attempts = min(15, len(still_unmatched))
        
        for i, unmatched in enumerate(still_unmatched[:max_attempts]):
            print(f"  Attempting LLM match for {unmatched['resource']} (Planned)...")
            
            # Get candidates
            bess_norm = normalize_name(unmatched['resource'])
            candidates = []
            
            for candidate in planned_candidates:
                score = 0
                if 'Project Name' in candidate and pd.notna(candidate['Project Name']):
                    score = calculate_similarity(bess_norm, normalize_name(candidate['Project Name']))
                
                if score > 0.25:  # Lower threshold for planned projects
                    candidate['_similarity'] = score
                    candidates.append(candidate)
            
            candidates = sorted(candidates, key=lambda x: x['_similarity'], reverse=True)
            
            if candidates:
                llm_match = call_llm_for_matching(unmatched, candidates[:15], api_key, focus_planned=True)
                
                if llm_match and llm_match.get('match_score', 0) > 40:  # Lower threshold for planned
                    llm_match['BESS_Gen_Resource'] = unmatched['resource']
                    llm_match['BESS_Load_Resource'] = unmatched['bess_row'].get('BESS_Load_Resource')
                    llm_match['Settlement_Point'] = unmatched['bess_row'].get('Settlement_Point')
                    llm_match['Substation'] = unmatched['substation']
                    llm_match['Load_Zone'] = unmatched['load_zone']
                    llm_match['Estimated_County'] = unmatched['county']
                    still_unmatched.remove(unmatched)
                    matches.append(llm_match)
                    llm_matches_p4 += 1
                    print(f"    ✓ Found match with {llm_match['match_score']}% confidence")
                else:
                    print(f"    ✗ No suitable match found")
            
            time.sleep(0.5)
        
        print(f'\nPass 4 Results:')
        print(f'  New matches found: {llm_matches_p4}')
        print(f'  Final unmatched: {len(still_unmatched)}')
    
    # Create final results DataFrame
    if matches:
        results_df = pd.DataFrame(matches)
        
        # Add unmatched with zero score
        for unmatched in still_unmatched:
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
        
        # Sort by match score
        results_df = results_df.sort_values('match_score', ascending=False)
    else:
        results_df = pd.DataFrame()
    
    # Save results
    output_file = '/home/enrico/projects/power_market_pipeline/BESS_FOUR_PASS_MATCHED.csv'
    results_df.to_csv(output_file, index=False)
    print(f'\n✅ Saved matched results to: {output_file}')
    
    # Show final statistics
    print('\n=== Final Four-Pass Matching Statistics ===')
    
    if len(results_df) > 0:
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
        
        # Show breakdown by source
        print('\n=== Matches by Source ===')
        source_counts = results_df[results_df['match_score'] > 0]['Source'].value_counts()
        for source, count in source_counts.items():
            print(f'{source}: {count}')
    
    return results_df

if __name__ == '__main__':
    results = main()