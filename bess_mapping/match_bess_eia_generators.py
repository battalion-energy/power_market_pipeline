#!/usr/bin/env python3
"""
Cross-reference BESS resources with EIA Generator monthly export
Four-pass matching: Heuristic Operating, LLM Operating, Heuristic Planned, LLM Planned
ONLY REAL DATA - NO FAKE MATCHES
"""

import pandas as pd
import numpy as np
from rapidfuzz import fuzz, process
import json
import os
from dotenv import load_dotenv
import requests
import time

# Load environment
load_dotenv('/home/enrico/projects/battalion-platform/.env')

def load_eia_data(sheet_name='Operating'):
    """Load EIA generator data from Excel"""
    file_path = '/home/enrico/projects/battalion-platform/data/EIA/generators/EIA_generators_latest.xlsx'
    
    # Read with correct header row (row 2)
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=2)
    
    # Filter for Texas battery storage
    df_tx = df[df['Plant State'] == 'TX'].copy()
    
    # Filter for battery/energy storage
    # Check Technology column and Energy Source Code
    df_bess = df_tx[
        (df_tx['Technology'].str.contains('Battery', na=False)) |
        (df_tx['Technology'].str.contains('Energy Storage', na=False)) |
        (df_tx['Energy Source Code'].str.contains('MWH', na=False)) |  # MWH is the code for battery storage
        (df_tx['Prime Mover Code'].str.contains('BA', na=False))  # BA is battery
    ].copy()
    
    print(f"Loaded {len(df_bess)} TX battery storage facilities from {sheet_name} tab")
    
    # Clean up names
    df_bess['Plant Name Clean'] = df_bess['Plant Name'].str.upper().str.replace(' ', '_')
    df_bess['Generator ID Clean'] = df_bess['Generator ID'].astype(str).str.upper()
    
    return df_bess

def load_bess_resources():
    """Load BESS resources from improved matched data"""
    df = pd.read_csv('/home/enrico/projects/power_market_pipeline/BESS_IMPROVED_MATCHED.csv')
    
    # Also load the resource mapping to get more info
    mapping_df = pd.read_csv('/home/enrico/projects/power_market_pipeline/BESS_RESOURCE_MAPPING_REAL_ONLY.csv')
    
    # Merge to get substation and load zone info
    df = df.merge(
        mapping_df[['BESS_Gen_Resource', 'Substation', 'Load_Zone']],
        on='BESS_Gen_Resource',
        how='left',
        suffixes=('', '_mapping')
    )
    
    # Use mapping data to fill missing values
    for col in ['Substation', 'Load_Zone']:
        if f'{col}_mapping' in df.columns:
            df[col] = df[col].fillna(df[f'{col}_mapping'])
            df = df.drop(columns=[f'{col}_mapping'])
    
    # Load county data from substation mapping if available
    try:
        substation_df = pd.read_csv('/home/enrico/projects/power_market_pipeline/substation_to_county_mapping.csv')
        df = df.merge(
            substation_df[['Substation', 'County']],
            on='Substation',
            how='left',
            suffixes=('', '_sub')
        )
        if 'County_sub' in df.columns:
            df['County'] = df['County'].fillna(df['County_sub'])
            df = df.drop(columns=['County_sub'])
    except:
        df['County'] = None
    
    return df

def normalize_county_name(county):
    """Normalize county names for matching"""
    if pd.isna(county):
        return ''
    
    # Remove 'County' suffix and standardize
    county = str(county).upper().replace(' COUNTY', '').replace(' CO.', '').strip()
    return county

def heuristic_match(bess_df, eia_df, pass_name="Heuristic"):
    """Heuristic matching based on name similarity and county"""
    matches = []
    
    for _, bess_row in bess_df.iterrows():
        bess_name = bess_row['BESS_Gen_Resource']
        bess_county = normalize_county_name(bess_row.get('County'))
        
        best_match = None
        best_score = 0
        best_reason = ""
        
        # Try different matching strategies
        for _, eia_row in eia_df.iterrows():
            eia_plant = eia_row['Plant Name Clean']
            eia_gen_id = eia_row['Generator ID Clean']
            eia_county = normalize_county_name(eia_row.get('County'))
            
            # Calculate name similarity
            plant_similarity = fuzz.ratio(bess_name, eia_plant) / 100
            
            # Also check generator ID similarity
            gen_similarity = fuzz.ratio(bess_name, eia_gen_id) / 100
            
            # Use the better similarity
            name_similarity = max(plant_similarity, gen_similarity)
            
            # County match bonus
            county_match = 1.0 if (bess_county and eia_county and bess_county == eia_county) else 0
            
            # Calculate combined score
            if county_match and name_similarity > 0.4:
                # Strong match with county verification
                score = 50 + (name_similarity * 50)
                reason = f"name similarity: {name_similarity:.2f}, county match: {bess_county}"
            elif name_similarity > 0.7:
                # Good name match without county
                score = name_similarity * 80
                reason = f"strong name similarity: {name_similarity:.2f}"
            elif name_similarity > 0.5 and county_match:
                # Moderate name match with county
                score = 40 + (name_similarity * 40)
                reason = f"name similarity: {name_similarity:.2f}, county match: {bess_county}"
            else:
                score = 0
            
            if score > best_score:
                best_score = score
                best_match = eia_row
                best_reason = reason
        
        if best_score >= 40:  # Minimum threshold
            matches.append({
                'BESS_Gen_Resource': bess_name,
                'EIA_Plant_Name': best_match['Plant Name'],
                'EIA_Generator_ID': best_match['Generator ID'],
                'EIA_County': best_match.get('County'),
                'EIA_Technology': best_match.get('Technology'),
                'EIA_Capacity_MW': best_match.get('Nameplate Capacity (MW)'),
                'EIA_Operating_Year': best_match.get('Operating Year'),
                'match_score': best_score,
                'match_reason': best_reason,
                'Pass': pass_name,
                'Source': 'EIA ' + eia_df.name if hasattr(eia_df, 'name') else 'EIA'
            })
    
    return pd.DataFrame(matches)

def llm_match(bess_df, eia_df, pass_name="LLM", sheet_type="Operating"):
    """LLM-based matching for difficult cases"""
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("Warning: OPENROUTER_API_KEY not found")
        return pd.DataFrame()
    
    matches = []
    
    # Prepare EIA data summary for context
    eia_summary = []
    for _, row in eia_df.iterrows():
        eia_summary.append({
            'plant_name': row['Plant Name'],
            'generator_id': row['Generator ID'],
            'county': row.get('County', 'Unknown'),
            'technology': row.get('Technology', 'Battery'),
            'capacity_mw': row.get('Nameplate Capacity (MW)', 0),
            'year': row.get('Operating Year' if sheet_type == 'Operating' else 'Planned Operation Year', 'Unknown')
        })
    
    # Process each unmatched BESS
    total_to_process = len(bess_df)
    for idx, (_, bess_row) in enumerate(bess_df.iterrows(), 1):
        if idx % 10 == 0:
            print(f"    Processing {idx}/{total_to_process}...")
        bess_name = bess_row['BESS_Gen_Resource']
        bess_county = bess_row.get('County', 'Unknown')
        bess_substation = bess_row.get('Substation', 'Unknown')
        
        prompt = f"""You are matching Texas BESS resources to EIA generator data.

BESS Resource to match:
- Name: {bess_name}
- County: {bess_county}
- Substation: {bess_substation}

Available EIA {sheet_type} Battery Facilities in Texas:
{json.dumps(eia_summary[:50], indent=2)}  # Limit to first 50 to avoid token limits

Match the BESS resource to the most likely EIA facility based on:
1. Name similarity (consider abbreviations, variations)
2. County location (must be same or adjacent counties)
3. Capacity if known
4. Developer patterns (e.g., BRP_ prefix = Broad Reach Power)

Texas-specific knowledge:
- SWOOSE projects are in Ward/Reeves counties
- BRP projects are Broad Reach Power
- MADERO/LOPENO are Rio Grande Valley (Hidalgo/Starr counties)
- HEIGHTTN is Houston area (Harris County)

Return ONLY a JSON object (no markdown, no explanation):
{{
  "matched": true/false,
  "plant_name": "exact EIA plant name if matched",
  "generator_id": "exact EIA generator ID if matched",
  "confidence": 0-100,
  "reason": "brief explanation"
}}

If no good match exists, return {{"matched": false, "confidence": 0}}
"""
        
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "HTTP-Referer": "https://github.com/enricoaquilina/power-market-pipeline",
                    "X-Title": "BESS EIA Matching"
                },
                json={
                    "model": "openai/gpt-4o",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 500
                }
            )
            
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content']
                
                # Clean JSON response
                if '```json' in content:
                    content = content.split('```json')[-1].split('```')[0]
                content = content.strip()
                
                result = json.loads(content)
                
                if result.get('matched') and result.get('confidence', 0) >= 70:
                    # Find the exact EIA row
                    eia_match = eia_df[
                        (eia_df['Plant Name'] == result['plant_name']) |
                        (eia_df['Generator ID'].astype(str) == str(result.get('generator_id', '')))
                    ]
                    
                    if not eia_match.empty:
                        eia_row = eia_match.iloc[0]
                        matches.append({
                            'BESS_Gen_Resource': bess_name,
                            'EIA_Plant_Name': eia_row['Plant Name'],
                            'EIA_Generator_ID': eia_row['Generator ID'],
                            'EIA_County': eia_row.get('County'),
                            'EIA_Technology': eia_row.get('Technology'),
                            'EIA_Capacity_MW': eia_row.get('Nameplate Capacity (MW)'),
                            'EIA_Operating_Year': eia_row.get('Operating Year' if sheet_type == 'Operating' else 'Planned Operation Year'),
                            'match_score': result['confidence'],
                            'match_reason': result.get('reason', 'LLM match'),
                            'Pass': pass_name,
                            'Source': f'EIA {sheet_type} (LLM)'
                        })
            
            # Rate limiting
            time.sleep(0.5)
            
        except Exception as e:
            print(f"LLM matching error for {bess_name}: {e}")
            continue
    
    return pd.DataFrame(matches)

def main(skip_llm=False):
    """Run the four-pass matching process"""
    
    print("="*70)
    print("BESS to EIA Generator Cross-Reference")
    print("="*70)
    
    # Load data
    print("\n📂 Loading data...")
    bess_df = load_bess_resources()
    eia_operating = load_eia_data('Operating')
    eia_planned = load_eia_data('Planned')
    
    print(f"\nTotal BESS resources to match: {len(bess_df)}")
    
    # Initialize results
    all_matches = []
    unmatched = set(bess_df['BESS_Gen_Resource'].unique())
    
    # Pass 1: Heuristic matching on Operating
    print("\n🔄 Pass 1: Heuristic matching on Operating tab...")
    print(f"  Processing {len(unmatched)} BESS resources against {len(eia_operating)} EIA facilities...")
    pass1_matches = heuristic_match(
        bess_df[bess_df['BESS_Gen_Resource'].isin(unmatched)],
        eia_operating,
        "Pass 1 (Operating Heuristic)"
    )
    if not pass1_matches.empty:
        all_matches.append(pass1_matches)
        unmatched -= set(pass1_matches['BESS_Gen_Resource'])
        print(f"  ✅ Matched: {len(pass1_matches)} (Remaining: {len(unmatched)})")
    else:
        print(f"  No matches found")
    
    # Pass 2: LLM matching on Operating (skip if requested)
    if not skip_llm:
        print("\n🤖 Pass 2: LLM matching on Operating tab...")
        if len(unmatched) > 0:
            print(f"  Processing {len(unmatched)} unmatched BESS resources...")
            pass2_matches = llm_match(
                bess_df[bess_df['BESS_Gen_Resource'].isin(unmatched)],
                eia_operating,
                "Pass 2 (Operating LLM)",
                "Operating"
            )
            if not pass2_matches.empty:
                all_matches.append(pass2_matches)
                unmatched -= set(pass2_matches['BESS_Gen_Resource'])
                print(f"  ✅ Matched: {len(pass2_matches)} (Remaining: {len(unmatched)})")
            else:
                print(f"  No matches found")
    else:
        print("\n⏭️  Skipping LLM passes (--skip-llm flag)")
    
    # Pass 3: Heuristic matching on Planned
    print("\n🔄 Pass 3: Heuristic matching on Planned tab...")
    if len(unmatched) > 0:
        print(f"  Processing {len(unmatched)} BESS resources against {len(eia_planned)} EIA planned facilities...")
        pass3_matches = heuristic_match(
            bess_df[bess_df['BESS_Gen_Resource'].isin(unmatched)],
            eia_planned,
            "Pass 3 (Planned Heuristic)"
        )
        if not pass3_matches.empty:
            all_matches.append(pass3_matches)
            unmatched -= set(pass3_matches['BESS_Gen_Resource'])
            print(f"  ✅ Matched: {len(pass3_matches)} (Remaining: {len(unmatched)})")
        else:
            print(f"  No matches found")
    
    # Pass 4: LLM matching on Planned (skip if requested)
    if not skip_llm:
        print("\n🤖 Pass 4: LLM matching on Planned tab...")
        if len(unmatched) > 0:
            print(f"  Processing {len(unmatched)} unmatched BESS resources...")
            pass4_matches = llm_match(
                bess_df[bess_df['BESS_Gen_Resource'].isin(unmatched)],
                eia_planned,
                "Pass 4 (Planned LLM)",
                "Planned"
            )
            if not pass4_matches.empty:
                all_matches.append(pass4_matches)
                unmatched -= set(pass4_matches['BESS_Gen_Resource'])
                print(f"  ✅ Matched: {len(pass4_matches)} (Remaining: {len(unmatched)})")
            else:
                print(f"  No matches found")
    
    # Combine all matches
    if all_matches:
        final_matches = pd.concat(all_matches, ignore_index=True)
    else:
        final_matches = pd.DataFrame()
    
    # Add unmatched resources
    for bess_name in unmatched:
        bess_info = bess_df[bess_df['BESS_Gen_Resource'] == bess_name].iloc[0]
        final_matches = pd.concat([final_matches, pd.DataFrame([{
            'BESS_Gen_Resource': bess_name,
            'EIA_Plant_Name': None,
            'EIA_Generator_ID': None,
            'EIA_County': None,
            'EIA_Technology': None,
            'EIA_Capacity_MW': None,
            'EIA_Operating_Year': None,
            'match_score': 0,
            'match_reason': 'No match found',
            'Pass': 'Unmatched',
            'Source': None
        }])], ignore_index=True)
    
    # Save results
    output_file = '/home/enrico/projects/power_market_pipeline/BESS_EIA_MATCHED.csv'
    final_matches.to_csv(output_file, index=False)
    
    # Summary
    print("\n" + "="*70)
    print("MATCHING SUMMARY")
    print("="*70)
    
    total = len(bess_df)
    matched = len(final_matches[final_matches['match_score'] > 0])
    unmatched_count = total - matched
    
    print(f"\nTotal BESS resources: {total}")
    print(f"Matched to EIA: {matched} ({100*matched/total:.1f}%)")
    print(f"Unmatched: {unmatched_count} ({100*unmatched_count/total:.1f}%)")
    
    if matched > 0:
        print("\nMatches by Pass:")
        pass_counts = final_matches[final_matches['match_score'] > 0]['Pass'].value_counts()
        for pass_name, count in pass_counts.items():
            print(f"  {pass_name}: {count}")
        
        print("\nMatch Quality Distribution:")
        excellent = len(final_matches[final_matches['match_score'] >= 90])
        good = len(final_matches[(final_matches['match_score'] >= 70) & (final_matches['match_score'] < 90)])
        fair = len(final_matches[(final_matches['match_score'] >= 50) & (final_matches['match_score'] < 70)])
        poor = len(final_matches[(final_matches['match_score'] > 0) & (final_matches['match_score'] < 50)])
        
        print(f"  Excellent (≥90%): {excellent}")
        print(f"  Good (70-89%): {good}")
        print(f"  Fair (50-69%): {fair}")
        print(f"  Poor (<50%): {poor}")
    
    print(f"\n✅ Results saved to: {output_file}")
    
    # Show sample matches
    if matched > 0:
        print("\nSample matches:")
        sample_cols = ['BESS_Gen_Resource', 'EIA_Plant_Name', 'EIA_County', 'match_score', 'Pass']
        print(final_matches[final_matches['match_score'] > 0][sample_cols].head(10).to_string(index=False))
    
    return final_matches

if __name__ == '__main__':
    import sys
    skip_llm = '--skip-llm' in sys.argv
    df = main(skip_llm=skip_llm)