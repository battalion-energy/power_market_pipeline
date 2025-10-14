#!/usr/bin/env python3
"""
Fast BESS to EIA cross-reference using batch LLM processing
Only uses REAL data from EIA generator reports
"""

import pandas as pd
import numpy as np
from rapidfuzz import fuzz
import json
import os
from dotenv import load_dotenv
import requests

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
    df_bess = df_tx[
        (df_tx['Technology'].str.contains('Battery', na=False)) |
        (df_tx['Technology'].str.contains('Energy Storage', na=False)) |
        (df_tx['Energy Source Code'].str.contains('MWH', na=False)) |
        (df_tx['Prime Mover Code'].str.contains('BA', na=False))
    ].copy()
    
    print(f"Loaded {len(df_bess)} TX battery storage facilities from {sheet_name} tab")
    
    # Clean up names
    df_bess['Plant Name Clean'] = df_bess['Plant Name'].str.upper().str.replace(' ', '_')
    
    return df_bess

def load_bess_resources():
    """Load BESS resources from improved matched data"""
    df = pd.read_csv('/home/enrico/projects/power_market_pipeline/BESS_IMPROVED_MATCHED.csv')
    
    # Load mapping for additional info
    mapping_df = pd.read_csv('/home/enrico/projects/power_market_pipeline/BESS_RESOURCE_MAPPING_REAL_ONLY.csv')
    df = df.merge(
        mapping_df[['BESS_Gen_Resource', 'Substation', 'Load_Zone']],
        on='BESS_Gen_Resource',
        how='left',
        suffixes=('', '_mapping')
    )
    
    # Try to get county from substation mapping
    try:
        sub_df = pd.read_csv('/home/enrico/projects/power_market_pipeline/substation_to_county_mapping.csv')
        df = df.merge(sub_df[['Substation', 'County']], on='Substation', how='left')
    except:
        df['County'] = None
    
    return df

def normalize_county(county):
    """Normalize county names"""
    if pd.isna(county):
        return ''
    return str(county).upper().replace(' COUNTY', '').replace(' CO.', '').strip()

def enhanced_heuristic_match(bess_df, eia_df, pass_name="Heuristic"):
    """Enhanced heuristic matching with multiple strategies"""
    matches = []
    
    # Pre-normalize EIA data
    eia_df['County_Norm'] = eia_df['County'].apply(normalize_county)
    
    for _, bess_row in bess_df.iterrows():
        bess_name = bess_row['BESS_Gen_Resource']
        bess_county = normalize_county(bess_row.get('County'))
        
        best_match = None
        best_score = 0
        best_reason = ""
        
        # Strategy 1: Direct name matching
        for _, eia_row in eia_df.iterrows():
            eia_plant = eia_row['Plant Name Clean']
            eia_county = eia_row['County_Norm']
            
            # Calculate similarity
            name_sim = fuzz.ratio(bess_name, eia_plant) / 100
            
            # Also check partial ratios for substring matches
            partial_sim = fuzz.partial_ratio(bess_name, eia_plant) / 100
            token_sim = fuzz.token_sort_ratio(bess_name, eia_plant) / 100
            
            # Use best similarity
            best_sim = max(name_sim, partial_sim * 0.9, token_sim * 0.85)
            
            # County bonus
            county_match = 1.0 if (bess_county and eia_county and bess_county == eia_county) else 0
            
            # Calculate score
            if county_match and best_sim > 0.3:
                score = 50 + (best_sim * 50)
                reason = f"name similarity: {best_sim:.2f}, county match"
            elif best_sim > 0.7:
                score = best_sim * 80
                reason = f"strong name similarity: {best_sim:.2f}"
            elif best_sim > 0.5 and county_match:
                score = 40 + (best_sim * 40)
                reason = f"name+county: {best_sim:.2f}"
            else:
                score = 0
            
            if score > best_score and score >= 40:
                best_score = score
                best_match = eia_row
                best_reason = reason
        
        if best_match is not None:
            matches.append({
                'BESS_Gen_Resource': bess_name,
                'EIA_Plant_Name': best_match['Plant Name'],
                'EIA_Generator_ID': best_match['Generator ID'],
                'EIA_County': best_match['County'],
                'EIA_Technology': best_match['Technology'],
                'EIA_Capacity_MW': best_match.get('Nameplate Capacity (MW)'),
                'match_score': best_score,
                'match_reason': best_reason,
                'Pass': pass_name,
                'Source': f'EIA {eia_df.name if hasattr(eia_df, "name") else ""}'
            })
    
    return pd.DataFrame(matches)

def batch_llm_match(bess_df, eia_df, pass_name="LLM Batch", sheet_type="Operating"):
    """Batch LLM matching - process multiple BESS at once"""
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("Warning: No OPENROUTER_API_KEY found, skipping LLM matching")
        return pd.DataFrame()
    
    matches = []
    
    # Prepare EIA summary
    eia_list = []
    for _, row in eia_df.head(100).iterrows():  # Limit to avoid token overflow
        eia_list.append({
            'name': row['Plant Name'],
            'gen_id': str(row['Generator ID']),
            'county': row.get('County', ''),
            'capacity': row.get('Nameplate Capacity (MW)', 0)
        })
    
    # Process in batches of 10
    batch_size = 10
    unmatched_list = bess_df.to_dict('records')
    
    for i in range(0, len(unmatched_list), batch_size):
        batch = unmatched_list[i:i+batch_size]
        print(f"    Processing batch {i//batch_size + 1}/{(len(unmatched_list)-1)//batch_size + 1}...")
        
        # Build batch request
        bess_batch = []
        for row in batch:
            bess_batch.append({
                'name': row['BESS_Gen_Resource'],
                'county': row.get('County', 'Unknown'),
                'substation': row.get('Substation', 'Unknown')
            })
        
        prompt = f"""Match BESS resources to EIA facilities. Use Texas knowledge:
- BRP_ = Broad Reach Power projects
- SWOOSE = Ward/Reeves counties
- County names must match or be adjacent

BESS Resources:
{json.dumps(bess_batch, indent=2)}

EIA {sheet_type} Facilities (partial list):
{json.dumps(eia_list[:50], indent=2)}

Return JSON array with one object per BESS:
[
  {{
    "bess_name": "exact BESS name",
    "matched": true/false,
    "eia_name": "exact EIA plant name if matched",
    "eia_gen_id": "exact generator ID if matched",
    "confidence": 0-100,
    "reason": "brief reason"
  }}
]

Only match if confidence >= 70. Return matched:false otherwise."""
        
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "HTTP-Referer": "https://github.com/enricoaquilina/power-market-pipeline",
                    "X-Title": "BESS EIA Batch Matching"
                },
                json={
                    "model": "openai/gpt-4o-mini",  # Use faster model
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 2000
                },
                timeout=30
            )
            
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content']
                
                # Clean response
                if '```json' in content:
                    content = content.split('```json')[-1].split('```')[0]
                
                results = json.loads(content.strip())
                
                # Process each result
                for result in results:
                    if result.get('matched') and result.get('confidence', 0) >= 70:
                        # Find exact EIA match
                        eia_match = eia_df[
                            (eia_df['Plant Name'] == result.get('eia_name')) |
                            (eia_df['Generator ID'].astype(str) == str(result.get('eia_gen_id', '')))
                        ]
                        
                        if not eia_match.empty:
                            eia_row = eia_match.iloc[0]
                            matches.append({
                                'BESS_Gen_Resource': result['bess_name'],
                                'EIA_Plant_Name': eia_row['Plant Name'],
                                'EIA_Generator_ID': eia_row['Generator ID'],
                                'EIA_County': eia_row.get('County'),
                                'EIA_Technology': eia_row.get('Technology'),
                                'EIA_Capacity_MW': eia_row.get('Nameplate Capacity (MW)'),
                                'match_score': result['confidence'],
                                'match_reason': result.get('reason', 'LLM match'),
                                'Pass': pass_name,
                                'Source': f'EIA {sheet_type} (LLM)'
                            })
            else:
                print(f"    API error: {response.status_code}")
                
        except Exception as e:
            print(f"    Batch processing error: {e}")
    
    return pd.DataFrame(matches)

def main(skip_llm=False):
    """Run fast cross-reference matching"""
    
    print("="*70)
    print("BESS to EIA Generator Cross-Reference (Fast Version)")
    print("="*70)
    
    # Load data
    print("\n📂 Loading data...")
    bess_df = load_bess_resources()
    eia_operating = load_eia_data('Operating')
    eia_planned = load_eia_data('Planned')
    
    print(f"\nTotal BESS resources: {len(bess_df)}")
    
    # Track results
    all_matches = []
    unmatched = set(bess_df['BESS_Gen_Resource'].unique())
    
    # Pass 1: Enhanced heuristic on Operating
    print("\n🔄 Pass 1: Enhanced heuristic matching on Operating...")
    pass1 = enhanced_heuristic_match(
        bess_df[bess_df['BESS_Gen_Resource'].isin(unmatched)],
        eia_operating,
        "Pass 1 (Operating)"
    )
    if not pass1.empty:
        all_matches.append(pass1)
        unmatched -= set(pass1['BESS_Gen_Resource'])
        print(f"  ✅ Matched: {len(pass1)} (Remaining: {len(unmatched)})")
    
    # Pass 2: Batch LLM on Operating
    if not skip_llm:
        print("\n🤖 Pass 2: Batch LLM matching on Operating...")
        if len(unmatched) > 0:
            pass2 = batch_llm_match(
                bess_df[bess_df['BESS_Gen_Resource'].isin(unmatched)],
                eia_operating,
                "Pass 2 (Operating LLM)",
                "Operating"
            )
            if not pass2.empty:
                all_matches.append(pass2)
                unmatched -= set(pass2['BESS_Gen_Resource'])
                print(f"  ✅ Matched: {len(pass2)} (Remaining: {len(unmatched)})")
    
    # Pass 3: Enhanced heuristic on Planned
    print("\n🔄 Pass 3: Enhanced heuristic matching on Planned...")
    if len(unmatched) > 0:
        pass3 = enhanced_heuristic_match(
            bess_df[bess_df['BESS_Gen_Resource'].isin(unmatched)],
            eia_planned,
            "Pass 3 (Planned)"
        )
        if not pass3.empty:
            all_matches.append(pass3)
            unmatched -= set(pass3['BESS_Gen_Resource'])
            print(f"  ✅ Matched: {len(pass3)} (Remaining: {len(unmatched)})")
    
    # Pass 4: Batch LLM on Planned
    if not skip_llm:
        print("\n🤖 Pass 4: Batch LLM matching on Planned...")
        if len(unmatched) > 0:
            pass4 = batch_llm_match(
                bess_df[bess_df['BESS_Gen_Resource'].isin(unmatched)],
                eia_planned,
                "Pass 4 (Planned LLM)",
                "Planned"
            )
            if not pass4.empty:
                all_matches.append(pass4)
                unmatched -= set(pass4['BESS_Gen_Resource'])
                print(f"  ✅ Matched: {len(pass4)} (Remaining: {len(unmatched)})")
    
    # Combine results
    if all_matches:
        final = pd.concat(all_matches, ignore_index=True)
    else:
        final = pd.DataFrame()
    
    # Add unmatched
    for bess in unmatched:
        final = pd.concat([final, pd.DataFrame([{
            'BESS_Gen_Resource': bess,
            'match_score': 0,
            'Pass': 'Unmatched'
        }])], ignore_index=True)
    
    # Save
    output = '/home/enrico/projects/power_market_pipeline/BESS_EIA_MATCHED.csv'
    final.to_csv(output, index=False)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    matched = len(final[final['match_score'] > 0])
    total = len(bess_df)
    
    print(f"\nTotal BESS: {total}")
    print(f"Matched to EIA: {matched} ({100*matched/total:.1f}%)")
    print(f"Unmatched: {total-matched} ({100*(total-matched)/total:.1f}%)")
    
    if matched > 0:
        print("\nBy Pass:")
        for pass_name, count in final[final['match_score'] > 0]['Pass'].value_counts().items():
            print(f"  {pass_name}: {count}")
    
    print(f"\n✅ Saved to: {output}")
    
    # Sample
    if matched > 0:
        print("\nSample matches:")
        cols = ['BESS_Gen_Resource', 'EIA_Plant_Name', 'match_score', 'Pass']
        print(final[final['match_score'] > 0][cols].head(10).to_string(index=False))
    
    return final

if __name__ == '__main__':
    import sys
    skip_llm = '--skip-llm' in sys.argv
    df = main(skip_llm=skip_llm)