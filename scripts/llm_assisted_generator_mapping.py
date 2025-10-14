#!/usr/bin/env python3
"""
LLM-Assisted Generator Mapping Pipeline
Uses Claude or other LLMs to intelligently decode ERCOT codes
"""

import pandas as pd
import numpy as np
import json
import re
import subprocess
import warnings
from typing import Dict, List, Tuple, Optional
warnings.filterwarnings('ignore')

# File paths
INTERCONNECTION_FILE = '/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/ERCOT_InterconnectionQueue/interconnection_gis_report.xlsx'
EIA_PLANT_FILE = '/home/enrico/experiments/ERCOT_SCED/pypsa-usa/workflow/repo_data/plants/eia860_ads_merged.csv'
DAM_RESOURCE_FILE = '/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/60-Day_DAM_Disclosure_Reports/csv/60d_DAM_Gen_Resource_Data-18-JAN-24.csv'

def extract_unique_ercot_codes(resources: List[str]) -> List[str]:
    """Extract unique ERCOT code prefixes"""
    codes = set()
    for resource in resources:
        # Extract the base code
        if '_' in resource:
            base = resource.split('_')[0]
        else:
            base = resource
        codes.add(base)
    return sorted(list(codes))

def get_llm_mapping_via_claude(ercot_codes: List[str], iq_names: List[str], eia_names: List[str]) -> Dict[str, str]:
    """Use Claude CLI to create intelligent mappings"""
    
    # Prepare the prompt
    prompt = f"""I need help matching ERCOT power plant codes to their full names. 

ERCOT uses abbreviated codes for power plants in Texas. I have:
1. ERCOT codes (abbreviated)
2. Interconnection Queue project names (full names)
3. EIA plant names (full names)

Please match these ERCOT codes to their most likely full plant names. Consider:
- Common abbreviations in the energy industry
- Texas power plant names
- Similar patterns (e.g., THW might be "T H Wharton")

ERCOT Codes to decode (first 50):
{json.dumps(ercot_codes[:50], indent=2)}

Available IQ Project Names (sample):
{json.dumps(iq_names[:100], indent=2)}

Available EIA Plant Names (sample):
{json.dumps(eia_names[:100], indent=2)}

Return a JSON object mapping ERCOT codes to their full names. For example:
{{
  "THW": "T H Wharton",
  "FORMOSA": "Formosa Utility Venture Ltd",
  "AMOCOOIL": "Amoco Oil",
  ...
}}

Focus on making intelligent matches based on abbreviation patterns and industry knowledge."""

    # Save prompt to file
    with open('/tmp/ercot_mapping_prompt.txt', 'w') as f:
        f.write(prompt)
    
    # Call Claude via CLI
    try:
        result = subprocess.run(
            ['claude', 'code', '--no-markdown'],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            # Parse the JSON response
            response = result.stdout.strip()
            # Extract JSON from response
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        else:
            print(f"Claude CLI error: {result.stderr}")
    except Exception as e:
        print(f"Error calling Claude: {e}")
    
    return {}

def use_llm_for_fuzzy_matching(ercot_code: str, candidates: List[str]) -> Tuple[str, float]:
    """Use LLM to find best match from candidates"""
    
    prompt = f"""Match the ERCOT code "{ercot_code}" to the most likely full name from these candidates:

{json.dumps(candidates[:20], indent=2)}

Consider:
- {ercot_code} is likely an abbreviation
- Common power industry naming patterns
- Geographic/company name abbreviations

Return just the best matching name and a confidence score (0-1).
Format: {{"match": "name", "confidence": 0.95}}"""
    
    try:
        result = subprocess.run(
            ['claude', 'code', '--no-markdown'],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            response = result.stdout.strip()
            json_match = re.search(r'\{[^{}]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                return data.get('match', ''), data.get('confidence', 0)
    except:
        pass
    
    return '', 0

def main():
    print("=" * 80)
    print("LLM-ASSISTED GENERATOR MAPPING PIPELINE")
    print("=" * 80)
    
    # 1. Load data
    print("\n1. Loading data sources...")
    
    # ERCOT resources
    dam_df = pd.read_csv(DAM_RESOURCE_FILE)
    resources = dam_df['Resource Name'].unique()
    ercot_codes = extract_unique_ercot_codes(resources)
    print(f"   Found {len(ercot_codes)} unique ERCOT code prefixes")
    
    # Load IQ
    iq_large = pd.read_excel(INTERCONNECTION_FILE, sheet_name='Project Details - Large Gen', skiprows=30)
    iq_large.columns = iq_large.columns.str.strip()
    iq_names = iq_large[iq_large['INR'].notna()]['Project Name'].dropna().unique().tolist()
    print(f"   Found {len(iq_names)} IQ project names")
    
    # Load EIA
    eia_df = pd.read_csv(EIA_PLANT_FILE, low_memory=False)
    eia_tx = eia_df[eia_df['state'] == 'TX']
    eia_names = eia_tx['plant_name'].dropna().unique().tolist()
    print(f"   Found {len(eia_names)} EIA plant names")
    
    # 2. Get LLM mappings
    print("\n2. Using LLM to decode ERCOT codes...")
    
    # Try to load cached mappings first
    cache_file = 'ercot_llm_mappings_cache.json'
    try:
        with open(cache_file, 'r') as f:
            llm_mappings = json.load(f)
        print(f"   Loaded {len(llm_mappings)} cached mappings")
    except:
        llm_mappings = {}
    
    # Process unmapped codes in batches
    unmapped = [code for code in ercot_codes if code not in llm_mappings]
    
    if unmapped:
        print(f"   Processing {len(unmapped)} unmapped codes with LLM...")
        
        # Process in batches of 50
        for i in range(0, min(len(unmapped), 200), 50):  # Limit to first 200 for demo
            batch = unmapped[i:i+50]
            print(f"   Processing batch {i//50 + 1}...")
            
            batch_mappings = get_llm_mapping_via_claude(batch, iq_names, eia_names)
            llm_mappings.update(batch_mappings)
            
            # Save cache after each batch
            with open(cache_file, 'w') as f:
                json.dump(llm_mappings, f, indent=2)
    
    # 3. Apply mappings to create matches
    print("\n3. Applying LLM mappings to resources...")
    
    results = []
    for resource in resources:
        code = resource.split('_')[0] if '_' in resource else resource
        
        if code in llm_mappings:
            full_name = llm_mappings[code]
            
            # Find in IQ
            iq_match = None
            for iq_name in iq_names:
                if full_name.lower() in iq_name.lower() or iq_name.lower() in full_name.lower():
                    iq_match = iq_name
                    break
            
            # Find in EIA
            eia_match = None
            for eia_name in eia_names:
                if full_name.lower() in eia_name.lower() or eia_name.lower() in full_name.lower():
                    eia_match = eia_name
                    break
            
            results.append({
                'Resource': resource,
                'Code': code,
                'LLM_Match': full_name,
                'IQ_Match': iq_match,
                'EIA_Match': eia_match
            })
        else:
            results.append({
                'Resource': resource,
                'Code': code,
                'LLM_Match': None,
                'IQ_Match': None,
                'EIA_Match': None
            })
    
    # 4. Create output
    results_df = pd.DataFrame(results)
    
    # Statistics
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    total = len(results_df)
    has_llm = results_df['LLM_Match'].notna().sum()
    has_iq = results_df['IQ_Match'].notna().sum()
    has_eia = results_df['EIA_Match'].notna().sum()
    
    print(f"Total Resources: {total}")
    print(f"With LLM Match: {has_llm} ({has_llm/total*100:.1f}%)")
    print(f"With IQ Match: {has_iq} ({has_iq/total*100:.1f}%)")
    print(f"With EIA Match: {has_eia} ({has_eia/total*100:.1f}%)")
    
    # Save results
    results_df.to_csv('llm_generator_mapping_results.csv', index=False)
    print("\nResults saved to: llm_generator_mapping_results.csv")
    
    # Show samples
    print("\nSample LLM Mappings:")
    for code, name in list(llm_mappings.items())[:10]:
        print(f"  {code:15} -> {name}")

if __name__ == "__main__":
    main()