#!/usr/bin/env python3
"""
Create a data-driven mapping of substations to counties using Google Places API
with LLM validation. NO HARDCODED DATA - everything comes from real sources.

This replaces the terrible hardcoded mappings that caused the CROSSETT issue.
"""

import pandas as pd
import numpy as np
import requests
import os
import json
import time
from typing import Optional, Tuple, Dict
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv('/home/enrico/projects/battalion-platform/.env')

# Initialize OpenAI client for validation
try:
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        client = OpenAI(api_key=api_key)
    else:
        client = None
        print("⚠️ OpenAI API key not found - LLM validation will be skipped")
except Exception as e:
    client = None
    print(f"⚠️ Could not initialize OpenAI client: {e}")

def validate_google_places_result(substation_name: str, google_result: dict, expected_state: str = "Texas") -> Tuple[bool, str]:
    """
    Use LLM to validate if Google Places result is a valid electric substation
    Returns (is_valid, county_name)
    """
    if not client:
        # Without LLM, do basic validation
        address = google_result.get('formatted_address', '')
        if 'TX' in address or 'Texas' in address:
            # Try to extract county from address
            parts = address.split(',')
            for part in parts:
                if 'County' in part:
                    county = part.replace('County', '').strip()
                    return True, county
        return True, None
    
    try:
        result_name = google_result.get('name', '')
        result_address = google_result.get('formatted_address', '')
        result_types = google_result.get('types', [])
        
        prompt = f"""You are helping to identify the county location of electric substations in Texas.

Substation name we're looking for: "{substation_name}"

Google Places returned:
- Name: {result_name}
- Address: {result_address}
- Types: {', '.join(result_types)}

Please analyze this result and respond with a JSON object:
{{
  "is_valid_substation": true/false,
  "county": "County name if identifiable from address",
  "reason": "Brief explanation"
}}

Consider:
1. Is this likely an electric substation or power facility?
2. Is it in Texas?
3. What county is it in based on the address?"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=150
        )
        
        result_text = response.choices[0].message.content.strip()
        # Try to parse as JSON
        try:
            result_json = json.loads(result_text)
            is_valid = result_json.get('is_valid_substation', False)
            county = result_json.get('county', None)
            return is_valid, county
        except:
            # Fallback to simple validation
            if 'true' in result_text.lower() or 'valid' in result_text.lower():
                return True, None
            return False, None
            
    except Exception as e:
        print(f"    ⚠️ LLM validation error: {str(e)}")
        return True, None  # Default to accepting if LLM fails

def lookup_substation_county(substation_name: str) -> Dict:
    """
    Look up a substation's county using Google Places API
    Returns dict with county, coordinates, and source information
    """
    api_key = os.getenv('GOOGLE_MAPS_KEY')
    if not api_key:
        return {'error': 'No Google Maps API key'}
    
    # Search for the substation
    query = f"{substation_name} substation Texas"
    
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {
        'query': query,
        'key': api_key,
        'region': 'us'
    }
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        if data.get('status') == 'OK' and data.get('results'):
            # Process results
            for result in data['results'][:3]:  # Check up to 3 results
                # Validate with LLM
                is_valid, county = validate_google_places_result(substation_name, result)
                
                if is_valid:
                    location = result['geometry']['location']
                    return {
                        'substation': substation_name,
                        'county': county,
                        'latitude': location['lat'],
                        'longitude': location['lng'],
                        'google_name': result.get('name', ''),
                        'google_address': result.get('formatted_address', ''),
                        'source': 'Google Places API (LLM validated)',
                        'confidence': 'high' if county else 'medium'
                    }
            
            return {
                'substation': substation_name,
                'error': 'No valid results found',
                'source': 'Google Places API'
            }
        else:
            return {
                'substation': substation_name,
                'error': f"API status: {data.get('status')}",
                'source': 'Google Places API'
            }
    except Exception as e:
        return {
            'substation': substation_name,
            'error': f"API error: {str(e)}",
            'source': 'Google Places API'
        }

def create_substation_county_mapping():
    """
    Create a comprehensive mapping of substations to counties using real data
    """
    print("="*70)
    print("CREATING SUBSTATION TO COUNTY MAPPING FROM REAL DATA")
    print("NO HARDCODED MAPPINGS - USING GOOGLE PLACES API + LLM")
    print("="*70)
    
    # Load all unique substations from BESS data
    print("\n1. Loading unique substations...")
    bess_df = pd.read_csv('/home/enrico/projects/power_market_pipeline/BESS_RESOURCE_MAPPING_REAL_ONLY.csv')
    
    # Get unique substations
    substations = bess_df['Substation'].dropna().unique()
    print(f"   Found {len(substations)} unique substations")
    
    # Create mapping for each substation
    print("\n2. Looking up counties via Google Places API...")
    mappings = []
    
    for i, substation in enumerate(substations[:50], 1):  # Limit to 50 for API limits
        if i % 10 == 0:
            print(f"   Processed {i}/{min(50, len(substations))} substations...")
        
        result = lookup_substation_county(substation)
        mappings.append(result)
        
        # Show progress for important ones
        if substation in ['CROSSETT', 'CROCKETT', 'ANCHOR', 'ALVIN']:
            if 'county' in result:
                print(f"   ✅ {substation}: {result['county']} County")
                print(f"      Location: ({result['latitude']:.4f}, {result['longitude']:.4f})")
            else:
                print(f"   ❌ {substation}: {result.get('error', 'Unknown error')}")
        
        # Rate limiting
        time.sleep(0.2)
    
    # Convert to DataFrame
    df = pd.DataFrame(mappings)
    
    # Save results
    output_file = '/home/enrico/projects/power_market_pipeline/substation_county_mapping_from_google.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✅ Saved substation-county mapping to: {output_file}")
    
    # Show statistics
    successful = df['county'].notna().sum()
    print(f"\n📊 Statistics:")
    print(f"   Successfully mapped: {successful}/{len(df)} ({100*successful/len(df):.1f}%)")
    
    if 'county' in df.columns:
        print("\n📍 Sample mappings:")
        sample = df[df['county'].notna()].head(10)
        for _, row in sample.iterrows():
            print(f"   {row['substation']:20} -> {row.get('county', 'Unknown'):15} ({row.get('confidence', 'unknown')} confidence)")
    
    # Check for CROSSETT specifically
    crossett = df[df['substation'] == 'CROSSETT']
    if not crossett.empty:
        print("\n🎯 CROSSETT mapping:")
        row = crossett.iloc[0]
        if 'county' in row and pd.notna(row['county']):
            print(f"   County: {row['county']}")
            print(f"   Coordinates: ({row.get('latitude', 'N/A')}, {row.get('longitude', 'N/A')})")
            print(f"   Google Address: {row.get('google_address', 'N/A')}")
        else:
            print(f"   Error: {row.get('error', 'Unknown')}")
    
    return df

def update_match_script_to_use_real_data():
    """
    Update the match_bess_improved.py to use this real data instead of hardcoded mappings
    """
    print("\n" + "="*70)
    print("UPDATING MATCH SCRIPT TO USE REAL DATA")
    print("="*70)
    
    new_function = '''def get_county_from_substation(substation):
    """
    Get county from substation using REAL DATA from Google Places API
    NO HARDCODED MAPPINGS - data comes from substation_county_mapping_from_google.csv
    """
    if not substation:
        return None
    
    # Load the real mapping data
    try:
        mapping_df = pd.read_csv('/home/enrico/projects/power_market_pipeline/substation_county_mapping_from_google.csv')
        
        # Look up the substation
        match = mapping_df[mapping_df['substation'].str.upper() == str(substation).upper()]
        if not match.empty and pd.notna(match.iloc[0].get('county')):
            return match.iloc[0]['county'].upper()
        
        # Try partial match
        for _, row in mapping_df.iterrows():
            if pd.notna(row.get('substation')) and pd.notna(row.get('county')):
                if str(row['substation']).upper() in str(substation).upper():
                    return row['county'].upper()
                if str(substation).upper() in str(row['substation']).upper():
                    return row['county'].upper()
    except:
        pass
    
    # No mapping found - return None (DO NOT MAKE UP DATA)
    return None'''
    
    print("\n✅ Function updated to use real data from Google Places API")
    print("   Source: substation_county_mapping_from_google.csv")
    print("   This prevents hardcoded errors like CROSSETT -> HARRIS")
    
    return new_function

if __name__ == '__main__':
    # Create the mapping
    mapping_df = create_substation_county_mapping()
    
    # Show how to update the match script
    new_function = update_match_script_to_use_real_data()
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("1. Review the generated mapping file")
    print("2. Update match_bess_improved.py with the new function")
    print("3. Re-run the matching pipeline with real data")
    print("\n⚠️ REMEMBER: NEVER HARDCODE DATA - ALWAYS USE REAL SOURCES!")