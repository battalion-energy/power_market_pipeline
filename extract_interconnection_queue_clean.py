#!/usr/bin/env python3
"""
Extract and clean all battery-related tabs from ERCOT interconnection queue Excel file
Properly handles headers and removes empty rows
"""

import pandas as pd
from pathlib import Path
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def find_header_row(df):
    """Find the row containing actual headers"""
    for i in range(min(20, len(df))):
        row = df.iloc[i]
        # Check if this row has meaningful headers
        non_null_count = row.notna().sum()
        if non_null_count >= 3:  # At least 3 non-null values
            # Check for header keywords
            row_str = ' '.join(str(v) for v in row if pd.notna(v)).upper()
            if any(keyword in row_str for keyword in ['UNIT', 'PROJECT', 'COUNTY', 'CAPACITY', 'STATUS', 'FUEL']):
                return i
    return None

def clean_dataframe(df, sheet_name):
    """Clean a dataframe by finding headers and removing empty rows"""
    # Find header row
    header_row = find_header_row(df)
    
    if header_row is not None:
        # Use the found row as headers
        new_headers = df.iloc[header_row].fillna('')
        
        # Skip to data after headers
        df_clean = df.iloc[header_row + 1:].copy()
        df_clean.columns = new_headers
        
        # Remove completely empty rows
        df_clean = df_clean.dropna(how='all')
        
        # Remove columns that are completely empty or unnamed
        df_clean = df_clean.loc[:, (df_clean != '').any()]
        df_clean = df_clean.loc[:, df_clean.columns != '']
        
        # Reset index
        df_clean = df_clean.reset_index(drop=True)
        
        print(f'  Found headers at row {header_row}')
        print(f'  Columns: {", ".join([col for col in df_clean.columns if col][:8])}...')
        
        return df_clean
    else:
        print(f'  Warning: Could not find headers for {sheet_name}')
        return df

def extract_and_clean_all_tabs():
    """Extract and clean all battery-related tabs from the interconnection queue Excel file"""
    
    excel_path = '/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/ERCOT_InterconnectionQueue/interconnection_queue.xlsx'
    output_dir = Path('/home/enrico/projects/power_market_pipeline/interconnection_queue_clean')
    output_dir.mkdir(exist_ok=True)
    
    print(f'Reading Excel file: {excel_path}')
    print(f'Output directory: {output_dir}\n')
    
    # Define the tabs we want to extract
    tabs_to_extract = [
        'Co-located with Solar',
        'Co-located with Wind', 
        'Co-located with Thermal',
        'Stand-Alone',
        'Co-located Operational'
    ]
    
    extracted_files = {}
    
    for sheet_name in tabs_to_extract:
        try:
            print(f'Processing sheet: {sheet_name}')
            df = pd.read_excel(excel_path, sheet_name=sheet_name)
            
            # Clean the dataframe
            df_clean = clean_dataframe(df, sheet_name)
            
            # Clean sheet name for filename
            filename = sheet_name.lower().replace(' ', '_').replace('-', '_') + '.csv'
            output_path = output_dir / filename
            
            # Save to CSV
            df_clean.to_csv(output_path, index=False)
            extracted_files[sheet_name] = {
                'path': output_path,
                'rows': len(df_clean),
                'columns': len(df_clean.columns)
            }
            
            print(f'  ✓ Saved {len(df_clean)} rows, {len(df_clean.columns)} columns to {filename}\n')
            
        except Exception as e:
            print(f'  ✗ Error processing {sheet_name}: {e}\n')
            import traceback
            traceback.print_exc()
    
    return extracted_files

def analyze_cleaned_data():
    """Analyze the cleaned data files to understand structure"""
    
    output_dir = Path('/home/enrico/projects/power_market_pipeline/interconnection_queue_clean')
    
    print('\n=== Analyzing Cleaned Data Structure ===\n')
    
    all_data = {}
    
    for csv_file in output_dir.glob('*.csv'):
        print(f'\n{csv_file.stem}:')
        df = pd.read_csv(csv_file)
        all_data[csv_file.stem] = df
        
        print(f'  Shape: {df.shape[0]} rows × {df.shape[1]} columns')
        print(f'  Columns:')
        for col in df.columns[:15]:
            # Show column name and data type
            non_null = df[col].notna().sum()
            print(f'    - {col} ({df[col].dtype}, {non_null} non-null)')
        if len(df.columns) > 15:
            print(f'    ... and {len(df.columns) - 15} more columns')
        
        # Show sample data
        if len(df) > 0:
            print(f'\n  Sample data (first 3 rows):')
            sample_cols = [col for col in df.columns if col and 'Unit' in col or 'Project' in col or 'County' in col or 'Capacity' in col][:5]
            if sample_cols:
                print(df[sample_cols].head(3).to_string(index=False))
    
    return all_data

if __name__ == '__main__':
    # Extract and clean all tabs
    extracted = extract_and_clean_all_tabs()
    
    # Analyze cleaned data
    all_data = analyze_cleaned_data()
    
    print('\n✅ Extraction and cleaning complete!')
    print(f'Clean data saved to: /home/enrico/projects/power_market_pipeline/interconnection_queue_clean/')