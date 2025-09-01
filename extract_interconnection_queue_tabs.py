#!/usr/bin/env python3
"""
Extract all battery-related tabs from ERCOT interconnection queue Excel file to CSV files
This will make it easier to analyze and match the data
"""

import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def extract_all_tabs():
    """Extract all battery-related tabs from the interconnection queue Excel file"""
    
    excel_path = '/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/ERCOT_InterconnectionQueue/interconnection_queue.xlsx'
    output_dir = Path('/home/enrico/projects/power_market_pipeline/interconnection_queue_data')
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
            print(f'Extracting sheet: {sheet_name}')
            df = pd.read_excel(excel_path, sheet_name=sheet_name)
            
            # Clean sheet name for filename
            filename = sheet_name.lower().replace(' ', '_').replace('-', '_') + '.csv'
            output_path = output_dir / filename
            
            # Save to CSV
            df.to_csv(output_path, index=False)
            extracted_files[sheet_name] = {
                'path': output_path,
                'rows': len(df),
                'columns': len(df.columns)
            }
            
            print(f'  ✓ Saved {len(df)} rows, {len(df.columns)} columns to {filename}')
            
            # Show first few columns
            print(f'  Columns (first 10):')
            for col in df.columns[:10]:
                print(f'    - {col}')
            if len(df.columns) > 10:
                print(f'    ... and {len(df.columns) - 10} more columns\n')
            
        except Exception as e:
            print(f'  ✗ Error extracting {sheet_name}: {e}\n')
    
    # Create a summary file
    summary_path = output_dir / 'extraction_summary.txt'
    with open(summary_path, 'w') as f:
        f.write('ERCOT Interconnection Queue Data Extraction Summary\n')
        f.write('=' * 50 + '\n\n')
        
        for sheet, info in extracted_files.items():
            f.write(f'{sheet}:\n')
            f.write(f'  File: {info["path"].name}\n')
            f.write(f'  Rows: {info["rows"]}\n')
            f.write(f'  Columns: {info["columns"]}\n\n')
    
    print(f'\nSummary saved to: {summary_path}')
    return extracted_files

def analyze_column_structure():
    """Analyze and compare column structure across all extracted files"""
    
    output_dir = Path('/home/enrico/projects/power_market_pipeline/interconnection_queue_data')
    
    print('\n=== Analyzing Column Structure Across Files ===\n')
    
    all_columns = {}
    common_columns = None
    
    csv_files = list(output_dir.glob('*.csv'))
    
    for csv_file in csv_files:
        if csv_file.name == 'column_analysis.csv':
            continue
            
        df = pd.read_csv(csv_file)
        all_columns[csv_file.stem] = list(df.columns)
        
        if common_columns is None:
            common_columns = set(df.columns)
        else:
            common_columns = common_columns.intersection(set(df.columns))
    
    print(f'Common columns across all files ({len(common_columns)}):')
    for col in sorted(common_columns):
        print(f'  - {col}')
    
    # Find key columns for matching
    print('\n=== Key Columns for Matching ===\n')
    
    key_patterns = [
        ('Project/Facility Name', ['PROJECT', 'FACILITY', 'NAME']),
        ('County', ['COUNTY']),
        ('POI/Interconnection', ['POI', 'INTERCONNECTION', 'SUBSTATION']),
        ('Capacity', ['CAPACITY', 'MW', 'SIZE']),
        ('Status', ['STATUS']),
        ('Dates', ['DATE', 'COD', 'COMMERCIAL']),
        ('Company/Developer', ['COMPANY', 'DEVELOPER', 'OWNER']),
        ('Resource Name', ['RESOURCE', 'UNIT'])
    ]
    
    column_mapping = {}
    
    for csv_file in csv_files:
        if csv_file.name == 'column_analysis.csv':
            continue
            
        df = pd.read_csv(csv_file)
        file_key = csv_file.stem
        column_mapping[file_key] = {}
        
        print(f'\n{file_key}:')
        
        for category, patterns in key_patterns:
            matching_cols = []
            for col in df.columns:
                col_upper = col.upper()
                if any(pattern in col_upper for pattern in patterns):
                    matching_cols.append(col)
            
            if matching_cols:
                column_mapping[file_key][category] = matching_cols
                print(f'  {category}: {", ".join(matching_cols)}')
    
    # Save column analysis
    analysis_path = output_dir / 'column_analysis.csv'
    
    # Create a DataFrame with all column mappings
    analysis_data = []
    for file_key, mappings in column_mapping.items():
        for category, cols in mappings.items():
            for col in cols:
                analysis_data.append({
                    'File': file_key,
                    'Category': category,
                    'Column': col
                })
    
    if analysis_data:
        analysis_df = pd.DataFrame(analysis_data)
        analysis_df.to_csv(analysis_path, index=False)
        print(f'\nColumn analysis saved to: {analysis_path}')
    
    return column_mapping

if __name__ == '__main__':
    # Extract all tabs
    extracted = extract_all_tabs()
    
    # Analyze column structure
    column_map = analyze_column_structure()
    
    print('\n✅ Extraction complete!')
    print(f'Data saved to: /home/enrico/projects/power_market_pipeline/interconnection_queue_data/')