#!/usr/bin/env python3
"""
Extract schemas from all parquet files and generate documentation.
"""

import pyarrow.parquet as pq
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

def get_parquet_info(file_path):
    """Extract schema and metadata from a parquet file."""
    try:
        # Read parquet file metadata
        parquet_file = pq.ParquetFile(file_path)
        
        # Get schema
        schema = parquet_file.schema
        
        # Get file metadata
        metadata = parquet_file.metadata
        
        # Get sample data to show data types
        df = pd.read_parquet(file_path).head(1)
        
        # Build column info
        columns = []
        for field in schema:
            col_info = {
                'name': field.name,
                'physical_type': str(field.physical_type),
                'logical_type': str(field.logical_type) if field.logical_type else None,
                'pandas_dtype': str(df[field.name].dtype) if field.name in df.columns else None,
                'nullable': field.max_definition_level > 0  # If definition level > 0, field is nullable
            }
            columns.append(col_info)
        
        return {
            'file': file_path.name,
            'path': str(file_path),
            'rows': metadata.num_rows,
            'columns': columns,
            'size_bytes': file_path.stat().st_size,
            'created': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
        }
    except Exception as e:
        return {
            'file': file_path.name,
            'path': str(file_path),
            'error': str(e)
        }

def format_size(bytes):
    """Format bytes to human readable size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024:
            return f"{bytes:.1f} {unit}"
        bytes /= 1024
    return f"{bytes:.1f} TB"

def generate_markdown(schemas_by_category):
    """Generate markdown documentation from schemas."""
    md = []
    md.append("# ERCOT Parquet File Documentation")
    md.append(f"\nGenerated: {datetime.now().isoformat()}")
    md.append("\n## Overview")
    md.append("\nThis document provides comprehensive documentation of all Parquet files in the ERCOT data pipeline.")
    md.append("\n---\n")
    
    # Table of contents
    md.append("## Table of Contents\n")
    for category in sorted(schemas_by_category.keys()):
        anchor = category.replace(" ", "-").replace("_", "-").lower()
        md.append(f"- [{category}](#{anchor})")
    md.append("\n---\n")
    
    # Document each category
    for category in sorted(schemas_by_category.keys()):
        md.append(f"## {category}\n")
        
        for file_info in schemas_by_category[category]:
            if 'error' in file_info:
                md.append(f"### ⚠️ {file_info['file']}")
                md.append(f"\n**Error**: {file_info['error']}\n")
                continue
            
            md.append(f"### {file_info['file']}")
            md.append(f"\n**Path**: `{file_info['path']}`")
            md.append(f"\n**Size**: {format_size(file_info['size_bytes'])}")
            md.append(f"\n**Rows**: {file_info['rows']:,}")
            md.append(f"\n**Last Modified**: {file_info['created']}")
            
            md.append("\n\n#### Schema\n")
            md.append("| Column | Type | Pandas Type | Nullable |")
            md.append("|--------|------|-------------|----------|")
            
            for col in file_info['columns']:
                logical = col['logical_type'] if col['logical_type'] else col['physical_type']
                pandas_type = col['pandas_dtype'] if col['pandas_dtype'] else 'N/A'
                nullable = "Yes" if col['nullable'] else "No"
                md.append(f"| {col['name']} | {logical} | {pandas_type} | {nullable} |")
            
            md.append("\n")
    
    return "\n".join(md)

def main():
    base_dir = Path("/home/enrico/data/ERCOT_data")
    
    # Find all parquet files
    parquet_files = list(base_dir.rglob("*.parquet"))
    
    print(f"Found {len(parquet_files)} parquet files")
    
    # Organize by category
    schemas_by_category = {}
    
    for file_path in sorted(parquet_files):
        # Determine category based on path
        relative_path = file_path.relative_to(base_dir)
        parts = relative_path.parts
        
        if "rollup_files" in parts:
            if "flattened" in parts:
                category = "Flattened Price Data"
            elif "combined" in parts:
                category = "Combined Price Data"
            elif "monthly" in parts:
                category = "Monthly Price Data"
            else:
                # Get the subdirectory name
                idx = parts.index("rollup_files")
                if idx + 1 < len(parts) - 1:
                    category = f"Rollup - {parts[idx + 1]}"
                else:
                    category = "Rollup Files"
        elif "bess_analysis" in str(relative_path):
            category = "BESS Analysis"
        else:
            # Use the first directory as category
            category = parts[0] if parts else "Other"
        
        # Skip very large directories or sample just a few
        if category not in schemas_by_category:
            schemas_by_category[category] = []
        
        # For categories with many files, sample more files including recent ones
        # Always include 2024 files and recent files
        should_include = (
            len(schemas_by_category[category]) < 5 or 
            "flattened" in category or 
            "combined" in category or
            "2024" in file_path.name or
            "2023" in file_path.name
        )
        
        if should_include:
            print(f"Processing {file_path.name} in {category}...")
            file_info = get_parquet_info(file_path)
            schemas_by_category[category].append(file_info)
    
    # Generate markdown
    markdown = generate_markdown(schemas_by_category)
    
    # Save to specs folder
    output_path = Path("/home/enrico/projects/power_market_pipeline/specs/ERCOT_Parquet_Schemas.md")
    output_path.write_text(markdown)
    
    print(f"\n✅ Documentation saved to {output_path}")
    
    # Also save a JSON version for programmatic access
    json_path = Path("/home/enrico/projects/power_market_pipeline/specs/parquet_schemas.json")
    with open(json_path, 'w') as f:
        json.dump(schemas_by_category, f, indent=2, default=str)
    
    print(f"✅ JSON schemas saved to {json_path}")

if __name__ == "__main__":
    main()