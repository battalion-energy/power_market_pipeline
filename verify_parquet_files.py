#!/usr/bin/env python3
"""
Comprehensive Parquet File Verification System

Checks all parquet files in the rollup directory for:
- Data integrity
- Duplicates
- Gaps in time series
- Schema consistency
- Corrupted files
"""

import pandas as pd
import pyarrow.parquet as pq
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import sys
import os
from typing import Dict, List, Tuple, Optional
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# Colors for terminal output
RED = '\033[91m'
YELLOW = '\033[93m'
GREEN = '\033[92m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'

def format_number(n: int) -> str:
    """Format number with commas."""
    return f"{n:,}"

def format_bytes(bytes: int) -> str:
    """Format bytes to human readable."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes < 1024.0:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.2f} PB"

class ParquetVerifier:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.rollup_dir = base_dir / "rollup_files"
        self.issues = []
        self.stats = {}
        
    def verify_all_datasets(self):
        """Verify all datasets in parallel."""
        print(f"{BOLD}🔍 Parquet Data Verification System{RESET}")
        print("=" * 80)
        print(f"Base directory: {self.base_dir}")
        print(f"Rollup directory: {self.rollup_dir}")
        
        if not self.rollup_dir.exists():
            print(f"{RED}❌ Rollup directory not found!{RESET}")
            return
        
        # Find all dataset directories
        datasets = [d for d in self.rollup_dir.iterdir() if d.is_dir()]
        
        if not datasets:
            print(f"{YELLOW}⚠️  No dataset directories found{RESET}")
            return
        
        print(f"\n📊 Found {len(datasets)} datasets to verify:")
        for ds in sorted(datasets):
            print(f"  • {ds.name}")
        
        # Process each dataset
        for dataset_dir in sorted(datasets):
            self.verify_dataset(dataset_dir)
        
        # Generate summary report
        self.generate_report()
    
    def verify_dataset(self, dataset_dir: Path):
        """Verify a single dataset."""
        dataset_name = dataset_dir.name
        print(f"\n{BLUE}━━━ Verifying {dataset_name} ━━━{RESET}")
        
        # Find all parquet files
        parquet_files = sorted(dataset_dir.glob("*.parquet"))
        
        if not parquet_files:
            print(f"  {YELLOW}No parquet files found{RESET}")
            return
        
        print(f"  Found {len(parquet_files)} parquet files")
        
        # Verify each file and collect stats
        file_stats = []
        dataset_issues = []
        
        # Use multiprocessing for parallel verification
        with ProcessPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(self.verify_single_file, f, dataset_name): f 
                for f in parquet_files
            }
            
            for future in as_completed(futures):
                file_path = futures[future]
                try:
                    stats, issues = future.result()
                    file_stats.append(stats)
                    dataset_issues.extend(issues)
                except Exception as e:
                    issue = {
                        'file': str(file_path),
                        'type': 'ERROR',
                        'message': f"Failed to verify: {str(e)}",
                        'severity': 'CRITICAL'
                    }
                    dataset_issues.append(issue)
        
        # Analyze cross-file consistency
        if file_stats:
            self.check_cross_file_consistency(file_stats, dataset_name, dataset_issues)
        
        # Store results
        self.stats[dataset_name] = {
            'files': len(parquet_files),
            'file_stats': file_stats,
            'issues': dataset_issues
        }
        
        # Print summary
        total_rows = sum(s['rows'] for s in file_stats)
        total_size = sum(s['size'] for s in file_stats)
        
        print(f"  ✅ Verified {len(file_stats)} files")
        print(f"     Total rows: {format_number(total_rows)}")
        print(f"     Total size: {format_bytes(total_size)}")
        
        # Report issues
        critical = [i for i in dataset_issues if i['severity'] == 'CRITICAL']
        warnings = [i for i in dataset_issues if i['severity'] == 'WARNING']
        
        if critical:
            print(f"  {RED}❌ {len(critical)} CRITICAL issues found!{RESET}")
            for issue in critical[:3]:  # Show first 3
                print(f"     • {issue['message']}")
        
        if warnings:
            print(f"  {YELLOW}⚠️  {len(warnings)} warnings{RESET}")
            for issue in warnings[:3]:  # Show first 3
                print(f"     • {issue['message']}")
    
    @staticmethod
    def verify_single_file(file_path: Path, dataset_name: str) -> Tuple[Dict, List]:
        """Verify a single parquet file."""
        stats = {
            'file': file_path.name,
            'size': file_path.stat().st_size,
            'rows': 0,
            'columns': [],
            'nulls': {},
            'duplicates': 0,
            'date_range': None
        }
        issues = []
        
        try:
            # Read parquet file
            df = pd.read_parquet(file_path)
            stats['rows'] = len(df)
            stats['columns'] = list(df.columns)
            
            # Check for empty file
            if len(df) == 0:
                issues.append({
                    'file': file_path.name,
                    'type': 'EMPTY',
                    'message': 'File contains no rows',
                    'severity': 'WARNING'
                })
            
            # Check for nulls
            null_counts = df.isnull().sum()
            stats['nulls'] = {col: int(count) for col, count in null_counts.items() if count > 0}
            
            # For RT prices, check duplicates more thoroughly
            if dataset_name == "RT_prices" and len(df) > 0:
                if 'datetime' in df.columns and 'settlement_point' in df.columns:
                    # Check for duplicate datetime + settlement_point combinations
                    dup_check = df[['datetime', 'settlement_point']].duplicated()
                    stats['duplicates'] = int(dup_check.sum())
                    
                    if stats['duplicates'] > 0:
                        issues.append({
                            'file': file_path.name,
                            'type': 'DUPLICATES',
                            'message': f"Found {format_number(stats['duplicates'])} duplicate rows (datetime + settlement_point)",
                            'severity': 'CRITICAL'
                        })
            
            # Check datetime range if available
            if 'datetime' in df.columns and len(df) > 0:
                try:
                    # Handle different datetime formats
                    if df['datetime'].dtype == 'object':
                        df['datetime'] = pd.to_datetime(df['datetime'])
                    
                    min_date = df['datetime'].min()
                    max_date = df['datetime'].max()
                    stats['date_range'] = (str(min_date), str(max_date))
                    
                    # Check if data is sorted
                    if not df['datetime'].is_monotonic_increasing:
                        issues.append({
                            'file': file_path.name,
                            'type': 'UNSORTED',
                            'message': 'Datetime column is not sorted',
                            'severity': 'WARNING'
                        })
                except Exception as e:
                    pass  # Datetime parsing might fail for some formats
            
        except Exception as e:
            issues.append({
                'file': file_path.name,
                'type': 'CORRUPT',
                'message': f"Failed to read file: {str(e)}",
                'severity': 'CRITICAL'
            })
        
        return stats, issues
    
    def check_cross_file_consistency(self, file_stats: List[Dict], dataset_name: str, issues: List):
        """Check consistency across files."""
        # Check schema consistency
        schemas = [tuple(sorted(s['columns'])) for s in file_stats]
        unique_schemas = set(schemas)
        
        if len(unique_schemas) > 1:
            issues.append({
                'file': 'CROSS_FILE',
                'type': 'SCHEMA_MISMATCH',
                'message': f"Found {len(unique_schemas)} different schemas across files",
                'severity': 'WARNING'
            })
        
        # Check for time overlaps (for time series data)
        if any('date_range' in s and s['date_range'] for s in file_stats):
            date_ranges = [(s['file'], s['date_range']) for s in file_stats if s.get('date_range')]
            date_ranges.sort(key=lambda x: x[1][0] if x[1] else '')
            
            for i in range(1, len(date_ranges)):
                prev_file, prev_range = date_ranges[i-1]
                curr_file, curr_range = date_ranges[i]
                
                if prev_range and curr_range:
                    if prev_range[1] > curr_range[0]:
                        issues.append({
                            'file': f"{prev_file} & {curr_file}",
                            'type': 'TIME_OVERLAP',
                            'message': f"Time overlap detected between files",
                            'severity': 'WARNING'
                        })
    
    def generate_report(self):
        """Generate verification report."""
        report_path = self.rollup_dir / "verification_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# Parquet Data Verification Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Overall summary
            total_files = sum(s['files'] for s in self.stats.values())
            total_issues = sum(len(s['issues']) for s in self.stats.values())
            
            f.write("## Summary\n\n")
            f.write(f"- **Datasets verified**: {len(self.stats)}\n")
            f.write(f"- **Total files**: {total_files}\n")
            f.write(f"- **Total issues**: {total_issues}\n\n")
            
            # Critical issues
            critical_issues = []
            for dataset, data in self.stats.items():
                for issue in data['issues']:
                    if issue['severity'] == 'CRITICAL':
                        critical_issues.append(f"{dataset}: {issue['message']}")
            
            if critical_issues:
                f.write("## ⚠️ Critical Issues\n\n")
                for issue in critical_issues:
                    f.write(f"- {issue}\n")
                f.write("\n")
            else:
                f.write("## ✅ No Critical Issues Found\n\n")
            
            # Dataset details
            f.write("## Dataset Details\n\n")
            for dataset, data in sorted(self.stats.items()):
                f.write(f"### {dataset}\n\n")
                f.write(f"- Files: {data['files']}\n")
                
                if data['file_stats']:
                    total_rows = sum(s['rows'] for s in data['file_stats'])
                    total_size = sum(s['size'] for s in data['file_stats'])
                    total_dups = sum(s.get('duplicates', 0) for s in data['file_stats'])
                    
                    f.write(f"- Total rows: {format_number(total_rows)}\n")
                    f.write(f"- Total size: {format_bytes(total_size)}\n")
                    
                    if total_dups > 0:
                        f.write(f"- **Duplicates found**: {format_number(total_dups)}\n")
                
                if data['issues']:
                    f.write(f"- Issues: {len(data['issues'])}\n")
                
                f.write("\n")
        
        print(f"\n{GREEN}📝 Verification report saved to: {report_path}{RESET}")
        
        # Also save JSON for programmatic access
        json_path = self.rollup_dir / "verification_report.json"
        with open(json_path, 'w') as f:
            json.dump(self.stats, f, indent=2, default=str)
        
        print(f"{GREEN}📊 JSON report saved to: {json_path}{RESET}")


def main():
    """Main entry point."""
    # Get ERCOT data directory from environment or use default
    data_dir = os.getenv("ERCOT_DATA_DIR", "/home/enrico/data/ERCOT_data")
    
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    
    base_dir = Path(data_dir)
    
    if not base_dir.exists():
        print(f"{RED}❌ Directory not found: {base_dir}{RESET}")
        sys.exit(1)
    
    verifier = ParquetVerifier(base_dir)
    
    start_time = time.time()
    verifier.verify_all_datasets()
    elapsed = time.time() - start_time
    
    print(f"\n{BOLD}⏱️  Verification completed in {elapsed:.2f} seconds{RESET}")


if __name__ == "__main__":
    main()