#!/usr/bin/env python3
"""
TBX Report Generator - Creates monthly and quarterly reports in MD and JSON formats
For display on the viewing website
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, date
import json
import argparse
from typing import Dict, List, Any

class TBXReportGenerator:
    def __init__(self, data_dir: Path, output_dir: Path):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.reports_dir = self.output_dir / "reports"
        self.monthly_dir = self.reports_dir / "monthly"
        self.quarterly_dir = self.reports_dir / "quarterly"
        
        # Create directories
        self.monthly_dir.mkdir(parents=True, exist_ok=True)
        self.quarterly_dir.mkdir(parents=True, exist_ok=True)
        
        # Load all data
        self.data = self.load_all_data()
        
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """Load all TBX data files"""
        data = {
            'daily': {},
            'monthly': {},
            'annual': {}
        }
        
        years = [2021, 2022, 2023, 2024, 2025]
        
        for year in years:
            # Load daily data
            daily_file = self.data_dir / f"tbx_daily_{year}.parquet"
            if daily_file.exists():
                df = pd.read_parquet(daily_file)
                df['date'] = pd.to_datetime(df['date'])
                data['daily'][year] = df
                
            # Load monthly data
            monthly_file = self.data_dir / f"tbx_monthly_{year}.parquet"
            if monthly_file.exists():
                df = pd.read_parquet(monthly_file)
                df['year'] = year
                data['monthly'][year] = df
                
            # Load annual data
            annual_file = self.data_dir / f"tbx_annual_{year}.parquet"
            if annual_file.exists():
                df = pd.read_parquet(annual_file)
                df['year'] = year
                data['annual'][year] = df
                
        return data
    
    def generate_monthly_report(self, year: int, month: int) -> Dict[str, Any]:
        """Generate report for a specific month"""
        
        # Get monthly data
        if year not in self.data['monthly']:
            return None
            
        monthly_df = self.data['monthly'][year]
        month_data = monthly_df[monthly_df['month'] == month]
        
        if month_data.empty:
            return None
            
        # Get daily data for detailed analysis
        daily_df = self.data['daily'].get(year, pd.DataFrame())
        if not daily_df.empty:
            month_daily = daily_df[daily_df['date'].dt.month == month]
        else:
            month_daily = pd.DataFrame()
        
        # Calculate statistics
        report = {
            'metadata': {
                'year': year,
                'month': month,
                'month_name': pd.Timestamp(year=year, month=month, day=1).strftime('%B'),
                'generated_at': datetime.now().isoformat(),
                'report_type': 'monthly'
            },
            'summary': {
                'total_tb2_da_revenue': float(month_data['tb2_da_revenue'].sum()),
                'total_tb2_rt_revenue': float(month_data['tb2_rt_revenue'].sum()),
                'total_tb4_da_revenue': float(month_data['tb4_da_revenue'].sum()),
                'total_tb4_rt_revenue': float(month_data['tb4_rt_revenue'].sum()),
                'avg_tb2_da_revenue': float(month_data['tb2_da_revenue'].mean()),
                'avg_tb4_da_revenue': float(month_data['tb4_da_revenue'].mean()),
                'days_in_month': int(month_data['days_count'].iloc[0]) if not month_data.empty else 0
            },
            'top_performers': {
                'tb2': month_data.nlargest(5, 'tb2_da_revenue')[['node', 'tb2_da_revenue', 'tb4_da_revenue']].to_dict('records'),
                'tb4': month_data.nlargest(5, 'tb4_da_revenue')[['node', 'tb2_da_revenue', 'tb4_da_revenue']].to_dict('records')
            },
            'daily_statistics': {}
        }
        
        # Add daily statistics if available
        if not month_daily.empty:
            daily_stats = {
                'best_day': {
                    'date': month_daily.groupby('date')['tb2_da_revenue'].sum().idxmax().isoformat(),
                    'total_revenue': float(month_daily.groupby('date')['tb2_da_revenue'].sum().max())
                },
                'worst_day': {
                    'date': month_daily.groupby('date')['tb2_da_revenue'].sum().idxmin().isoformat(),
                    'total_revenue': float(month_daily.groupby('date')['tb2_da_revenue'].sum().min())
                },
                'daily_avg': float(month_daily.groupby('date')['tb2_da_revenue'].sum().mean()),
                'daily_std': float(month_daily.groupby('date')['tb2_da_revenue'].sum().std())
            }
            report['daily_statistics'] = daily_stats
            
        # Compare to previous month
        if month > 1:
            prev_month_data = monthly_df[monthly_df['month'] == month - 1]
        else:
            # Check previous year December
            if year - 1 in self.data['monthly']:
                prev_monthly_df = self.data['monthly'][year - 1]
                prev_month_data = prev_monthly_df[prev_monthly_df['month'] == 12]
            else:
                prev_month_data = pd.DataFrame()
                
        if not prev_month_data.empty:
            current_total = month_data['tb2_da_revenue'].sum()
            prev_total = prev_month_data['tb2_da_revenue'].sum()
            report['comparison'] = {
                'vs_previous_month': {
                    'change_dollars': float(current_total - prev_total),
                    'change_percent': float((current_total - prev_total) / prev_total * 100) if prev_total != 0 else 0
                }
            }
            
        return report
    
    def generate_quarterly_report(self, year: int, quarter: int) -> Dict[str, Any]:
        """Generate report for a specific quarter"""
        
        # Define quarter months
        quarter_months = {
            1: [1, 2, 3],
            2: [4, 5, 6],
            3: [7, 8, 9],
            4: [10, 11, 12]
        }
        
        months = quarter_months[quarter]
        
        # Get quarterly data
        if year not in self.data['monthly']:
            return None
            
        monthly_df = self.data['monthly'][year]
        quarter_data = monthly_df[monthly_df['month'].isin(months)]
        
        if quarter_data.empty:
            return None
            
        # Aggregate by node
        quarter_agg = quarter_data.groupby('node').agg({
            'tb2_da_revenue': 'sum',
            'tb2_rt_revenue': 'sum',
            'tb4_da_revenue': 'sum',
            'tb4_rt_revenue': 'sum',
            'days_count': 'sum'
        }).reset_index()
        
        # Get daily data for detailed analysis
        daily_df = self.data['daily'].get(year, pd.DataFrame())
        if not daily_df.empty:
            quarter_daily = daily_df[daily_df['date'].dt.month.isin(months)]
        else:
            quarter_daily = pd.DataFrame()
        
        report = {
            'metadata': {
                'year': year,
                'quarter': quarter,
                'quarter_name': f'Q{quarter} {year}',
                'months': months,
                'generated_at': datetime.now().isoformat(),
                'report_type': 'quarterly'
            },
            'summary': {
                'total_tb2_da_revenue': float(quarter_agg['tb2_da_revenue'].sum()),
                'total_tb2_rt_revenue': float(quarter_agg['tb2_rt_revenue'].sum()),
                'total_tb4_da_revenue': float(quarter_agg['tb4_da_revenue'].sum()),
                'total_tb4_rt_revenue': float(quarter_agg['tb4_rt_revenue'].sum()),
                'avg_tb2_da_revenue': float(quarter_agg['tb2_da_revenue'].mean()),
                'avg_tb4_da_revenue': float(quarter_agg['tb4_da_revenue'].mean()),
                'total_days': int(quarter_agg['days_count'].iloc[0]) if not quarter_agg.empty else 0
            },
            'top_performers': {
                'tb2': quarter_agg.nlargest(10, 'tb2_da_revenue')[['node', 'tb2_da_revenue', 'tb4_da_revenue']].to_dict('records'),
                'tb4': quarter_agg.nlargest(10, 'tb4_da_revenue')[['node', 'tb2_da_revenue', 'tb4_da_revenue']].to_dict('records')
            },
            'monthly_breakdown': []
        }
        
        # Add monthly breakdown
        for month in months:
            month_subset = quarter_data[quarter_data['month'] == month]
            if not month_subset.empty:
                month_summary = {
                    'month': month,
                    'month_name': pd.Timestamp(year=year, month=month, day=1).strftime('%B'),
                    'total_tb2_revenue': float(month_subset['tb2_da_revenue'].sum()),
                    'total_tb4_revenue': float(month_subset['tb4_da_revenue'].sum()),
                    'top_node': month_subset.nlargest(1, 'tb2_da_revenue')['node'].iloc[0]
                }
                report['monthly_breakdown'].append(month_summary)
        
        # Compare to previous quarter
        if quarter > 1:
            prev_quarter_months = quarter_months[quarter - 1]
            prev_quarter_data = monthly_df[monthly_df['month'].isin(prev_quarter_months)]
        else:
            # Check previous year Q4
            if year - 1 in self.data['monthly']:
                prev_monthly_df = self.data['monthly'][year - 1]
                prev_quarter_data = prev_monthly_df[prev_monthly_df['month'].isin([10, 11, 12])]
            else:
                prev_quarter_data = pd.DataFrame()
                
        if not prev_quarter_data.empty:
            current_total = quarter_agg['tb2_da_revenue'].sum()
            prev_total = prev_quarter_data['tb2_da_revenue'].sum()
            report['comparison'] = {
                'vs_previous_quarter': {
                    'change_dollars': float(current_total - prev_total),
                    'change_percent': float((current_total - prev_total) / prev_total * 100) if prev_total != 0 else 0
                }
            }
            
        # Year-over-year comparison
        if year - 1 in self.data['monthly']:
            prev_year_df = self.data['monthly'][year - 1]
            prev_year_quarter = prev_year_df[prev_year_df['month'].isin(months)]
            if not prev_year_quarter.empty:
                current_total = quarter_agg['tb2_da_revenue'].sum()
                prev_year_total = prev_year_quarter['tb2_da_revenue'].sum()
                report['comparison']['vs_previous_year'] = {
                    'change_dollars': float(current_total - prev_year_total),
                    'change_percent': float((current_total - prev_year_total) / prev_year_total * 100) if prev_year_total != 0 else 0
                }
                
        return report
    
    def report_to_markdown(self, report: Dict[str, Any]) -> str:
        """Convert report to Markdown format"""
        
        if report is None:
            return ""
            
        md = []
        meta = report['metadata']
        
        # Header
        if meta['report_type'] == 'monthly':
            md.append(f"# TBX Monthly Report - {meta['month_name']} {meta['year']}")
        else:
            md.append(f"# TBX Quarterly Report - {meta['quarter_name']}")
            
        md.append(f"\n*Generated: {meta['generated_at'][:10]}*\n")
        
        # Summary
        md.append("## Executive Summary\n")
        summary = report['summary']
        
        if meta['report_type'] == 'monthly':
            md.append(f"- **Period**: {meta['month_name']} {meta['year']} ({summary.get('days_in_month', 'N/A')} days)")
        else:
            md.append(f"- **Period**: Q{meta['quarter']} {meta['year']} ({summary.get('total_days', 'N/A')} days)")
            
        md.append(f"- **Total TB2 Revenue**: ${summary['total_tb2_da_revenue']:,.2f}")
        md.append(f"- **Total TB4 Revenue**: ${summary['total_tb4_da_revenue']:,.2f}")
        md.append(f"- **Average TB2 Revenue/Node**: ${summary['avg_tb2_da_revenue']:,.2f}")
        md.append(f"- **TB4 Premium**: {(summary['avg_tb4_da_revenue']/summary['avg_tb2_da_revenue']-1)*100:.1f}%")
        
        # Comparison
        if 'comparison' in report:
            md.append("\n## Period Comparison\n")
            comp = report['comparison']
            
            if 'vs_previous_month' in comp:
                vs_prev = comp['vs_previous_month']
                direction = "↑" if vs_prev['change_dollars'] > 0 else "↓"
                md.append(f"### vs Previous Month")
                md.append(f"- Change: {direction} ${abs(vs_prev['change_dollars']):,.2f} ({vs_prev['change_percent']:+.1f}%)")
                
            if 'vs_previous_quarter' in comp:
                vs_prev = comp['vs_previous_quarter']
                direction = "↑" if vs_prev['change_dollars'] > 0 else "↓"
                md.append(f"### vs Previous Quarter")
                md.append(f"- Change: {direction} ${abs(vs_prev['change_dollars']):,.2f} ({vs_prev['change_percent']:+.1f}%)")
                
            if 'vs_previous_year' in comp:
                vs_prev = comp['vs_previous_year']
                direction = "↑" if vs_prev['change_dollars'] > 0 else "↓"
                md.append(f"### vs Same Quarter Last Year")
                md.append(f"- Change: {direction} ${abs(vs_prev['change_dollars']):,.2f} ({vs_prev['change_percent']:+.1f}%)")
        
        # Top Performers
        md.append("\n## Top Performing Nodes\n")
        
        md.append("### TB2 Leaders")
        md.append("| Rank | Node | TB2 Revenue | TB4 Revenue |")
        md.append("|------|------|-------------|-------------|")
        
        for i, node in enumerate(report['top_performers']['tb2'][:5], 1):
            md.append(f"| {i} | {node['node']} | ${node['tb2_da_revenue']:,.2f} | ${node['tb4_da_revenue']:,.2f} |")
            
        # Monthly breakdown for quarterly reports
        if 'monthly_breakdown' in report and report['monthly_breakdown']:
            md.append("\n## Monthly Breakdown\n")
            md.append("| Month | TB2 Revenue | TB4 Revenue | Top Node |")
            md.append("|-------|-------------|-------------|----------|")
            
            for month_data in report['monthly_breakdown']:
                md.append(f"| {month_data['month_name']} | ${month_data['total_tb2_revenue']:,.2f} | ${month_data['total_tb4_revenue']:,.2f} | {month_data['top_node']} |")
        
        # Daily statistics for monthly reports
        if 'daily_statistics' in report and report['daily_statistics']:
            daily = report['daily_statistics']
            md.append("\n## Daily Statistics\n")
            md.append(f"- **Best Day**: {daily['best_day']['date']} (${daily['best_day']['total_revenue']:,.2f})")
            md.append(f"- **Worst Day**: {daily['worst_day']['date']} (${daily['worst_day']['total_revenue']:,.2f})")
            md.append(f"- **Daily Average**: ${daily['daily_avg']:,.2f}")
            md.append(f"- **Daily Std Dev**: ${daily['daily_std']:,.2f}")
        
        # Footer
        md.append("\n---")
        md.append("*TBX Calculator | 90% Round-Trip Efficiency | Day-Ahead Market*")
        
        return "\n".join(md)
    
    def save_report(self, report: Dict[str, Any], report_type: str, year: int, period: int):
        """Save report in both JSON and Markdown formats"""
        
        if report is None:
            return
            
        # Determine output directory and filename
        if report_type == 'monthly':
            output_dir = self.monthly_dir
            filename_base = f"tbx_monthly_{year}_{period:02d}"
        else:
            output_dir = self.quarterly_dir
            filename_base = f"tbx_quarterly_{year}_Q{period}"
            
        # Save JSON
        json_file = output_dir / f"{filename_base}.json"
        with open(json_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        # Save Markdown
        md_content = self.report_to_markdown(report)
        md_file = output_dir / f"{filename_base}.md"
        with open(md_file, 'w') as f:
            f.write(md_content)
            
        print(f"  ✅ Saved {filename_base}.json and .md")
    
    def generate_all_reports(self):
        """Generate all monthly and quarterly reports"""
        
        print("📊 Generating TBX Reports")
        print("=" * 60)
        
        # Generate monthly reports
        print("\n📅 Monthly Reports:")
        for year in sorted(self.data['monthly'].keys()):
            print(f"\n  Year {year}:")
            for month in range(1, 13):
                report = self.generate_monthly_report(year, month)
                if report:
                    self.save_report(report, 'monthly', year, month)
                    
        # Generate quarterly reports
        print("\n📊 Quarterly Reports:")
        for year in sorted(self.data['monthly'].keys()):
            print(f"\n  Year {year}:")
            for quarter in range(1, 5):
                report = self.generate_quarterly_report(year, quarter)
                if report:
                    self.save_report(report, 'quarterly', year, quarter)
                    
        # Generate index files
        self.generate_index_files()
        
        print("\n✅ All reports generated successfully!")
        print(f"📁 Reports saved to: {self.reports_dir}")
        
    def generate_index_files(self):
        """Generate index files for easy navigation"""
        
        # Monthly index
        monthly_index = {
            'title': 'TBX Monthly Reports Index',
            'generated_at': datetime.now().isoformat(),
            'reports': []
        }
        
        for file in sorted(self.monthly_dir.glob('*.json')):
            if 'index' not in file.name:
                parts = file.stem.split('_')
                year = int(parts[2])
                month = int(parts[3])
                monthly_index['reports'].append({
                    'year': year,
                    'month': month,
                    'filename': file.name,
                    'path': str(file.relative_to(self.reports_dir))
                })
                
        with open(self.monthly_dir / 'index.json', 'w') as f:
            json.dump(monthly_index, f, indent=2)
            
        # Quarterly index
        quarterly_index = {
            'title': 'TBX Quarterly Reports Index',
            'generated_at': datetime.now().isoformat(),
            'reports': []
        }
        
        for file in sorted(self.quarterly_dir.glob('*.json')):
            if 'index' not in file.name:
                parts = file.stem.split('_')
                year = int(parts[2])
                quarter = int(parts[3].replace('Q', ''))
                quarterly_index['reports'].append({
                    'year': year,
                    'quarter': quarter,
                    'filename': file.name,
                    'path': str(file.relative_to(self.reports_dir))
                })
                
        with open(self.quarterly_dir / 'index.json', 'w') as f:
            json.dump(quarterly_index, f, indent=2)
            
        # Master index
        master_index = {
            'title': 'TBX Reports Master Index',
            'generated_at': datetime.now().isoformat(),
            'monthly_reports': monthly_index['reports'],
            'quarterly_reports': quarterly_index['reports'],
            'statistics': {
                'total_monthly_reports': len(monthly_index['reports']),
                'total_quarterly_reports': len(quarterly_index['reports']),
                'years_covered': sorted(list(set([r['year'] for r in monthly_index['reports']]))),
                'latest_report': max([r['year'] * 100 + r['month'] for r in monthly_index['reports']]) if monthly_index['reports'] else 0
            }
        }
        
        with open(self.reports_dir / 'index.json', 'w') as f:
            json.dump(master_index, f, indent=2)
            
        # Create README
        readme_content = f"""# TBX Reports Directory

This directory contains automated monthly and quarterly TBX (Battery Arbitrage) reports.

## Directory Structure
- `monthly/` - Monthly reports in JSON and Markdown formats
- `quarterly/` - Quarterly reports in JSON and Markdown formats
- `index.json` - Master index of all reports

## Report Formats
- `.json` - Machine-readable format for website integration
- `.md` - Human-readable Markdown format

## Report Contents
- Executive summary with total revenues
- Top performing nodes
- Period-over-period comparisons
- Daily/monthly statistics

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
"""
        
        with open(self.reports_dir / 'README.md', 'w') as f:
            f.write(readme_content)

def main():
    parser = argparse.ArgumentParser(description='Generate TBX monthly and quarterly reports')
    parser.add_argument('--data-dir', type=str,
                       default='/home/enrico/data/ERCOT_data/tbx_results',
                       help='TBX data directory')
    parser.add_argument('--output-dir', type=str,
                       default='/home/enrico/data/ERCOT_data/tbx_results',
                       help='Output directory for reports')
    
    args = parser.parse_args()
    
    generator = TBXReportGenerator(args.data_dir, args.output_dir)
    generator.generate_all_reports()

if __name__ == "__main__":
    main()