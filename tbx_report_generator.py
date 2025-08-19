#!/usr/bin/env python3
"""
TBX Report Generator - Creates comprehensive monthly and quarterly reports
Designed to work with Claude Code subagent for automated analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class TBXReportGenerator:
    def __init__(self, data_dir: Path = None, output_dir: Path = None):
        self.data_dir = data_dir or Path("/home/enrico/data/ERCOT_data/tbx_results_all_nodes")
        self.output_dir = output_dir or Path("/home/enrico/data/ERCOT_data/tbx_results_all_nodes/reports")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create subdirectories
        self.monthly_dir = self.output_dir / "monthly"
        self.quarterly_dir = self.output_dir / "quarterly"
        self.email_dir = self.output_dir / "email_templates"
        self.blog_dir = self.output_dir / "blog_posts"
        
        for dir in [self.monthly_dir, self.quarterly_dir, self.email_dir, self.blog_dir]:
            dir.mkdir(exist_ok=True, parents=True)
    
    def load_all_data(self) -> pd.DataFrame:
        """Load all TBX results"""
        all_data = []
        
        # Load all daily files
        for file in sorted(self.data_dir.glob("tbx_daily_*_all_nodes.parquet")):
            df = pd.read_parquet(file)
            all_data.append(df)
        
        if not all_data:
            raise ValueError("No TBX data files found")
        
        combined = pd.concat(all_data, ignore_index=True)
        combined['date'] = pd.to_datetime(combined['date'])
        return combined
    
    def generate_monthly_report(self, year: int, month: int, data: pd.DataFrame) -> Dict:
        """Generate comprehensive monthly report"""
        
        # Filter data for the month
        month_data = data[(data['year'] == year) & (data['month'] == month)]
        
        if month_data.empty:
            return None
        
        # Calculate key metrics
        report = {
            'period': f"{year}-{month:02d}",
            'period_type': 'monthly',
            'generated_at': datetime.now().isoformat(),
            'summary': {},
            'top_performers': [],
            'statistics': {},
            'insights': [],
            'comparisons': {}
        }
        
        # Summary metrics
        report['summary'] = {
            'total_tb2_revenue': float(month_data['tb2_revenue'].sum()),
            'total_tb4_revenue': float(month_data['tb4_revenue'].sum()),
            'avg_tb2_daily': float(month_data['tb2_revenue'].mean()),
            'avg_tb4_daily': float(month_data['tb4_revenue'].mean()),
            'unique_nodes': int(month_data['node'].nunique()),
            'total_days': int(month_data['date'].nunique()),
            'tb4_premium': float((month_data['tb4_revenue'].sum() / month_data['tb2_revenue'].sum() - 1) * 100)
        }
        
        # Top performers
        node_revenues = month_data.groupby('node').agg({
            'tb2_revenue': 'sum',
            'tb4_revenue': 'sum',
            'price_mean': 'mean',
            'price_std': 'mean'
        }).sort_values('tb4_revenue', ascending=False)
        
        for node in node_revenues.head(20).index:
            report['top_performers'].append({
                'node': node,
                'tb2_revenue': float(node_revenues.loc[node, 'tb2_revenue']),
                'tb4_revenue': float(node_revenues.loc[node, 'tb4_revenue']),
                'avg_price': float(node_revenues.loc[node, 'price_mean']),
                'volatility': float(node_revenues.loc[node, 'price_std'])
            })
        
        # Statistical analysis
        report['statistics'] = {
            'price_stats': {
                'mean': float(month_data['price_mean'].mean()),
                'std': float(month_data['price_std'].mean()),
                'min': float(month_data['price_min'].min()),
                'max': float(month_data['price_max'].max())
            },
            'revenue_distribution': {
                'tb2_p25': float(month_data['tb2_revenue'].quantile(0.25)),
                'tb2_p50': float(month_data['tb2_revenue'].quantile(0.50)),
                'tb2_p75': float(month_data['tb2_revenue'].quantile(0.75)),
                'tb4_p25': float(month_data['tb4_revenue'].quantile(0.25)),
                'tb4_p50': float(month_data['tb4_revenue'].quantile(0.50)),
                'tb4_p75': float(month_data['tb4_revenue'].quantile(0.75))
            }
        }
        
        # Previous month comparison
        if month > 1:
            prev_month_data = data[(data['year'] == year) & (data['month'] == month - 1)]
        else:
            prev_month_data = data[(data['year'] == year - 1) & (data['month'] == 12)]
        
        if not prev_month_data.empty:
            prev_tb4_total = prev_month_data['tb4_revenue'].sum()
            curr_tb4_total = month_data['tb4_revenue'].sum()
            
            report['comparisons']['vs_previous_month'] = {
                'tb4_revenue_change': float(curr_tb4_total - prev_tb4_total),
                'tb4_revenue_change_pct': float((curr_tb4_total / prev_tb4_total - 1) * 100),
                'volatility_change': float(month_data['price_std'].mean() - prev_month_data['price_std'].mean())
            }
        
        # Generate insights
        report['insights'] = self.generate_insights(month_data, report)
        
        return report
    
    def generate_quarterly_report(self, year: int, quarter: int, data: pd.DataFrame) -> Dict:
        """Generate comprehensive quarterly report"""
        
        # Define quarter months
        quarter_months = {
            1: [1, 2, 3],
            2: [4, 5, 6],
            3: [7, 8, 9],
            4: [10, 11, 12]
        }
        
        # Filter data for the quarter
        months = quarter_months[quarter]
        quarter_data = data[(data['year'] == year) & (data['month'].isin(months))]
        
        if quarter_data.empty:
            return None
        
        report = {
            'period': f"{year}-Q{quarter}",
            'period_type': 'quarterly',
            'generated_at': datetime.now().isoformat(),
            'summary': {},
            'top_performers': [],
            'monthly_breakdown': [],
            'market_analysis': {},
            'investment_metrics': {}
        }
        
        # Summary metrics
        report['summary'] = {
            'total_tb2_revenue': float(quarter_data['tb2_revenue'].sum()),
            'total_tb4_revenue': float(quarter_data['tb4_revenue'].sum()),
            'avg_tb2_daily': float(quarter_data['tb2_revenue'].mean()),
            'avg_tb4_daily': float(quarter_data['tb4_revenue'].mean()),
            'unique_nodes': int(quarter_data['node'].nunique()),
            'total_days': int(quarter_data['date'].nunique()),
            'tb4_premium': float((quarter_data['tb4_revenue'].sum() / quarter_data['tb2_revenue'].sum() - 1) * 100)
        }
        
        # Monthly breakdown
        for month in months:
            month_subset = quarter_data[quarter_data['month'] == month]
            if not month_subset.empty:
                report['monthly_breakdown'].append({
                    'month': month,
                    'tb2_revenue': float(month_subset['tb2_revenue'].sum()),
                    'tb4_revenue': float(month_subset['tb4_revenue'].sum()),
                    'days': int(month_subset['date'].nunique())
                })
        
        # Top performers
        node_revenues = quarter_data.groupby('node').agg({
            'tb2_revenue': 'sum',
            'tb4_revenue': 'sum',
            'price_mean': 'mean',
            'price_std': 'mean',
            'date': 'count'
        }).sort_values('tb4_revenue', ascending=False)
        
        for node in node_revenues.head(25).index:
            report['top_performers'].append({
                'node': node,
                'tb2_revenue': float(node_revenues.loc[node, 'tb2_revenue']),
                'tb4_revenue': float(node_revenues.loc[node, 'tb4_revenue']),
                'tb2_per_mw_month': float(node_revenues.loc[node, 'tb2_revenue'] / 3),
                'tb4_per_mw_month': float(node_revenues.loc[node, 'tb4_revenue'] / 3),
                'days_active': int(node_revenues.loc[node, 'date'])
            })
        
        # Investment metrics
        report['investment_metrics'] = self.calculate_investment_metrics(quarter_data)
        
        return report
    
    def generate_insights(self, data: pd.DataFrame, report: Dict) -> List[str]:
        """Generate AI-style insights from the data"""
        insights = []
        
        # Top performer insight
        if report['top_performers']:
            top_node = report['top_performers'][0]
            insights.append(
                f"Top performing node {top_node['node']} generated ${top_node['tb4_revenue']:,.2f} "
                f"in TB4 revenue, representing {top_node['tb4_revenue']/report['summary']['total_tb4_revenue']*100:.1f}% "
                f"of total market opportunity."
            )
        
        # Volatility insight
        avg_volatility = data['price_std'].mean()
        if avg_volatility > 30:
            insights.append(
                f"High price volatility (σ=${avg_volatility:.2f}) indicates strong arbitrage opportunities "
                f"for battery storage systems."
            )
        elif avg_volatility < 15:
            insights.append(
                f"Low price volatility (σ=${avg_volatility:.2f}) suggests limited arbitrage opportunities "
                f"in current market conditions."
            )
        
        # Geographic patterns
        if 'RN' in ' '.join([p['node'] for p in report['top_performers'][:10]]):
            rn_count = sum(1 for p in report['top_performers'][:10] if 'RN' in p['node'])
            insights.append(
                f"{rn_count} of top 10 nodes are renewable sites (_RN), highlighting co-location benefits."
            )
        
        # TB4 premium insight
        tb4_premium = report['summary']['tb4_premium']
        if tb4_premium > 60:
            insights.append(
                f"TB4 systems show {tb4_premium:.1f}% premium over TB2, strongly favoring longer-duration storage."
            )
        
        return insights
    
    def calculate_investment_metrics(self, data: pd.DataFrame) -> Dict:
        """Calculate investment metrics for BESS"""
        
        # Assume costs from bess_cost_mapping
        tb2_cost_per_mw = 500000  # $500/kW for 2-hour
        tb4_cost_per_mw = 800000  # $800/kW for 4-hour
        
        # Top node metrics
        top_nodes = data.groupby('node')['tb4_revenue'].sum().nlargest(10)
        
        metrics = {
            'tb2_metrics': {
                'capex_per_mw': tb2_cost_per_mw,
                'annual_revenue_top10_avg': float(data[data['node'].isin(top_nodes.index)]['tb2_revenue'].sum() / 10 * 365 / data['date'].nunique()),
                'simple_payback_years': float(tb2_cost_per_mw / (data[data['node'].isin(top_nodes.index)]['tb2_revenue'].sum() / 10 * 365 / data['date'].nunique()))
            },
            'tb4_metrics': {
                'capex_per_mw': tb4_cost_per_mw,
                'annual_revenue_top10_avg': float(data[data['node'].isin(top_nodes.index)]['tb4_revenue'].sum() / 10 * 365 / data['date'].nunique()),
                'simple_payback_years': float(tb4_cost_per_mw / (data[data['node'].isin(top_nodes.index)]['tb4_revenue'].sum() / 10 * 365 / data['date'].nunique()))
            }
        }
        
        return metrics
    
    def create_markdown_report(self, report: Dict) -> str:
        """Convert report dict to markdown format"""
        
        md = []
        
        # Header
        if report['period_type'] == 'monthly':
            md.append(f"# TBX Market Report - {report['period']}")
        else:
            md.append(f"# TBX Quarterly Report - {report['period']}")
        
        md.append(f"\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n")
        
        # Executive Summary
        md.append("## Executive Summary\n")
        md.append(f"- **Total TB2 Revenue**: ${report['summary']['total_tb2_revenue']:,.2f}")
        md.append(f"- **Total TB4 Revenue**: ${report['summary']['total_tb4_revenue']:,.2f}")
        md.append(f"- **TB4 Premium**: {report['summary']['tb4_premium']:.1f}%")
        md.append(f"- **Unique Settlement Points**: {report['summary']['unique_nodes']:,}")
        md.append(f"- **Analysis Period**: {report['summary']['total_days']} days\n")
        
        # Key Insights
        if 'insights' in report and report['insights']:
            md.append("## Key Insights\n")
            for insight in report['insights']:
                md.append(f"- {insight}")
            md.append("")
        
        # Top Performers
        md.append("## Top Performing Nodes\n")
        md.append("| Rank | Settlement Point | TB2 Revenue | TB4 Revenue | Volatility |")
        md.append("|------|------------------|-------------|-------------|------------|")
        
        for i, node in enumerate(report['top_performers'][:10], 1):
            md.append(f"| {i} | {node['node'][:20]} | ${node['tb2_revenue']:,.2f} | ${node['tb4_revenue']:,.2f} | ${node.get('volatility', 0):.2f} |")
        md.append("")
        
        # Period Comparison
        if 'comparisons' in report and 'vs_previous_month' in report['comparisons']:
            comp = report['comparisons']['vs_previous_month']
            md.append("## Period Comparison\n")
            md.append("### vs Previous Month")
            direction = "↑" if comp['tb4_revenue_change'] > 0 else "↓"
            md.append(f"- Revenue Change: {direction} ${abs(comp['tb4_revenue_change']):,.2f} ({comp['tb4_revenue_change_pct']:+.1f}%)")
            md.append(f"- Volatility Change: {comp['volatility_change']:+.2f}\n")
        
        # Investment Metrics (for quarterly)
        if 'investment_metrics' in report:
            md.append("## Investment Analysis\n")
            metrics = report['investment_metrics']
            md.append("### TB2 System (2-hour)")
            md.append(f"- CAPEX: ${metrics['tb2_metrics']['capex_per_mw']:,.0f}/MW")
            md.append(f"- Annual Revenue (Top 10 avg): ${metrics['tb2_metrics']['annual_revenue_top10_avg']:,.0f}")
            md.append(f"- Simple Payback: {metrics['tb2_metrics']['simple_payback_years']:.1f} years\n")
            
            md.append("### TB4 System (4-hour)")
            md.append(f"- CAPEX: ${metrics['tb4_metrics']['capex_per_mw']:,.0f}/MW")
            md.append(f"- Annual Revenue (Top 10 avg): ${metrics['tb4_metrics']['annual_revenue_top10_avg']:,.0f}")
            md.append(f"- Simple Payback: {metrics['tb4_metrics']['simple_payback_years']:.1f} years\n")
        
        # Market Statistics
        if 'statistics' in report:
            md.append("## Market Statistics\n")
            stats = report['statistics']
            md.append("### Price Statistics")
            md.append(f"- Average: ${stats['price_stats']['mean']:.2f}/MWh")
            md.append(f"- Std Dev: ${stats['price_stats']['std']:.2f}")
            md.append(f"- Range: ${stats['price_stats']['min']:.2f} - ${stats['price_stats']['max']:.2f}\n")
        
        return "\n".join(md)
    
    def create_email_template(self, report: Dict) -> str:
        """Create email template from report"""
        
        period = report['period']
        tb4_total = report['summary']['total_tb4_revenue']
        top_node = report['top_performers'][0] if report['top_performers'] else None
        
        top_node_name = top_node['node'] if top_node else 'N/A'
        top_node_revenue = f"${top_node['tb4_revenue']:,.2f}" if top_node else "$0"
        
        template = f"""Subject: ERCOT TBX Market Report - {period}

Dear Stakeholder,

Please find below the key highlights from the {period} TBX market analysis:

**Performance Summary:**
• Total TB4 Revenue Opportunity: ${tb4_total:,.0f}
• TB4 Premium over TB2: {report['summary']['tb4_premium']:.1f}%
• Settlement Points Analyzed: {report['summary']['unique_nodes']:,}

**Top Performer:**
{top_node_name}: {top_node_revenue}

**Key Insights:**
{chr(10).join(['• ' + i for i in report.get('insights', [])[:3]])}

**Investment Outlook:**
Battery storage continues to show strong arbitrage opportunities in ERCOT markets, 
with 4-hour systems demonstrating superior economics at high-volatility nodes.

For the full report and detailed analysis, please see the attached documents.

Best regards,
TBX Analytics Team

---
This is an automated report generated by the TBX Analysis System
"""
        return template
    
    def create_blog_post(self, report: Dict) -> str:
        """Create blog post draft from report"""
        
        period = report['period']
        
        post = f"""# ERCOT Battery Storage Arbitrage Analysis: {period} Update

## Market Overview

The ERCOT electricity market continues to present significant opportunities for battery energy storage systems (BESS), 
with our latest analysis of {report['summary']['unique_nodes']:,} settlement points revealing total TB4 arbitrage 
revenues of ${report['summary']['total_tb4_revenue']:,.0f} for the period.

## Key Findings

### 1. Duration Matters
Our analysis shows that 4-hour battery systems (TB4) generated a {report['summary']['tb4_premium']:.1f}% premium 
over 2-hour systems (TB2), highlighting the value of longer-duration storage in capturing price spreads.

### 2. Location is Critical
The top-performing settlement point, {report['top_performers'][0]['node'] if report['top_performers'] else 'N/A'}, 
achieved ${report['top_performers'][0]['tb4_revenue']:,.2f} in TB4 revenue, 
demonstrating the importance of strategic siting.

### 3. Market Volatility Drives Value
{"High volatility" if 'statistics' in report and report['statistics']['price_stats']['std'] > 25 else "Market conditions"} during the period created 
{"favorable" if report['summary']['tb4_premium'] > 50 else "moderate"} arbitrage opportunities 
for battery storage systems.

## Top 5 Settlement Points

"""
        
        # Add top 5 table
        for i, node in enumerate(report['top_performers'][:5], 1):
            post += f"{i}. **{node['node']}**: ${node['tb4_revenue']:,.2f} TB4 revenue\n"
        
        post += f"""

## Investment Implications

Based on current market conditions and assuming standard BESS costs:
- Simple payback periods range from 5-8 years for well-sited projects
- Co-location with renewable generation sites shows particular promise
- 4-hour systems demonstrate superior economics despite higher upfront costs

## Looking Ahead

As ERCOT continues to integrate more renewable energy and retire thermal generation, 
we expect price volatility and arbitrage opportunities to {"increase" if report['summary']['tb4_premium'] > 50 else "remain stable"}.

Investors and developers should focus on:
1. High-volatility nodes with strong grid connections
2. 4-hour or longer duration systems
3. Co-location opportunities with renewable generation

---

*This analysis is based on historical ERCOT settlement point prices with 90% round-trip efficiency. 
Actual revenues will vary based on operational strategies and market participation.*

**Data Source**: ERCOT DAM Settlement Point Prices
**Analysis Period**: {report['period']}
**Methodology**: TB2/TB4 perfect foresight arbitrage with 90% efficiency
"""
        
        return post
    
    def generate_all_reports(self):
        """Generate all monthly and quarterly reports"""
        
        print("📊 Loading all TBX data...")
        data = self.load_all_data()
        
        # Get unique year-month combinations
        data['year_month'] = data['date'].dt.to_period('M')
        periods = data['year_month'].unique()
        
        print(f"📅 Found {len(periods)} months of data")
        
        # Generate monthly reports
        monthly_reports = []
        for period in periods:
            year = period.year
            month = period.month
            
            print(f"  📝 Generating report for {year}-{month:02d}...")
            report = self.generate_monthly_report(year, month, data)
            
            if report:
                # Save JSON
                json_file = self.monthly_dir / f"tbx_monthly_{year}_{month:02d}.json"
                with open(json_file, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                
                # Save Markdown
                md_content = self.create_markdown_report(report)
                md_file = self.monthly_dir / f"tbx_monthly_{year}_{month:02d}.md"
                with open(md_file, 'w') as f:
                    f.write(md_content)
                
                # Save email template
                email_content = self.create_email_template(report)
                email_file = self.email_dir / f"tbx_email_{year}_{month:02d}.txt"
                with open(email_file, 'w') as f:
                    f.write(email_content)
                
                monthly_reports.append(report)
        
        # Generate quarterly reports
        quarters = data.groupby([data['date'].dt.year, data['date'].dt.quarter]).size()
        
        for (year, quarter), _ in quarters.items():
            print(f"  📊 Generating report for {year}-Q{quarter}...")
            report = self.generate_quarterly_report(year, quarter, data)
            
            if report:
                # Save JSON
                json_file = self.quarterly_dir / f"tbx_quarterly_{year}_Q{quarter}.json"
                with open(json_file, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                
                # Save Markdown
                md_content = self.create_markdown_report(report)
                md_file = self.quarterly_dir / f"tbx_quarterly_{year}_Q{quarter}.md"
                with open(md_file, 'w') as f:
                    f.write(md_content)
                
                # Save blog post
                blog_content = self.create_blog_post(report)
                blog_file = self.blog_dir / f"tbx_blog_{year}_Q{quarter}.md"
                with open(blog_file, 'w') as f:
                    f.write(blog_content)
        
        # Create index file
        self.create_index_file(monthly_reports)
        
        print(f"\n✅ Generated {len(monthly_reports)} monthly reports")
        print(f"✅ Generated {len(quarters)} quarterly reports")
        print(f"📁 Reports saved to: {self.output_dir}")
        
        return len(monthly_reports), len(quarters)
    
    def create_index_file(self, reports: List[Dict]):
        """Create an index file for all reports"""
        
        index = {
            'generated_at': datetime.now().isoformat(),
            'total_reports': len(reports),
            'report_list': []
        }
        
        for report in reports:
            index['report_list'].append({
                'period': report['period'],
                'type': report['period_type'],
                'total_tb4_revenue': report['summary']['total_tb4_revenue'],
                'top_node': report['top_performers'][0]['node'] if report['top_performers'] else None
            })
        
        with open(self.output_dir / 'index.json', 'w') as f:
            json.dump(index, f, indent=2)

if __name__ == "__main__":
    generator = TBXReportGenerator()
    generator.generate_all_reports()