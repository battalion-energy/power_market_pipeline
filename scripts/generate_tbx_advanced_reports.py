#!/usr/bin/env python3
"""
TBX Advanced Report Generator - Modo Energy Style Analytics
Creates comprehensive market analysis reports with advanced metrics
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, date, timedelta
import json
import argparse
from typing import Dict, List, Any, Tuple
# from scipy import stats  # Optional for advanced statistics

class TBXAdvancedReportGenerator:
    def __init__(self, data_dir: Path, output_dir: Path):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.reports_dir = self.output_dir / "advanced_reports"
        self.monthly_dir = self.reports_dir / "monthly"
        self.quarterly_dir = self.reports_dir / "quarterly"
        self.market_dir = self.reports_dir / "market_insights"
        
        # Create directories
        for dir in [self.monthly_dir, self.quarterly_dir, self.market_dir]:
            dir.mkdir(parents=True, exist_ok=True)
        
        # Load all data
        self.data = self.load_all_data()
        
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """Load all TBX data files with enhanced processing"""
        data = {
            'daily': {},
            'monthly': {},
            'annual': {},
            'combined': pd.DataFrame()
        }
        
        years = [2021, 2022, 2023, 2024, 2025]
        all_daily = []
        
        for year in years:
            # Load daily data
            daily_file = self.data_dir / f"tbx_daily_{year}.parquet"
            if daily_file.exists():
                df = pd.read_parquet(daily_file)
                df['date'] = pd.to_datetime(df['date'])
                df['year'] = year
                df['month'] = df['date'].dt.month
                df['quarter'] = df['date'].dt.quarter
                df['day_of_week'] = df['date'].dt.dayofweek
                df['week_of_year'] = df['date'].dt.isocalendar().week
                data['daily'][year] = df
                all_daily.append(df)
                
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
        
        # Create combined dataset for advanced analytics
        if all_daily:
            data['combined'] = pd.concat(all_daily, ignore_index=True)
            
        return data
    
    def calculate_advanced_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate advanced metrics similar to Modo Energy reports"""
        
        metrics = {}
        
        # Revenue Distribution Analysis
        metrics['revenue_distribution'] = {
            'percentiles': {
                'p10': float(df['tb2_da_revenue'].quantile(0.10)),
                'p25': float(df['tb2_da_revenue'].quantile(0.25)),
                'p50': float(df['tb2_da_revenue'].quantile(0.50)),
                'p75': float(df['tb2_da_revenue'].quantile(0.75)),
                'p90': float(df['tb2_da_revenue'].quantile(0.90)),
                'p95': float(df['tb2_da_revenue'].quantile(0.95)),
                'p99': float(df['tb2_da_revenue'].quantile(0.99))
            },
            'mean': float(df['tb2_da_revenue'].mean()),
            'std': float(df['tb2_da_revenue'].std()),
            'cv': float(df['tb2_da_revenue'].std() / df['tb2_da_revenue'].mean()) if df['tb2_da_revenue'].mean() != 0 else 0,
            'skewness': float(df['tb2_da_revenue'].skew()),
            'kurtosis': float(df['tb2_da_revenue'].kurtosis())
        }
        
        # Volatility Metrics
        if 'date' in df.columns:
            daily_revenues = df.groupby('date')['tb2_da_revenue'].sum()
            if len(daily_revenues) > 1:
                daily_returns = daily_revenues.pct_change().dropna()
                metrics['volatility'] = {
                    'daily_volatility': float(daily_returns.std()),
                    'annualized_volatility': float(daily_returns.std() * np.sqrt(365)),
                    'sharpe_ratio': float(daily_revenues.mean() / daily_revenues.std()) if daily_revenues.std() != 0 else 0,
                    'max_daily_revenue': float(daily_revenues.max()),
                    'min_daily_revenue': float(daily_revenues.min()),
                    'revenue_range': float(daily_revenues.max() - daily_revenues.min())
                }
        
        # Capacity Factor Equivalent
        # Assuming 1MW battery, calculate equivalent capacity factor
        avg_daily_revenue = df['tb2_da_revenue'].mean()
        # Approximate MWh throughput based on 2-hour battery cycling once per day
        estimated_mwh = 2.0  # 1MW * 2 hours
        metrics['capacity_metrics'] = {
            'avg_daily_revenue_per_mw': float(avg_daily_revenue),
            'avg_revenue_per_mwh': float(avg_daily_revenue / estimated_mwh) if estimated_mwh > 0 else 0,
            'annualized_revenue_per_mw': float(avg_daily_revenue * 365),
            'cycling_assumption': '1 cycle per day',
            'efficiency': 0.9
        }
        
        # Node Performance Rankings
        node_performance = df.groupby('node').agg({
            'tb2_da_revenue': ['mean', 'std', 'sum'],
            'tb4_da_revenue': ['mean', 'std', 'sum']
        }).round(2)
        
        top_nodes = node_performance.sort_values(('tb2_da_revenue', 'sum'), ascending=False).head(5)
        
        # Convert to simpler dict structure for JSON serialization
        top_nodes_dict = {}
        for node in top_nodes.index:
            top_nodes_dict[node] = {
                'tb2_mean': float(node_performance.loc[node, ('tb2_da_revenue', 'mean')]),
                'tb2_std': float(node_performance.loc[node, ('tb2_da_revenue', 'std')]),
                'tb2_sum': float(node_performance.loc[node, ('tb2_da_revenue', 'sum')]),
                'tb4_mean': float(node_performance.loc[node, ('tb4_da_revenue', 'mean')]),
                'tb4_std': float(node_performance.loc[node, ('tb4_da_revenue', 'std')]),
                'tb4_sum': float(node_performance.loc[node, ('tb4_da_revenue', 'sum')])
            }
        
        metrics['node_rankings'] = {
            'top_5_nodes': top_nodes_dict,
            'total_nodes_analyzed': len(node_performance),
            'node_concentration': float(top_nodes[('tb2_da_revenue', 'sum')].sum() / 
                                       node_performance[('tb2_da_revenue', 'sum')].sum())
        }
        
        # Seasonal Patterns
        if 'month' in df.columns:
            monthly_avg = df.groupby('month')['tb2_da_revenue'].mean()
            metrics['seasonality'] = {
                'monthly_averages': monthly_avg.to_dict(),
                'peak_month': int(monthly_avg.idxmax()),
                'trough_month': int(monthly_avg.idxmin()),
                'seasonal_spread': float(monthly_avg.max() - monthly_avg.min())
            }
        
        # Day of Week Analysis
        if 'day_of_week' in df.columns:
            dow_avg = df.groupby('day_of_week')['tb2_da_revenue'].mean()
            dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            metrics['day_of_week'] = {
                dow_names[i]: float(val) for i, val in dow_avg.items()
            }
        
        return metrics
    
    def generate_market_insight_report(self) -> Dict[str, Any]:
        """Generate comprehensive market insights report"""
        
        if self.data['combined'].empty:
            return None
            
        df = self.data['combined']
        
        report = {
            'metadata': {
                'title': 'ERCOT Battery Storage Market Intelligence Report',
                'generated_at': datetime.now().isoformat(),
                'data_period': {
                    'start': df['date'].min().isoformat(),
                    'end': df['date'].max().isoformat(),
                    'total_days': len(df['date'].unique())
                }
            },
            'executive_summary': {},
            'market_trends': {},
            'revenue_analysis': {},
            'location_analysis': {},
            'outlook': {}
        }
        
        # Executive Summary
        total_tb2_revenue = df['tb2_da_revenue'].sum()
        total_tb4_revenue = df['tb4_da_revenue'].sum()
        
        report['executive_summary'] = {
            'total_market_value': {
                'tb2_total': float(total_tb2_revenue),
                'tb4_total': float(total_tb4_revenue),
                'tb4_premium': float((total_tb4_revenue / total_tb2_revenue - 1) * 100)
            },
            'daily_averages': {
                'tb2_mean': float(df.groupby('date')['tb2_da_revenue'].sum().mean()),
                'tb4_mean': float(df.groupby('date')['tb4_da_revenue'].sum().mean())
            },
            'key_findings': []
        }
        
        # Market Trends
        yearly_stats = df.groupby('year').agg({
            'tb2_da_revenue': ['mean', 'sum', 'std'],
            'tb4_da_revenue': ['mean', 'sum', 'std']
        })
        
        # Convert yearly stats to simple dict
        yearly_stats_dict = {}
        for year in yearly_stats.index:
            yearly_stats_dict[str(year)] = {
                'tb2_mean': float(yearly_stats.loc[year, ('tb2_da_revenue', 'mean')]),
                'tb2_sum': float(yearly_stats.loc[year, ('tb2_da_revenue', 'sum')]),
                'tb2_std': float(yearly_stats.loc[year, ('tb2_da_revenue', 'std')]),
                'tb4_mean': float(yearly_stats.loc[year, ('tb4_da_revenue', 'mean')]),
                'tb4_sum': float(yearly_stats.loc[year, ('tb4_da_revenue', 'sum')]),
                'tb4_std': float(yearly_stats.loc[year, ('tb4_da_revenue', 'std')])
            }
        
        # Calculate year-over-year growth
        yoy_growth = []
        years = sorted(df['year'].unique())
        for i in range(1, len(years)):
            prev_year = yearly_stats.loc[years[i-1], ('tb2_da_revenue', 'sum')]
            curr_year = yearly_stats.loc[years[i], ('tb2_da_revenue', 'sum')]
            growth = (curr_year - prev_year) / prev_year * 100
            yoy_growth.append({
                'year': years[i],
                'growth_percent': float(growth),
                'absolute_change': float(curr_year - prev_year)
            })
        
        report['market_trends'] = {
            'yearly_performance': yearly_stats_dict,
            'year_over_year_growth': yoy_growth,
            'trend_direction': 'increasing' if yoy_growth and yoy_growth[-1]['growth_percent'] > 0 else 'decreasing'
        }
        
        # Revenue Analysis with Advanced Metrics
        report['revenue_analysis'] = self.calculate_advanced_metrics(df)
        
        # Location Analysis
        location_stats = df.groupby('node').agg({
            'tb2_da_revenue': ['mean', 'std', 'sum', 'count'],
            'tb4_da_revenue': ['mean', 'std', 'sum']
        }).round(2)
        
        # Identify zones vs hubs vs DC ties
        zones = [n for n in location_stats.index if n.startswith('LZ_')]
        hubs = [n for n in location_stats.index if n.startswith('HB_')]
        dc_ties = [n for n in location_stats.index if n.startswith('DC_')]
        
        report['location_analysis'] = {
            'by_type': {
                'zones': {
                    'count': len(zones),
                    'avg_revenue': float(location_stats.loc[zones, ('tb2_da_revenue', 'mean')].mean()) if zones else 0,
                    'total_revenue': float(location_stats.loc[zones, ('tb2_da_revenue', 'sum')].sum()) if zones else 0
                },
                'hubs': {
                    'count': len(hubs),
                    'avg_revenue': float(location_stats.loc[hubs, ('tb2_da_revenue', 'mean')].mean()) if hubs else 0,
                    'total_revenue': float(location_stats.loc[hubs, ('tb2_da_revenue', 'sum')].sum()) if hubs else 0
                },
                'dc_ties': {
                    'count': len(dc_ties),
                    'avg_revenue': float(location_stats.loc[dc_ties, ('tb2_da_revenue', 'mean')].mean()) if dc_ties else 0,
                    'total_revenue': float(location_stats.loc[dc_ties, ('tb2_da_revenue', 'sum')].sum()) if dc_ties else 0
                }
            },
            'top_performing_locations': {
                idx: {
                    'tb2_mean': float(location_stats.loc[idx, ('tb2_da_revenue', 'mean')]),
                    'tb2_sum': float(location_stats.loc[idx, ('tb2_da_revenue', 'sum')]),
                    'tb4_mean': float(location_stats.loc[idx, ('tb4_da_revenue', 'mean')]),
                    'tb4_sum': float(location_stats.loc[idx, ('tb4_da_revenue', 'sum')])
                } for idx in location_stats.nlargest(10, ('tb2_da_revenue', 'sum')).index
            },
            'geographic_concentration': self.calculate_geographic_concentration(location_stats)
        }
        
        # Market Outlook
        recent_data = df[df['year'] == df['year'].max()]
        recent_avg = recent_data['tb2_da_revenue'].mean()
        historical_avg = df[df['year'] < df['year'].max()]['tb2_da_revenue'].mean()
        
        report['outlook'] = {
            'recent_performance': {
                'current_year': int(df['year'].max()),
                'avg_daily_revenue': float(recent_avg),
                'vs_historical': float((recent_avg - historical_avg) / historical_avg * 100) if historical_avg != 0 else 0
            },
            'market_maturity': self.assess_market_maturity(df),
            'revenue_stability': self.assess_revenue_stability(df)
        }
        
        # Key Findings
        key_findings = []
        
        # Finding 1: Best performing location
        best_node = location_stats.nlargest(1, ('tb2_da_revenue', 'sum')).index[0]
        best_revenue = location_stats.loc[best_node, ('tb2_da_revenue', 'sum')]
        key_findings.append(f"{best_node} is the highest revenue location with ${best_revenue:,.0f} total TB2 revenue")
        
        # Finding 2: Market trend
        if yoy_growth and yoy_growth[-1]['growth_percent'] > 0:
            key_findings.append(f"Market growing at {yoy_growth[-1]['growth_percent']:.1f}% year-over-year")
        else:
            key_findings.append(f"Market declining by {abs(yoy_growth[-1]['growth_percent']):.1f}% year-over-year")
        
        # Finding 3: TB4 advantage
        tb4_premium = (total_tb4_revenue / total_tb2_revenue - 1) * 100
        key_findings.append(f"4-hour batteries generate {tb4_premium:.0f}% more revenue than 2-hour systems")
        
        # Finding 4: Volatility
        cv = report['revenue_analysis']['revenue_distribution']['cv']
        if cv > 1.0:
            key_findings.append("High revenue volatility presents both opportunities and risks")
        elif cv > 0.5:
            key_findings.append("Moderate revenue volatility indicates stable but variable market")
        else:
            key_findings.append("Low revenue volatility suggests mature and stable market conditions")
        
        report['executive_summary']['key_findings'] = key_findings
        
        return report
    
    def calculate_geographic_concentration(self, location_stats: pd.DataFrame) -> Dict[str, float]:
        """Calculate Herfindahl-Hirschman Index for geographic concentration"""
        
        total_revenue = location_stats[('tb2_da_revenue', 'sum')].sum()
        market_shares = location_stats[('tb2_da_revenue', 'sum')] / total_revenue
        
        # HHI calculation
        hhi = (market_shares ** 2).sum() * 10000
        
        # Gini coefficient
        sorted_shares = np.sort(market_shares)
        n = len(sorted_shares)
        index = np.arange(1, n + 1)
        gini = (2 * index - n - 1).dot(sorted_shares) / (n * sorted_shares.sum())
        
        return {
            'herfindahl_index': float(hhi),
            'gini_coefficient': float(gini),
            'top_3_concentration': float(market_shares.nlargest(3).sum()),
            'top_5_concentration': float(market_shares.nlargest(5).sum()),
            'concentration_assessment': 'high' if hhi > 2500 else 'moderate' if hhi > 1500 else 'low'
        }
    
    def assess_market_maturity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess market maturity based on various indicators"""
        
        # Calculate rolling volatility
        daily_rev = df.groupby('date')['tb2_da_revenue'].sum()
        rolling_vol = daily_rev.rolling(30).std()
        
        # Trend analysis using numpy polyfit instead of scipy
        x = np.arange(len(daily_rev))
        coeffs = np.polyfit(x, daily_rev.values, 1)
        slope = coeffs[0]
        intercept = coeffs[1]
        # Calculate R-squared
        y_pred = np.polyval(coeffs, x)
        ss_res = np.sum((daily_rev.values - y_pred) ** 2)
        ss_tot = np.sum((daily_rev.values - np.mean(daily_rev.values)) ** 2)
        r_value = np.sqrt(1 - (ss_res / ss_tot)) if ss_tot != 0 else 0
        
        # Revenue stability over time
        yearly_cv = df.groupby('year')['tb2_da_revenue'].apply(lambda x: x.std() / x.mean())
        
        maturity_score = 0
        factors = []
        
        # Factor 1: Declining volatility
        if len(rolling_vol.dropna()) > 0 and rolling_vol.iloc[-1] < rolling_vol.mean():
            maturity_score += 1
            factors.append("Declining price volatility")
        
        # Factor 2: Stable or declining growth
        if slope < 0:
            maturity_score += 1
            factors.append("Revenue growth stabilizing")
        
        # Factor 3: Lower CV in recent years
        if len(yearly_cv) > 2 and yearly_cv.iloc[-1] < yearly_cv.mean():
            maturity_score += 1
            factors.append("Reduced revenue variability")
        
        return {
            'maturity_score': maturity_score,
            'maturity_level': 'mature' if maturity_score >= 2 else 'developing' if maturity_score == 1 else 'emerging',
            'indicators': factors,
            'trend_slope': float(slope),
            'trend_r_squared': float(r_value ** 2)
        }
    
    def assess_revenue_stability(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess revenue stability and predictability"""
        
        daily_rev = df.groupby('date')['tb2_da_revenue'].sum()
        
        # Calculate various stability metrics
        stability_metrics = {
            'coefficient_of_variation': float(daily_rev.std() / daily_rev.mean()),
            'interquartile_range': float(daily_rev.quantile(0.75) - daily_rev.quantile(0.25)),
            'range_to_mean_ratio': float((daily_rev.max() - daily_rev.min()) / daily_rev.mean()),
            'outlier_days': int(((daily_rev > daily_rev.quantile(0.95)) | 
                                 (daily_rev < daily_rev.quantile(0.05))).sum()),
            'outlier_percentage': float(((daily_rev > daily_rev.quantile(0.95)) | 
                                        (daily_rev < daily_rev.quantile(0.05))).mean() * 100)
        }
        
        # Stability assessment
        if stability_metrics['coefficient_of_variation'] < 0.5:
            stability_metrics['assessment'] = 'highly_stable'
        elif stability_metrics['coefficient_of_variation'] < 1.0:
            stability_metrics['assessment'] = 'moderately_stable'
        else:
            stability_metrics['assessment'] = 'volatile'
        
        return stability_metrics
    
    def save_market_insight_report(self, report: Dict[str, Any]):
        """Save market insight report in multiple formats"""
        
        if report is None:
            return
            
        # Save JSON
        json_file = self.market_dir / f"market_insights_{datetime.now().strftime('%Y%m')}.json"
        with open(json_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        # Generate and save Markdown
        md_content = self.market_report_to_markdown(report)
        md_file = self.market_dir / f"market_insights_{datetime.now().strftime('%Y%m')}.md"
        with open(md_file, 'w') as f:
            f.write(md_content)
            
        print(f"  ✅ Saved market insights report")
    
    def market_report_to_markdown(self, report: Dict[str, Any]) -> str:
        """Convert market insight report to professional Markdown format"""
        
        md = []
        
        # Header
        md.append("# ERCOT Battery Storage Market Intelligence Report")
        md.append(f"\n*{datetime.now().strftime('%B %Y')} | TBX Analytics Platform*\n")
        
        # Executive Summary
        md.append("## Executive Summary\n")
        for finding in report['executive_summary']['key_findings']:
            md.append(f"- **{finding}**")
        
        md.append(f"\n### Market Size")
        summary = report['executive_summary']['total_market_value']
        md.append(f"- Total TB2 Market Value: **${summary['tb2_total']:,.0f}**")
        md.append(f"- Total TB4 Market Value: **${summary['tb4_total']:,.0f}**")
        md.append(f"- TB4 Premium: **{summary['tb4_premium']:.1f}%**")
        
        # Market Trends
        md.append("\n## Market Trends\n")
        
        if report['market_trends']['year_over_year_growth']:
            md.append("### Year-over-Year Growth")
            md.append("| Year | Growth % | Absolute Change |")
            md.append("|------|----------|-----------------|")
            for growth in report['market_trends']['year_over_year_growth']:
                direction = "📈" if growth['growth_percent'] > 0 else "📉"
                md.append(f"| {growth['year']} | {direction} {growth['growth_percent']:+.1f}% | ${growth['absolute_change']:,.0f} |")
        
        # Revenue Analysis
        md.append("\n## Revenue Distribution Analysis\n")
        dist = report['revenue_analysis']['revenue_distribution']
        
        md.append("### Statistical Distribution")
        md.append(f"- Mean: ${dist['mean']:,.2f}")
        md.append(f"- Standard Deviation: ${dist['std']:,.2f}")
        md.append(f"- Coefficient of Variation: {dist['cv']:.2f}")
        
        md.append("\n### Percentiles")
        md.append("| Percentile | Daily Revenue |")
        md.append("|------------|---------------|")
        for p, val in dist['percentiles'].items():
            md.append(f"| {p.upper()} | ${val:,.2f} |")
        
        # Volatility Metrics
        if 'volatility' in report['revenue_analysis']:
            vol = report['revenue_analysis']['volatility']
            md.append("\n### Volatility Metrics")
            md.append(f"- Daily Volatility: {vol['daily_volatility']:.1%}")
            md.append(f"- Annualized Volatility: {vol['annualized_volatility']:.1%}")
            md.append(f"- Revenue Range: ${vol['revenue_range']:,.0f}")
        
        # Location Analysis
        md.append("\n## Geographic Analysis\n")
        loc = report['location_analysis']
        
        md.append("### Market Segmentation")
        md.append("| Segment | Count | Avg Revenue | Total Revenue | Market Share |")
        md.append("|---------|-------|-------------|---------------|--------------|")
        
        total_rev = sum([loc['by_type'][t]['total_revenue'] for t in ['zones', 'hubs', 'dc_ties']])
        for segment, data in loc['by_type'].items():
            share = data['total_revenue'] / total_rev * 100 if total_rev > 0 else 0
            md.append(f"| {segment.replace('_', ' ').title()} | {data['count']} | ${data['avg_revenue']:,.0f} | ${data['total_revenue']:,.0f} | {share:.1f}% |")
        
        # Geographic Concentration
        conc = loc['geographic_concentration']
        md.append("\n### Market Concentration")
        md.append(f"- Herfindahl Index: **{conc['herfindahl_index']:.0f}** ({conc['concentration_assessment']})")
        md.append(f"- Top 3 Nodes Control: **{conc['top_3_concentration']:.1%}** of revenue")
        md.append(f"- Top 5 Nodes Control: **{conc['top_5_concentration']:.1%}** of revenue")
        
        # Market Outlook
        md.append("\n## Market Outlook\n")
        outlook = report['outlook']
        
        md.append("### Current Performance")
        recent = outlook['recent_performance']
        direction = "above" if recent['vs_historical'] > 0 else "below"
        md.append(f"- {recent['current_year']} YTD Average: ${recent['avg_daily_revenue']:,.2f}/day")
        md.append(f"- Performance vs Historical: {abs(recent['vs_historical']):.1f}% {direction} average")
        
        # Market Maturity
        maturity = outlook['market_maturity']
        md.append(f"\n### Market Maturity: **{maturity['maturity_level'].upper()}**")
        for indicator in maturity['indicators']:
            md.append(f"- {indicator}")
        
        # Revenue Stability
        stability = outlook['revenue_stability']
        md.append(f"\n### Revenue Stability: **{stability['assessment'].replace('_', ' ').upper()}**")
        md.append(f"- Coefficient of Variation: {stability['coefficient_of_variation']:.2f}")
        md.append(f"- Outlier Days: {stability['outlier_percentage']:.1f}% of trading days")
        
        # Footer
        md.append("\n---")
        md.append("*Generated by TBX Advanced Analytics Platform*")
        md.append(f"*Data Period: {report['metadata']['data_period']['start'][:10]} to {report['metadata']['data_period']['end'][:10]}*")
        
        return "\n".join(md)
    
    def generate_all_reports(self):
        """Generate all advanced reports"""
        
        print("🚀 Generating Advanced TBX Reports")
        print("=" * 60)
        
        # Generate market insights report
        print("\n📈 Market Intelligence Report:")
        market_report = self.generate_market_insight_report()
        if market_report:
            self.save_market_insight_report(market_report)
        
        print("\n✅ Advanced reports generated successfully!")
        print(f"📁 Reports saved to: {self.reports_dir}")

def main():
    parser = argparse.ArgumentParser(description='Generate advanced TBX reports with market insights')
    parser.add_argument('--data-dir', type=str,
                       default='/home/enrico/data/ERCOT_data/tbx_results',
                       help='TBX data directory')
    parser.add_argument('--output-dir', type=str,
                       default='/home/enrico/data/ERCOT_data/tbx_results',
                       help='Output directory for reports')
    
    args = parser.parse_args()
    
    generator = TBXAdvancedReportGenerator(args.data_dir, args.output_dir)
    generator.generate_all_reports()

if __name__ == "__main__":
    main()