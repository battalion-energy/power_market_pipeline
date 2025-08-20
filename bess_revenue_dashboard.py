#!/usr/bin/env python3
"""
BESS Revenue Dashboard Generator
Creates visualizations and summary reports for BESS revenue analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class BESSRevenueDashboard:
    def __init__(self):
        self.data_dir = Path('/home/enrico/data/ERCOT_data/bess_analysis')
        self.output_dir = self.data_dir / 'dashboard'
        self.output_dir.mkdir(exist_ok=True)
        
    def generate_dashboard_summary(self):
        """Generate dashboard summary JSON for frontend"""
        
        # Load corrected revenue data
        df = pd.read_csv(self.data_dir / 'corrected_bess_revenues.csv')
        
        # Calculate summary metrics
        summary = {
            'timestamp': datetime.now().isoformat(),
            'data_period': '2024',
            'fleet_overview': {
                'total_units': len(df),
                'profitable_units': len(df[df['total_net_revenue'] > 0]),
                'loss_making_units': len(df[df['total_net_revenue'] < 0]),
                'total_fleet_revenue': float(df['total_net_revenue'].sum()),
                'average_revenue_per_unit': float(df['total_net_revenue'].mean()),
                'median_revenue_per_unit': float(df['total_net_revenue'].median())
            },
            'revenue_breakdown': {
                'dam_net': float(df['dam_net'].sum()),
                'rt_net': float(df['rt_net'].sum()),
                'as_revenue': float(df['as_revenue'].sum()),
                'total_discharge_revenue': float(df['dam_discharge_revenue'].sum() + df['rt_discharge_revenue'].sum()),
                'total_charging_cost': float(df['dam_charge_cost'].sum() + df['rt_charge_cost'].sum())
            },
            'revenue_mix_percentages': {
                'dam_pct': float(df['dam_net'].sum() / df['total_net_revenue'].sum() * 100) if df['total_net_revenue'].sum() > 0 else 0,
                'rt_pct': float(df['rt_net'].sum() / df['total_net_revenue'].sum() * 100) if df['total_net_revenue'].sum() > 0 else 0,
                'as_pct': float(df['as_revenue'].sum() / df['total_net_revenue'].sum() * 100) if df['total_net_revenue'].sum() > 0 else 0
            },
            'top_performers': [],
            'bottom_performers': [],
            'efficiency_metrics': {}
        }
        
        # Add top performers
        top_5 = df.nlargest(5, 'total_net_revenue')
        for _, row in top_5.iterrows():
            summary['top_performers'].append({
                'name': row['resource_name'],
                'settlement_point': row['settlement_point'],
                'total_revenue': float(row['total_net_revenue']),
                'dam_net': float(row['dam_net']),
                'as_revenue': float(row['as_revenue']),
                'profit_margin': float((row['total_net_revenue'] / (row['dam_discharge_revenue'] + row['rt_discharge_revenue'])) * 100) if (row['dam_discharge_revenue'] + row['rt_discharge_revenue']) > 0 else 0
            })
        
        # Add bottom performers
        bottom_5 = df.nsmallest(5, 'total_net_revenue')
        for _, row in bottom_5.iterrows():
            summary['bottom_performers'].append({
                'name': row['resource_name'],
                'settlement_point': row['settlement_point'],
                'total_revenue': float(row['total_net_revenue']),
                'dam_net': float(row['dam_net']),
                'as_revenue': float(row['as_revenue'])
            })
        
        # Calculate efficiency metrics
        total_discharge = df['dam_discharge_revenue'].sum()
        total_charge_cost = df['dam_charge_cost'].sum()
        
        if total_charge_cost > 0:
            implied_efficiency = (1 - (total_charge_cost / total_discharge)) * 100
        else:
            implied_efficiency = 0
            
        summary['efficiency_metrics'] = {
            'implied_round_trip_efficiency': float(implied_efficiency),
            'average_dam_spread': float((total_discharge - total_charge_cost) / len(df)) if len(df) > 0 else 0,
            'units_with_positive_dam': len(df[df['dam_net'] > 0]),
            'units_with_negative_dam': len(df[df['dam_net'] < 0])
        }
        
        # Add risk metrics
        summary['risk_metrics'] = {
            'revenue_concentration': {
                'top_1_share': float(top_5.iloc[0]['total_net_revenue'] / df['total_net_revenue'].sum() * 100) if len(top_5) > 0 and df['total_net_revenue'].sum() > 0 else 0,
                'top_3_share': float(top_5.head(3)['total_net_revenue'].sum() / df['total_net_revenue'].sum() * 100) if len(top_5) >= 3 and df['total_net_revenue'].sum() > 0 else 0,
                'herfindahl_index': float(((df['total_net_revenue'] / df['total_net_revenue'].sum()) ** 2).sum() * 10000) if df['total_net_revenue'].sum() > 0 else 0
            },
            'as_dependency': {
                'units_over_80pct': len(df[(df['as_revenue'] / df['total_net_revenue']) > 0.8]),
                'average_as_dependency': float((df['as_revenue'] / df['total_net_revenue']).mean() * 100) if (df['total_net_revenue'] != 0).any() else 0
            }
        }
        
        # Save dashboard data
        with open(self.output_dir / 'dashboard_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary
    
    def generate_performance_report(self):
        """Generate text-based performance report"""
        
        summary = self.generate_dashboard_summary()
        
        report = []
        report.append("="*80)
        report.append("BESS FLEET PERFORMANCE DASHBOARD")
        report.append("="*80)
        report.append(f"Generated: {summary['timestamp']}")
        report.append(f"Data Period: {summary['data_period']}")
        report.append("")
        
        # Fleet Overview
        report.append("FLEET OVERVIEW")
        report.append("-"*40)
        fo = summary['fleet_overview']
        report.append(f"Total Units: {fo['total_units']}")
        report.append(f"Profitable Units: {fo['profitable_units']} ({fo['profitable_units']/fo['total_units']*100:.1f}%)")
        report.append(f"Loss-Making Units: {fo['loss_making_units']} ({fo['loss_making_units']/fo['total_units']*100:.1f}%)")
        report.append(f"Total Fleet Revenue: ${fo['total_fleet_revenue']:,.0f}")
        report.append(f"Average Revenue/Unit: ${fo['average_revenue_per_unit']:,.0f}")
        report.append(f"Median Revenue/Unit: ${fo['median_revenue_per_unit']:,.0f}")
        report.append("")
        
        # Revenue Mix
        report.append("REVENUE MIX")
        report.append("-"*40)
        rm = summary['revenue_mix_percentages']
        report.append(f"Ancillary Services: {rm['as_pct']:.1f}%")
        report.append(f"DAM Net: {rm['dam_pct']:.1f}%")
        report.append(f"RT Net: {rm['rt_pct']:.1f}%")
        report.append("")
        
        # Top Performers
        report.append("TOP 5 PERFORMERS")
        report.append("-"*40)
        for i, unit in enumerate(summary['top_performers'], 1):
            report.append(f"{i}. {unit['name']}: ${unit['total_revenue']:,.0f}")
            report.append(f"   DAM: ${unit['dam_net']:,.0f}, AS: ${unit['as_revenue']:,.0f}")
        report.append("")
        
        # Risk Metrics
        report.append("RISK METRICS")
        report.append("-"*40)
        rc = summary['risk_metrics']['revenue_concentration']
        report.append(f"Top 1 Unit Share: {rc['top_1_share']:.1f}%")
        report.append(f"Top 3 Units Share: {rc['top_3_share']:.1f}%")
        report.append(f"Herfindahl Index: {rc['herfindahl_index']:.0f} (>2500 = high concentration)")
        
        asd = summary['risk_metrics']['as_dependency']
        report.append(f"Units with >80% AS dependency: {asd['units_over_80pct']}")
        report.append(f"Average AS dependency: {asd['average_as_dependency']:.1f}%")
        report.append("")
        
        # Efficiency Metrics
        report.append("EFFICIENCY METRICS")
        report.append("-"*40)
        em = summary['efficiency_metrics']
        report.append(f"Implied Round-Trip Efficiency: {em['implied_round_trip_efficiency']:.1f}%")
        report.append(f"Units with Positive DAM Arbitrage: {em['units_with_positive_dam']}")
        report.append(f"Units with Negative DAM Arbitrage: {em['units_with_negative_dam']}")
        report.append(f"Average DAM Spread per Unit: ${em['average_dam_spread']:,.0f}")
        report.append("")
        
        # Key Insights
        report.append("KEY INSIGHTS")
        report.append("-"*40)
        
        if rc['top_1_share'] > 50:
            report.append("⚠️  EXTREME revenue concentration - single unit dominates fleet")
        
        if asd['average_as_dependency'] > 70:
            report.append("⚠️  HIGH dependency on Ancillary Services across fleet")
        
        if em['units_with_negative_dam'] > em['units_with_positive_dam']:
            report.append("⚠️  MAJORITY of units losing money on DAM arbitrage")
        
        if em['implied_round_trip_efficiency'] < 50:
            report.append("⚠️  LOW implied efficiency suggests data or operational issues")
        
        report.append("")
        report.append("="*80)
        
        # Save report
        report_text = "\n".join(report)
        with open(self.output_dir / 'performance_report.txt', 'w') as f:
            f.write(report_text)
        
        print(report_text)
        
        return report_text
    
    def generate_leaderboard_csv(self):
        """Generate leaderboard CSV for easy sharing"""
        
        df = pd.read_csv(self.data_dir / 'corrected_bess_revenues.csv')
        
        # Create leaderboard
        leaderboard = df[['resource_name', 'settlement_point', 'total_net_revenue', 
                         'dam_net', 'rt_net', 'as_revenue']].copy()
        
        # Add calculated columns
        leaderboard['dam_efficiency'] = (df['dam_discharge_revenue'] - df['dam_charge_cost']) / df['dam_discharge_revenue'] * 100
        leaderboard['as_dependency'] = df['as_revenue'] / df['total_net_revenue'] * 100
        leaderboard['rank'] = leaderboard['total_net_revenue'].rank(ascending=False, method='min').astype(int)
        
        # Sort by rank
        leaderboard = leaderboard.sort_values('rank')
        
        # Format columns
        currency_cols = ['total_net_revenue', 'dam_net', 'rt_net', 'as_revenue']
        for col in currency_cols:
            leaderboard[f'{col}_formatted'] = leaderboard[col].apply(lambda x: f"${x:,.0f}")
        
        percent_cols = ['dam_efficiency', 'as_dependency']
        for col in percent_cols:
            leaderboard[f'{col}_formatted'] = leaderboard[col].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A")
        
        # Select final columns
        final_columns = ['rank', 'resource_name', 'settlement_point', 
                        'total_net_revenue_formatted', 'dam_net_formatted', 
                        'rt_net_formatted', 'as_revenue_formatted',
                        'dam_efficiency_formatted', 'as_dependency_formatted']
        
        leaderboard_final = leaderboard[final_columns].copy()
        leaderboard_final.columns = ['Rank', 'Resource', 'Settlement Point', 
                                     'Total Revenue', 'DAM Net', 'RT Net', 
                                     'AS Revenue', 'DAM Efficiency', 'AS Dependency']
        
        # Save leaderboard
        leaderboard_final.to_csv(self.output_dir / 'bess_leaderboard.csv', index=False)
        
        print("\nLEADERBOARD GENERATED")
        print(leaderboard_final.to_string(index=False))
        
        return leaderboard_final

def main():
    dashboard = BESSRevenueDashboard()
    
    print("Generating BESS Revenue Dashboard...")
    print("="*80)
    
    # Generate all outputs
    dashboard.generate_performance_report()
    dashboard.generate_leaderboard_csv()
    
    print("\n✅ Dashboard generation complete!")
    print(f"   Output directory: {dashboard.output_dir}")
    print("   Files generated:")
    print("   - dashboard_summary.json")
    print("   - performance_report.txt")
    print("   - bess_leaderboard.csv")

if __name__ == '__main__':
    main()