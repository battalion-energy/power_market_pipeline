#!/usr/bin/env python3
"""Analyze battery results and create summary report."""

import pandas as pd
from pathlib import Path

# Load data
summary_df = pd.read_csv('battery_analysis_output/ERCOT_RT_Battery_Summary.csv')
monthly_df = pd.read_csv('battery_analysis_output/ERCOT_RT_Battery_Monthly.csv')

# Create HTML report
html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>ERCOT Battery Analysis Report</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 40px;
            background-color: #f5f5f5;
        }
        .header {
            background-color: #366092;
            color: white;
            padding: 20px;
            text-align: center;
            border-radius: 8px;
        }
        .section {
            background-color: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        th {
            background-color: #366092;
            color: white;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .negative {
            color: #d62728;
        }
        .positive {
            color: #2ca02c;
        }
        .insight {
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 20px 0;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>ERCOT Battery Energy Storage Analysis</h1>
        <h2>Real-Time Market Arbitrage Opportunities (2022)</h2>
    </div>
"""

# Executive Summary
html_content += """
    <div class="section">
        <h2>Executive Summary</h2>
        <div class="insight">
            <p><strong>Key Finding:</strong> All analyzed battery configurations show negative annual revenues when operating solely in the ERCOT real-time market. This is primarily due to:</p>
            <ul>
                <li>Battery degradation costs ($10/MWh) exceeding typical arbitrage margins</li>
                <li>Limited intraday price volatility in 5-minute RT markets</li>
                <li>Absence of day-ahead market optimization opportunities</li>
            </ul>
        </div>
    </div>
"""

# Annual Summary Table
html_content += """
    <div class="section">
        <h2>Annual Revenue Summary</h2>
        <table>
            <thead>
                <tr>
                    <th>Zone</th>
                    <th>Battery Type</th>
                    <th>Annual Revenue</th>
                    <th>Avg Daily Revenue</th>
                    <th>Total Cycles</th>
                    <th>Revenue per Cycle</th>
                </tr>
            </thead>
            <tbody>
"""

for _, row in summary_df.iterrows():
    revenue_class = 'negative' if row['Total Annual Revenue'] < 0 else 'positive'
    revenue_per_cycle = row['Total Annual Revenue'] / row['Total Cycles']
    html_content += f"""
                <tr>
                    <td>{row['Zone']}</td>
                    <td>{row['Battery']}</td>
                    <td class="{revenue_class}">${row['Total Annual Revenue']:,.2f}</td>
                    <td class="{revenue_class}">${row['Avg Daily Revenue']:,.2f}</td>
                    <td>{row['Total Cycles']:.1f}</td>
                    <td class="{revenue_class}">${revenue_per_cycle:,.2f}</td>
                </tr>
"""

html_content += """
            </tbody>
        </table>
    </div>
"""

# Zone Comparison
best_tb2 = summary_df[summary_df['Battery'] == 'TB2'].loc[summary_df[summary_df['Battery'] == 'TB2']['Total Annual Revenue'].idxmax()]
best_tb4 = summary_df[summary_df['Battery'] == 'TB4'].loc[summary_df[summary_df['Battery'] == 'TB4']['Total Annual Revenue'].idxmax()]

html_content += f"""
    <div class="section">
        <h2>Zone Performance Comparison</h2>
        <p><strong>Best Performing Zones (Least Negative Revenue):</strong></p>
        <ul>
            <li>TB2 (2-hour): {best_tb2['Zone']} with ${best_tb2['Total Annual Revenue']:,.2f} annual revenue</li>
            <li>TB4 (4-hour): {best_tb4['Zone']} with ${best_tb4['Total Annual Revenue']:,.2f} annual revenue</li>
        </ul>
        <p>The West zone shows the best performance, likely due to higher renewable penetration creating more price volatility.</p>
    </div>
"""

# Monthly Analysis
monthly_summary = monthly_df.groupby('zone_battery')['total_revenue'].agg(['sum', 'mean', 'std', 'min', 'max'])
best_months = monthly_df.nlargest(5, 'total_revenue')

html_content += """
    <div class="section">
        <h2>Monthly Revenue Analysis</h2>
        <p><strong>Top 5 Revenue Months (Least Negative):</strong></p>
        <table>
            <thead>
                <tr>
                    <th>Zone/Battery</th>
                    <th>Month</th>
                    <th>Revenue</th>
                </tr>
            </thead>
            <tbody>
"""

for _, row in best_months.iterrows():
    html_content += f"""
                <tr>
                    <td>{row['zone_battery']}</td>
                    <td>{row['month']}</td>
                    <td class="negative">${row['total_revenue']:,.2f}</td>
                </tr>
"""

html_content += """
            </tbody>
        </table>
    </div>
"""

# Recommendations
html_content += """
    <div class="section">
        <h2>Recommendations</h2>
        <ol>
            <li><strong>Combined DA/RT Strategy:</strong> Integrate day-ahead market positions with real-time adjustments to capture larger price spreads</li>
            <li><strong>Ancillary Services:</strong> Participate in regulation and spinning reserve markets for additional revenue streams</li>
            <li><strong>Reduce Degradation Costs:</strong> Consider batteries with lower cycling costs or longer lifespans</li>
            <li><strong>Location Selection:</strong> Focus on West Texas locations with higher renewable penetration and price volatility</li>
            <li><strong>Duration Optimization:</strong> 2-hour batteries show better economics than 4-hour for pure energy arbitrage</li>
        </ol>
    </div>
"""

# Technical Details
html_content += f"""
    <div class="section">
        <h2>Technical Details</h2>
        <ul>
            <li>Analysis Period: January 1, 2022 - December 31, 2022</li>
            <li>Market: ERCOT Real-Time (5-minute intervals)</li>
            <li>Battery Configurations: 1 MW / 2 MWh (TB2) and 1 MW / 4 MWh (TB4)</li>
            <li>Round-trip Efficiency: 85%</li>
            <li>Degradation Cost: $10/MWh</li>
            <li>Optimization Algorithm: Greedy selection of lowest/highest price intervals</li>
            <li>Days Analyzed: 358 per zone (missing data excluded)</li>
        </ul>
    </div>
"""

html_content += """
    <div style="text-align: center; margin-top: 40px; color: #666;">
        <p>Report generated on """ + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
    </div>
</body>
</html>
"""

# Save HTML report
with open('battery_analysis_output/ERCOT_Battery_Analysis_Report.html', 'w') as f:
    f.write(html_content)

print("HTML report saved to battery_analysis_output/ERCOT_Battery_Analysis_Report.html")

# Print text summary
print("\n" + "="*60)
print("ERCOT BATTERY ANALYSIS SUMMARY")
print("="*60)

# Load a daily file to get more insights
houston_daily = pd.read_csv('battery_analysis_output/HOUSTON_TB2_daily_results.csv')

print(f"\nAnalysis Period: 2022 (358 days per zone)")
print(f"Battery Configurations: TB2 (1MW/2MWh) and TB4 (1MW/4MWh)")
print(f"\nKey Findings:")
print(f"- All zones show negative annual revenues")
print(f"- Best performing zone: WEST (${summary_df[summary_df['Zone'] == 'WEST']['Total Annual Revenue'].max():,.2f})")
print(f"- Average price in Houston: ${houston_daily['avg_price'].mean():.2f}/MWh")
print(f"- Average price spread: ${houston_daily['price_spread'].mean():.2f}/MWh")
print(f"- Days with positive revenue: {(houston_daily['total_revenue'] > 0).sum()} out of {len(houston_daily)}")

print(f"\nRevenue Breakdown:")
print(f"- Revenue without degradation: ~${(houston_daily['total_revenue'].sum() + 10 * houston_daily['cycles'].sum() * 2):,.2f}")
print(f"- Degradation cost: ${10 * houston_daily['cycles'].sum() * 2:,.2f}")
print(f"- Net revenue: ${houston_daily['total_revenue'].sum():,.2f}")

print("\nConclusion: Pure RT arbitrage is not economically viable with current assumptions.")
print("Combined DA/RT optimization and ancillary services participation are essential.")