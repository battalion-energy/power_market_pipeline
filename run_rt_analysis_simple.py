#!/usr/bin/env python3
"""Run RT battery analysis with simple output."""

from ercot_rt_battery_analysis import ERCOTRTBatteryAnalyzer
from pathlib import Path
import pandas as pd

# Run analysis
analyzer = ERCOTRTBatteryAnalyzer()
results = analyzer.run_full_analysis(2022)

# Create output directory
output_dir = Path("battery_analysis_output")
output_dir.mkdir(exist_ok=True)

# Print summary statistics
print("\n=== SUMMARY STATISTICS (RT-Only Analysis) ===")
summary_data = []

for key, df in results.items():
    if len(df) > 0:
        total_revenue = df['total_revenue'].sum()
        avg_daily_revenue = df['total_revenue'].mean()
        total_cycles = df['cycles'].sum()
        
        parts = key.split('_')
        zone = '_'.join(parts[:-1])
        battery = parts[-1]
        
        summary_data.append({
            'Zone': zone,
            'Battery': battery,
            'Total Annual Revenue': total_revenue,
            'Avg Daily Revenue': avg_daily_revenue,
            'Total Cycles': total_cycles,
            'Days Analyzed': len(df)
        })
        
        print(f"\n{key}:")
        print(f"  Total Annual Revenue: ${total_revenue:,.2f}")
        print(f"  Average Daily Revenue: ${avg_daily_revenue:,.2f}")
        print(f"  Total Cycles: {total_cycles:,.1f}")
        print(f"  Days Analyzed: {len(df)}")

# Save summary as CSV
summary_df = pd.DataFrame(summary_data)
summary_csv = output_dir / "ERCOT_RT_Battery_Summary.csv"
summary_df.to_csv(summary_csv, index=False)
print(f"\nSummary saved to {summary_csv}")

# Save detailed results
for key, df in results.items():
    if len(df) > 0:
        csv_file = output_dir / f"{key}_daily_results.csv"
        df.to_csv(csv_file, index=False)
        print(f"Saved {key} results to {csv_file}")

# Create monthly summary
print("\n=== MONTHLY SUMMARY ===")
monthly_summaries = []

for key, df in results.items():
    if len(df) > 0:
        df_copy = df.copy()
        df_copy['month'] = pd.to_datetime(df_copy['date']).dt.to_period('M')
        monthly = df_copy.groupby('month').agg({
            'total_revenue': 'sum',
            'cycles': 'sum',
            'avg_price': 'mean',
            'price_spread': 'mean'
        }).reset_index()
        
        monthly['zone_battery'] = key
        monthly_summaries.append(monthly)

if monthly_summaries:
    all_monthly = pd.concat(monthly_summaries)
    monthly_csv = output_dir / "ERCOT_RT_Battery_Monthly.csv"
    all_monthly.to_csv(monthly_csv, index=False)
    print(f"\nMonthly summary saved to {monthly_csv}")
    
    # Print top revenue months
    print("\nTop 10 Revenue Months:")
    top_months = all_monthly.nlargest(10, 'total_revenue')
    for _, row in top_months.iterrows():
        print(f"  {row['zone_battery']} - {row['month']}: ${row['total_revenue']:,.2f}")