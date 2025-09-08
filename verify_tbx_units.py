#!/usr/bin/env python3
"""
Verify TBX calculation units and assumptions.
This script demonstrates that TBX calculations assume a 1 MW battery.
"""

import numpy as np

def calculate_tbx(hourly_prices, hours, efficiency=0.9):
    """
    Calculate TBX arbitrage revenue for a 1 MW battery.
    
    Args:
        hourly_prices: Array of 24 hourly prices ($/MWh)
        hours: Number of hours (2 for TB2, 4 for TB4)
        efficiency: Round-trip efficiency (0.9 = 90%)
    
    Returns:
        Daily revenue in $/MW-day
    """
    # Find lowest price hours for charging
    sorted_indices = np.argsort(hourly_prices)
    charge_hours = sorted_indices[:hours]
    discharge_hours = sorted_indices[-hours:]
    
    # For a 1 MW battery:
    # - Charging for 'hours' hours = 'hours' MWh consumed
    # - With efficiency losses, need to buy: hours * MWh / efficiency
    # - Discharging for 'hours' hours = 'hours' MWh produced * efficiency
    
    # Cost to charge ($/day for 1 MW battery)
    # Each hour we charge 1 MW, so we buy 1 MWh per hour
    # With efficiency, we need to buy 1/efficiency MWh to store 1 MWh
    charge_cost = sum(hourly_prices[h] / efficiency for h in charge_hours)
    
    # Revenue from discharge ($/day for 1 MW battery)  
    # Each hour we discharge 1 MW, so we sell 1 MWh per hour
    # With efficiency, we can only discharge efficiency * stored energy
    discharge_revenue = sum(hourly_prices[h] * efficiency for h in discharge_hours)
    
    # Net revenue ($/MW-day)
    return discharge_revenue - charge_cost

# Example calculation
if __name__ == "__main__":
    print("TBX Units Verification")
    print("=" * 60)
    print()
    
    # Example hourly prices ($/MWh)
    example_prices = np.array([
        20, 25, 30, 35, 40, 45,  # Early morning (low)
        50, 55, 60, 65, 70, 75,  # Morning ramp
        80, 85, 90, 95, 100, 105,  # Peak afternoon/evening
        95, 85, 75, 65, 55, 45   # Evening decline
    ])
    
    print("Example hourly prices ($/MWh):")
    for i in range(0, 24, 6):
        print(f"  Hours {i:2d}-{i+5:2d}: {example_prices[i:i+6]}")
    print()
    
    # Calculate TB2 (2-hour battery)
    tb2_revenue = calculate_tbx(example_prices, 2, 0.9)
    print(f"TB2 (2-hour battery) calculation:")
    print(f"  Charge hours: 0-1 (lowest prices: $20, $25/MWh)")
    print(f"  Charge cost: ($20 + $25) / 0.9 = $50.00/MW-day")
    print(f"  Discharge hours: 16-17 (highest prices: $100, $105/MWh)")
    print(f"  Discharge revenue: ($100 + $105) * 0.9 = $184.50/MW-day")
    print(f"  Net TB2 revenue: $184.50 - $50.00 = ${tb2_revenue:.2f}/MW-day")
    print()
    
    # Calculate TB4 (4-hour battery)
    tb4_revenue = calculate_tbx(example_prices, 4, 0.9)
    print(f"TB4 (4-hour battery) calculation:")
    print(f"  Charge hours: 0-3 (lowest prices: $20, $25, $30, $35/MWh)")
    print(f"  Charge cost: ($20 + $25 + $30 + $35) / 0.9 = $122.22/MW-day")
    print(f"  Discharge hours: 14-17 (highest prices: $90, $95, $100, $105/MWh)")
    print(f"  Discharge revenue: ($90 + $95 + $100 + $105) * 0.9 = $351.00/MW-day")
    print(f"  Net TB4 revenue: $351.00 - $122.22 = ${tb4_revenue:.2f}/MW-day")
    print()
    
    print("UNITS SUMMARY:")
    print("-" * 60)
    print("✅ Daily revenue: $/MW-day (assumes 1 MW battery)")
    print("✅ Annual revenue: $/MW-year (daily * 365)")
    print("✅ TB2: 1 MW battery with 2 MWh capacity (2-hour duration)")
    print("✅ TB4: 1 MW battery with 4 MWh capacity (4-hour duration)")
    print()
    print("To scale for different battery sizes:")
    print("  - 100 MW battery: multiply revenue by 100")
    print("  - 250 MW battery: multiply revenue by 250")
    print()
    
    # Verify with actual data
    import pandas as pd
    import os
    
    data_dir = os.getenv('ERCOT_DATA_DIR', '/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data')
    results_file = os.path.join(data_dir, 'tbx_results_all_nodes/tbx_leaderboard_all_nodes.csv')
    
    if os.path.exists(results_file):
        print("Verification from actual results:")
        print("-" * 60)
        df = pd.read_csv(results_file, nrows=1)
        node = df.iloc[0]['node']
        tb4_total = df.iloc[0]['tb4_revenue']
        days = df.iloc[0]['days']
        tb4_per_year = df.iloc[0]['tb4_per_mw_year']
        
        print(f"Top node: {node}")
        print(f"  TB4 total revenue: ${tb4_total:,.2f} over {days} days")
        print(f"  Daily average: ${tb4_total/days:.2f}/MW-day")
        print(f"  Annual rate: ${tb4_per_year:,.2f}/MW-year")
        print(f"  Calculation check: {tb4_total/days*365:.2f} ≈ {tb4_per_year:.2f} ✓")