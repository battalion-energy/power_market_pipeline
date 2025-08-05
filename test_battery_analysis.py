#!/usr/bin/env python3
"""Test battery analysis with a small date range."""

import pandas as pd
from pathlib import Path
from battery_optimizer import BatteryConfig, BatteryOptimizer
from ercot_battery_analysis import ERCOTBatteryAnalyzer
from report_generator import ReportGenerator
import logging

logging.basicConfig(level=logging.INFO)

# Test with 2022 data which might have better coverage
analyzer = ERCOTBatteryAnalyzer()
analyzer.load_data(2022)

# Test Houston zone for January 2023
zone_key = 'HOUSTON'
zone_name = analyzer.LOAD_ZONES[zone_key]

print(f"Testing {zone_key} ({zone_name}) for January 2022")

# Get price data
prices = analyzer.get_zone_prices(zone_name, '2022-01-01', '2022-01-31')

print(f"\nDA prices shape: {prices['da'].shape if isinstance(prices['da'], pd.Series) else 'No data'}")
print(f"RT prices shape: {prices['rt'].shape if isinstance(prices['rt'], pd.Series) else 'No data'}")

if len(prices['da']) > 0:
    print(f"\nDA price sample:")
    print(prices['da'].head())
    print(f"\nDA price stats: min={prices['da'].min():.2f}, max={prices['da'].max():.2f}, mean={prices['da'].mean():.2f}")

if len(prices['rt']) > 0:
    print(f"\nRT price sample:")
    print(prices['rt'].head())
    print(f"\nRT price stats: min={prices['rt'].min():.2f}, max={prices['rt'].max():.2f}, mean={prices['rt'].mean():.2f}")

# Test battery optimization for one day
if len(prices['da']) >= 24 and len(prices['rt']) >= 288:
    print("\nTesting battery optimization for 2022-01-15...")
    
    battery_config = BatteryConfig(power_mw=1.0, duration_hours=2.0)
    results = analyzer.analyze_battery(
        zone_key,
        battery_config,
        '2022-01-15',
        '2022-01-15'
    )
    
    if len(results) > 0:
        print("\nOptimization results:")
        print(results)
    else:
        print("\nNo optimization results returned")
else:
    print("\nInsufficient data for optimization test")