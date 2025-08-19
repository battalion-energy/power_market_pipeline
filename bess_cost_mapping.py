#!/usr/bin/env python3
"""
BESS Cost Mapping - Industry estimates for battery storage costs over time
Includes economies of scale calculations using power law
"""

import json
import numpy as np
from pathlib import Path

def generate_bess_cost_mapping():
    """
    Generate BESS cost estimates based on industry data
    Sources: NREL, BloombergNEF, Lazard LCOS
    """
    
    # Base costs per kWh for 4-hour systems (most common configuration)
    # These are total installed costs including all components
    base_costs = {
        2020: {
            'min': 345,      # $/kWh - Best in class
            'typical': 420,  # $/kWh - Industry average
            'max': 550,      # $/kWh - High cost projects
            'source': 'NREL Cost and Performance Database 2020'
        },
        2021: {
            'min': 330,
            'typical': 395,
            'max': 520,
            'source': 'NREL/BloombergNEF 2021'
        },
        2022: {
            'min': 315,
            'typical': 375,
            'max': 490,
            'source': 'Lazard LCOS v7.0 (2022)'
        },
        2023: {
            'min': 300,
            'typical': 360,
            'max': 470,
            'source': 'BloombergNEF Battery Price Survey 2023'
        },
        2024: {
            'min': 285,
            'typical': 340,
            'max': 445,
            'source': 'Industry estimates 2024'
        },
        2025: {
            'min': 270,
            'typical': 325,
            'max': 420,
            'source': 'Projected based on learning curve'
        }
    }
    
    # Cost breakdown components (as % of total)
    cost_breakdown = {
        'battery_cells': 0.40,      # Battery cells/modules
        'inverter_pcs': 0.15,       # Power conversion system
        'bop_balance': 0.20,        # Balance of plant
        'epc_costs': 0.15,          # Engineering, procurement, construction
        'developer_margin': 0.10    # Developer overhead and profit
    }
    
    # Economies of scale factors using power law: cost = base_cost * (size/base_size)^scale_factor
    # Scale factor typically between -0.1 and -0.2 for utility-scale projects
    scale_parameters = {
        'base_size_mwh': 4.0,       # Base size for cost estimates (1MW * 4 hours)
        'scale_factor': -0.15,      # Power law exponent
        'min_size_mwh': 0.5,        # Minimum project size
        'max_size_mwh': 400.0       # Maximum project size
    }
    
    def calculate_scaled_cost(base_cost, size_mwh, base_size_mwh=4.0, scale_factor=-0.15):
        """Calculate cost with economies of scale using power law"""
        return base_cost * (size_mwh / base_size_mwh) ** scale_factor
    
    # Generate detailed cost mapping
    cost_mapping = {
        'base_costs_per_kwh': base_costs,
        'cost_breakdown': cost_breakdown,
        'scale_parameters': scale_parameters,
        'project_sizes': {},
        'duration_adjustments': {
            '1_hour': 1.25,   # 25% premium for 1-hour systems
            '2_hour': 1.10,   # 10% premium for 2-hour systems
            '4_hour': 1.00,   # Base case
            '6_hour': 0.95,   # 5% discount for 6-hour systems
            '8_hour': 0.92    # 8% discount for 8-hour systems
        },
        'regional_adjustments': {
            'texas': 0.95,        # Lower costs due to favorable regulations
            'california': 1.15,   # Higher costs due to regulations and land
            'northeast': 1.10,    # Higher labor and land costs
            'midwest': 1.00,      # Average costs
            'southeast': 0.98     # Slightly below average
        }
    }
    
    # Calculate costs for different project sizes
    project_sizes = [1, 5, 10, 20, 50, 100, 200, 400]  # MWh
    
    for year in base_costs.keys():
        cost_mapping['project_sizes'][year] = {}
        
        for size in project_sizes:
            base_cost = base_costs[year]['typical']
            scaled_cost = calculate_scaled_cost(
                base_cost, 
                size, 
                scale_parameters['base_size_mwh'],
                scale_parameters['scale_factor']
            )
            
            # Calculate per MW costs for different durations
            # For a 1MW system with X hours duration
            durations = {
                '2_hour': {
                    'mwh': 2,
                    'total_cost': scaled_cost * 2 * 1000,  # Convert to total $
                    'cost_per_mw': scaled_cost * 2 * 1000,
                    'cost_per_kwh': scaled_cost * cost_mapping['duration_adjustments']['2_hour']
                },
                '4_hour': {
                    'mwh': 4,
                    'total_cost': scaled_cost * 4 * 1000,
                    'cost_per_mw': scaled_cost * 4 * 1000,
                    'cost_per_kwh': scaled_cost
                }
            }
            
            cost_mapping['project_sizes'][year][f'{size}_mwh'] = {
                'size_mwh': size,
                'base_cost_per_kwh': base_cost,
                'scaled_cost_per_kwh': scaled_cost,
                'scale_discount': (1 - scaled_cost/base_cost) * 100,
                'durations': durations
            }
    
    # Add standardized metrics
    cost_mapping['standardized_metrics'] = {}
    
    for year in base_costs.keys():
        # For 1MW systems
        tb2_cost = base_costs[year]['typical'] * 2 * 1000 * cost_mapping['duration_adjustments']['2_hour']
        tb4_cost = base_costs[year]['typical'] * 4 * 1000
        
        cost_mapping['standardized_metrics'][year] = {
            'tb2_system_1mw': {
                'capex_total': tb2_cost,
                'capex_per_mw': tb2_cost,
                'capex_per_kwh': tb2_cost / 2000,  # 2 MWh capacity
                'o_and_m_per_mw_year': tb2_cost * 0.015,  # 1.5% of capex annually
                'expected_lifetime_years': 15,
                'cycles_per_year': 365,
                'total_lifetime_cycles': 5475
            },
            'tb4_system_1mw': {
                'capex_total': tb4_cost,
                'capex_per_mw': tb4_cost,
                'capex_per_kwh': tb4_cost / 4000,  # 4 MWh capacity
                'o_and_m_per_mw_year': tb4_cost * 0.015,
                'expected_lifetime_years': 15,
                'cycles_per_year': 365,
                'total_lifetime_cycles': 5475
            },
            'levelized_costs': {
                'discount_rate': 0.07,  # 7% discount rate
                'tb2_lcoe': calculate_lcoe(tb2_cost, 2, 15, 0.07),
                'tb4_lcoe': calculate_lcoe(tb4_cost, 4, 15, 0.07)
            }
        }
    
    return cost_mapping

def calculate_lcoe(capex, capacity_mwh, lifetime_years, discount_rate):
    """Calculate levelized cost of energy storage"""
    # Annual O&M as % of capex
    annual_om = capex * 0.015
    
    # Total discounted costs
    total_cost = capex
    for year in range(1, lifetime_years + 1):
        total_cost += annual_om / ((1 + discount_rate) ** year)
    
    # Total discounted energy throughput (assuming 1 cycle per day, 90% efficiency)
    total_energy = 0
    annual_energy = capacity_mwh * 365 * 0.9  # MWh per year with efficiency
    for year in range(1, lifetime_years + 1):
        total_energy += annual_energy / ((1 + discount_rate) ** year)
    
    # LCOE in $/MWh
    return total_cost / total_energy if total_energy > 0 else 0

def save_cost_mapping(output_dir: Path):
    """Generate and save BESS cost mapping"""
    
    cost_mapping = generate_bess_cost_mapping()
    
    # Save as JSON
    output_file = output_dir / "bess_cost_mapping.json"
    with open(output_file, 'w') as f:
        json.dump(cost_mapping, f, indent=2)
    
    print(f"✅ Saved BESS cost mapping to {output_file}")
    
    # Generate summary report
    summary = []
    summary.append("BESS Cost Mapping Summary")
    summary.append("=" * 50)
    summary.append("\nTypical Costs per kWh (4-hour system):")
    
    for year, costs in cost_mapping['base_costs_per_kwh'].items():
        summary.append(f"{year}: ${costs['typical']}/kWh (range: ${costs['min']}-${costs['max']})")
    
    summary.append("\n1MW System Costs:")
    for year, metrics in cost_mapping['standardized_metrics'].items():
        tb2 = metrics['tb2_system_1mw']
        tb4 = metrics['tb4_system_1mw']
        summary.append(f"\n{year}:")
        summary.append(f"  TB2 (2-hour): ${tb2['capex_total']:,.0f}")
        summary.append(f"  TB4 (4-hour): ${tb4['capex_total']:,.0f}")
        summary.append(f"  LCOE TB2: ${metrics['levelized_costs']['tb2_lcoe']:.2f}/MWh")
        summary.append(f"  LCOE TB4: ${metrics['levelized_costs']['tb4_lcoe']:.2f}/MWh")
    
    summary_text = "\n".join(summary)
    print(summary_text)
    
    # Save summary
    summary_file = output_dir / "bess_cost_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(summary_text)
    
    return cost_mapping

if __name__ == "__main__":
    output_dir = Path("/home/enrico/data/ERCOT_data/tbx_results")
    save_cost_mapping(output_dir)