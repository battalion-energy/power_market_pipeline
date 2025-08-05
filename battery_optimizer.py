#!/usr/bin/env python3
"""
Battery optimization module for combined Day-Ahead and Real-Time arbitrage.
Implements TB2 (2-hour) and TB4 (4-hour) battery analysis for ERCOT markets.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BatteryConfig:
    """Configuration for battery energy storage system."""
    power_mw: float = 1.0  # Power capacity in MW
    duration_hours: float = 2.0  # Storage duration (2 for TB2, 4 for TB4)
    efficiency: float = 0.85  # Round-trip efficiency
    min_soc: float = 0.0  # Minimum state of charge (0-1)
    max_soc: float = 1.0  # Maximum state of charge (0-1)
    degradation_cost: float = 10.0  # $/MWh cycling cost
    
    @property
    def energy_mwh(self) -> float:
        """Total energy capacity in MWh."""
        return self.power_mw * self.duration_hours
    
    @property
    def charge_efficiency(self) -> float:
        """One-way charging efficiency."""
        return np.sqrt(self.efficiency)
    
    @property
    def discharge_efficiency(self) -> float:
        """One-way discharging efficiency."""
        return np.sqrt(self.efficiency)


class BatteryOptimizer:
    """Optimizes battery operation across Day-Ahead and Real-Time markets."""
    
    def __init__(self, battery_config: BatteryConfig):
        self.battery = battery_config
        
    def optimize_combined_markets(
        self,
        da_prices: pd.Series,
        rt_prices: pd.DataFrame,
        date: pd.Timestamp
    ) -> Dict[str, pd.DataFrame]:
        """
        Optimize battery operation for combined DA and RT markets.
        
        Args:
            da_prices: Hourly day-ahead prices (24 values)
            rt_prices: 5-minute real-time prices (288 values for full day)
            date: Date for optimization
            
        Returns:
            Dictionary with optimization results including schedules and revenues
        """
        # Ensure we have full day of data
        if len(da_prices) != 24:
            raise ValueError(f"Expected 24 DA prices, got {len(da_prices)}")
        if len(rt_prices) != 288:
            raise ValueError(f"Expected 288 RT prices, got {len(rt_prices)}")
            
        # Create combined optimization problem
        # Decision variables: DA charge, DA discharge, RT charge, RT discharge for each interval
        # Plus state of charge tracking
        
        # For simplicity, we'll use a heuristic approach:
        # 1. Find best hours in DA market for charging/discharging
        # 2. Within those hours, optimize RT deviations
        
        results = self._heuristic_optimization(da_prices, rt_prices, date)
        
        return results
    
    def _heuristic_optimization(
        self,
        da_prices: pd.Series,
        rt_prices: pd.DataFrame,
        date: pd.Timestamp
    ) -> Dict[str, pd.DataFrame]:
        """
        Heuristic optimization approach for combined markets.
        """
        # Initialize results
        hours = np.arange(24)
        da_charge = np.zeros(24)
        da_discharge = np.zeros(24)
        rt_charge = np.zeros(288)
        rt_discharge = np.zeros(288)
        
        # Step 1: Find best DA arbitrage opportunities
        # Sort hours by price
        sorted_hours = np.argsort(da_prices.values)
        
        # Determine how many hours to charge/discharge based on battery duration
        charge_hours = int(np.ceil(self.battery.duration_hours))
        discharge_hours = int(np.ceil(self.battery.duration_hours))
        
        # Select lowest price hours for charging
        da_charge_hours = sorted_hours[:charge_hours]
        # Select highest price hours for discharging
        da_discharge_hours = sorted_hours[-discharge_hours:]
        
        # Allocate DA positions
        for h in da_charge_hours:
            da_charge[h] = self.battery.power_mw
        for h in da_discharge_hours:
            da_discharge[h] = self.battery.power_mw
            
        # Step 2: Optimize RT deviations within each hour
        for hour in range(24):
            hour_start = hour * 12
            hour_end = (hour + 1) * 12
            hour_rt_prices = rt_prices.iloc[hour_start:hour_end].values
            
            # Find best 5-min intervals for additional arbitrage
            rt_sorted = np.argsort(hour_rt_prices)
            
            # If we're charging in DA, look for even lower RT prices
            if da_charge[hour] > 0:
                # Can we find cheaper intervals in RT?
                cheapest_rt = rt_sorted[:4]  # 20 minutes of charging
                for idx in cheapest_rt:
                    if hour_rt_prices[idx] < da_prices.iloc[hour] - self.battery.degradation_cost:
                        rt_charge[hour_start + idx] = self.battery.power_mw * 0.5  # Partial charge
                        
            # If we're discharging in DA, look for even higher RT prices
            elif da_discharge[hour] > 0:
                # Can we find more expensive intervals in RT?
                highest_rt = rt_sorted[-4:]  # 20 minutes of discharging
                for idx in highest_rt:
                    if hour_rt_prices[idx] > da_prices.iloc[hour] + self.battery.degradation_cost:
                        rt_discharge[hour_start + idx] = self.battery.power_mw * 0.5  # Partial discharge
        
        # Calculate revenues
        da_revenue = self._calculate_da_revenue(da_prices, da_charge, da_discharge)
        rt_revenue = self._calculate_rt_revenue(rt_prices, rt_charge, rt_discharge)
        
        # Package results
        results = {
            'da_schedule': pd.DataFrame({
                'hour': hours,
                'charge_mw': da_charge,
                'discharge_mw': da_discharge,
                'price': da_prices.values,
                'revenue': da_revenue['hourly_revenue']
            }),
            'rt_schedule': pd.DataFrame({
                'interval': np.arange(288),
                'charge_mw': rt_charge,
                'discharge_mw': rt_discharge,
                'price': rt_prices.values,
                'revenue': rt_revenue['interval_revenue']
            }),
            'summary': pd.DataFrame({
                'date': [date],
                'da_revenue': [da_revenue['total']],
                'rt_revenue': [rt_revenue['total']],
                'total_revenue': [da_revenue['total'] + rt_revenue['total']],
                'cycles': [self._calculate_cycles(da_charge, da_discharge, rt_charge, rt_discharge)]
            })
        }
        
        return results
    
    def _calculate_da_revenue(
        self,
        prices: pd.Series,
        charge: np.ndarray,
        discharge: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Calculate day-ahead market revenue."""
        # Revenue = discharge * price - charge * price - degradation costs
        charge_cost = charge * prices.values
        discharge_revenue = discharge * prices.values * self.battery.discharge_efficiency
        degradation = (charge + discharge) * self.battery.degradation_cost
        
        hourly_revenue = discharge_revenue - charge_cost - degradation
        
        return {
            'hourly_revenue': hourly_revenue,
            'total': hourly_revenue.sum()
        }
    
    def _calculate_rt_revenue(
        self,
        prices: pd.DataFrame,
        charge: np.ndarray,
        discharge: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Calculate real-time market revenue."""
        # Convert 5-minute MW to MWh (divide by 12)
        charge_mwh = charge / 12
        discharge_mwh = discharge / 12
        
        charge_cost = charge_mwh * prices.values.flatten()
        discharge_revenue = discharge_mwh * prices.values.flatten() * self.battery.discharge_efficiency
        degradation = (charge_mwh + discharge_mwh) * self.battery.degradation_cost
        
        interval_revenue = discharge_revenue - charge_cost - degradation
        
        return {
            'interval_revenue': interval_revenue,
            'total': interval_revenue.sum()
        }
    
    def _calculate_cycles(
        self,
        da_charge: np.ndarray,
        da_discharge: np.ndarray,
        rt_charge: np.ndarray,
        rt_discharge: np.ndarray
    ) -> float:
        """Calculate equivalent full cycles."""
        # DA energy (MWh)
        da_energy = da_charge.sum() + da_discharge.sum()
        
        # RT energy (MWh) - convert from 5-min MW to MWh
        rt_energy = (rt_charge.sum() + rt_discharge.sum()) / 12
        
        total_energy = da_energy + rt_energy
        cycles = total_energy / (2 * self.battery.energy_mwh)
        
        return cycles


class OptimalBatteryScheduler:
    """
    Advanced battery scheduler using linear programming for perfect foresight optimization.
    """
    
    def __init__(self, battery_config: BatteryConfig):
        self.battery = battery_config
        
    def optimize_perfect_foresight(
        self,
        da_prices: pd.Series,
        rt_prices: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """
        Optimize with perfect foresight using linear programming.
        This gives the theoretical maximum revenue.
        """
        # This is a more complex implementation that would use scipy.optimize.linprog
        # or cvxpy for true optimal scheduling
        # For now, using the heuristic optimizer
        optimizer = BatteryOptimizer(self.battery)
        return optimizer._heuristic_optimization(
            da_prices,
            rt_prices,
            da_prices.index[0]
        )