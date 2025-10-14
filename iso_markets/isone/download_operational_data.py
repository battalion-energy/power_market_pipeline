#!/usr/bin/env python3
"""
ISO-NE Operational Data Downloader

Downloads comprehensive operational data from ISO-NE Web Services API including:
- Load forecasts (hourly, 5-minute zonal, reliability region)
- Wind and solar forecasts (7-day)
- Fuel mix and generation data
- Demand data (combined hourly, day-ahead, realtime)
- Capacity scarcity conditions
- Constraints and congestion
- Reserve prices (hourly final, 5-minute)
- Zonal load estimates
- Outage data

Usage:
    python download_operational_data.py --start-date 2024-01-01 --end-date 2024-12-31 \
        --data-types load_forecast fuel_mix demand --max-concurrent 1
"""

import argparse
import asyncio
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import httpx
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

# Load environment variables
load_dotenv()

# API Configuration
BASE_URL = "https://webservices.iso-ne.com/api/v1.1"
USERNAME = os.getenv("ISONE_USERNAME")
PASSWORD = os.getenv("ISONE_PASSWORD")
DATA_DIR = Path(os.getenv("ISONE_DATA_DIR", "/pool/ssd8tb/data/iso/ISONE"))


class ISONEOperationalDataDownloader:
    """Downloader for ISO-NE operational data."""

    def __init__(self, max_concurrent: int = 1):
        self.auth = (USERNAME, PASSWORD)
        self.max_concurrent = max_concurrent
        self.session: Optional[httpx.AsyncClient] = None
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def __aenter__(self):
        self.session = httpx.AsyncClient(timeout=60.0)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.aclose()

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=300))
    async def fetch_data(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Fetch data from ISO-NE API with retry logic."""
        url = f"{BASE_URL}/{endpoint}"

        async with self.semaphore:
            response = await self.session.get(url, auth=self.auth, params=params or {})
            response.raise_for_status()
            return response.json()

    async def save_data(self, data: Dict, filepath: Path):
        """Save JSON data to file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        async with asyncio.Lock():
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)

    # ==================== LOAD FORECASTS ====================

    async def download_hourly_load_forecast(self, date: datetime):
        """Download hourly load forecast for a specific day."""
        try:
            endpoint = f"hourlyloadforecast/day/{date.strftime('%Y%m%d')}"
            data = await self.fetch_data(endpoint)

            if data and "HourlyLoadForecasts" in data:
                output_dir = DATA_DIR / "load_forecast" / "hourly" / str(date.year)
                filepath = output_dir / f"hourly_load_forecast_{date.strftime('%Y%m%d')}.json"
                await self.save_data(data, filepath)
                return True, len(data["HourlyLoadForecasts"]["HourlyLoadForecast"])
            return False, 0
        except Exception as e:
            print(f"Error downloading hourly load forecast for {date.date()}: {e}")
            return False, 0

    async def download_zonal_load_forecast(self, date: datetime):
        """Download 5-minute zonal load forecast."""
        try:
            endpoint = f"fiveminutezonalloadforecast/day/{date.strftime('%Y%m%d')}"
            data = await self.fetch_data(endpoint)

            if data and "FiveMinuteZonalLoadForecasts" in data:
                output_dir = DATA_DIR / "load_forecast" / "zonal_5min" / str(date.year)
                filepath = output_dir / f"zonal_5min_forecast_{date.strftime('%Y%m%d')}.json"
                await self.save_data(data, filepath)
                return True, len(data["FiveMinuteZonalLoadForecasts"]["FiveMinuteZonalLoadForecast"])
            return False, 0
        except Exception as e:
            print(f"Error downloading zonal load forecast for {date.date()}: {e}")
            return False, 0

    async def download_reliability_region_forecast(self, date: datetime):
        """Download reliability region load forecast."""
        try:
            endpoint = f"reliabilityregionloadforecast/day/{date.strftime('%Y%m%d')}"
            data = await self.fetch_data(endpoint)

            if data and "ReliabilityRegionLoadForecasts" in data:
                output_dir = DATA_DIR / "load_forecast" / "reliability_region" / str(date.year)
                filepath = output_dir / f"reliability_forecast_{date.strftime('%Y%m%d')}.json"
                await self.save_data(data, filepath)
                return True, len(data["ReliabilityRegionLoadForecasts"]["ReliabilityRegionLoadForecast"])
            return False, 0
        except Exception as e:
            print(f"Error downloading reliability region forecast for {date.date()}: {e}")
            return False, 0

    # ==================== WIND/SOLAR FORECASTS ====================

    async def download_wind_forecast(self, date: datetime):
        """Download 7-day wind power forecast."""
        try:
            endpoint = f"sevendaywindpowerforecast/day/{date.strftime('%Y%m%d')}"
            data = await self.fetch_data(endpoint)

            if data and "SevenDayWindPowerForecasts" in data:
                output_dir = DATA_DIR / "forecasts" / "wind_7day" / str(date.year)
                filepath = output_dir / f"wind_7day_forecast_{date.strftime('%Y%m%d')}.json"
                await self.save_data(data, filepath)
                return True, len(data["SevenDayWindPowerForecasts"]["SevenDayWindPowerForecast"])
            return False, 0
        except Exception as e:
            print(f"Error downloading wind forecast for {date.date()}: {e}")
            return False, 0

    # ==================== FUEL MIX ====================

    async def download_fuel_mix(self, date: datetime):
        """Download generation fuel mix data."""
        try:
            endpoint = f"genfuelmix/day/{date.strftime('%Y%m%d')}"
            data = await self.fetch_data(endpoint)

            if data and "GenFuelMixes" in data:
                output_dir = DATA_DIR / "fuel_mix" / str(date.year)
                filepath = output_dir / f"fuel_mix_{date.strftime('%Y%m%d')}.json"
                await self.save_data(data, filepath)
                return True, len(data["GenFuelMixes"]["GenFuelMix"])
            return False, 0
        except Exception as e:
            print(f"Error downloading fuel mix for {date.date()}: {e}")
            return False, 0

    # ==================== DEMAND DATA ====================

    async def download_combined_hourly_demand(self, date: datetime):
        """Download combined hourly demand data."""
        try:
            endpoint = f"combinedhourlydemand/day/{date.strftime('%Y%m%d')}"
            data = await self.fetch_data(endpoint)

            if data and "CombinedHourlyDemands" in data:
                output_dir = DATA_DIR / "demand" / "combined_hourly" / str(date.year)
                filepath = output_dir / f"combined_hourly_demand_{date.strftime('%Y%m%d')}.json"
                await self.save_data(data, filepath)
                return True, len(data["CombinedHourlyDemands"]["CombinedHourlyDemand"])
            return False, 0
        except Exception as e:
            print(f"Error downloading combined hourly demand for {date.date()}: {e}")
            return False, 0

    async def download_dayahead_hourly_demand(self, date: datetime):
        """Download day-ahead hourly demand data."""
        try:
            endpoint = f"dayaheadhourlydemand/day/{date.strftime('%Y%m%d')}"
            data = await self.fetch_data(endpoint)

            if data and "DayAheadHourlyDemands" in data:
                output_dir = DATA_DIR / "demand" / "dayahead_hourly" / str(date.year)
                filepath = output_dir / f"dayahead_hourly_demand_{date.strftime('%Y%m%d')}.json"
                await self.save_data(data, filepath)
                return True, len(data["DayAheadHourlyDemands"]["DayAheadHourlyDemand"])
            return False, 0
        except Exception as e:
            print(f"Error downloading day-ahead hourly demand for {date.date()}: {e}")
            return False, 0

    async def download_realtime_hourly_demand(self, date: datetime):
        """Download realtime hourly demand data."""
        try:
            endpoint = f"realtimehourlydemand/day/{date.strftime('%Y%m%d')}"
            data = await self.fetch_data(endpoint)

            if data and "RealTimeHourlyDemands" in data:
                output_dir = DATA_DIR / "demand" / "realtime_hourly" / str(date.year)
                filepath = output_dir / f"realtime_hourly_demand_{date.strftime('%Y%m%d')}.json"
                await self.save_data(data, filepath)
                return True, len(data["RealTimeHourlyDemands"]["RealTimeHourlyDemand"])
            return False, 0
        except Exception as e:
            print(f"Error downloading realtime hourly demand for {date.date()}: {e}")
            return False, 0

    # ==================== ZONAL LOAD ====================

    async def download_zonal_load(self, date: datetime):
        """Download 5-minute estimated zonal load."""
        try:
            endpoint = f"fiveminuteestimatedzonalload/day/{date.strftime('%Y%m%d')}"
            data = await self.fetch_data(endpoint)

            if data and "FiveMinuteEstimatedZonalLoads" in data:
                output_dir = DATA_DIR / "load" / "zonal_5min" / str(date.year)
                filepath = output_dir / f"zonal_5min_load_{date.strftime('%Y%m%d')}.json"
                await self.save_data(data, filepath)
                return True, len(data["FiveMinuteEstimatedZonalLoads"]["FiveMinuteEstimatedZonalLoad"])
            return False, 0
        except Exception as e:
            print(f"Error downloading zonal load for {date.date()}: {e}")
            return False, 0

    # ==================== CAPACITY & RESERVES ====================

    async def download_capacity_scarcity(self, date: datetime):
        """Download capacity scarcity condition data."""
        try:
            endpoint = f"capacityscarcitycondition/day/{date.strftime('%Y%m%d')}"
            data = await self.fetch_data(endpoint)

            if data and "CapacityScarcityConditions" in data:
                output_dir = DATA_DIR / "capacity" / "scarcity" / str(date.year)
                filepath = output_dir / f"capacity_scarcity_{date.strftime('%Y%m%d')}.json"
                await self.save_data(data, filepath)
                return True, len(data["CapacityScarcityConditions"]["CapacityScarcityCondition"])
            return False, 0
        except Exception as e:
            print(f"Error downloading capacity scarcity for {date.date()}: {e}")
            return False, 0

    async def download_hourly_final_reserve_price(self, date: datetime):
        """Download hourly final reserve prices."""
        try:
            endpoint = f"hourlyfinalreserveprice/day/{date.strftime('%Y%m%d')}"
            data = await self.fetch_data(endpoint)

            if data and "HourlyFinalReservePrices" in data:
                output_dir = DATA_DIR / "reserves" / "hourly_final_price" / str(date.year)
                filepath = output_dir / f"hourly_final_reserve_price_{date.strftime('%Y%m%d')}.json"
                await self.save_data(data, filepath)
                return True, len(data["HourlyFinalReservePrices"]["HourlyFinalReservePrice"])
            return False, 0
        except Exception as e:
            print(f"Error downloading hourly final reserve price for {date.date()}: {e}")
            return False, 0

    async def download_fiveminute_reserve_price(self, date: datetime):
        """Download 5-minute reserve prices."""
        try:
            endpoint = f"fiveminutereserveprice/day/{date.strftime('%Y%m%d')}"
            data = await self.fetch_data(endpoint)

            if data and "FiveMinuteReservePrices" in data:
                output_dir = DATA_DIR / "reserves" / "fiveminute_price" / str(date.year)
                filepath = output_dir / f"fiveminute_reserve_price_{date.strftime('%Y%m%d')}.json"
                await self.save_data(data, filepath)
                return True, len(data["FiveMinuteReservePrices"]["FiveMinuteReservePrice"])
            return False, 0
        except Exception as e:
            print(f"Error downloading 5-minute reserve price for {date.date()}: {e}")
            return False, 0

    # ==================== CONSTRAINTS & CONGESTION ====================

    async def download_dayahead_constraints(self, date: datetime):
        """Download day-ahead constraints."""
        try:
            endpoint = f"dayaheadconstraints/day/{date.strftime('%Y%m%d')}"
            data = await self.fetch_data(endpoint)

            if data and "DayAheadConstraints" in data:
                output_dir = DATA_DIR / "constraints" / "dayahead" / str(date.year)
                filepath = output_dir / f"dayahead_constraints_{date.strftime('%Y%m%d')}.json"
                await self.save_data(data, filepath)
                return True, len(data["DayAheadConstraints"]["DayAheadConstraint"])
            return False, 0
        except Exception as e:
            print(f"Error downloading day-ahead constraints for {date.date()}: {e}")
            return False, 0

    async def download_dayahead_lmp_congestion(self, date: datetime):
        """Download day-ahead LMP average congestion."""
        try:
            # Monthly data - only download on first day of month
            if date.day != 1:
                return False, 0

            endpoint = f"dayaheadlmpavgcongestion/month/{date.strftime('%Y%m01')}"
            data = await self.fetch_data(endpoint)

            if data and "DayAheadLmpAvgCongestions" in data:
                output_dir = DATA_DIR / "congestion" / "dayahead_lmp_avg" / str(date.year)
                filepath = output_dir / f"dayahead_lmp_congestion_{date.strftime('%Y%m')}.json"
                await self.save_data(data, filepath)
                return True, len(data["DayAheadLmpAvgCongestions"]["DayAheadLmpAvgCongestion"])
            return False, 0
        except Exception as e:
            print(f"Error downloading day-ahead LMP congestion for {date.strftime('%Y-%m')}: {e}")
            return False, 0

    # ==================== OUTAGES ====================

    async def download_outages(self, date: datetime):
        """Download outage data."""
        try:
            endpoint = f"outages/day/{date.strftime('%Y%m%d')}"
            data = await self.fetch_data(endpoint)

            if data and "Outages" in data:
                output_dir = DATA_DIR / "outages" / str(date.year)
                filepath = output_dir / f"outages_{date.strftime('%Y%m%d')}.json"
                await self.save_data(data, filepath)
                return True, len(data["Outages"]["Outage"])
            return False, 0
        except Exception as e:
            print(f"Error downloading outages for {date.date()}: {e}")
            return False, 0

    # ==================== REAL-TIME CONSTRAINTS ====================

    async def download_realtime_constraints(self, date: datetime):
        """Download real-time transmission constraints."""
        try:
            endpoint = f"realtimeconstraints/day/{date.strftime('%Y%m%d')}"
            data = await self.fetch_data(endpoint)

            if data and "RealTimeConstraints" in data:
                output_dir = DATA_DIR / "constraints" / "realtime" / str(date.year)
                filepath = output_dir / f"realtime_constraints_{date.strftime('%Y%m%d')}.json"
                await self.save_data(data, filepath)
                return True, len(data["RealTimeConstraints"]["RealTimeConstraint"])
            return False, 0
        except Exception as e:
            print(f"Error downloading realtime constraints for {date.date()}: {e}")
            return False, 0

    async def download_fiveminute_constraints(self, date: datetime):
        """Download 5-minute transmission constraints."""
        try:
            endpoint = f"fiveminuteconstraints/day/{date.strftime('%Y%m%d')}"
            data = await self.fetch_data(endpoint)

            if data and "FiveMinuteConstraints" in data:
                output_dir = DATA_DIR / "constraints" / "fiveminute" / str(date.year)
                filepath = output_dir / f"fiveminute_constraints_{date.strftime('%Y%m%d')}.json"
                await self.save_data(data, filepath)
                return True, len(data["FiveMinuteConstraints"]["FiveMinuteConstraint"])
            return False, 0
        except Exception as e:
            print(f"Error downloading 5-minute constraints for {date.date()}: {e}")
            return False, 0

    # ==================== INTERCHANGE & FLOWS ====================

    async def download_fifteenminute_interchange(self, date: datetime):
        """Download 15-minute interchange data (tie line flows)."""
        try:
            endpoint = f"fifteenminuteinterchange/day/{date.strftime('%Y%m%d')}"
            data = await self.fetch_data(endpoint)

            if data and "FifteenMinuteInterchanges" in data:
                output_dir = DATA_DIR / "interchange" / "fifteenminute" / str(date.year)
                filepath = output_dir / f"interchange_15min_{date.strftime('%Y%m%d')}.json"
                await self.save_data(data, filepath)
                return True, len(data["FifteenMinuteInterchanges"]["FifteenMinuteInterchange"])
            return False, 0
        except Exception as e:
            print(f"Error downloading 15-minute interchange for {date.date()}: {e}")
            return False, 0

    async def download_hourly_ba_interchange(self, date: datetime):
        """Download hourly balancing area interchange."""
        try:
            endpoint = f"hourlybainterchange/day/{date.strftime('%Y%m%d')}"
            data = await self.fetch_data(endpoint)

            if data and "HourlyBaInterchanges" in data:
                output_dir = DATA_DIR / "interchange" / "hourly_ba" / str(date.year)
                filepath = output_dir / f"ba_interchange_hourly_{date.strftime('%Y%m%d')}.json"
                await self.save_data(data, filepath)
                return True, len(data["HourlyBaInterchanges"]["HourlyBaInterchange"])
            return False, 0
        except Exception as e:
            print(f"Error downloading hourly BA interchange for {date.date()}: {e}")
            return False, 0

    async def download_fiveminute_external_flow(self, date: datetime):
        """Download 5-minute external flow data."""
        try:
            endpoint = f"fiveminuteexternalflow/day/{date.strftime('%Y%m%d')}"
            data = await self.fetch_data(endpoint)

            if data and "FiveMinuteExternalFlows" in data:
                output_dir = DATA_DIR / "interchange" / "fiveminute_external" / str(date.year)
                filepath = output_dir / f"external_flow_5min_{date.strftime('%Y%m%d')}.json"
                await self.save_data(data, filepath)
                return True, len(data["FiveMinuteExternalFlows"]["FiveMinuteExternalFlow"])
            return False, 0
        except Exception as e:
            print(f"Error downloading 5-minute external flow for {date.date()}: {e}")
            return False, 0

    # ==================== SYSTEM LOAD & FORECASTS ====================

    async def download_fiveminute_system_load(self, date: datetime):
        """Download 5-minute system load (actual)."""
        try:
            endpoint = f"fiveminutesystemload/day/{date.strftime('%Y%m%d')}"
            data = await self.fetch_data(endpoint)

            if data and "FiveMinuteSystemLoads" in data:
                output_dir = DATA_DIR / "load" / "system_5min" / str(date.year)
                filepath = output_dir / f"system_load_5min_{date.strftime('%Y%m%d')}.json"
                await self.save_data(data, filepath)
                return True, len(data["FiveMinuteSystemLoads"]["FiveMinuteSystemLoad"])
            return False, 0
        except Exception as e:
            print(f"Error downloading 5-minute system load for {date.date()}: {e}")
            return False, 0

    async def download_sevenday_forecast(self, date: datetime):
        """Download 7-day system forecast."""
        try:
            endpoint = f"sevendayforecast/day/{date.strftime('%Y%m%d')}"
            data = await self.fetch_data(endpoint)

            if data and "SevenDayForecasts" in data:
                output_dir = DATA_DIR / "forecasts" / "sevenday_system" / str(date.year)
                filepath = output_dir / f"sevenday_forecast_{date.strftime('%Y%m%d')}.json"
                await self.save_data(data, filepath)
                return True, len(data["SevenDayForecasts"]["SevenDayForecast"])
            return False, 0
        except Exception as e:
            print(f"Error downloading 7-day forecast for {date.date()}: {e}")
            return False, 0

    # ==================== SYSTEM CONDITIONS & CAPACITY ====================

    async def download_power_system_conditions(self, date: datetime):
        """Download power system conditions and alerts."""
        try:
            endpoint = f"powersystemconditions/day/{date.strftime('%Y%m%d')}"
            data = await self.fetch_data(endpoint)

            if data and "PowerSystemConditions" in data:
                output_dir = DATA_DIR / "system" / "conditions" / str(date.year)
                filepath = output_dir / f"system_conditions_{date.strftime('%Y%m%d')}.json"
                await self.save_data(data, filepath)
                return True, len(data["PowerSystemConditions"]["PowerSystemCondition"])
            return False, 0
        except Exception as e:
            print(f"Error downloading power system conditions for {date.date()}: {e}")
            return False, 0

    async def download_dayahead_operating_reserve(self, date: datetime):
        """Download day-ahead hourly operating reserve requirements."""
        try:
            endpoint = f"dayaheadhourlyoperatingreserve/day/{date.strftime('%Y%m%d')}"
            data = await self.fetch_data(endpoint)

            if data and "DayAheadHourlyOperatingReserves" in data:
                output_dir = DATA_DIR / "reserves" / "dayahead_hourly" / str(date.year)
                filepath = output_dir / f"da_hourly_operating_reserve_{date.strftime('%Y%m%d')}.json"
                await self.save_data(data, filepath)
                return True, len(data["DayAheadHourlyOperatingReserves"]["DayAheadHourlyOperatingReserve"])
            return False, 0
        except Exception as e:
            print(f"Error downloading day-ahead operating reserve for {date.date()}: {e}")
            return False, 0

    async def download_nextday_operational_capacity(self, date: datetime):
        """Download next-day operational capacity report."""
        try:
            endpoint = f"nextdayoperationalcapacityreport/day/{date.strftime('%Y%m%d')}"
            data = await self.fetch_data(endpoint)

            if data and "NextDayOperationalCapacityReports" in data:
                output_dir = DATA_DIR / "capacity" / "nextday_operational" / str(date.year)
                filepath = output_dir / f"nextday_operational_capacity_{date.strftime('%Y%m%d')}.json"
                await self.save_data(data, filepath)
                return True, len(data["NextDayOperationalCapacityReports"]["NextDayOperationalCapacityReport"])
            return False, 0
        except Exception as e:
            print(f"Error downloading next-day operational capacity for {date.date()}: {e}")
            return False, 0

    # ==================== ORCHESTRATION ====================

    async def download_for_date(self, date: datetime, data_types: List[str]):
        """Download all requested data types for a specific date."""
        results = {}

        for data_type in data_types:
            if data_type == "load_forecast_hourly":
                success, count = await self.download_hourly_load_forecast(date)
                results[data_type] = {"success": success, "count": count}
            elif data_type == "load_forecast_zonal":
                success, count = await self.download_zonal_load_forecast(date)
                results[data_type] = {"success": success, "count": count}
            elif data_type == "load_forecast_reliability":
                success, count = await self.download_reliability_region_forecast(date)
                results[data_type] = {"success": success, "count": count}
            elif data_type == "wind_forecast":
                success, count = await self.download_wind_forecast(date)
                results[data_type] = {"success": success, "count": count}
            elif data_type == "fuel_mix":
                success, count = await self.download_fuel_mix(date)
                results[data_type] = {"success": success, "count": count}
            elif data_type == "demand_combined_hourly":
                success, count = await self.download_combined_hourly_demand(date)
                results[data_type] = {"success": success, "count": count}
            elif data_type == "demand_dayahead_hourly":
                success, count = await self.download_dayahead_hourly_demand(date)
                results[data_type] = {"success": success, "count": count}
            elif data_type == "demand_realtime_hourly":
                success, count = await self.download_realtime_hourly_demand(date)
                results[data_type] = {"success": success, "count": count}
            elif data_type == "zonal_load":
                success, count = await self.download_zonal_load(date)
                results[data_type] = {"success": success, "count": count}
            elif data_type == "capacity_scarcity":
                success, count = await self.download_capacity_scarcity(date)
                results[data_type] = {"success": success, "count": count}
            elif data_type == "reserve_price_hourly":
                success, count = await self.download_hourly_final_reserve_price(date)
                results[data_type] = {"success": success, "count": count}
            elif data_type == "reserve_price_5min":
                success, count = await self.download_fiveminute_reserve_price(date)
                results[data_type] = {"success": success, "count": count}
            elif data_type == "dayahead_constraints":
                success, count = await self.download_dayahead_constraints(date)
                results[data_type] = {"success": success, "count": count}
            elif data_type == "dayahead_lmp_congestion":
                success, count = await self.download_dayahead_lmp_congestion(date)
                results[data_type] = {"success": success, "count": count}
            elif data_type == "outages":
                success, count = await self.download_outages(date)
                results[data_type] = {"success": success, "count": count}
            elif data_type == "realtime_constraints":
                success, count = await self.download_realtime_constraints(date)
                results[data_type] = {"success": success, "count": count}
            elif data_type == "fiveminute_constraints":
                success, count = await self.download_fiveminute_constraints(date)
                results[data_type] = {"success": success, "count": count}
            elif data_type == "interchange_15min":
                success, count = await self.download_fifteenminute_interchange(date)
                results[data_type] = {"success": success, "count": count}
            elif data_type == "interchange_hourly_ba":
                success, count = await self.download_hourly_ba_interchange(date)
                results[data_type] = {"success": success, "count": count}
            elif data_type == "interchange_5min_external":
                success, count = await self.download_fiveminute_external_flow(date)
                results[data_type] = {"success": success, "count": count}
            elif data_type == "system_load_5min":
                success, count = await self.download_fiveminute_system_load(date)
                results[data_type] = {"success": success, "count": count}
            elif data_type == "forecast_7day_system":
                success, count = await self.download_sevenday_forecast(date)
                results[data_type] = {"success": success, "count": count}
            elif data_type == "system_conditions":
                success, count = await self.download_power_system_conditions(date)
                results[data_type] = {"success": success, "count": count}
            elif data_type == "dayahead_operating_reserve":
                success, count = await self.download_dayahead_operating_reserve(date)
                results[data_type] = {"success": success, "count": count}
            elif data_type == "nextday_operational_capacity":
                success, count = await self.download_nextday_operational_capacity(date)
                results[data_type] = {"success": success, "count": count}

        return results

    async def download_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        data_types: List[str],
        reverse: bool = False,
    ):
        """Download data for a date range."""
        # Generate list of dates
        dates = []
        current_date = start_date
        while current_date <= end_date:
            dates.append(current_date)
            current_date += timedelta(days=1)

        if reverse:
            dates.reverse()

        print(f"\nDownloading {len(dates)} days of data for {len(data_types)} data types")
        print(f"Data types: {', '.join(data_types)}")
        print(f"Date range: {start_date.date()} to {end_date.date()}\n")

        total_success = 0
        total_failed = 0

        for i, date in enumerate(dates, 1):
            print(f"[{i}/{len(dates)}] Processing {date.date()}...")

            results = await self.download_for_date(date, data_types)

            day_success = sum(1 for r in results.values() if r["success"])
            day_failed = len(results) - day_success
            total_success += day_success
            total_failed += day_failed

            # Print summary for this day
            for dtype, result in results.items():
                status = "✓" if result["success"] else "✗"
                print(f"  {status} {dtype}: {result['count']} records")

        print(f"\n{'='*80}")
        print(f"Download Summary")
        print(f"{'='*80}")
        print(f"Total successful: {total_success}")
        print(f"Total failed: {total_failed}")
        print(f"{'='*80}\n")


async def main():
    parser = argparse.ArgumentParser(
        description="Download ISO-NE operational data"
    )
    parser.add_argument(
        "--start-date",
        required=True,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        required=True,
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--data-types",
        nargs="+",
        default=["all"],
        choices=[
            "all",
            "load_forecast_hourly",
            "load_forecast_zonal",
            "load_forecast_reliability",
            "wind_forecast",
            "fuel_mix",
            "demand_combined_hourly",
            "demand_dayahead_hourly",
            "demand_realtime_hourly",
            "zonal_load",
            "capacity_scarcity",
            "reserve_price_hourly",
            "reserve_price_5min",
            "dayahead_constraints",
            "dayahead_lmp_congestion",
            "outages",
            "realtime_constraints",
            "fiveminute_constraints",
            "interchange_15min",
            "interchange_hourly_ba",
            "interchange_5min_external",
            "system_load_5min",
            "forecast_7day_system",
            "system_conditions",
            "dayahead_operating_reserve",
            "nextday_operational_capacity",
        ],
        help="Data types to download",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=1,
        help="Maximum concurrent requests",
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Download in reverse chronological order",
    )

    args = parser.parse_args()

    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")

    # Handle "all" data types
    if "all" in args.data_types:
        data_types = [
            "load_forecast_hourly",
            "load_forecast_zonal",
            "load_forecast_reliability",
            "wind_forecast",
            "fuel_mix",
            "demand_combined_hourly",
            "demand_dayahead_hourly",
            "demand_realtime_hourly",
            "zonal_load",
            "capacity_scarcity",
            "reserve_price_hourly",
            "reserve_price_5min",
            "dayahead_constraints",
            "dayahead_lmp_congestion",
            "outages",
            "realtime_constraints",
            "fiveminute_constraints",
            "interchange_15min",
            "interchange_hourly_ba",
            "interchange_5min_external",
            "system_load_5min",
            "forecast_7day_system",
            "system_conditions",
            "dayahead_operating_reserve",
            "nextday_operational_capacity",
        ]
    else:
        data_types = args.data_types

    async with ISONEOperationalDataDownloader(max_concurrent=args.max_concurrent) as downloader:
        await downloader.download_date_range(
            start_date,
            end_date,
            data_types,
            reverse=args.reverse,
        )


if __name__ == "__main__":
    asyncio.run(main())
