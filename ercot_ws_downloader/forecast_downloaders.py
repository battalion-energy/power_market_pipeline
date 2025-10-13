"""
ERCOT Forecast and Generation Downloader implementations.

This module implements downloaders for:
- Wind power forecasts and actuals
- Solar power forecasts and actuals
- Load forecasts
- Actual system load
- Fuel mix data
- Resource outages
- System metrics

All data is used for price forecasting models.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict

from .base_downloader import BaseDownloader


class WindPowerDownloader(BaseDownloader):
    """
    Wind Power Production - Hourly Averaged Actual and Forecasted Values.

    Report NP4-732-CD includes:
    - System-wide and regional actual hourly averaged wind power production (GEN)
    - STWPF (Short-Term Wind Power Forecast)
    - WGRPP (Wind Generation Resource Power Potential)
    - COP HSLs for Online WGRs
    - Historical 48 hours + Future 168 hours
    """

    def get_dataset_name(self) -> str:
        return "Wind_Power_Production"

    def get_endpoint(self) -> str:
        return "np4-732-cd/wpp_hrly_avrg_actl_fcast"

    def get_output_dir(self) -> Path:
        return self.output_dir / "Wind_Power_Production"

    def format_params(self, start_date: datetime, end_date: datetime) -> Dict:
        return {
            "deliveryDateFrom": start_date.strftime("%Y-%m-%d"),
            "deliveryDateTo": end_date.strftime("%Y-%m-%d"),
        }

    def get_chunk_size(self) -> int:
        # Hourly data with forecasts, moderate size
        return 30  # 30 days per chunk

    def get_page_size(self) -> int:
        # System-wide + regions * 24 hours * 30 days + forecasts
        return 50000


class SolarPowerDownloader(BaseDownloader):
    """
    Solar Power Production - Hourly Averaged Actual and Forecasted Values by Geographical Region.

    Report NP4-745-CD includes:
    - System-wide and regional actual hourly averaged solar power production (GEN)
    - STPPF (Short-Term PhotoVoltaic Power Forecast)
    - PVGRPP (PhotoVoltaic Generation Resource Power Potential)
    - COP HSLs for Online PVGRs
    - Historical 48 hours + Future 168 hours
    """

    def get_dataset_name(self) -> str:
        return "Solar_Power_Production"

    def get_endpoint(self) -> str:
        return "np4-745-cd/spp_hrly_actual_fcast_geo"

    def get_output_dir(self) -> Path:
        return self.output_dir / "Solar_Power_Production"

    def format_params(self, start_date: datetime, end_date: datetime) -> Dict:
        return {
            "deliveryDateFrom": start_date.strftime("%Y-%m-%d"),
            "deliveryDateTo": end_date.strftime("%Y-%m-%d"),
        }

    def get_chunk_size(self) -> int:
        return 30  # 30 days per chunk

    def get_page_size(self) -> int:
        return 50000


class LoadForecastByForecastZoneDownloader(BaseDownloader):
    """
    Seven-Day Load Forecast by Forecast Zone.

    Report NP3-565-CD includes:
    - Hourly load forecasts by forecast zone
    - 7-day rolling forecast (168 hours)
    - Updated hourly
    """

    def get_dataset_name(self) -> str:
        return "Load_Forecast_By_Forecast_Zone"

    def get_endpoint(self) -> str:
        # FIXED: Correct endpoint is lf_by_model_weather_zone
        return "np3-565-cd/lf_by_model_weather_zone"

    def get_output_dir(self) -> Path:
        return self.output_dir / "Load_Forecast_By_Forecast_Zone"

    def format_params(self, start_date: datetime, end_date: datetime) -> Dict:
        # This endpoint uses postedDatetime instead of deliveryDate
        return {
            "postedDatetimeFrom": start_date.strftime("%Y-%m-%dT00:00:00"),
            "postedDatetimeTo": end_date.strftime("%Y-%m-%dT23:59:59"),
        }

    def get_chunk_size(self) -> int:
        # Forecasts are posted hourly, so lots of data
        return 7  # 7 days per chunk

    def get_page_size(self) -> int:
        # Multiple forecast zones * 24 posts/day * 168 forecast hours
        return 50000


class LoadForecastByWeatherZoneDownloader(BaseDownloader):
    """
    Seven-Day Load Forecast by Weather Zone.

    Report NP3-566-CD includes:
    - Hourly load forecasts by weather zone
    - 7-day rolling forecast (168 hours)
    - Updated hourly
    """

    def get_dataset_name(self) -> str:
        return "Load_Forecast_By_Weather_Zone"

    def get_endpoint(self) -> str:
        # FIXED: Correct endpoint is lf_by_model_study_area
        return "np3-566-cd/lf_by_model_study_area"

    def get_output_dir(self) -> Path:
        return self.output_dir / "Load_Forecast_By_Weather_Zone"

    def format_params(self, start_date: datetime, end_date: datetime) -> Dict:
        return {
            "postedDatetimeFrom": start_date.strftime("%Y-%m-%dT00:00:00"),
            "postedDatetimeTo": end_date.strftime("%Y-%m-%dT23:59:59"),
        }

    def get_chunk_size(self) -> int:
        return 7  # 7 days per chunk

    def get_page_size(self) -> int:
        return 50000


class ActualSystemLoadByWeatherZoneDownloader(BaseDownloader):
    """
    Actual System Load by Weather Zone.

    Report NP6-345-CD includes:
    - Actual system load by weather zone
    - Hourly intervals (not 5-minute as originally thought)
    - System-wide and zone-level data
    """

    def get_dataset_name(self) -> str:
        return "Actual_System_Load_By_Weather_Zone"

    def get_endpoint(self) -> str:
        # FIXED: Endpoint name is abbreviated to wzn (not wzones)
        return "np6-345-cd/act_sys_load_by_wzn"

    def get_output_dir(self) -> Path:
        return self.output_dir / "Actual_System_Load_By_Weather_Zone"

    def format_params(self, start_date: datetime, end_date: datetime) -> Dict:
        # FIXED: Uses operatingDay parameters (not deliveryDate)
        return {
            "operatingDayFrom": start_date.strftime("%Y-%m-%d"),
            "operatingDayTo": end_date.strftime("%Y-%m-%d"),
        }

    def get_chunk_size(self) -> int:
        # Hourly data
        return 30  # 30 days per chunk

    def get_page_size(self) -> int:
        # Multiple zones * 24 hours/day * 30 days
        return 50000


class ActualSystemLoadByForecastZoneDownloader(BaseDownloader):
    """
    Actual System Load by Forecast Zone.

    Report NP6-346-CD includes:
    - Actual system load by forecast zone
    - Hourly intervals
    - System-wide and zone-level data
    """

    def get_dataset_name(self) -> str:
        return "Actual_System_Load_By_Forecast_Zone"

    def get_endpoint(self) -> str:
        # FIXED: Endpoint name is abbreviated to fzn (not fzones) - following NP6-345 pattern
        return "np6-346-cd/act_sys_load_by_fzn"

    def get_output_dir(self) -> Path:
        return self.output_dir / "Actual_System_Load_By_Forecast_Zone"

    def format_params(self, start_date: datetime, end_date: datetime) -> Dict:
        # FIXED: Uses operatingDay parameters (same as NP6-345-CD)
        return {
            "operatingDayFrom": start_date.strftime("%Y-%m-%d"),
            "operatingDayTo": end_date.strftime("%Y-%m-%d"),
        }

    def get_chunk_size(self) -> int:
        # Hourly data
        return 30  # 30 days per chunk

    def get_page_size(self) -> int:
        return 50000


class UnplannedResourceOutagesDownloader(BaseDownloader):
    """
    Unplanned Resource Outages Report.

    Report NP3-233-CD includes:
    - Unplanned outages for generation resources
    - Start/end times
    - Resource name and capacity
    - Nature of outage
    """

    def get_dataset_name(self) -> str:
        return "Unplanned_Resource_Outages"

    def get_endpoint(self) -> str:
        return "np3-233-cd/unpl_res_outages"

    def get_output_dir(self) -> Path:
        return self.output_dir / "Unplanned_Resource_Outages"

    def format_params(self, start_date: datetime, end_date: datetime) -> Dict:
        # Outages use publishDatetime
        return {
            "publishDatetimeFrom": start_date.strftime("%Y-%m-%dT00:00:00"),
            "publishDatetimeTo": end_date.strftime("%Y-%m-%dT23:59:59"),
        }

    def get_chunk_size(self) -> int:
        return 30  # 30 days per chunk

    def get_page_size(self) -> int:
        # Variable, depends on number of outages
        return 10000

    def get_lag_days(self) -> int:
        # No lag for outages
        return 0


class DAMSystemLambdaDownloader(BaseDownloader):
    """
    DAM System Lambda.

    Report NP4-523-CD includes:
    - Day-Ahead Market System Lambda (shadow price)
    - Hourly values
    - Important indicator of system constraints
    """

    def get_dataset_name(self) -> str:
        return "DAM_System_Lambda"

    def get_endpoint(self) -> str:
        # FIXED: Correct endpoint is dam_system_lambda (full word "system", not "sys")
        return "np4-523-cd/dam_system_lambda"

    def get_output_dir(self) -> Path:
        return self.output_dir / "DAM_System_Lambda"

    def format_params(self, start_date: datetime, end_date: datetime) -> Dict:
        return {
            "deliveryDateFrom": start_date.strftime("%Y-%m-%d"),
            "deliveryDateTo": end_date.strftime("%Y-%m-%d"),
        }

    def get_chunk_size(self) -> int:
        return 30  # 30 days per chunk

    def get_page_size(self) -> int:
        # Hourly, single value per hour
        return 10000


class FuelMixDownloader(BaseDownloader):
    """
    Fuel Mix Report (2-Day Aggregate Generation Summary).

    Report NP3-910-ER includes:
    - Aggregate generation by resource type (15-minute intervals)
    - NonIRR (Non-Intermittent Renewable Resources)
    - WGR (Wind Generation Resources)
    - PVGR (PhotoVoltaic Generation Resources)
    - REMRES (Remaining Resources)
    - Total generation telemetry
    """

    def get_dataset_name(self) -> str:
        return "Fuel_Mix"

    def get_endpoint(self) -> str:
        # FIXED: Use NP3-910-ER (2-day aggregate generation summary)
        # Original NP6-787-CD doesn't exist in API
        return "np3-910-er/2d_agg_gen_summary"

    def get_output_dir(self) -> Path:
        return self.output_dir / "Fuel_Mix"

    def format_params(self, start_date: datetime, end_date: datetime) -> Dict:
        # FIXED: Uses SCEDTimestamp parameters (not deliveryDate)
        return {
            "SCEDTimestampFrom": start_date.strftime("%Y-%m-%dT%H:%M"),
            "SCEDTimestampTo": end_date.strftime("%Y-%m-%dT23:55"),
        }

    def get_chunk_size(self) -> int:
        # 15-minute data (96 intervals/day)
        return 7  # 7 days per chunk

    def get_page_size(self) -> int:
        # 96 intervals/day * 7 days = 672 records per chunk
        return 50000


class SystemWideDemandDownloader(BaseDownloader):
    """
    System Wide Demand - Calculated from Weather Zone data.

    Since NP6-322-CD/act_sys_load_5_min doesn't exist in the API,
    this downloader uses NP6-345-CD (Actual System Load by Weather Zone)
    which already includes a "total" column with system-wide demand.

    The weather zone data includes:
    - Individual zone loads (coast, east, farWest, north, northC, southern, southC, west)
    - Total system load (already summed)
    - Hourly intervals
    """

    def get_dataset_name(self) -> str:
        return "System_Wide_Demand"

    def get_endpoint(self) -> str:
        # WORKAROUND: Use weather zone endpoint which includes system total
        return "np6-345-cd/act_sys_load_by_wzn"

    def get_output_dir(self) -> Path:
        return self.output_dir / "System_Wide_Demand"

    def format_params(self, start_date: datetime, end_date: datetime) -> Dict:
        # Uses operatingDay parameters (same as weather zone data)
        return {
            "operatingDayFrom": start_date.strftime("%Y-%m-%d"),
            "operatingDayTo": end_date.strftime("%Y-%m-%d"),
        }

    def get_chunk_size(self) -> int:
        # Hourly data
        return 30  # 30 days per chunk

    def get_page_size(self) -> int:
        # 24 hours/day * 30 days
        return 50000

    async def download_chunk(
        self, start_date: datetime, end_date: datetime
    ):
        """
        Download and process weather zone data to extract system-wide demand.

        Override to process data after download:
        - Downloads full weather zone data (NP6-345-CD)
        - Extracts only system-wide total column
        - Returns processed data with reduced columns
        """
        # Call parent download_chunk to get weather zone data
        data = await super().download_chunk(start_date, end_date)

        if not data:
            return data

        # Process to extract only system-wide demand
        # Weather zone data has columns: operatingDay, hourEnding, coast, east,
        # farWest, north, northC, southern, southC, west, total, DSTFlag

        processed = []
        for record in data:
            if isinstance(record, list) and len(record) >= 11:
                # Array format: extract operatingDay, hourEnding, total, DSTFlag
                processed.append([
                    record[0],   # operatingDay
                    record[1],   # hourEnding
                    record[9],   # total (system-wide demand)
                    record[10] if len(record) > 10 else None   # DSTFlag
                ])
            elif isinstance(record, dict):
                # Dict format
                processed.append({
                    "operatingDay": record.get("operatingDay"),
                    "hourEnding": record.get("hourEnding"),
                    "systemLoad": record.get("total"),
                    "DSTFlag": record.get("DSTFlag")
                })
            else:
                # Unknown format, keep as is
                processed.append(record)

        logger = logging.getLogger(__name__)
        logger.info(
            f"{self.dataset_name}: Extracted system-wide demand from {len(processed)} weather zone records"
        )

        return processed
