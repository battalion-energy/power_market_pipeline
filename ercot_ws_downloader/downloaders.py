"""
Specific downloader implementations for each ERCOT dataset type.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict

from .base_downloader import BaseDownloader


class DAMPriceDownloader(BaseDownloader):
    """Day-Ahead Market Settlement Point Prices downloader."""

    def get_dataset_name(self) -> str:
        return "DAM_Prices"

    def get_endpoint(self) -> str:
        return "np4-190-cd/dam_stlmnt_pnt_prices"

    def get_output_dir(self) -> Path:
        return self.output_dir / "DAM_Settlement_Point_Prices"

    def format_params(self, start_date: datetime, end_date: datetime) -> Dict:
        return {
            "deliveryDateFrom": start_date.strftime("%Y-%m-%d"),
            "deliveryDateTo": end_date.strftime("%Y-%m-%d"),
        }

    def get_chunk_size(self) -> int:
        # DAM prices are hourly, can handle larger chunks
        return 30  # 30 days per chunk

    def get_page_size(self) -> int:
        # ~50 settlement points * 24 hours * 30 days = ~36,000 records
        return 50000


class RTMPriceDownloader(BaseDownloader):
    """Real-Time Market Settlement Point Prices downloader."""

    def get_dataset_name(self) -> str:
        return "RTM_Prices"

    def get_endpoint(self) -> str:
        # ERCOT RT prices endpoint
        return "np6-785-cd/rtm_spp"

    def get_output_dir(self) -> Path:
        return self.output_dir / "RTM_Settlement_Point_Prices"

    def format_params(self, start_date: datetime, end_date: datetime) -> Dict:
        return {
            "deliveryDateFrom": start_date.strftime("%Y-%m-%d"),
            "deliveryDateTo": end_date.strftime("%Y-%m-%d"),
        }

    def get_chunk_size(self) -> int:
        # RT prices are 5-minute, much more data
        return 7  # 7 days per chunk

    def get_page_size(self) -> int:
        # ~50 settlement points * 288 intervals/day * 7 days = ~100,000 records
        return 50000


class ASPriceDownloader(BaseDownloader):
    """
    Ancillary Services Prices downloader (all AS products).

    This endpoint returns DAM clearing prices for ALL AS products:
    - REGUP (Regulation Up)
    - REGDN (Regulation Down)
    - RRS (Responsive Reserve Service)
    - NSPIN (Non-Spinning Reserve)
    - ECRS (ERCOT Contingency Reserve Service)
    """

    def get_dataset_name(self) -> str:
        return "AS_Prices"

    def get_endpoint(self) -> str:
        # NP4-188-CD: DAM Clearing Prices for Capacity
        return "np4-188-cd/dam_clear_price_for_cap"

    def get_output_dir(self) -> Path:
        return self.output_dir / "AS_Prices"

    def format_params(self, start_date: datetime, end_date: datetime) -> Dict:
        return {
            "deliveryDateFrom": start_date.strftime("%Y-%m-%d"),
            "deliveryDateTo": end_date.strftime("%Y-%m-%d"),
        }

    def get_chunk_size(self) -> int:
        return 30  # 30 days per chunk

    def get_page_size(self) -> int:
        # ~5 AS types * 24 hours * 30 days = ~3,600 records per chunk
        return 50000


class DAMDisclosureDownloader(BaseDownloader):
    """
    60-Day DAM Generation Resource Data downloader.

    This is the CRITICAL dataset for BESS revenue analysis!
    Contains DAM awards, prices, and ancillary service awards.
    """

    def get_dataset_name(self) -> str:
        return "60d_DAM_Gen_Resources"

    def get_endpoint(self) -> str:
        return "np3-966-er/60_dam_gen_res_data"

    def get_output_dir(self) -> Path:
        return self.output_dir / "60-Day_DAM_Disclosure_Reports" / "Gen_Resources"

    def format_params(self, start_date: datetime, end_date: datetime) -> Dict:
        # 60-day disclosure uses different date format sometimes
        return {
            "deliveryDateFrom": start_date.strftime("%Y-%m-%d"),
            "deliveryDateTo": end_date.strftime("%Y-%m-%d"),
        }

    def get_chunk_size(self) -> int:
        # 60-day disclosure has HUGE files, use small chunks
        return 3  # 3 days per chunk

    def get_page_size(self) -> int:
        # Lots of generation resources, many columns
        return 10000

    def get_lag_days(self) -> int:
        # 60-day disclosure lag
        return 60


class DAMLoadResourceDownloader(BaseDownloader):
    """60-Day DAM Load Resource Data downloader (for BESS charging)."""

    def get_dataset_name(self) -> str:
        return "60d_DAM_Load_Resources"

    def get_endpoint(self) -> str:
        return "np3-966-er/60_dam_load_res_data"

    def get_output_dir(self) -> Path:
        return self.output_dir / "60-Day_DAM_Disclosure_Reports" / "Load_Resources"

    def format_params(self, start_date: datetime, end_date: datetime) -> Dict:
        return {
            "deliveryDateFrom": start_date.strftime("%Y-%m-%d"),
            "deliveryDateTo": end_date.strftime("%Y-%m-%d"),
        }

    def get_chunk_size(self) -> int:
        return 3

    def get_page_size(self) -> int:
        return 10000

    def get_lag_days(self) -> int:
        return 60


class SCEDDisclosureDownloader(BaseDownloader):
    """
    60-Day SCED Generation Resource Data downloader.

    CRITICAL for BESS revenue analysis!
    Contains actual dispatch, base points, telemetered output, and SOC tracking.
    """

    def get_dataset_name(self) -> str:
        return "60d_SCED_Gen_Resources"

    def get_endpoint(self) -> str:
        return "np3-965-er/60_sced_gen_res_data"

    def get_output_dir(self) -> Path:
        return self.output_dir / "60-Day_SCED_Disclosure_Reports" / "Gen_Resources"

    def format_params(self, start_date: datetime, end_date: datetime) -> Dict:
        # SCED endpoints use SCEDTimestamp parameters, not deliveryDate
        return {
            "SCEDTimestampFrom": start_date.strftime("%Y-%m-%dT%H:%M"),
            "SCEDTimestampTo": end_date.strftime("%Y-%m-%dT23:55"),  # End of day
        }

    def get_chunk_size(self) -> int:
        # SCED data is MASSIVE (5-minute intervals for all resources)
        return 1  # 1 day per chunk

    def get_page_size(self) -> int:
        # Tons of resources, tons of columns
        return 5000

    def get_lag_days(self) -> int:
        return 60


class SCEDLoadResourceDownloader(BaseDownloader):
    """60-Day SCED Load Resource Data downloader (for BESS charging)."""

    def get_dataset_name(self) -> str:
        return "60d_SCED_Load_Resources"

    def get_endpoint(self) -> str:
        return "np3-965-er/60_load_res_data_in_sced"

    def get_output_dir(self) -> Path:
        return self.output_dir / "60-Day_SCED_Disclosure_Reports" / "Load_Resources"

    def format_params(self, start_date: datetime, end_date: datetime) -> Dict:
        # SCED endpoints use SCEDTimestamp parameters, not deliveryDate
        return {
            "SCEDTimestampFrom": start_date.strftime("%Y-%m-%dT%H:%M"),
            "SCEDTimestampTo": end_date.strftime("%Y-%m-%dT23:55"),  # End of day
        }

    def get_chunk_size(self) -> int:
        return 1  # 1 day per chunk

    def get_page_size(self) -> int:
        return 5000

    def get_lag_days(self) -> int:
        return 60
