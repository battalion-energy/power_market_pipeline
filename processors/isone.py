"""ISONE data processor for standardizing and cleaning data."""

from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import pytz


class ISONEProcessor:
    """Process and standardize ISONE data."""
    
    def __init__(self):
        self.tz = pytz.timezone('US/Eastern')
    
    def process_energy_data(
        self, 
        data: Dict, 
        market_type: str, 
        location: str
    ) -> pd.DataFrame:
        """Process ISONE energy price data from API response."""
        records = []
        
        # Handle different response structures
        if market_type == "DAM":
            # Day-ahead hourly data
            if "HourlyLmps" in data and "HourlyLmp" in data["HourlyLmps"]:
                lmp_data = data["HourlyLmps"]["HourlyLmp"]
                if not isinstance(lmp_data, list):
                    lmp_data = [lmp_data]
                
                for item in lmp_data:
                    record = {
                        "timestamp": self._parse_timestamp(item["BeginDate"]),
                        "location": location,
                        "market_type": market_type,
                        "lmp_total": float(item.get("LmpTotal", 0)),
                        "energy_component": float(item.get("EnergyComponent", 0)),
                        "congestion_component": float(item.get("CongestionComponent", 0)),
                        "loss_component": float(item.get("LossComponent", 0))
                    }
                    records.append(record)
        
        elif market_type == "RTM":
            # Real-time 5-minute data
            if "FiveMinLmps" in data and "FiveMinLmp" in data["FiveMinLmps"]:
                lmp_data = data["FiveMinLmps"]["FiveMinLmp"]
                if not isinstance(lmp_data, list):
                    lmp_data = [lmp_data]
                
                for item in lmp_data:
                    record = {
                        "timestamp": self._parse_timestamp(item["BeginDate"]),
                        "location": location,
                        "market_type": market_type,
                        "lmp_total": float(item.get("LmpTotal", 0)),
                        "energy_component": float(item.get("EnergyComponent", 0)),
                        "congestion_component": float(item.get("CongestionComponent", 0)),
                        "loss_component": float(item.get("LossComponent", 0))
                    }
                    records.append(record)
        
        return pd.DataFrame(records)
    
    def process_reserve_data(
        self,
        data: Dict,
        service_type: str,
        market_type: str,
        zone: str
    ) -> pd.DataFrame:
        """Process ISONE reserve price data."""
        records = []
        
        if "FiveMinReservePrices" in data and "FiveMinReservePrice" in data["FiveMinReservePrices"]:
            reserve_data = data["FiveMinReservePrices"]["FiveMinReservePrice"]
            if not isinstance(reserve_data, list):
                reserve_data = [reserve_data]
            
            for item in reserve_data:
                # Map service types to price fields
                if service_type == "TMSR":
                    price = float(item.get("TmsrClearingPrice", 0))
                    quantity = float(item.get("TmsrDesignatedMw", 0))
                elif service_type == "TMNSR":
                    price = float(item.get("TmnsrClearingPrice", 0))
                    quantity = float(item.get("TmnsrDesignatedMw", 0))
                elif service_type == "TMOR":
                    price = float(item.get("TmorClearingPrice", 0))
                    quantity = float(item.get("TmorDesignatedMw", 0))
                else:
                    continue
                
                record = {
                    "timestamp": self._parse_timestamp(item["BeginDate"]),
                    "service_type": service_type,
                    "market_type": market_type,
                    "zone": zone,
                    "price": price,
                    "quantity_mw": quantity
                }
                records.append(record)
        
        return pd.DataFrame(records)
    
    def process_regulation_data(
        self,
        data: Dict,
        market_type: str
    ) -> pd.DataFrame:
        """Process ISONE frequency regulation data."""
        records = []
        
        if "FiveMinRcps" in data and "FiveMinRcp" in data["FiveMinRcps"]:
            reg_data = data["FiveMinRcps"]["FiveMinRcp"]
            if not isinstance(reg_data, list):
                reg_data = [reg_data]
            
            for item in reg_data:
                # Regulation service price
                records.append({
                    "timestamp": self._parse_timestamp(item["BeginDate"]),
                    "service_type": "REG_SERVICE",
                    "market_type": market_type,
                    "zone": "SYSTEM",
                    "price": float(item.get("RegServiceClearingPrice", 0)),
                    "quantity_mw": None
                })
                
                # Regulation capacity price
                records.append({
                    "timestamp": self._parse_timestamp(item["BeginDate"]),
                    "service_type": "REG_CAPACITY",
                    "market_type": market_type,
                    "zone": "SYSTEM",
                    "price": float(item.get("RegCapacityClearingPrice", 0)),
                    "quantity_mw": None
                })
        
        return pd.DataFrame(records)
    
    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """Parse ISONE timestamp format."""
        # ISONE timestamps are in format: "2024-01-01T00:00:00.000-05:00"
        # Parse and convert to timezone-aware datetime
        try:
            # Remove milliseconds if present
            if "." in timestamp_str:
                timestamp_str = timestamp_str.split(".")[0] + timestamp_str[-6:]
            
            dt = datetime.fromisoformat(timestamp_str)
            
            # Ensure it's in Eastern timezone
            if dt.tzinfo is None:
                dt = self.tz.localize(dt)
            else:
                dt = dt.astimezone(self.tz)
            
            return dt
        except Exception as e:
            # Fallback parsing
            dt_str = timestamp_str.replace("T", " ").split(".")[0].split("-")[0:3]
            dt = datetime.strptime(" ".join(dt_str), "%Y %m %d %H:%M:%S")
            return self.tz.localize(dt)