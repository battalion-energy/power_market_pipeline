"""ERCOT-specific constants and configurations."""

from datetime import datetime

# ERCOT market types
MARKET_TYPES = {
    "DAM": "Day-Ahead Market",
    "RTM": "Real-Time Market",
    "SCED": "Security Constrained Economic Dispatch"
}

# ERCOT ancillary service types
ANCILLARY_SERVICES = {
    "REGUP": "Regulation Up",
    "REGDN": "Regulation Down", 
    "SPIN": "Spinning Reserve",
    "NON_SPIN": "Non-Spinning Reserve",
    "RRS": "Responsive Reserve Service",
    "ECRS": "ERCOT Contingency Reserve Service"
}

# ERCOT trading hubs
TRADING_HUBS = [
    "HB_BUSAVG",
    "HB_HOUSTON", 
    "HB_NORTH",
    "HB_SOUTH",
    "HB_WEST",
    "HB_PAN"
]

# ERCOT zones
LOAD_ZONES = [
    "LZ_AEN",
    "LZ_CPS", 
    "LZ_HOUSTON",
    "LZ_NORTH",
    "LZ_RAYBN",
    "LZ_SOUTH",
    "LZ_WEST"
]

# ERCOT data products for selenium scraping
DATA_PRODUCTS = {
    "DAM_SPP": {
        "name": "DAM Settlement Point Prices",
        "url": "https://www.ercot.com/mp/data-products/data-product-details?id=NP6-785-CD",
        "requires_auth": False
    },
    "RTM_SPP": {
        "name": "RTM Settlement Point Prices", 
        "url": "https://www.ercot.com/mp/data-products/data-product-details?id=NP6-786-CD",
        "requires_auth": False
    },
    "SCED_SHADOW_PRICES": {
        "name": "SCED Shadow Prices and Binding Constraints",
        "url": "https://www.ercot.com/mp/data-products/data-product-details?id=NP6-788-CD",
        "requires_auth": False
    },
    "DAM_ANCILLARY": {
        "name": "DAM Clearing Prices for CRR Auction",
        "url": "https://www.ercot.com/mp/data-products/data-product-details?id=NP4-183-CD",
        "requires_auth": False
    },
    "HISTORICAL_RTM_SPP": {
        "name": "Historical RTM SPP",
        "url": "https://www.ercot.com/mp/data-products/data-product-details?id=NP6-786-FH",
        "requires_auth": True
    }
}

# ERCOT webservice API cutoff date
WEBSERVICE_CUTOFF_DATE = datetime(2023, 12, 11)

# File naming patterns
FILE_PATTERNS = {
    "DAM_SPP": "dam_spp_{date}.csv",
    "RTM_SPP": "rtm_spp_{date}.csv",
    "ANCILLARY": "ancillary_{service}_{date}.csv"
}