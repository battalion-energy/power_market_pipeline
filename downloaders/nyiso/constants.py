"""NYISO-specific constants and configurations."""

# NYISO market types
MARKET_TYPES = {
    "DAM": "Day-Ahead Market",
    "RTM": "Real-Time Market (5-minute)",
    "RTC": "Real-Time Commitment"
}

# NYISO zones
NYISO_ZONES = [
    "CAPITL",  # Capital Zone
    "CENTRL",  # Central Zone
    "DUNWOD",  # Dunwoodie Zone
    "GENESE",  # Genesee Zone
    "HUD VL",  # Hudson Valley Zone
    "LONGIL",  # Long Island Zone
    "MHK VL",  # Mohawk Valley Zone
    "MILLWD",  # Millwood Zone
    "N.Y.C.",  # New York City Zone
    "NORTH",   # North Zone
    "WEST"     # West Zone
]

# NYISO reference bus
NYISO_REF_BUS = "NYISO"

# NYISO ancillary services
ANCILLARY_SERVICES = {
    "SPIN_10": "10-Minute Spinning Reserve",
    "NON_SYNC_10": "10-Minute Non-Synchronized Reserve",
    "OPER_30": "30-Minute Operating Reserve",
    "REG": "Regulation",
    "REG_CAPACITY": "Regulation Capacity",
    "REG_MOVEMENT": "Regulation Movement"
}

# Data URLs
NYISO_DATA_URL = "http://mis.nyiso.com/public/csv/{type}/{filename}"

# File naming patterns
FILE_PATTERNS = {
    "DAM_LMP_ZONE": "{date}damlbmp_zone_csv.zip",
    "DAM_LMP_GEN": "{date}damlbmp_gen_csv.zip",
    "RTM_LMP_ZONE": "{date}realtime_zone_csv.zip",
    "RTM_LMP_GEN": "{date}realtime_gen_csv.zip",
    "DAM_ASP": "{date}damasp_csv.zip",
    "RTM_ASP": "{date}rtasp_csv.zip",
    "DAM_RESERVE": "{date}damreserve_csv.zip",
    "RTM_RESERVE": "{date}rtreserve_csv.zip",
}

# Data type mappings
DATA_TYPE_MAP = {
    "damlbmp": {"market": "DAM", "type": "lbmp"},
    "realtime": {"market": "RTM", "type": "lbmp"},
    "damasp": {"market": "DAM", "type": "asp"},
    "rtasp": {"market": "RTM", "type": "asp"},
    "damreserve": {"market": "DAM", "type": "reserve"},
    "rtreserve": {"market": "RTM", "type": "reserve"},
}

# CSV column mappings
COLUMN_MAPPINGS = {
    "LBMP": {
        "Time Stamp": "timestamp",
        "Name": "location",
        "PTID": "ptid",
        "LBMP ($/MWHr)": "lmp",
        "Marginal Cost Losses ($/MWHr)": "loss_component",
        "Marginal Cost Congestion ($/MWHr)": "congestion_component",
    },
    "ASP": {
        "Time Stamp": "timestamp",
        "Regulation Capacity Market Clearing Price ($/MW)": "reg_capacity_price",
        "Regulation Movement Clearing Price ($/MWh)": "reg_movement_price",
        "Regulation Service Clearing Price": "reg_service_price",
    },
    "RESERVE": {
        "Time Stamp": "timestamp",
        "Reserve Zone": "zone",
        "10 Min Spin": "spin_10_price",
        "10 Min Non-Sync": "non_sync_10_price",
        "30 Min Oper Rsv": "oper_30_price",
    }
}