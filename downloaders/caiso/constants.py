"""CAISO-specific constants and configurations."""

# CAISO market types
MARKET_TYPES = {
    "DAM": "Day-Ahead Market",
    "RTM": "Real-Time Market (5-minute)",
    "HASP": "Hour-Ahead Scheduling Process"
}

# CAISO node types
NODE_TYPES = {
    "TH": "Trading Hub",
    "ASP": "Aggregated System Pricing",
    "PNODE": "Pricing Node",
    "APNODE": "Aggregated Pricing Node",
    "DLAP": "Default Load Aggregation Point"
}

# Major CAISO trading hubs
TRADING_HUBS = [
    "TH_NP15_GEN-APND",
    "TH_SP15_GEN-APND",
    "TH_ZP26_GEN-APND"
]

# CAISO DLAPs (Default Load Aggregation Points)
DLAPS = [
    "DLAP_PGAE",
    "DLAP_SCE",
    "DLAP_SDGE",
    "DLAP_VEA"
]

# CAISO Ancillary Services
ANCILLARY_SERVICES = {
    "SPIN": "Spinning Reserve",
    "NON_SPIN": "Non-Spinning Reserve",
    "REG_UP": "Regulation Up",
    "REG_DN": "Regulation Down",
    "REG_MILEAGE_UP": "Regulation Mileage Up",
    "REG_MILEAGE_DN": "Regulation Mileage Down"
}

# CAISO OASIS API endpoints
OASIS_ENDPOINTS = {
    "LMP": {
        "DAM": "PRC_LMP",
        "RTM": "PRC_INTVL_LMP",
        "HASP": "PRC_HASP_LMP"
    },
    "AS_PRICES": {
        "DAM": "PRC_AS",
        "RTM": "PRC_INTVL_AS",
        "HASP": "PRC_HASP_AS"
    },
    "MILEAGE": {
        "DAM": "AS_MILEAGE_CALC",
        "RTM": "AS_MILEAGE_CALC"
    }
}

# API configuration
OASIS_BASE_URL = "http://oasis.caiso.com/oasisapi/SingleZip"
OASIS_TIMEOUT = 60  # seconds
MAX_RETRIES = 3