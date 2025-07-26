"""ISONE-specific constants and configurations."""

# ISONE market types
MARKET_TYPES = {
    "DAM": "Day-Ahead Market",
    "RTM": "Real-Time Market (5-minute)"
}

# ISONE zones and hubs
ISONE_ZONES = [
    "4001",  # Maine (ME)
    "4002",  # New Hampshire (NH)
    "4003",  # Vermont (VT)
    "4004",  # Connecticut (CT)
    "4005",  # Rhode Island (RI)
    "4006",  # Southeast Mass (SEMA)
    "4007",  # Western/Central Mass (WCMA)
    "4008",  # Northeast Mass/Boston (NEMA)
]

# ISONE hub
ISONE_HUB = ".H.INTERNAL_HUB"

# Major interfaces
ISONE_INTERFACES = [
    ".I.ROSETON345_SOUTH",
    ".I.HQ_PHASE2",
    ".I.NORTHPORT-NORWALK",
    ".I.NEW_BRUNSWICK_HQ"
]

# ISONE ancillary services
ANCILLARY_SERVICES = {
    "TMSR": "Ten Minute Spinning Reserve",
    "TMNSR": "Ten Minute Non-Spinning Reserve",
    "TMOR": "Thirty Minute Operating Reserve",
    "REG": "Regulation",
    "REG_CAPACITY": "Regulation Capacity",
    "REG_SERVICE": "Regulation Service"
}

# Reserve zones
RESERVE_ZONES = [
    "SYSTEM",
    "CONNECTICUT",
    "NEMA",
    "SWCT"
]

# API configuration
ISONE_API_BASE = "https://webservices.iso-ne.com/api/v1.1"
API_TIMEOUT = 60
MAX_RETRIES = 3

# Endpoints
ENDPOINTS = {
    "DAM_LMP": "/hourlylmp/da/final/day/{date}/location/{location}.json",
    "RTM_LMP": "/fiveminutelmp/final/day/{date}/location/{location}.json",
    "DAM_AS": "/hourlydayaheadconstraint/day/{date}.json",
    "RTM_AS": "/fiveminutereserveprice/final/day/{date}/reserveZone/{zone}.json",
    "FREQ_REG": "/fiveminutercp/final/day/{date}.json",
    "LOCATIONS": "/locations/current.json"
}