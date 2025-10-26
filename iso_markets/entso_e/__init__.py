"""
ENTSO-E Transparency Platform & Regelleistung.net Data Pipeline

This package provides tools for downloading and processing electricity market data
from European markets, with primary focus on Germany for BESS projects.

Data Sources:
1. ENTSO-E Transparency Platform - Pan-European market data
   - Day-ahead prices (hourly)
   - Imbalance prices (15-minute, RT equivalent)
   - Load, generation, balancing energy

2. Regelleistung.net - German ancillary services
   - FCR (Frequency Containment Reserve)
   - aFRR (automatic Frequency Restoration Reserve) - BEST BESS opportunity
   - mFRR (manual Frequency Restoration Reserve)
   - Capacity and energy prices, 4-hour block structure

Main components:
- ENTSOEAPIClient: ENTSO-E API client with rate limiting
- RegelleistungAPIClient: Regelleistung.net API client
- european_zones: Configuration for all European bidding zones
- download_da_prices: Day-ahead market price downloader
- download_imbalance_prices: Imbalance price downloader
- download_ancillary_services: FCR/aFRR/mFRR downloader
- update_germany_with_resume: Unified Germany updater (all data sources)

Usage:
    from iso_markets.entso_e import ENTSOEAPIClient, RegelleistungAPIClient
    from iso_markets.entso_e import BIDDING_ZONES, get_germany_zone
"""

__version__ = '0.2.0'

# Import core components that don't require external dependencies
from .regelleistung_api_client import RegelleistungAPIClient
from .european_zones import (
    BIDDING_ZONES,
    BiddingZone,
    get_zone,
    get_priority_1_zones,
    get_zones_by_priority,
    get_germany_zone,
    REGION_GROUPS,
    get_region_zones
)

# Try to import ENTSO-E client (requires entsoe-py)
try:
    from .entso_e_api_client import ENTSOEAPIClient
    _ENTSO_E_AVAILABLE = True
except ImportError:
    ENTSOEAPIClient = None
    _ENTSO_E_AVAILABLE = False

__all__ = [
    'RegelleistungAPIClient',
    'BIDDING_ZONES',
    'BiddingZone',
    'get_zone',
    'get_priority_1_zones',
    'get_zones_by_priority',
    'get_germany_zone',
    'REGION_GROUPS',
    'get_region_zones',
]

# Add ENTSO-E client to exports only if available
if _ENTSO_E_AVAILABLE:
    __all__.append('ENTSOEAPIClient')
