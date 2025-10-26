#!/usr/bin/env python3
"""
European Bidding Zone Configuration for ENTSO-E Transparency Platform

This module contains all European bidding zone codes and metadata for use with
the ENTSO-E API. Germany is prioritized, with support for all European zones.

Reference: https://transparency.entsoe.eu/content/static_content/Static%20content/web%20api/Guide.html
"""

from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class BiddingZone:
    """Configuration for a European bidding zone."""
    code: str
    name: str
    country: str
    timezone: str
    priority: int  # 1 = highest priority (focus markets like Germany)
    notes: Optional[str] = None


# European Bidding Zones
# Priority 1: Focus markets for BESS projects
# Priority 2: Secondary markets
# Priority 3: Additional coverage

BIDDING_ZONES = {
    # GERMANY - Priority 1 (Primary Focus)
    'DE_LU': BiddingZone(
        code='10Y1001A1001A82H',
        name='Germany-Luxembourg',
        country='Germany/Luxembourg',
        timezone='Europe/Berlin',
        priority=1,
        notes='Primary BESS market. Includes FCR, aFRR, mFRR opportunities'
    ),

    # PRIORITY 1 MARKETS - High BESS opportunity
    'FR': BiddingZone(
        code='10YFR-RTE------C',
        name='France',
        country='France',
        timezone='Europe/Paris',
        priority=1,
        notes='Large market, good FCR/aFRR opportunities'
    ),

    'NL': BiddingZone(
        code='10YNL----------L',
        name='Netherlands',
        country='Netherlands',
        timezone='Europe/Amsterdam',
        priority=1,
        notes='High price volatility, strong BESS market'
    ),

    'BE': BiddingZone(
        code='10YBE----------2',
        name='Belgium',
        country='Belgium',
        timezone='Europe/Brussels',
        priority=1,
        notes='Good ancillary services market'
    ),

    'AT': BiddingZone(
        code='10YAT-APG------L',
        name='Austria',
        country='Austria',
        timezone='Europe/Vienna',
        priority=1,
        notes='Connected to German market'
    ),

    'CH': BiddingZone(
        code='10YCH-SWISSGRIDZ',
        name='Switzerland',
        country='Switzerland',
        timezone='Europe/Zurich',
        priority=1,
        notes='High prices, pumped hydro dominance'
    ),

    'IT_NORTH': BiddingZone(
        code='10Y1001A1001A73I',
        name='Italy North',
        country='Italy',
        timezone='Europe/Rome',
        priority=1,
        notes='High prices, good arbitrage opportunities'
    ),

    # PRIORITY 2 MARKETS - Secondary focus
    'GB': BiddingZone(
        code='10YGB----------A',
        name='Great Britain',
        country='United Kingdom',
        timezone='Europe/London',
        priority=2,
        notes='Large BESS market, separate from EU since Brexit'
    ),

    'DK_1': BiddingZone(
        code='10YDK-1--------W',
        name='Denmark West (DK1)',
        country='Denmark',
        timezone='Europe/Copenhagen',
        priority=2,
        notes='High wind penetration, price volatility'
    ),

    'DK_2': BiddingZone(
        code='10YDK-2--------M',
        name='Denmark East (DK2)',
        country='Denmark',
        timezone='Europe/Copenhagen',
        priority=2,
        notes='Connected to Sweden and Germany'
    ),

    'SE_1': BiddingZone(
        code='10Y1001A1001A44P',
        name='Sweden SE1 (North)',
        country='Sweden',
        timezone='Europe/Stockholm',
        priority=2,
        notes='Hydro-dominated, low prices'
    ),

    'SE_2': BiddingZone(
        code='10Y1001A1001A45N',
        name='Sweden SE2',
        country='Sweden',
        timezone='Europe/Stockholm',
        priority=2,
        notes='Mid-Sweden zone'
    ),

    'SE_3': BiddingZone(
        code='10Y1001A1001A46L',
        name='Sweden SE3',
        country='Sweden',
        timezone='Europe/Stockholm',
        priority=2,
        notes='South-central Sweden'
    ),

    'SE_4': BiddingZone(
        code='10Y1001A1001A47J',
        name='Sweden SE4 (South)',
        country='Sweden',
        timezone='Europe/Stockholm',
        priority=2,
        notes='Connected to Denmark and Poland, higher prices'
    ),

    'NO_1': BiddingZone(
        code='10YNO-1--------2',
        name='Norway NO1 (Oslo)',
        country='Norway',
        timezone='Europe/Oslo',
        priority=2,
        notes='Hydro-dominated'
    ),

    'NO_2': BiddingZone(
        code='10YNO-2--------T',
        name='Norway NO2 (Kristiansand)',
        country='Norway',
        timezone='Europe/Oslo',
        priority=2,
        notes='Southern Norway'
    ),

    'NO_3': BiddingZone(
        code='10YNO-3--------J',
        name='Norway NO3 (Trondheim)',
        country='Norway',
        timezone='Europe/Oslo',
        priority=2,
        notes='Central Norway'
    ),

    'NO_4': BiddingZone(
        code='10YNO-4--------9',
        name='Norway NO4 (Tromsø)',
        country='Norway',
        timezone='Europe/Oslo',
        priority=2,
        notes='Northern Norway'
    ),

    'NO_5': BiddingZone(
        code='10Y1001A1001A48H',
        name='Norway NO5 (Bergen)',
        country='Norway',
        timezone='Europe/Oslo',
        priority=2,
        notes='Western Norway'
    ),

    'ES': BiddingZone(
        code='10YES-REE------0',
        name='Spain',
        country='Spain',
        timezone='Europe/Madrid',
        priority=2,
        notes='High solar penetration, growing BESS market'
    ),

    'PT': BiddingZone(
        code='10YPT-REN------W',
        name='Portugal',
        country='Portugal',
        timezone='Europe/Lisbon',
        priority=2,
        notes='Coupled with Spain (MIBEL market)'
    ),

    'PL': BiddingZone(
        code='10YPL-AREA-----S',
        name='Poland',
        country='Poland',
        timezone='Europe/Warsaw',
        priority=2,
        notes='Coal-dominated, transitioning'
    ),

    'CZ': BiddingZone(
        code='10YCZ-CEPS-----N',
        name='Czech Republic',
        country='Czech Republic',
        timezone='Europe/Prague',
        priority=2,
        notes='Connected to German market'
    ),

    # PRIORITY 3 MARKETS - Additional coverage
    'IT_CNOR': BiddingZone(
        code='10Y1001A1001A70O',
        name='Italy Central North',
        country='Italy',
        timezone='Europe/Rome',
        priority=3
    ),

    'IT_CSUD': BiddingZone(
        code='10Y1001A1001A71M',
        name='Italy Central South',
        country='Italy',
        timezone='Europe/Rome',
        priority=3
    ),

    'IT_SUD': BiddingZone(
        code='10Y1001A1001A788',
        name='Italy South',
        country='Italy',
        timezone='Europe/Rome',
        priority=3
    ),

    'IT_SICI': BiddingZone(
        code='10Y1001A1001A77A',
        name='Italy Sicily',
        country='Italy',
        timezone='Europe/Rome',
        priority=3
    ),

    'IT_SARD': BiddingZone(
        code='10Y1001A1001A74G',
        name='Italy Sardinia',
        country='Italy',
        timezone='Europe/Rome',
        priority=3
    ),

    'FI': BiddingZone(
        code='10YFI-1--------U',
        name='Finland',
        country='Finland',
        timezone='Europe/Helsinki',
        priority=3
    ),

    'EE': BiddingZone(
        code='10Y1001A1001A39I',
        name='Estonia',
        country='Estonia',
        timezone='Europe/Tallinn',
        priority=3,
        notes='Baltic market'
    ),

    'LV': BiddingZone(
        code='10YLV-1001A00074',
        name='Latvia',
        country='Latvia',
        timezone='Europe/Riga',
        priority=3,
        notes='Baltic market'
    ),

    'LT': BiddingZone(
        code='10YLT-1001A0008Q',
        name='Lithuania',
        country='Lithuania',
        timezone='Europe/Vilnius',
        priority=3,
        notes='Baltic market'
    ),

    'IE': BiddingZone(
        code='10YIE-1001A00010',
        name='Ireland (SEM)',
        country='Ireland',
        timezone='Europe/Dublin',
        priority=3,
        notes='Single Electricity Market with Northern Ireland'
    ),

    'HU': BiddingZone(
        code='10YHU-MAVIR----U',
        name='Hungary',
        country='Hungary',
        timezone='Europe/Budapest',
        priority=3
    ),

    'RO': BiddingZone(
        code='10YRO-TEL------P',
        name='Romania',
        country='Romania',
        timezone='Europe/Bucharest',
        priority=3
    ),

    'BG': BiddingZone(
        code='10YCA-BULGARIA-R',
        name='Bulgaria',
        country='Bulgaria',
        timezone='Europe/Sofia',
        priority=3
    ),

    'GR': BiddingZone(
        code='10YGR-HTSO-----Y',
        name='Greece',
        country='Greece',
        timezone='Europe/Athens',
        priority=3
    ),

    'HR': BiddingZone(
        code='10YHR-HEP------M',
        name='Croatia',
        country='Croatia',
        timezone='Europe/Zagreb',
        priority=3
    ),

    'SI': BiddingZone(
        code='10YSI-ELES-----O',
        name='Slovenia',
        country='Slovenia',
        timezone='Europe/Ljubljana',
        priority=3
    ),

    'SK': BiddingZone(
        code='10YSK-SEPS-----K',
        name='Slovakia',
        country='Slovakia',
        timezone='Europe/Bratislava',
        priority=3
    ),
}


# Convenience groupings
def get_zones_by_priority(priority: int) -> List[BiddingZone]:
    """Get all bidding zones with a specific priority level."""
    return [zone for zone in BIDDING_ZONES.values() if zone.priority == priority]


def get_priority_1_zones() -> List[BiddingZone]:
    """Get priority 1 zones (primary BESS markets)."""
    return get_zones_by_priority(1)


def get_zone(zone_name: str) -> Optional[BiddingZone]:
    """Get a specific bidding zone by name."""
    return BIDDING_ZONES.get(zone_name)


def get_all_zone_codes() -> Dict[str, str]:
    """Get mapping of zone names to ENTSO-E codes."""
    return {name: zone.code for name, zone in BIDDING_ZONES.items()}


def get_germany_zone() -> BiddingZone:
    """Get the Germany-Luxembourg bidding zone."""
    return BIDDING_ZONES['DE_LU']


# Regional groupings for bulk operations
REGION_GROUPS = {
    'DACH': ['DE_LU', 'AT', 'CH'],  # Germany, Austria, Switzerland
    'BENELUX': ['BE', 'NL'],  # Belgium, Netherlands (LU included in DE_LU)
    'NORDIC': ['DK_1', 'DK_2', 'SE_1', 'SE_2', 'SE_3', 'SE_4', 'NO_1', 'NO_2', 'NO_3', 'NO_4', 'NO_5', 'FI'],
    'BALTIC': ['EE', 'LV', 'LT'],
    'IBERIA': ['ES', 'PT'],
    'ITALY': ['IT_NORTH', 'IT_CNOR', 'IT_CSUD', 'IT_SUD', 'IT_SICI', 'IT_SARD'],
    'CENTRAL_EUROPE': ['PL', 'CZ', 'SK', 'HU'],
    'BALKANS': ['SI', 'HR', 'RO', 'BG', 'GR'],
}


def get_region_zones(region: str) -> List[BiddingZone]:
    """Get all bidding zones in a specific region."""
    zone_names = REGION_GROUPS.get(region, [])
    return [BIDDING_ZONES[name] for name in zone_names if name in BIDDING_ZONES]


if __name__ == '__main__':
    # Print summary of configured zones
    print("European Bidding Zones Configuration")
    print("=" * 80)
    print(f"\nTotal zones configured: {len(BIDDING_ZONES)}")
    print(f"Priority 1 markets: {len(get_priority_1_zones())}")
    print(f"Priority 2 markets: {len(get_zones_by_priority(2))}")
    print(f"Priority 3 markets: {len(get_zones_by_priority(3))}")

    print("\n" + "=" * 80)
    print("PRIORITY 1 MARKETS (Primary BESS Focus)")
    print("=" * 80)
    for zone in get_priority_1_zones():
        print(f"\n{zone.name} ({zone.country})")
        print(f"  Code: {zone.code}")
        print(f"  Timezone: {zone.timezone}")
        if zone.notes:
            print(f"  Notes: {zone.notes}")

    print("\n" + "=" * 80)
    print("REGIONAL GROUPINGS")
    print("=" * 80)
    for region, zones in REGION_GROUPS.items():
        print(f"\n{region}: {', '.join(zones)}")
