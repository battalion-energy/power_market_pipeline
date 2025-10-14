#!/usr/bin/env python3
"""
Check which ERCOT API endpoints are actually available.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ercot_ws_downloader.client import ERCOTWebServiceClient
import httpx

async def check_endpoint(client: ERCOTWebServiceClient, report_code: str, endpoint: str):
    """Check if an endpoint exists."""
    full_url = f"{client.base_url}/np{report_code[2:]}/{endpoint}"

    try:
        # Try a simple GET request with minimal params
        response = await client._make_request(
            endpoint=f"{report_code}/{endpoint}",
            params={"page": 1, "size": 1}
        )
        return True, None
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return False, "404 Not Found"
        else:
            return False, f"{e.response.status_code} {e.response.reason_phrase}"
    except Exception as e:
        return False, str(e)

async def main():
    """Check all endpoints."""

    # Endpoints to check (from forecast_downloaders.py)
    endpoints_to_check = [
        ("NP3-565-CD", "lf_by_fzones", "Load Forecast by Forecast Zone"),
        ("NP3-566-CD", "lf_by_wzones", "Load Forecast by Weather Zone"),
        ("NP6-345-CD", "act_sys_load_by_wzones", "Actual Load by Weather Zone"),
        ("NP6-346-CD", "act_sys_load_by_fzones", "Actual Load by Forecast Zone"),
        ("NP6-787-CD", "fuel_mix", "Fuel Mix"),
        ("NP6-322-CD", "act_sys_load_5_min", "System Wide Demand"),
        ("NP3-233-CD", "unpl_res_outages", "Unplanned Outages"),
        ("NP4-191-CD", "dam_sys_lambda", "DAM System Lambda"),
        ("NP4-732-CD", "wpp_hrly_avrg_actl_fcast", "Wind Power"),
        ("NP4-745-CD", "spp_hrly_actual_fcast_geo", "Solar Power"),
        ("NP4-190-CD", "dam_stlmnt_pnt_prices", "DAM Prices"),
        ("NP6-785-CD", "rtm_spp", "RTM Prices"),
        ("NP4-188-CD", "as_prices", "AS Prices"),
    ]

    print("\n" + "="*80)
    print("CHECKING ERCOT API ENDPOINTS")
    print("="*80 + "\n")

    client = ERCOTWebServiceClient()
    await client.authenticate()

    results = []
    for report_code, endpoint, name in endpoints_to_check:
        full_endpoint = f"{report_code.lower()}/{endpoint}"
        print(f"Checking: {name:40s} ({full_endpoint})")

        exists, error = await check_endpoint(client, report_code.lower(), endpoint)

        if exists:
            print(f"  ✅ EXISTS\n")
            results.append((name, report_code, endpoint, "✅ EXISTS"))
        else:
            print(f"  ❌ {error}\n")
            results.append((name, report_code, endpoint, f"❌ {error}"))

        await asyncio.sleep(0.5)  # Be polite to API

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80 + "\n")

    available = [r for r in results if "✅" in r[3]]
    unavailable = [r for r in results if "❌" in r[3]]

    print(f"Available endpoints: {len(available)}/{len(results)}\n")

    if unavailable:
        print("❌ UNAVAILABLE ENDPOINTS:\n")
        for name, report, endpoint, status in unavailable:
            print(f"  - {name}")
            print(f"    {report.lower()}/{endpoint}")
            print(f"    {status}\n")

    print("="*80 + "\n")

if __name__ == "__main__":
    asyncio.run(main())
