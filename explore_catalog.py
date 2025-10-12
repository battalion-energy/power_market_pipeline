#!/usr/bin/env python3
"""
Explore ERCOT API catalog structure.
"""

import os
import asyncio
import httpx
import json
from dotenv import load_dotenv

load_dotenv()

async def authenticate():
    """Authenticate with ERCOT and get access token."""
    AUTH_URL = "https://ercotb2c.b2clogin.com/ercotb2c.onmicrosoft.com/B2C_1_PUBAPI-ROPC-FLOW/oauth2/v2.0/token"
    CLIENT_ID = "fec253ea-0d06-4272-a5e6-b478baeecd70"

    params = {
        "username": os.getenv("ERCOT_USERNAME"),
        "password": os.getenv("ERCOT_PASSWORD"),
        "grant_type": "password",
        "scope": f"openid {CLIENT_ID} offline_access",
        "client_id": CLIENT_ID,
        "response_type": "token",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(AUTH_URL, data=params)
        response.raise_for_status()
        return response.json()


async def main():
    print("\n" + "="*80)
    print("EXPLORING ERCOT API CATALOG")
    print("="*80)

    # Authenticate
    print("\nAuthenticating...")
    token_data = await authenticate()
    print("✅ Authenticated\n")

    subscription_key = os.getenv("ERCOT_SUBSCRIPTION_KEY")
    BASE_URL = "https://api.ercot.com/api/public-reports"

    headers = {
        "Authorization": f"Bearer {token_data['access_token']}",
        "Ocp-Apim-Subscription-Key": subscription_key,
        "Accept": "application/json",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(BASE_URL, headers=headers)

        if response.status_code == 200:
            catalog = response.json()

            print("Catalog structure:")
            print(f"  Top-level keys: {list(catalog.keys())}")

            if "_embedded" in catalog:
                embedded = catalog["_embedded"]
                print(f"\n_embedded keys: {list(embedded.keys())}")

                if "products" in embedded:
                    products = embedded["products"]
                    print(f"\nTotal products: {len(products)}")

                    # Each product might contain reports
                    all_reports = []
                    for product in products:
                        if "_embedded" in product and "publicReports" in product["_embedded"]:
                            reports = product["_embedded"]["publicReports"]
                            all_reports.extend(reports)

                    reports = all_reports
                    print(f"Total reports across all products: {len(reports)}")

                    # Search for outage, lambda, demand related reports
                    keywords = ["outage", "lambda", "demand", "load", "constraint"]

                    print(f"\nSearching for reports with keywords: {keywords}")
                    print("="*80)

                    found_reports = []
                    for report in reports:
                        report_name = report.get("reportDisplayName", "").lower()
                        report_id = report.get("reportId", "")
                        report_emil = report.get("reportEMIL", "")

                        for keyword in keywords:
                            if keyword in report_name:
                                found_reports.append({
                                    "name": report.get("reportDisplayName"),
                                    "id": report_id,
                                    "emil": report_emil,
                                    "endpoint": report.get("reportName"),
                                    "keyword": keyword
                                })
                                break

                    if found_reports:
                        print(f"\nFound {len(found_reports)} matching reports:\n")

                        # Group by keyword
                        for keyword in keywords:
                            keyword_reports = [r for r in found_reports if r["keyword"] == keyword]
                            if keyword_reports:
                                print(f"\n{keyword.upper()} reports:")
                                for r in keyword_reports:
                                    print(f"  ✅ {r['emil']}: {r['name']}")
                                    print(f"     Endpoint: {r['endpoint']}")
                    else:
                        print("\n❌ No reports found with these keywords")

                    # Save full catalog for reference
                    with open("/home/enrico/projects/power_market_pipeline/ercot_catalog.json", "w") as f:
                        json.dump(catalog, f, indent=2)
                    print(f"\n💾 Full catalog saved to ercot_catalog.json")

            if "_links" in catalog:
                links = catalog["_links"]
                print(f"\n_links keys: {list(links.keys())}")


if __name__ == "__main__":
    asyncio.run(main())
