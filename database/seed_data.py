"""Seed initial data for the database."""

from datetime import datetime

from .connection import get_db
from .models_v2 import ISO
from .dataset_models import DatasetCategory


def seed_isos():
    """Seed ISO data."""
    isos_data = [
        {
            "code": "ERCOT",
            "name": "Electric Reliability Council of Texas",
            "timezone": "America/Chicago",
        },
        {
            "code": "CAISO",
            "name": "California Independent System Operator",
            "timezone": "America/Los_Angeles",
        },
        {
            "code": "ISONE",
            "name": "ISO New England",
            "timezone": "America/New_York",
        },
        {
            "code": "NYISO",
            "name": "New York Independent System Operator",
            "timezone": "America/New_York",
        },
        {
            "code": "PJM",
            "name": "PJM Interconnection",
            "timezone": "America/New_York",
        },
        {
            "code": "MISO",
            "name": "Midcontinent Independent System Operator",
            "timezone": "America/New_York",
        },
        {
            "code": "SPP",
            "name": "Southwest Power Pool",
            "timezone": "America/Chicago",
        },
    ]
    
    with get_db() as db:
        for iso_data in isos_data:
            # Check if ISO already exists
            existing = db.query(ISO).filter(ISO.code == iso_data["code"]).first()
            if not existing:
                iso = ISO(**iso_data)
                db.add(iso)
        
        db.commit()
        print(f"✓ Seeded {len(isos_data)} ISOs")


def seed_dataset_categories():
    """Seed dataset categories."""
    categories = [
        {
            "name": "energy_prices",
            "description": "Locational marginal prices and energy market data",
            "display_order": 1,
        },
        {
            "name": "ancillary_services",
            "description": "Regulation, reserves, and other ancillary service markets",
            "display_order": 2,
        },
        {
            "name": "load",
            "description": "System load, demand forecasts, and consumption data",
            "display_order": 3,
        },
        {
            "name": "generation",
            "description": "Generation by fuel type, unit availability, and capacity",
            "display_order": 4,
        },
        {
            "name": "transmission",
            "description": "Transmission constraints, congestion, and flow data",
            "display_order": 5,
        },
        {
            "name": "weather",
            "description": "Weather data and forecasts for market regions",
            "display_order": 6,
        },
        {
            "name": "renewable",
            "description": "Wind, solar, and other renewable generation forecasts",
            "display_order": 7,
        },
        {
            "name": "storage",
            "description": "Battery and other energy storage operations",
            "display_order": 8,
        },
        {
            "name": "emissions",
            "description": "Carbon emissions and environmental data",
            "display_order": 9,
        },
    ]
    
    with get_db() as db:
        for cat_data in categories:
            # Check if category already exists
            existing = db.query(DatasetCategory).filter(
                DatasetCategory.name == cat_data["name"]
            ).first()
            if not existing:
                category = DatasetCategory(**cat_data)
                db.add(category)
        
        db.commit()
        print(f"✓ Seeded {len(categories)} dataset categories")


def seed_all():
    """Seed all initial data."""
    print("Seeding database...")
    seed_isos()
    seed_dataset_categories()
    print("✓ Database seeding complete")


if __name__ == "__main__":
    seed_all()