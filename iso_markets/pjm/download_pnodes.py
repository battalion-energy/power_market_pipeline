#!/usr/bin/env python3
"""
Download complete list of PJM Pricing Nodes (pnodes)

This script downloads the master list of all pnodes in the PJM system,
including hubs, zones, generators, loads, aggregates, etc.

Usage:
    python download_pnodes.py
"""

import os
import sys
import logging
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

# Add parent directory to path to import pjm_api_client
sys.path.insert(0, str(Path(__file__).parent))
from pjm_api_client import PJMAPIClient

# Load environment variables
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_pnodes(client: PJMAPIClient, output_path: Path):
    """
    Download all pnodes from PJM.

    Args:
        client: PJMAPIClient instance
        output_path: Path to save CSV file
    """
    logger.info("Downloading all PJM pnodes...")

    try:
        # Get all pnodes (no filters)
        data = client.get_pnodes()

        if not data:
            logger.warning("No data returned")
            return

        # Check if data is in expected format
        if isinstance(data, dict):
            if 'items' in data:
                items = data['items']
                total_rows = data.get('totalRows', len(items))
            elif 'data' in data:
                items = data['data']
                total_rows = len(items)
            else:
                items = [data]
                total_rows = 1
        else:
            items = data
            total_rows = len(items)

        if not items:
            logger.warning("No items in response")
            return

        # Convert to DataFrame
        df = pd.DataFrame(items)

        if df.empty:
            logger.warning("Empty dataframe")
            return

        # Save to CSV
        df.to_csv(output_path, index=False)

        logger.info(f"✓ Downloaded {len(df)} pnodes (total rows: {total_rows})")
        logger.info(f"✓ Saved to {output_path}")

        # Display summary statistics
        if 'pnode_type' in df.columns:
            logger.info("\nPnode Types:")
            for ptype, count in df['pnode_type'].value_counts().items():
                logger.info(f"  {ptype}: {count}")

        if 'pnode_subtype' in df.columns:
            logger.info("\nPnode Subtypes:")
            for subtype, count in df['pnode_subtype'].value_counts().head(10).items():
                logger.info(f"  {subtype}: {count}")

        if 'zone' in df.columns:
            logger.info("\nTop Zones:")
            for zone, count in df['zone'].value_counts().head(10).items():
                logger.info(f"  {zone}: {count}")

    except Exception as e:
        logger.error(f"Error downloading pnodes: {e}")
        raise


def main():
    # Determine data directory
    data_dir = Path(os.getenv('PJM_DATA_DIR', '/home/enrico/data/PJM_data'))
    reference_dir = data_dir / 'reference'
    reference_dir.mkdir(parents=True, exist_ok=True)

    output_path = reference_dir / 'pnodes.csv'

    logger.info(f"Output file: {output_path}")

    # Initialize API client
    try:
        client = PJMAPIClient(requests_per_minute=6)
    except ValueError as e:
        logger.error(f"Failed to initialize API client: {e}")
        logger.error("Please set PJM_API_KEY in your .env file")
        return

    # Download pnodes
    download_pnodes(client, output_path)

    logger.info("\n✓ Download complete!")


if __name__ == "__main__":
    main()
