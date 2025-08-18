#!/usr/bin/env python3
"""Test script to verify ERCOT_DATA_DIR environment variable configuration."""

import os
import sys
from pathlib import Path

# Try to load dotenv if available
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✓ dotenv loaded successfully")
except ImportError:
    print("⚠ dotenv not available, reading from system environment only")

def get_ercot_data_dir():
    """Get ERCOT data directory from environment or platform-specific default."""
    data_dir = os.getenv("ERCOT_DATA_DIR")
    if data_dir:
        return Path(data_dir)
    # Default based on platform
    if sys.platform == "linux":
        return Path("/home/enrico/data/ERCOT_data")
    else:
        return Path("/Users/enrico/data/ERCOT_data")

if __name__ == "__main__":
    print("=" * 60)
    print("ERCOT Data Directory Configuration Test")
    print("=" * 60)
    
    # Check environment variable
    env_value = os.getenv("ERCOT_DATA_DIR")
    print(f"ERCOT_DATA_DIR from environment: {env_value}")
    
    # Get resolved path
    data_dir = get_ercot_data_dir()
    print(f"Resolved data directory: {data_dir}")
    
    # Check if directory exists
    if data_dir.exists():
        print(f"✓ Directory exists: {data_dir}")
    else:
        print(f"✗ Directory does not exist: {data_dir}")
    
    # Check platform
    print(f"Platform: {sys.platform}")
    
    print("=" * 60)