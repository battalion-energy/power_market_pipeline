#!/usr/bin/env python3
"""
Inspect 60-day ERCOT data files to understand structure before rollup.
"""

from pathlib import Path
import re
from collections import defaultdict

BASE_DIR = Path("/Users/enrico/data/ERCOT_data")

def inspect_dam_disclosures():
    """Inspect DAM disclosure files."""
    print("\n60-DAY DAM DISCLOSURE FILES")
    print("=" * 60)
    
    dam_dir = BASE_DIR / "60-Day_DAM_Disclosure_Reports" / "csv"
    if not dam_dir.exists():
        dam_dir = BASE_DIR / "60-Day_DAM_Disclosure_Reports"
    
    if not dam_dir.exists():
        print(f"Directory not found: {dam_dir}")
        return
    
    # Count files by type
    file_types = defaultdict(list)
    
    for file_path in sorted(dam_dir.glob("*.csv"))[:100]:  # Limit to first 100
        # Extract file type from name
        if file_path.name.startswith("60d_DAM_"):
            # Remove date part to get file type
            match = re.match(r'(60d_DAM_[^-]+)', file_path.name)
            if match:
                file_type = match.group(1)
                file_types[file_type].append(file_path.name)
    
    for file_type, files in sorted(file_types.items()):
        print(f"\n{file_type}: {len(files)} files")
        # Show first 3 examples
        for example in files[:3]:
            print(f"  - {example}")

def inspect_cop_updates():
    """Inspect COP All Updates files."""
    print("\n60-DAY COP ALL UPDATES FILES")
    print("=" * 60)
    
    cop_dir = BASE_DIR / "60-Day_COP_All_Updates" / "csv"
    if not cop_dir.exists():
        cop_dir = BASE_DIR / "60-Day_COP_All_Updates"
    
    if not cop_dir.exists():
        print(f"Directory not found: {cop_dir}")
        return
    
    cop_files = list(cop_dir.glob("*.csv"))[:20]  # First 20 files
    print(f"Found {len(list(cop_dir.glob('*.csv')))} total COP files")
    print("\nFirst few files:")
    for f in cop_files[:5]:
        print(f"  - {f.name}")

def main():
    """Main inspection."""
    print("INSPECTING 60-DAY ERCOT DATA FILES")
    print("=" * 60)
    
    inspect_dam_disclosures()
    inspect_cop_updates()
    
    print("\n" + "=" * 60)
    print("Inspection complete!")

if __name__ == "__main__":
    main()