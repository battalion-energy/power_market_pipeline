#!/usr/bin/env python3
"""
Copy top-level zip files and .sh files from ERCOT data directories to external drive.
Only copies zip files in the root of each directory, not from subdirectories.
"""

import os
import shutil
from pathlib import Path
from typing import List, Tuple

def copy_top_level_files(source_dirs: List[str], dest_base: str) -> Tuple[int, int]:
    """
    Copy only top-level .zip and .sh files from source directories to destination.
    
    Returns:
        Tuple of (files_copied, files_skipped)
    """
    files_copied = 0
    files_skipped = 0
    
    # Ensure destination base exists
    dest_base_path = Path(dest_base)
    dest_base_path.mkdir(parents=True, exist_ok=True)
    
    for source_dir in source_dirs:
        source_path = Path(source_dir)
        
        if not source_path.exists():
            print(f"⚠️  Source directory does not exist: {source_dir}")
            continue
            
        # Create corresponding destination directory
        dest_dir = dest_base_path / source_path.name
        dest_dir.mkdir(exist_ok=True)
        print(f"\n📁 Processing: {source_path.name}")
        print(f"   From: {source_dir}")
        print(f"   To:   {dest_dir}")
        
        # List only files in the top level (not in subdirectories)
        try:
            for item in source_path.iterdir():
                # Skip directories
                if item.is_dir():
                    continue
                    
                # Only process .zip and .sh files
                if item.suffix.lower() in ['.zip', '.sh']:
                    dest_file = dest_dir / item.name
                    
                    # Check if file already exists at destination
                    if dest_file.exists():
                        # Compare sizes to see if it's the same file
                        source_size = item.stat().st_size
                        dest_size = dest_file.stat().st_size
                        if source_size == dest_size:
                            print(f"   ⏭️  Skipping (already exists): {item.name}")
                            files_skipped += 1
                            continue
                    
                    # Copy the file with metadata
                    try:
                        print(f"   📋 Copying: {item.name} ({item.stat().st_size / (1024*1024):.1f} MB)")
                        # copy2 preserves metadata, but we'll also explicitly set times
                        shutil.copy2(item, dest_file)
                        
                        # Explicitly preserve timestamps (creation and modification)
                        stat = item.stat()
                        os.utime(dest_file, (stat.st_atime, stat.st_mtime))
                        
                        # On macOS, also try to preserve creation time
                        if hasattr(stat, 'st_birthtime'):
                            try:
                                # Use SetFile command on macOS to set creation date
                                from datetime import datetime
                                creation_time = datetime.fromtimestamp(stat.st_birthtime)
                                creation_str = creation_time.strftime("%m/%d/%Y %H:%M:%S")
                                os.system(f'SetFile -d "{creation_str}" "{dest_file}" 2>/dev/null')
                            except:
                                pass  # Silently ignore if SetFile is not available
                        
                        files_copied += 1
                    except Exception as e:
                        print(f"   ❌ Error copying {item.name}: {e}")
                        
        except PermissionError as e:
            print(f"   ❌ Permission error accessing {source_dir}: {e}")
            
    return files_copied, files_skipped


def main():
    # Source directories
    source_dirs = [
        "/Users/enrico/data/ERCOT_data/DAM_Settlement_Point_Prices",
        "/Users/enrico/data/ERCOT_data/60-Day_COP_All_Updates",
        "/Users/enrico/data/ERCOT_data/60-Day_SASM_Disclosure_Reports",
        "/Users/enrico/data/ERCOT_data/DAM_Clearing_Prices_for_Capacity",
        "/Users/enrico/data/ERCOT_data/Settlement_Point_Prices_at_Resource_Nodes,_Hubs_and_Load_Zones"
    ]

    source_dirs = ["/Users/enrico/data/ERCOT_data/60-Day_COP_Adjustment_Period_Snapshot",
                   "/Users/enrico/data/ERCOT_data/60-Day_COP_All_Updates",
                   "/Users/enrico/data/ERCOT_data/60-Day_DAM_Disclosure_Reports"]
    
    # Destination base directory
    dest_base = "/Volumes/SamsungX52T/ERCOT_data"
    
    print("🚀 ERCOT Data Copy Utility")
    print("=" * 60)
    print(f"Destination: {dest_base}")
    print("=" * 60)
    
    # Check if destination volume is mounted
    if not Path("/Volumes/SamsungX52T").exists():
        print("❌ Error: External drive /Volumes/SamsungX52T is not mounted!")
        print("Please connect your Samsung external drive and try again.")
        return 1
    
    # Perform the copy
    files_copied, files_skipped = copy_top_level_files(source_dirs, dest_base)
    
    print("\n" + "=" * 60)
    print("✅ Copy operation completed!")
    print(f"   Files copied: {files_copied}")
    print(f"   Files skipped (already exist): {files_skipped}")
    print(f"   Total processed: {files_copied + files_skipped}")
    
    return 0


if __name__ == "__main__":
    exit(main())
