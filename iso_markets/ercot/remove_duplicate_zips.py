#!/usr/bin/env python3
"""
Remove duplicate zip files with patterns like "Filename (1).zip", "Filename (2).zip", etc.
where "Filename.zip" exists and has the same size.

Uses multiprocessing to handle large numbers of files efficiently.
"""

import os
import re
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Set
import time
import multiprocessing
from itertools import chain

def find_zip_files_in_directory(directory: Path) -> List[Path]:
    """Find all zip files in a single directory (non-recursive)."""
    try:
        return [f for f in directory.iterdir() if f.is_file() and f.suffix.lower() == '.zip']
    except PermissionError:
        print(f"Permission denied: {directory}")
        return []

def get_all_directories(root_path: Path) -> List[Path]:
    """Get all directories recursively from root path."""
    directories = [root_path]
    for dirpath, dirnames, _ in os.walk(root_path):
        for dirname in dirnames:
            directories.append(Path(dirpath) / dirname)
    return directories

def chunk_directories(directories: List[Path], num_chunks: int) -> List[List[Path]]:
    """Split directories into roughly equal chunks for better load balancing."""
    chunk_size = max(1, len(directories) // num_chunks)
    chunks = []
    for i in range(0, len(directories), chunk_size):
        chunks.append(directories[i:i + chunk_size])
    return chunks

def process_directory_batch(directories: List[Path]) -> Tuple[List[Path], int]:
    """Process a batch of directories."""
    all_files_to_delete = []
    total_bytes = 0
    
    for directory in directories:
        files, bytes_freed = process_directory(directory)
        all_files_to_delete.extend(files)
        total_bytes += bytes_freed
    
    return all_files_to_delete, total_bytes

def process_directory(directory: Path) -> Tuple[List[Path], int]:
    """
    Process a single directory to find and identify duplicate zip files.
    Returns tuple of (files_to_delete, bytes_to_free)
    """
    zip_files = find_zip_files_in_directory(directory)
    if not zip_files:
        return [], 0
    
    # Group files by their base name
    file_groups = {}
    duplicate_pattern = re.compile(r'^(.+?)\s*\(\d+\)\.zip$', re.IGNORECASE)
    
    for zip_file in zip_files:
        filename = zip_file.name
        
        # Check if this is a duplicate pattern
        match = duplicate_pattern.match(filename)
        if match:
            base_name = match.group(1)
            key = base_name.lower()
        else:
            # This could be an original file
            if filename.lower().endswith('.zip'):
                base_name = filename[:-4]  # Remove .zip extension
                key = base_name.lower()
            else:
                continue
        
        if key not in file_groups:
            file_groups[key] = []
        file_groups[key].append(zip_file)
    
    files_to_delete = []
    bytes_to_free = 0
    
    # For each group, check if original exists and duplicates have same size
    for base_key, files in file_groups.items():
        # Find the original file (without parentheses number)
        original = None
        duplicates = []
        
        for f in files:
            if duplicate_pattern.match(f.name):
                duplicates.append(f)
            else:
                # This should be the original
                original = f
        
        if original and duplicates:
            try:
                original_size = original.stat().st_size
                
                # Check each duplicate
                for dup in duplicates:
                    try:
                        dup_size = dup.stat().st_size
                        if dup_size == original_size:
                            files_to_delete.append(dup)
                            bytes_to_free += dup_size
                    except OSError as e:
                        print(f"Error accessing {dup}: {e}")
            except OSError as e:
                print(f"Error accessing original {original}: {e}")
    
    return files_to_delete, bytes_to_free

def delete_files(files: List[Path], dry_run: bool = True) -> int:
    """Delete the specified files. Returns count of deleted files."""
    deleted_count = 0
    for file in files:
        try:
            if dry_run:
                print(f"[DRY RUN] Would delete: {file}")
            else:
                file.unlink()
                print(f"Deleted: {file}")
            deleted_count += 1
        except OSError as e:
            print(f"Error deleting {file}: {e}")
    return deleted_count

def main():
    parser = argparse.ArgumentParser(description='Remove duplicate zip files')
    parser.add_argument('path', nargs='?', default='.',
                      help='Root directory to scan (default: current directory)')
    parser.add_argument('--dry-run', action='store_true',
                      help='Show what would be deleted without actually deleting')
    parser.add_argument('--workers', type=int, default=None,
                      help='Number of worker processes (default: number of CPU cores)')
    args = parser.parse_args()
    
    root_path = Path(args.path).resolve()
    
    if not root_path.exists():
        print(f"Error: Path {root_path} does not exist")
        return 1
    
    # Determine number of workers
    num_workers = args.workers if args.workers else multiprocessing.cpu_count()
    
    print(f"Scanning for duplicate zip files in: {root_path}")
    print(f"Using {num_workers} worker processes (CPU cores: {multiprocessing.cpu_count()})")
    if args.dry_run:
        print("DRY RUN MODE - No files will be deleted")
    print("-" * 60)
    
    start_time = time.time()
    
    # Get all directories to process
    print("Collecting directories...")
    directories = get_all_directories(root_path)
    print(f"Found {len(directories)} directories to process")
    
    # Process directories in parallel with better load balancing
    all_files_to_delete = []
    total_bytes_to_free = 0
    
    if len(directories) <= num_workers:
        # If we have fewer directories than workers, process individually
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_directory, d): d for d in directories}
            
            for i, future in enumerate(as_completed(futures), 1):
                if i % 100 == 0:
                    print(f"Processed {i}/{len(directories)} directories...")
                
                try:
                    files_to_delete, bytes_to_free = future.result()
                    if files_to_delete:
                        all_files_to_delete.extend(files_to_delete)
                        total_bytes_to_free += bytes_to_free
                except Exception as e:
                    print(f"Error processing directory: {e}")
    else:
        # Chunk directories for better load balancing
        dir_chunks = chunk_directories(directories, num_workers * 4)  # Create more chunks than workers
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_directory_batch, chunk) for chunk in dir_chunks]
            
            completed_chunks = 0
            for future in as_completed(futures):
                completed_chunks += 1
                print(f"Completed chunk {completed_chunks}/{len(dir_chunks)}...")
                
                try:
                    files_to_delete, bytes_to_free = future.result()
                    if files_to_delete:
                        all_files_to_delete.extend(files_to_delete)
                        total_bytes_to_free += bytes_to_free
                except Exception as e:
                    print(f"Error processing chunk: {e}")
    
    print(f"\nScanning complete in {time.time() - start_time:.2f} seconds")
    print("-" * 60)
    
    if not all_files_to_delete:
        print("No duplicate zip files found")
        return 0
    
    # Sort files for consistent output
    all_files_to_delete.sort()
    
    print(f"Found {len(all_files_to_delete)} duplicate files")
    print(f"Total space to free: {total_bytes_to_free / (1024**3):.2f} GB")
    print("-" * 60)
    
    # Delete files
    if not args.dry_run:
        response = input(f"Delete {len(all_files_to_delete)} files? [y/N]: ")
        if response.lower() != 'y':
            print("Deletion cancelled")
            return 0
    
    deleted_count = delete_files(all_files_to_delete, args.dry_run)
    
    print("-" * 60)
    if args.dry_run:
        print(f"Dry run complete. {len(all_files_to_delete)} files would be deleted")
        print(f"Space that would be freed: {total_bytes_to_free / (1024**3):.2f} GB")
    else:
        print(f"Deleted {deleted_count} files")
        print(f"Freed approximately {total_bytes_to_free / (1024**3):.2f} GB")
    
    print(f"Total time: {time.time() - start_time:.2f} seconds")
    
    return 0

if __name__ == '__main__':
    exit(main())