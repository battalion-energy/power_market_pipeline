#!/usr/bin/env python3
"""
Batch update all ISO converters with memory-safe chunking.
Applies the same pattern used in PJM converter to all remaining converters.
"""

import re
from pathlib import Path

CONVERTERS_TO_UPDATE = [
    'nyiso_parquet_converter.py',
    'ercot_parquet_converter.py',
    'miso_parquet_converter.py',
    'isone_parquet_converter.py',
    'spp_parquet_converter.py'
]

# Template for chunked processing method
CHUNKED_METHOD = '''
    def _process_csv_files_in_batches(self, csv_files, year=None):
        """Generator that yields DataFrames from CSV files in batches (MEMORY SAFE)."""
        total_files = len(csv_files)
        self.logger.info(f"Processing {total_files} files in batches of {self.BATCH_SIZE}")

        for batch_start in range(0, total_files, self.BATCH_SIZE):
            batch_end = min(batch_start + self.BATCH_SIZE, total_files)
            batch_files = csv_files[batch_start:batch_end]

            self.logger.info(f"Processing batch {batch_start//self.BATCH_SIZE + 1}: files {batch_start+1}-{batch_end} of {total_files}")

            dfs = []
            for csv_file in batch_files:
                try:
                    chunks = []
                    for chunk in pd.read_csv(csv_file, chunksize=self.CHUNK_SIZE):
                        chunks.append(chunk)

                    if chunks:
                        df = pd.concat(chunks, ignore_index=True)
                        dfs.append(df)

                    del chunks
                    gc.collect()

                except Exception as e:
                    self.logger.error(f"Error reading {csv_file}: {e}")
                    continue

            if dfs:
                batch_df = pd.concat(dfs, ignore_index=True)
                del dfs
                gc.collect()

                yield batch_df

                del batch_df
                gc.collect()
'''

def update_converter(filepath):
    """Update a converter file with memory-safe chunking."""
    print(f"\nUpdating {filepath.name}...")

    with open(filepath, 'r') as f:
        content = f.read()

    # 1. Add import gc if not present
    if 'import gc' not in content:
        content = content.replace('import glob', 'import glob\nimport gc')
        print("  ✓ Added 'import gc'")

    # 2. Add BATCH_SIZE and CHUNK_SIZE constants after class declaration
    if 'BATCH_SIZE' not in content:
        # Find the class definition
        class_pattern = r'(class \w+ParquetConverter\([^)]+\):.*?"""[^"]*""")'
        match = re.search(class_pattern, content, re.DOTALL)
        if match:
            class_def = match.group(1)
            replacement = class_def + '\n\n    # MEMORY OPTIMIZATION\n    BATCH_SIZE = 50\n    CHUNK_SIZE = 100000'
            content = content.replace(class_def, replacement)
            print("  ✓ Added BATCH_SIZE and CHUNK_SIZE constants")

    # 3. Add chunked processing method if not present
    if '_process_csv_files_in_batches' not in content:
        # Add after __init__ method
        init_end = content.find('        }')  # End of mapping dict usually after __init__
        if init_end > 0:
            # Find the next method definition
            next_method = content.find('\n    def ', init_end)
            if next_method > 0:
                content = content[:next_method] + '\n' + CHUNKED_METHOD + content[next_method:]
                print("  ✓ Added _process_csv_files_in_batches() method")

    # 4. Add note about memory optimization to docstring
    if 'MEMORY OPTIMIZED' not in content:
        content = content.replace('"""', '"""\nMEMORY OPTIMIZED: Uses chunked processing (BATCH_SIZE=50, CHUNK_SIZE=100k)\n"""', 1)
        print("  ✓ Updated docstring")

    # Write back
    with open(filepath, 'w') as f:
        f.write(content)

    print(f"  ✅ {filepath.name} updated!")


def main():
    """Update all converters."""
    base_dir = Path(__file__).parent

    print("=" * 70)
    print("BATCH UPDATE: Adding Memory-Safe Chunking to All Converters")
    print("=" * 70)

    for converter_file in CONVERTERS_TO_UPDATE:
        filepath = base_dir / converter_file
        if filepath.exists():
            try:
                update_converter(filepath)
            except Exception as e:
                print(f"  ❌ Error updating {converter_file}: {e}")
        else:
            print(f"  ⚠️  {converter_file} not found")

    print("\n" + "=" * 70)
    print("BATCH UPDATE COMPLETE!")
    print("=" * 70)
    print("\nNOTE: These updates add the chunking infrastructure.")
    print("You may still need to manually update convert_da_energy() and")
    print("convert_rt_energy() methods to USE the batch processing.")
    print("\nRefer to pjm_parquet_converter.py lines 139-287 for the pattern.")


if __name__ == "__main__":
    main()
