#!/usr/bin/env python3
"""
Master script to run all ISO parquet converters in parallel.

This script launches converter processes for each ISO that has data available.
"""

import os
import sys
import argparse
import subprocess
import logging
from pathlib import Path
from datetime import datetime
import time

# ISO configuration
ISO_CONFIGS = {
    'PJM': {
        'script': 'pjm_parquet_converter.py',
        'csv_dir': '/pool/ssd8tb/data/iso/PJM_data/csv_files',
        'has_data': True,
        'description': 'PJM Interconnection - DA/RT hourly & 5min nodal, AS'
    },
    'CAISO': {
        'script': 'caiso_parquet_converter.py',
        'csv_dir': '/pool/ssd8tb/data/iso/CAISO_data/csv_files',
        'has_data': True,
        'description': 'CAISO - DA hourly & RT 15min nodal, AS'
    },
    'ERCOT': {
        'script': 'ercot_parquet_converter.py',
        'csv_dir': '/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data',
        'has_data': True,
        'description': 'ERCOT - DA hourly & RT 15min (unified format, separate from legacy)'
    },
    'MISO': {
        'script': 'miso_parquet_converter.py',
        'csv_dir': '/pool/ssd8tb/data/iso/MISO/csv_files',
        'has_data': True,  # 1,298 DA files available
        'description': 'MISO - DA ex-post/ex-ante, RT hourly & 5min, AS'
    },
    'NYISO': {
        'script': 'nyiso_parquet_converter.py',
        'csv_dir': '/pool/ssd8tb/data/iso/NYISO_data/csv_files',
        'has_data': True,
        'description': 'NYISO - DA & RT zonal (11 zones), AS'
    },
    'ISONE': {
        'script': 'isone_parquet_converter.py',
        'csv_dir': '/pool/ssd8tb/data/iso/ISONE_data/csv_files',
        'has_data': True,  # 2,476 DA files available
        'description': 'ISO-NE - DA & RT hourly nodal, AS'
    },
    'SPP': {
        'script': 'spp_parquet_converter.py',
        'csv_dir': '/pool/ssd8tb/data/iso/SPP/csv_files',
        'has_data': True,  # 651 DA files available
        'description': 'SPP - DA & RT hourly settlement locations, AS'
    }
}


def setup_logger():
    """Setup logger."""
    logger = logging.getLogger('ISO_Converter_Master')
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def run_converter(iso_name, config, year=None, da_only=False, logger=None):
    """
    Run converter for a specific ISO.

    Args:
        iso_name: ISO name
        config: ISO configuration dict
        year: Optional year to process
        da_only: Only convert DA prices
        logger: Logger instance

    Returns:
        subprocess.Popen object or tuple (None, log_file) if skipped
    """
    if logger is None:
        logger = logging.getLogger()

    script_path = Path(__file__).parent / config['script']

    if not script_path.exists():
        logger.error(f"Script not found: {script_path}")
        return None, None

    if not config['has_data']:
        logger.warning(f"Skipping {iso_name} - no data available")
        return None, None

    # Check if CSV directory exists
    csv_dir = Path(config['csv_dir'])
    if not csv_dir.exists():
        logger.warning(f"Skipping {iso_name} - CSV directory not found: {csv_dir}")
        return None, None

    # Build command
    cmd = ['python3', str(script_path)]

    if year:
        cmd.extend(['--year', str(year)])
    else:
        cmd.append('--all')

    if da_only:
        cmd.append('--da-only')

    # Create log file
    log_dir = Path('/pool/ssd8tb/data/iso/unified_iso_data/logs')
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f"{iso_name.lower()}_conversion_{timestamp}.log"

    logger.info(f"{'='*80}")
    logger.info(f"Starting {iso_name} converter")
    logger.info(f"Command: {' '.join(cmd)}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"{'='*80}")

    # Run process
    with open(log_file, 'w') as log_f:
        process = subprocess.Popen(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            text=True
        )

    return process, log_file


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run ISO parquet converters (SEQUENTIAL by default)')
    parser.add_argument('--year', type=int, help='Specific year to process')
    parser.add_argument('--isos', nargs='+', help='Specific ISOs to process (e.g., PJM CAISO)')
    parser.add_argument('--sequential', action='store_true', default=True, help='Run sequentially (DEFAULT)')
    parser.add_argument('--parallel', action='store_true', help='Run in parallel (max 2)')
    parser.add_argument('--max-parallel', type=int, default=1, help='Max parallel processes (default: 1)')
    parser.add_argument('--da-only', action='store_true', help='Only convert DA prices')

    args = parser.parse_args()

    logger = setup_logger()

    # Override sequential if parallel flag is set
    if args.parallel:
        args.sequential = False
        args.max_parallel = min(args.max_parallel, 2)  # Never exceed 2

    # MEMORY SAFETY: Default to sequential, limit to 2 parallel max
    max_parallel = 1 if args.sequential else min(args.max_parallel, 2)

    # Determine which ISOs to process
    if args.isos:
        isos_to_process = {iso: config for iso, config in ISO_CONFIGS.items() if iso in args.isos}
    else:
        isos_to_process = {iso: config for iso, config in ISO_CONFIGS.items() if config['has_data']}

    logger.info(f"Processing ISOs: {', '.join(isos_to_process.keys())}")
    logger.info(f"Max parallel processes: {max_parallel if not args.sequential else 1}")

    # Run converters with limited parallelism
    processes = {}
    pending_isos = list(isos_to_process.items())

    completed_isos = []
    failed_isos = []

    while pending_isos or processes:
        # Start new processes up to max_parallel
        while len(processes) < max_parallel and pending_isos and not args.sequential:
            iso_name, config = pending_isos.pop(0)
            process, log_file = run_converter(iso_name, config, args.year, args.da_only, logger)

            if process:
                processes[iso_name] = {'process': process, 'log_file': log_file}
                logger.info(f"Started {iso_name} ({len(processes)}/{max_parallel} running)")

        # Sequential mode
        if args.sequential and pending_isos:
            iso_name, config = pending_isos.pop(0)
            process, log_file = run_converter(iso_name, config, args.year, args.da_only, logger)

            if process:
                start_time = time.time()
                logger.info(f"Waiting for {iso_name} to complete...")
                process.wait()
                elapsed = time.time() - start_time
                logger.info(f"{iso_name} completed in {elapsed/60:.1f} minutes with return code {process.returncode}")

                if process.returncode == 0:
                    completed_isos.append(iso_name)
                else:
                    failed_isos.append(iso_name)
                    logger.error(f"{iso_name} FAILED! Check log: {log_file}")

        # Check for completed processes (parallel mode)
        completed = []
        for iso_name, info in processes.items():
            retcode = info['process'].poll()
            if retcode is not None:
                if retcode == 0:
                    logger.info(f"{iso_name} completed successfully")
                    completed_isos.append(iso_name)
                else:
                    logger.error(f"{iso_name} FAILED with return code {retcode}")
                    failed_isos.append(iso_name)
                completed.append(iso_name)

        for iso_name in completed:
            del processes[iso_name]

        if processes or pending_isos:
            time.sleep(5)  # Check every 5 seconds

    logger.info("="*80)
    logger.info("ALL CONVERSIONS COMPLETE!")
    logger.info("="*80)

    # Print summary
    logger.info(f"\n📊 CONVERSION SUMMARY:")
    logger.info(f"  ✅ Completed: {len(completed_isos)}")
    logger.info(f"  ❌ Failed: {len(failed_isos)}")

    if completed_isos:
        logger.info(f"\n✅ Successful conversions:")
        for iso in completed_isos:
            logger.info(f"  - {iso}")

    if failed_isos:
        logger.info(f"\n❌ Failed conversions:")
        for iso in failed_isos:
            logger.info(f"  - {iso}")

    log_dir = Path('/pool/ssd8tb/data/iso/unified_iso_data/logs')
    logger.info("\n📝 Log files:")
    for log_file in sorted(log_dir.glob("*_conversion_*.log"), reverse=True)[:len(isos_to_process)]:
        size_mb = log_file.stat().st_size / 1024 / 1024
        logger.info(f"  {log_file.name} ({size_mb:.1f} MB)")

    logger.info("\n📦 Parquet files created:")
    parquet_dir = Path('/pool/ssd8tb/data/iso/unified_iso_data/parquet')
    total_size = 0
    for iso_dir in sorted(parquet_dir.glob("*")):
        if iso_dir.is_dir():
            parquet_files = list(iso_dir.rglob("*.parquet"))
            iso_size = sum(f.stat().st_size for f in parquet_files) / 1024 / 1024
            total_size += iso_size
            if parquet_files:
                logger.info(f"  {iso_dir.name.upper()}: {len(parquet_files)} files ({iso_size:.1f} MB)")

    logger.info(f"\n💾 Total parquet size: {total_size:.1f} MB ({total_size/1024:.2f} GB)")
    logger.info(f"\n✨ Run 'python3 check_conversion_status.py' for detailed analysis")


if __name__ == "__main__":
    main()
