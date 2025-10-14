#!/usr/bin/env python3
"""
ISO-NE Download Orchestration Script

Automatically chains ISO-NE downloads sequentially, waiting for each to complete
before starting the next. Handles failures gracefully and provides detailed logging.

Usage:
    python orchestrate_downloads.py --downloads 2025 2024 2019-2023
    python orchestrate_downloads.py --downloads 2024 --data-types lmp as
    python orchestrate_downloads.py --config downloads.json
"""

import argparse
import asyncio
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class DownloadOrchestrator:
    """Orchestrates sequential ISO-NE data downloads."""

    def __init__(
        self,
        log_dir: Path = Path("/tmp"),
        max_retries: int = 2,
        retry_delay: int = 300,  # 5 minutes
    ):
        self.log_dir = Path(log_dir)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.results = []

    async def run_download(
        self,
        script: str,
        name: str,
        args: List[str],
        log_file: Optional[Path] = None,
    ) -> Tuple[bool, int, str]:
        """
        Run a single download script and wait for completion.

        Returns:
            (success, exit_code, log_path)
        """
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = self.log_dir / f"{name}_{timestamp}.log"

        cmd = ["uv", "run", "python", script] + args

        print(f"\n{'='*80}")
        print(f"Starting: {name}")
        print(f"Command: {' '.join(cmd)}")
        print(f"Log file: {log_file}")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")

        # Run with tee to both display and log
        process = await asyncio.create_subprocess_shell(
            f"{' '.join(cmd)} 2>&1 | tee {log_file}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await process.communicate()
            exit_code = process.returncode

            success = exit_code == 0

            print(f"\n{'='*80}")
            print(f"Completed: {name}")
            print(f"Exit code: {exit_code}")
            print(f"Status: {'✓ SUCCESS' if success else '✗ FAILED'}")
            print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*80}\n")

            return success, exit_code, str(log_file)

        except Exception as e:
            print(f"\n✗ ERROR: {name} failed with exception: {e}")
            return False, -1, str(log_file)

    async def run_download_with_retries(
        self,
        script: str,
        name: str,
        args: List[str],
    ) -> Dict:
        """Run a download with automatic retries on failure."""
        result = {
            "name": name,
            "script": script,
            "args": args,
            "attempts": 0,
            "success": False,
            "exit_code": -1,
            "log_files": [],
        }

        for attempt in range(1, self.max_retries + 1):
            result["attempts"] = attempt

            if attempt > 1:
                print(f"\n⏳ Retry #{attempt-1} for {name} after {self.retry_delay}s delay...")
                await asyncio.sleep(self.retry_delay)

            success, exit_code, log_file = await self.run_download(
                script, f"{name}_attempt{attempt}", args
            )

            result["log_files"].append(log_file)
            result["exit_code"] = exit_code

            if success:
                result["success"] = True
                break

            if attempt < self.max_retries:
                print(f"⚠ Download failed (attempt {attempt}/{self.max_retries})")

        return result

    async def run_sequential_downloads(
        self,
        downloads: List[Dict[str, any]],
    ) -> List[Dict]:
        """Run downloads sequentially, one after another."""
        results = []

        print(f"\n{'#'*80}")
        print(f"# ISO-NE Download Orchestration")
        print(f"# Total downloads: {len(downloads)}")
        print(f"# Max retries per download: {self.max_retries}")
        print(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'#'*80}\n")

        for i, download in enumerate(downloads, 1):
            print(f"\n📊 Download {i}/{len(downloads)}: {download['name']}")

            result = await self.run_download_with_retries(
                script=download["script"],
                name=download["name"],
                args=download["args"],
            )

            results.append(result)

            # Stop on failure if configured
            if not result["success"] and download.get("stop_on_failure", False):
                print(f"\n🛑 Stopping orchestration due to failure: {download['name']}")
                break

        # Print summary
        self._print_summary(results)

        return results

    def _print_summary(self, results: List[Dict]):
        """Print a summary of all downloads."""
        successful = sum(1 for r in results if r["success"])
        failed = len(results) - successful

        print(f"\n{'#'*80}")
        print(f"# Download Summary")
        print(f"{'#'*80}")
        print(f"Total downloads: {len(results)}")
        print(f"✓ Successful: {successful}")
        print(f"✗ Failed: {failed}")
        print(f"\nDetailed results:")
        print(f"{'─'*80}")

        for i, result in enumerate(results, 1):
            status = "✓ SUCCESS" if result["success"] else "✗ FAILED"
            print(f"{i}. {result['name']}: {status} (attempts: {result['attempts']})")
            if result["log_files"]:
                print(f"   Log: {result['log_files'][-1]}")

        print(f"{'#'*80}\n")


def create_download_config(year_ranges: List[str], data_types: List[str]) -> List[Dict]:
    """
    Create download configuration from year ranges and data types.

    Args:
        year_ranges: List like ["2025", "2024", "2019-2023"]
        data_types: List like ["lmp", "as"] or ["all"]
    """
    downloads = []
    base_path = Path("iso_markets/isone")

    # Define download configurations
    configs = {
        "lmp": {
            "script": str(base_path / "download_lmp.py"),
            "args_template": [
                "--market-types", "da", "rt",
                "--hubs-only",
                "--max-concurrent", "1",
                "--reverse",
            ],
        },
        "as": {
            "script": str(base_path / "download_ancillary_services.py"),
            "args_template": [
                "--data-types", "freq_reg", "reserves",
                "--reserve-zones", "7000", "7001", "7002", "7003", "7004", "7005",
                "7006", "7007", "7008", "7009", "7010", "7011",
                "--max-concurrent", "1",
                "--reverse",
            ],
        },
    }

    # Parse year ranges
    for year_range in year_ranges:
        if "-" in year_range:
            # Range like "2019-2023"
            start, end = year_range.split("-")
            start_date = f"{start}-01-01"
            end_date = f"{end}-12-31"
        else:
            # Single year like "2025"
            start_date = f"{year_range}-01-01"
            end_date = f"{year_range}-10-12"  # Current date

        # Create downloads for each data type
        for data_type in data_types:
            if data_type not in configs:
                continue

            config = configs[data_type]
            args = config["args_template"].copy()
            args.extend(["--start-date", start_date, "--end-date", end_date])

            downloads.append({
                "name": f"isone_{data_type}_{year_range}",
                "script": config["script"],
                "args": args,
                "stop_on_failure": False,
            })

    return downloads


async def main():
    parser = argparse.ArgumentParser(
        description="Orchestrate ISO-NE data downloads sequentially"
    )
    parser.add_argument(
        "--downloads",
        nargs="+",
        help="Year ranges to download (e.g., 2025 2024 2019-2023)",
    )
    parser.add_argument(
        "--data-types",
        nargs="+",
        default=["lmp", "as"],
        choices=["lmp", "as", "all"],
        help="Data types to download",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="JSON config file with custom download specifications",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("/tmp"),
        help="Directory for log files",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Maximum retries per download on failure",
    )
    parser.add_argument(
        "--retry-delay",
        type=int,
        default=300,
        help="Delay in seconds between retries",
    )

    args = parser.parse_args()

    # Load download configuration
    if args.config:
        with open(args.config) as f:
            downloads = json.load(f)
    elif args.downloads:
        data_types = args.data_types
        if "all" in data_types:
            data_types = ["lmp", "as"]
        downloads = create_download_config(args.downloads, data_types)
    else:
        print("Error: Must specify either --downloads or --config")
        sys.exit(1)

    # Create orchestrator and run
    orchestrator = DownloadOrchestrator(
        log_dir=args.log_dir,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
    )

    results = await orchestrator.run_sequential_downloads(downloads)

    # Exit with error code if any downloads failed
    if any(not r["success"] for r in results):
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
