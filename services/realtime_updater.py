"""Real-time data update service with smart polling."""

import asyncio
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from database import get_db
from downloaders.base_v2 import DownloadConfig
from services.data_fetcher import DataFetcher


class RealtimeUpdater:
    """Real-time update service that polls ISOs for latest data."""
    
    def __init__(self, isos: List[str], data_types: Optional[List[str]] = None):
        self.isos = isos
        self.data_types = data_types or ["lmp", "load", "generation"]
        self.logger = structlog.get_logger()
        self.scheduler = AsyncIOScheduler()
        
        # Polling configuration
        self.polling_interval = 5  # seconds
        self.max_polling_attempts = 12  # 1 minute max
        self.update_window = timedelta(hours=2)  # Look back 2 hours for updates
        
        # Track last successful update times
        self.last_update_times: Dict[str, Dict[str, datetime]] = {}
        
    async def start(self):
        """Start the real-time update scheduler."""
        self.logger.info("Starting real-time updater", isos=self.isos)
        
        # Schedule updates at 5-minute intervals (00, 05, 10, 15, etc.)
        self.scheduler.add_job(
            self._run_update_cycle,
            CronTrigger(minute="*/5", second=0),
            id="realtime_update",
            replace_existing=True,
            max_instances=1
        )
        
        self.scheduler.start()
        self.logger.info("Real-time updater started")
        
        # Run initial update
        await self._run_update_cycle()
        
    async def stop(self):
        """Stop the scheduler."""
        self.scheduler.shutdown(wait=False)
        self.logger.info("Real-time updater stopped")
        
    async def _run_update_cycle(self):
        """Run a complete update cycle with smart polling."""
        cycle_start = datetime.now()
        self.logger.info("Starting update cycle", time=cycle_start)
        
        # Create tasks for each ISO
        tasks = []
        for iso in self.isos:
            task = asyncio.create_task(self._update_iso_with_polling(iso))
            tasks.append(task)
        
        # Wait for all ISOs to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Log results
        for iso, result in zip(self.isos, results):
            if isinstance(result, Exception):
                self.logger.error(f"Update failed for {iso}", error=str(result))
            else:
                self.logger.info(f"Update completed for {iso}", records=result)
        
        cycle_duration = (datetime.now() - cycle_start).total_seconds()
        self.logger.info("Update cycle completed", duration_seconds=cycle_duration)
        
    async def _update_iso_with_polling(self, iso: str) -> Dict[str, int]:
        """Update a single ISO with smart polling for new data."""
        self.logger.info(f"Starting update for {iso}")
        
        # Determine time range for update
        end_time = datetime.now()
        start_time = end_time - self.update_window
        
        # Get last successful update time for this ISO
        last_update = self.last_update_times.get(iso, {})
        
        # Create config for data fetcher
        config = DownloadConfig(
            start_date=start_time,
            end_date=end_time,
            data_types=self.data_types,
            output_dir=f"/tmp/power_market_pipeline/{iso.lower()}",
            batch_size=1000,
            retry_attempts=3,
            retry_delay=30
        )
        
        # Initialize data fetcher
        fetcher = DataFetcher(config)
        
        # Poll for new data
        total_records = {}
        for attempt in range(self.max_polling_attempts):
            try:
                # Fetch data
                results = await fetcher.fetch_all_data(
                    isos=[iso],
                    start_date=start_time,
                    end_date=end_time,
                    data_types=self.data_types
                )
                
                if iso in results:
                    iso_results = results[iso]
                    
                    # Check if we got new data
                    new_data = False
                    for data_type, count in iso_results.items():
                        if count > 0:
                            new_data = True
                            if data_type not in total_records:
                                total_records[data_type] = 0
                            total_records[data_type] += count
                    
                    if new_data:
                        self.logger.info(
                            f"{iso}: New data found",
                            attempt=attempt + 1,
                            records=iso_results
                        )
                        
                        # Update last successful time
                        for data_type in iso_results:
                            if iso not in self.last_update_times:
                                self.last_update_times[iso] = {}
                            self.last_update_times[iso][data_type] = end_time
                    else:
                        self.logger.debug(
                            f"{iso}: No new data",
                            attempt=attempt + 1
                        )
                
                # For real-time data, always wait between polls
                if attempt < self.max_polling_attempts - 1:
                    await asyncio.sleep(self.polling_interval)
                    
            except Exception as e:
                self.logger.error(
                    f"{iso}: Error during polling",
                    attempt=attempt + 1,
                    error=str(e)
                )
                
                # Wait before retry
                if attempt < self.max_polling_attempts - 1:
                    await asyncio.sleep(self.polling_interval)
        
        return total_records
    
    async def run_forever(self):
        """Run the updater forever."""
        try:
            await self.start()
            # Keep running until interrupted
            while True:
                await asyncio.sleep(60)
        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal")
        finally:
            await self.stop()


class ERCOTRealtimeUpdater(RealtimeUpdater):
    """Specialized real-time updater for ERCOT using WebService API."""
    
    def __init__(self):
        super().__init__(
            isos=["ERCOT"],
            data_types=["lmp", "load", "ancillary"]
        )
        
        # ERCOT-specific configuration
        self.polling_interval = 5  # Poll every 5 seconds
        self.max_polling_attempts = 24  # 2 minutes max
        
    async def _update_iso_with_polling(self, iso: str) -> Dict[str, int]:
        """ERCOT-specific update using WebService API for real-time data."""
        self.logger.info("Starting ERCOT real-time update")
        
        # For ERCOT, we want to get the most recent 5-minute interval
        now = datetime.now()
        
        # Round down to nearest 5-minute interval
        minutes = (now.minute // 5) * 5
        current_interval = now.replace(minute=minutes, second=0, microsecond=0)
        
        # Look for data from the current and previous interval
        start_time = current_interval - timedelta(minutes=5)
        end_time = current_interval + timedelta(minutes=5)
        
        self.logger.info(
            "ERCOT: Polling for data",
            interval_start=start_time,
            interval_end=end_time
        )
        
        # Use parent implementation with ERCOT-specific time range
        config = DownloadConfig(
            start_date=start_time,
            end_date=end_time,
            data_types=["lmp"],  # Focus on LMP for real-time
            output_dir="/tmp/power_market_pipeline/ercot",
            batch_size=1000,
            retry_attempts=1,  # Quick retries for real-time
            retry_delay=5
        )
        
        fetcher = DataFetcher(config)
        
        # Poll aggressively for new data
        total_records = {}
        got_current_interval = False
        
        for attempt in range(self.max_polling_attempts):
            try:
                results = await fetcher.fetch_all_data(
                    isos=["ERCOT"],
                    start_date=start_time,
                    end_date=end_time,
                    data_types=["lmp"]
                )
                
                if "ERCOT" in results:
                    iso_results = results["ERCOT"]
                    
                    # Check if we got data for the current interval
                    if iso_results.get("lmp_rt5m", 0) > 0:
                        got_current_interval = True
                        total_records.update(iso_results)
                        self.logger.info(
                            "ERCOT: Got current interval data",
                            attempt=attempt + 1,
                            records=iso_results
                        )
                        break
                
                # Wait before next poll
                await asyncio.sleep(self.polling_interval)
                
            except Exception as e:
                self.logger.error(
                    "ERCOT: Polling error",
                    attempt=attempt + 1,
                    error=str(e)
                )
                await asyncio.sleep(self.polling_interval)
        
        if not got_current_interval:
            self.logger.warning(
                "ERCOT: Failed to get current interval data",
                interval=current_interval
            )
        
        return total_records


async def main():
    """Run real-time updater as standalone service."""
    import sys
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Check database connection
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("Error: DATABASE_URL not set")
        sys.exit(1)
    
    # Initialize database
    from database import init_db
    init_db()
    
    # Determine which ISOs to update
    isos = os.getenv("REALTIME_ISOS", "ERCOT").split(",")
    
    # Create updater
    if len(isos) == 1 and isos[0] == "ERCOT":
        # Use specialized ERCOT updater
        updater = ERCOTRealtimeUpdater()
    else:
        # Use general updater
        updater = RealtimeUpdater(isos=isos)
    
    # Run forever
    await updater.run_forever()


if __name__ == "__main__":
    asyncio.run(main())