#!/usr/bin/env python3
"""Real-time data collector for power market data.

Runs every 5 minutes to collect the latest real-time data from all ISOs.
Designed to trigger right at the minute crossover for fastest data retrieval.
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List
import signal

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from dotenv import load_dotenv
import structlog

load_dotenv()

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import init_db, get_db
from database.models_v2 import LMP
from downloaders.ercot.downloader_v2 import ERCOTDownloaderV2
from downloaders.base_v2 import DownloadConfig
from sqlalchemy import func

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.dict_tracebacks.DictTracebackProcessor(),
        structlog.dev.ConsoleRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


class RealtimeCollector:
    """Collects real-time power market data from all ISOs."""
    
    def __init__(self):
        self.logger = logger.bind(component="RealtimeCollector")
        self.downloaders = {}
        self.scheduler = AsyncIOScheduler()
        self.running = True
        
    def initialize_downloaders(self):
        """Initialize downloaders for each ISO."""
        # Configuration for real-time data - look back 1 hour
        config = DownloadConfig(
            start_date=datetime.now() - timedelta(hours=1),
            end_date=datetime.now(),
            data_types=['lmp'],
            output_dir='./data/realtime'
        )
        
        # Initialize ERCOT downloader
        try:
            self.downloaders['ERCOT'] = ERCOTDownloaderV2(config)
            self.logger.info("Initialized ERCOT downloader")
        except Exception as e:
            self.logger.error("Failed to initialize ERCOT downloader", error=str(e))
        
        # TODO: Initialize other ISO downloaders when implemented
        # self.downloaders['CAISO'] = CAISODownloaderV2(config)
        # self.downloaders['ISONE'] = ISONEDownloaderV2(config)
        # self.downloaders['NYISO'] = NYISODownloaderV2(config)
        
    async def collect_realtime_data(self):
        """Collect real-time data from all ISOs."""
        start_time = datetime.now()
        self.logger.info("Starting real-time data collection", timestamp=start_time.isoformat())
        
        # Update config with current time window
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=15)  # Look back 15 minutes for RT data
        
        results = {}
        
        for iso, downloader in self.downloaders.items():
            try:
                self.logger.info(f"Collecting {iso} real-time data", 
                               start=start_time.isoformat(), 
                               end=end_time.isoformat())
                
                # Update downloader config
                downloader.config.start_date = start_time
                downloader.config.end_date = end_time
                
                # Download real-time LMP data
                count = await downloader.download_lmp('RT5M', start_time, end_time)
                results[iso] = count
                
                self.logger.info(f"{iso} collection complete", records=count)
                
            except Exception as e:
                self.logger.error(f"{iso} collection failed", error=str(e), exc_info=True)
                results[iso] = 0
        
        # Log summary
        total_records = sum(results.values())
        duration = (datetime.now() - start_time).total_seconds()
        
        self.logger.info("Real-time collection complete", 
                        duration_seconds=duration,
                        total_records=total_records,
                        results=results)
        
        # Check database state
        await self.check_database_state()
        
    async def collect_dayahead_updates(self):
        """Collect day-ahead market updates (runs hourly)."""
        start_time = datetime.now()
        self.logger.info("Starting day-ahead data collection", timestamp=start_time.isoformat())
        
        # Day-ahead markets publish data for tomorrow
        tomorrow = datetime.now().date() + timedelta(days=1)
        start_date = datetime.combine(tomorrow, datetime.min.time())
        end_date = start_date + timedelta(days=1)
        
        results = {}
        
        for iso, downloader in self.downloaders.items():
            try:
                self.logger.info(f"Collecting {iso} day-ahead data", 
                               date=tomorrow.isoformat())
                
                # Update downloader config
                downloader.config.start_date = start_date
                downloader.config.end_date = end_date
                
                # Download DAM data
                count = await downloader.download_lmp('DAM', start_date, end_date)
                results[iso] = count
                
                self.logger.info(f"{iso} DAM collection complete", records=count)
                
            except Exception as e:
                self.logger.error(f"{iso} DAM collection failed", error=str(e))
                results[iso] = 0
        
        total_records = sum(results.values())
        self.logger.info("Day-ahead collection complete", 
                        total_records=total_records,
                        results=results)
    
    async def check_database_state(self):
        """Check and log current database state."""
        with get_db() as db:
            # Get recent data stats
            recent_cutoff = datetime.now() - timedelta(hours=1)
            recent_stats = db.query(
                LMP.iso,
                LMP.market,
                func.count(LMP.iso).label('count'),
                func.max(LMP.interval_start).label('latest')
            ).filter(
                LMP.interval_start >= recent_cutoff
            ).group_by(LMP.iso, LMP.market).all()
            
            if recent_stats:
                for iso, market, count, latest in recent_stats:
                    lag = (datetime.now() - latest.replace(tzinfo=None)).total_seconds() / 60
                    self.logger.info(
                        "Recent data status",
                        iso=iso,
                        market=market,
                        records_last_hour=count,
                        latest_data=latest.isoformat(),
                        lag_minutes=round(lag, 1)
                    )
    
    def setup_schedules(self):
        """Set up scheduled tasks."""
        # Real-time data collection - every 5 minutes, right after the minute
        # Run at :00, :05, :10, :15, :20, :25, :30, :35, :40, :45, :50, :55
        self.scheduler.add_job(
            self.collect_realtime_data,
            CronTrigger(minute='*/5', second='2'),  # 2 seconds after the minute
            id='realtime_collection',
            name='Real-time data collection',
            misfire_grace_time=30
        )
        
        # Day-ahead updates - every hour at :30
        self.scheduler.add_job(
            self.collect_dayahead_updates,
            CronTrigger(minute='30'),
            id='dayahead_collection',
            name='Day-ahead data collection',
            misfire_grace_time=300
        )
        
        self.logger.info("Scheduled tasks configured")
    
    async def run(self):
        """Run the real-time collector."""
        self.logger.info("Starting Power Market Real-time Collector")
        
        # Initialize database
        init_db()
        
        # Initialize downloaders
        self.initialize_downloaders()
        
        if not self.downloaders:
            self.logger.error("No downloaders available, exiting")
            return
        
        # Set up schedules
        self.setup_schedules()
        
        # Start scheduler
        self.scheduler.start()
        self.logger.info("Scheduler started, waiting for scheduled tasks...")
        
        # Run initial collection immediately
        await self.collect_realtime_data()
        
        # Keep running until interrupted
        try:
            while self.running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            self.logger.info("Shutting down...")
        finally:
            self.scheduler.shutdown()
            self.logger.info("Scheduler stopped")
    
    def stop(self):
        """Stop the collector."""
        self.running = False


async def main():
    """Main entry point."""
    collector = RealtimeCollector()
    
    # Set up signal handlers
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}, shutting down...")
        collector.stop()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Print startup information
    print("\n" + "=" * 60)
    print("🚀 Power Market Real-time Data Collector")
    print("=" * 60)
    print("\nSchedule:")
    print("  • Real-time data: Every 5 minutes (:00, :05, :10, ...)")
    print("  • Day-ahead data: Every hour at :30")
    print("\nPress Ctrl+C to stop")
    print("=" * 60 + "\n")
    
    # Run collector
    await collector.run()


if __name__ == "__main__":
    asyncio.run(main())