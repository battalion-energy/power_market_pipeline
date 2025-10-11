#!/usr/bin/env python3
"""
Shadow Bidding System - Main Orchestrator

THIS IS THE MAIN SCRIPT FOR YOUR DAUGHTER'S FUTURE.

Runs complete shadow bidding cycle:
1. Fetch real-time forecasts from ERCOT
2. Run ML models (price forecasting + spike prediction)
3. Generate optimal bids (DA energy + AS)
4. Log bids (shadow mode - don't actually submit)
5. Wait for actual results
6. Calculate revenue (what we would have made)
7. Generate performance report

Run daily at 9:00 AM (before 10 AM DA deadline)
"""

import asyncio
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import json
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shadow_bidding.real_time_data_fetcher import RealTimeDataFetcher
from shadow_bidding.model_inference import ModelInferencePipeline
from shadow_bidding.bid_generator import BidGenerator, BatterySpec
from shadow_bidding.revenue_calculator import RevenueCalculator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('shadow_bidding/logs/shadow_bidding.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ShadowBiddingSystem:
    """
    Complete shadow bidding system orchestrator.

    This is the MAIN SYSTEM that runs daily to prove the strategy works
    before going live with real money.
    """

    def __init__(self, battery_spec: BatterySpec):
        self.battery = battery_spec
        self.data_fetcher = RealTimeDataFetcher()
        self.model_pipeline = ModelInferencePipeline()
        self.bid_generator = BidGenerator(battery_spec)
        self.revenue_calculator = RevenueCalculator()

        logger.info("="*80)
        logger.info("🚀 SHADOW BIDDING SYSTEM INITIALIZED")
        logger.info("="*80)
        logger.info(f"   Battery: {battery_spec.name}")
        logger.info(f"   Power: {battery_spec.power_mw} MW")
        logger.info(f"   Energy: {battery_spec.energy_mwh} MWh")
        logger.info(f"   Current SOC: {battery_spec.current_soc*100:.0f}%")
        logger.info("="*80 + "\n")

    async def run_daily_bidding(self) -> bool:
        """
        Run complete daily bidding cycle.

        Returns:
            True if successful, False if error
        """
        try:
            logger.info("\n" + "="*80)
            logger.info(f"🌅 STARTING DAILY SHADOW BIDDING - {datetime.now()}")
            logger.info("="*80 + "\n")

            start_time = datetime.now()

            # ============================================================
            # STEP 1: FETCH REAL-TIME FORECASTS FROM ERCOT
            # ============================================================
            logger.info("📡 STEP 1: Fetching real-time forecasts from ERCOT...")

            await self.data_fetcher.initialize_downloaders()
            forecast_data = await self.data_fetcher.fetch_all_forecasts()

            logger.info(f"✅ Forecasts fetched successfully\n")

            # ============================================================
            # STEP 2: RUN ML MODELS (PRICE FORECASTING + SPIKE PREDICTION)
            # ============================================================
            logger.info("🧠 STEP 2: Running ML models...")

            self.model_pipeline.load_models()
            predictions = self.model_pipeline.run_all_predictions(forecast_data)

            logger.info(f"✅ All models executed successfully\n")

            # ============================================================
            # STEP 3: GENERATE OPTIMAL BIDS (DA + AS)
            # ============================================================
            logger.info("⚡ STEP 3: Generating optimal bids...")

            bidding_strategy = self.bid_generator.optimize_bidding_strategy(predictions)

            logger.info(f"✅ Bidding strategy optimized\n")

            # ============================================================
            # STEP 4: LOG BIDS (SHADOW MODE - DON'T SUBMIT)
            # ============================================================
            logger.info("💾 STEP 4: Logging bids (shadow mode)...")

            self._save_bids(bidding_strategy, forecast_data, predictions)

            logger.info(f"✅ Bids logged successfully\n")

            # ============================================================
            # STEP 5: SUMMARY
            # ============================================================
            elapsed = (datetime.now() - start_time).total_seconds()

            logger.info("="*80)
            logger.info("✅ DAILY SHADOW BIDDING COMPLETE")
            logger.info("="*80)
            logger.info(f"   Total Time: {elapsed:.1f} seconds")
            logger.info(f"   Expected Revenue: ${bidding_strategy.expected_total_revenue:,.0f}")
            logger.info(f"   DA Bids: {len(bidding_strategy.da_energy_bids)}")
            logger.info(f"   AS Offers: {len(bidding_strategy.reg_up_offers + bidding_strategy.reg_down_offers)}")
            logger.info(f"   Next Step: Wait for DA awards at ~1:30 PM, then calculate actual revenue")
            logger.info("="*80 + "\n")

            return True

        except Exception as e:
            logger.error(f"❌ CRITICAL ERROR in daily bidding: {e}", exc_info=True)
            return False

    async def run_revenue_calculation(self, date: Optional[datetime] = None) -> bool:
        """
        Calculate revenue after DA awards are posted.

        Run this after 1:30 PM when DA awards are available.

        Args:
            date: Date to calculate revenue for (default: yesterday)
        """
        try:
            if date is None:
                date = datetime.now() - timedelta(days=1)

            logger.info("\n" + "="*80)
            logger.info(f"💰 CALCULATING REVENUE FOR {date.date()}")
            logger.info("="*80 + "\n")

            # Load bidding strategy from yesterday
            bidding_strategy = self._load_bids(date)

            if bidding_strategy is None:
                logger.error(f"❌ No bidding strategy found for {date.date()}")
                return False

            # Calculate revenue
            revenue_result = self.revenue_calculator.calculate_revenue(
                date=date,
                bidding_strategy=bidding_strategy,
                battery_spec=self.battery
            )

            logger.info(f"✅ Revenue calculation complete\n")

            return True

        except Exception as e:
            logger.error(f"❌ CRITICAL ERROR calculating revenue: {e}", exc_info=True)
            return False

    def generate_performance_report(self, days: int = 30):
        """
        Generate performance report for last N days.

        Shows how well the shadow bidding system is performing.
        """
        try:
            logger.info("\n" + "="*80)
            logger.info(f"📊 GENERATING {days}-DAY PERFORMANCE REPORT")
            logger.info("="*80 + "\n")

            report_df = self.revenue_calculator.generate_performance_report(days=days)

            if report_df.empty:
                logger.warning("⚠️ No data available for report")
                return

            # Additional analysis
            logger.info("📈 TREND ANALYSIS:")
            logger.info(f"   Best Day: {report_df['date'].iloc[report_df['actual_total'].idxmax()].date()} - ${report_df['actual_total'].max():,.0f}")
            logger.info(f"   Worst Day: {report_df['date'].iloc[report_df['actual_total'].idxmin()].date()} - ${report_df['actual_total'].min():,.0f}")

            # Forecast accuracy trend
            recent_7_days = report_df.tail(7)
            logger.info(f"\n   Last 7 Days Avg Error: {recent_7_days['revenue_error_pct'].mean():+.1f}%")

            # Winning percentage
            winning_days = (report_df['revenue_error'] > 0).sum()
            logger.info(f"   Days Beat Forecast: {winning_days}/{len(report_df)} ({winning_days/len(report_df):.0%})")

            logger.info("\n" + "="*80 + "\n")

        except Exception as e:
            logger.error(f"❌ Error generating performance report: {e}")

    def _save_bids(self, bidding_strategy, forecast_data, predictions):
        """Save bids to disk for audit trail."""
        try:
            bids_dir = Path("shadow_bidding/bids")
            bids_dir.mkdir(parents=True, exist_ok=True)

            date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = bids_dir / f"bids_{date_str}_{self.battery.name}.json"

            bid_data = {
                'timestamp': datetime.now().isoformat(),
                'battery': self.battery.name,
                'forecast_data': forecast_data.to_dict(),
                'predictions': predictions.to_dict(),
                'bidding_strategy': bidding_strategy.to_dict(),
                'da_bids': [
                    {
                        'hour': bid.hour,
                        'price_quantity_pairs': bid.price_quantity_pairs,
                        'expected_clearing_price': bid.expected_clearing_price,
                        'expected_award': bid.expected_award
                    }
                    for bid in bidding_strategy.da_energy_bids
                ],
                'as_offers': {
                    'reg_up': [
                        {
                            'hour': offer.hour,
                            'price': offer.expected_clearing_price,
                            'quantity': offer.expected_award
                        }
                        for offer in bidding_strategy.reg_up_offers
                    ],
                    'reg_down': [
                        {
                            'hour': offer.hour,
                            'price': offer.expected_clearing_price,
                            'quantity': offer.expected_award
                        }
                        for offer in bidding_strategy.reg_down_offers
                    ]
                }
            }

            with open(filename, 'w') as f:
                json.dump(bid_data, f, indent=2)

            logger.info(f"💾 Bids saved to {filename}")

        except Exception as e:
            logger.error(f"⚠️ Failed to save bids: {e}")

    def _load_bids(self, date: datetime):
        """Load bidding strategy from disk."""
        try:
            bids_dir = Path("shadow_bidding/bids")
            date_str = date.strftime('%Y%m%d')

            # Find bid file for this date
            bid_files = list(bids_dir.glob(f"bids_{date_str}_*.json"))

            if not bid_files:
                logger.warning(f"⚠️ No bid file found for {date_str}")
                return None

            # Load most recent if multiple
            bid_file = sorted(bid_files)[-1]

            with open(bid_file, 'r') as f:
                bid_data = json.load(f)

            logger.info(f"📂 Loaded bids from {bid_file}")

            # Reconstruct bidding strategy
            # TODO: Implement proper deserialization
            # For now, return the raw data
            return bid_data['bidding_strategy']

        except Exception as e:
            logger.error(f"❌ Failed to load bids: {e}")
            return None


async def main():
    """Main entry point for shadow bidding system."""

    parser = argparse.ArgumentParser(
        description="Shadow Bidding System - For Your Daughter's Future",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run daily bidding (before 10 AM)
    python shadow_bidding/run_shadow_bidding.py --mode bidding

    # Calculate revenue (after 1:30 PM)
    python shadow_bidding/run_shadow_bidding.py --mode revenue

    # Generate 30-day performance report
    python shadow_bidding/run_shadow_bidding.py --mode report --days 30

    # Run complete cycle (bidding + wait + revenue)
    python shadow_bidding/run_shadow_bidding.py --mode full
        """
    )

    parser.add_argument('--mode', choices=['bidding', 'revenue', 'report', 'full'],
                       default='bidding',
                       help='Mode to run: bidding, revenue calculation, or report')
    parser.add_argument('--battery', type=str, default='MOSS1_UNIT1',
                       help='Battery name')
    parser.add_argument('--power-mw', type=float, default=10.0,
                       help='Battery power rating (MW)')
    parser.add_argument('--energy-mwh', type=float, default=20.0,
                       help='Battery energy capacity (MWh)')
    parser.add_argument('--soc', type=float, default=0.5,
                       help='Current state of charge (0-1)')
    parser.add_argument('--days', type=int, default=30,
                       help='Number of days for performance report')

    args = parser.parse_args()

    # Create battery specification
    battery = BatterySpec(
        name=args.battery,
        power_mw=args.power_mw,
        energy_mwh=args.energy_mwh,
        efficiency=0.9,
        soc_min=0.1,
        soc_max=0.9,
        current_soc=args.soc
    )

    # Initialize system
    system = ShadowBiddingSystem(battery)

    # Run requested mode
    if args.mode == 'bidding':
        # Run daily bidding
        success = await system.run_daily_bidding()
        sys.exit(0 if success else 1)

    elif args.mode == 'revenue':
        # Calculate revenue for yesterday
        success = await system.run_revenue_calculation()
        sys.exit(0 if success else 1)

    elif args.mode == 'report':
        # Generate performance report
        system.generate_performance_report(days=args.days)
        sys.exit(0)

    elif args.mode == 'full':
        # Run complete cycle
        logger.info("🔄 Running complete shadow bidding cycle...")

        # 1. Run daily bidding
        success = await system.run_daily_bidding()
        if not success:
            logger.error("❌ Daily bidding failed")
            sys.exit(1)

        # 2. Calculate revenue for yesterday (if available)
        yesterday = datetime.now() - timedelta(days=1)
        await system.run_revenue_calculation(date=yesterday)

        # 3. Generate report
        system.generate_performance_report(days=args.days)

        logger.info("✅ Complete cycle finished successfully")
        sys.exit(0)


if __name__ == "__main__":
    # Check if running at appropriate time
    current_hour = datetime.now().hour

    if 9 <= current_hour < 10:
        logger.info("⏰ Perfect timing! Running before DA deadline (10 AM)")
    elif 13 <= current_hour < 15:
        logger.info("⏰ Good time for revenue calculation (after DA awards at 1:30 PM)")
    else:
        logger.warning(f"⏰ Running at {datetime.now().strftime('%I:%M %p')} - unusual time")
        logger.warning("   Typical schedule: 9 AM (bidding), 2 PM (revenue calculation)")

    asyncio.run(main())
