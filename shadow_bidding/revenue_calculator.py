"""
Revenue Calculator for Shadow Bidding

Calculates actual vs. expected revenue:
1. Fetch actual DA/RT/AS clearing prices
2. Determine what would have been awarded
3. Calculate actual revenue if bids were submitted
4. Compare to expected revenue
5. Generate performance report

FOR YOUR DAUGHTER - Every dollar matters.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RevenueResult:
    """Complete revenue calculation for a single day."""
    date: datetime
    battery_name: str

    # Expected revenue (what we predicted)
    expected_da_revenue: float
    expected_as_revenue: float
    expected_rt_revenue: float
    expected_total: float

    # Actual revenue (what we would have made)
    actual_da_revenue: float
    actual_as_revenue: float
    actual_rt_revenue: float
    actual_total: float

    # Performance metrics
    revenue_error: float  # actual - expected
    revenue_error_pct: float  # (actual - expected) / expected * 100
    price_forecast_mae: float  # MAE of price forecasts

    # Bid statistics
    da_bids_submitted: int
    da_bids_cleared: int
    da_clearing_rate: float
    as_offers_submitted: int
    as_offers_cleared: int

    # Detailed breakdown
    hourly_breakdown: List[Dict]

    def to_dict(self) -> Dict:
        """Convert to dictionary for logging."""
        return {
            'date': self.date.isoformat(),
            'battery_name': self.battery_name,
            'expected_total': self.expected_total,
            'actual_total': self.actual_total,
            'revenue_error': self.revenue_error,
            'revenue_error_pct': self.revenue_error_pct,
            'da_clearing_rate': self.da_clearing_rate,
        }


class RevenueCalculator:
    """
    Calculate actual revenue from shadow bids.

    Compares what we WOULD have made to what we EXPECTED to make.
    This validates model accuracy and bid optimization quality.
    """

    def __init__(self, data_dir: Path = Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data")):
        self.data_dir = data_dir
        logger.info("✅ RevenueCalculator initialized")

    def load_actual_prices(self, date: datetime) -> Dict[str, pd.DataFrame]:
        """
        Load actual clearing prices for the given date.

        Returns:
            {
                'da': DataFrame with hourly DA prices,
                'rt': DataFrame with 5-min RT prices,
                'as': DataFrame with hourly AS prices
            }
        """
        try:
            logger.info(f"Loading actual prices for {date.date()}...")

            # Load DA prices
            da_files = sorted((self.data_dir / "DAM_Prices").glob("*.csv"))
            if da_files:
                da_df = pd.read_csv(da_files[-1])
                da_df['timestamp'] = pd.to_datetime(da_df['deliveryDate'])
                da_df = da_df[da_df['timestamp'].dt.date == date.date()]
            else:
                logger.warning("⚠️ No DA price files found")
                da_df = pd.DataFrame()

            # Load RT prices
            rt_files = sorted((self.data_dir / "RTM_Prices").glob("*.csv"))
            if rt_files:
                rt_df = pd.read_csv(rt_files[-1])
                rt_df['timestamp'] = pd.to_datetime(rt_df['deliveryDate'])
                rt_df = rt_df[rt_df['timestamp'].dt.date == date.date()]
            else:
                logger.warning("⚠️ No RT price files found")
                rt_df = pd.DataFrame()

            # Load AS prices
            as_files = sorted((self.data_dir / "AS_Prices").glob("*.csv"))
            if as_files:
                as_df = pd.read_csv(as_files[-1])
                as_df['timestamp'] = pd.to_datetime(as_df['deliveryDate'])
                as_df = as_df[as_df['timestamp'].dt.date == date.date()]
            else:
                logger.warning("⚠️ No AS price files found")
                as_df = pd.DataFrame()

            logger.info(f"✅ Loaded prices: DA={len(da_df)}, RT={len(rt_df)}, AS={len(as_df)}")

            return {
                'da': da_df,
                'rt': rt_df,
                'as': as_df
            }

        except Exception as e:
            logger.error(f"❌ Error loading actual prices: {e}")
            return {'da': pd.DataFrame(), 'rt': pd.DataFrame(), 'as': pd.DataFrame()}

    def calculate_da_revenue(
        self,
        bids: List,
        actual_prices: pd.DataFrame,
        battery_spec
    ) -> Tuple[float, int, int]:
        """
        Calculate DA revenue if bids were submitted.

        For each bid:
        - Check if bid price < clearing price → CLEARED
        - Revenue = award MW × clearing price × 1 hour

        Returns:
            (revenue, bids_submitted, bids_cleared)
        """
        total_revenue = 0.0
        bids_submitted = 0
        bids_cleared = 0

        for bid in bids:
            if not bid.price_quantity_pairs:
                continue  # No bid

            bids_submitted += 1

            # Get actual clearing price for this hour
            hour_prices = actual_prices[actual_prices['timestamp'].dt.hour == bid.hour]

            if hour_prices.empty:
                logger.warning(f"⚠️ No price data for hour {bid.hour}")
                continue

            # Use settlement point price (or average if multiple)
            # TODO: Use battery's actual settlement point
            clearing_price = hour_prices['settlementPointPrice'].mean()

            # Check if any bid segments cleared
            for bid_price, bid_quantity in bid.price_quantity_pairs:
                if bid_quantity > 0:  # Discharge bid
                    if bid_price <= clearing_price:
                        # Bid cleared
                        revenue = bid_quantity * clearing_price * 1.0  # 1 hour
                        total_revenue += revenue
                        bids_cleared += 1
                        logger.debug(f"  Hour {bid.hour}: Discharged {bid_quantity:.1f} MW @ ${clearing_price:.2f} = ${revenue:.0f}")
                        break  # Only count once per hour

                elif bid_quantity < 0:  # Charge bid
                    if bid_price >= clearing_price:
                        # Charge bid cleared
                        cost = abs(bid_quantity) * clearing_price * 1.0
                        total_revenue -= cost  # Negative revenue (cost)
                        bids_cleared += 1
                        logger.debug(f"  Hour {bid.hour}: Charged {abs(bid_quantity):.1f} MW @ ${clearing_price:.2f} = -${cost:.0f}")
                        break

        return total_revenue, bids_submitted, bids_cleared

    def calculate_as_revenue(
        self,
        offers: Dict[str, List],
        actual_prices: pd.DataFrame
    ) -> Tuple[float, int, int]:
        """
        Calculate AS revenue if offers were submitted.

        AS revenue = capacity payment only (no energy)
        Revenue = offered MW × capacity price × 1 hour

        Returns:
            (revenue, offers_submitted, offers_cleared)
        """
        total_revenue = 0.0
        offers_submitted = 0
        offers_cleared = 0

        for product, offer_list in offers.items():
            for offer in offer_list:
                if not offer.price_quantity_pairs:
                    continue

                offers_submitted += 1

                # Get actual AS price for this hour and product
                hour_prices = actual_prices[
                    (actual_prices['timestamp'].dt.hour == offer.hour) &
                    (actual_prices['ancillaryType'] == product.upper().replace('_', ''))
                ]

                if hour_prices.empty:
                    logger.warning(f"⚠️ No {product} price for hour {offer.hour}")
                    continue

                clearing_price = hour_prices['mcpc'].mean()  # Market Clearing Price for Capacity

                # Check if offer cleared
                offer_price, offer_quantity = offer.price_quantity_pairs[0]

                if offer_price <= clearing_price:
                    # Offer cleared
                    revenue = offer_quantity * clearing_price * 1.0  # 1 hour
                    total_revenue += revenue
                    offers_cleared += 1
                    logger.debug(f"  Hour {offer.hour} {product}: {offer_quantity:.1f} MW @ ${clearing_price:.2f} = ${revenue:.0f}")

        return total_revenue, offers_submitted, offers_cleared

    def calculate_revenue(
        self,
        date: datetime,
        bidding_strategy,
        battery_spec
    ) -> RevenueResult:
        """
        Main entry point: Calculate actual vs. expected revenue.

        This is the critical validation - did our strategy work?
        """
        try:
            logger.info("\n" + "="*80)
            logger.info(f"💰 CALCULATING REVENUE FOR {date.date()}")
            logger.info("="*80)

            # Load actual prices
            actual_prices = self.load_actual_prices(date)

            # Calculate DA revenue
            da_revenue, da_submitted, da_cleared = self.calculate_da_revenue(
                bidding_strategy.da_energy_bids,
                actual_prices['da'],
                battery_spec
            )

            # Calculate AS revenue
            as_offers = {
                'reg_up': bidding_strategy.reg_up_offers,
                'reg_down': bidding_strategy.reg_down_offers,
                'rrs': bidding_strategy.rrs_offers,
                'ecrs': bidding_strategy.ecrs_offers
            }

            as_revenue, as_submitted, as_cleared = self.calculate_as_revenue(
                as_offers,
                actual_prices['as']
            )

            # RT revenue (TODO: Implement RT arbitrage)
            rt_revenue = 0.0

            # Total actual revenue
            actual_total = da_revenue + as_revenue + rt_revenue

            # Compare to expected
            expected_total = bidding_strategy.expected_total_revenue
            revenue_error = actual_total - expected_total
            revenue_error_pct = (revenue_error / expected_total * 100) if expected_total > 0 else 0

            # Create result
            result = RevenueResult(
                date=date,
                battery_name=bidding_strategy.battery_name,
                expected_da_revenue=bidding_strategy.expected_da_revenue,
                expected_as_revenue=bidding_strategy.expected_as_revenue,
                expected_rt_revenue=bidding_strategy.expected_rt_revenue,
                expected_total=expected_total,
                actual_da_revenue=da_revenue,
                actual_as_revenue=as_revenue,
                actual_rt_revenue=rt_revenue,
                actual_total=actual_total,
                revenue_error=revenue_error,
                revenue_error_pct=revenue_error_pct,
                price_forecast_mae=0.0,  # TODO: Calculate
                da_bids_submitted=da_submitted,
                da_bids_cleared=da_cleared,
                da_clearing_rate=da_cleared / da_submitted if da_submitted > 0 else 0,
                as_offers_submitted=as_submitted,
                as_offers_cleared=as_cleared,
                hourly_breakdown=[]  # TODO: Add hourly details
            )

            logger.info("="*80)
            logger.info("💰 REVENUE CALCULATION COMPLETE")
            logger.info("="*80)
            logger.info(f"   Expected Total: ${expected_total:,.0f}")
            logger.info(f"   Actual Total:   ${actual_total:,.0f}")
            logger.info(f"   Error:          ${revenue_error:+,.0f} ({revenue_error_pct:+.1f}%)")
            logger.info(f"   DA Clearing:    {da_cleared}/{da_submitted} ({result.da_clearing_rate:.0%})")
            logger.info(f"   AS Clearing:    {as_cleared}/{as_submitted}")
            logger.info("="*80 + "\n")

            # Save result
            self._save_result(result)

            return result

        except Exception as e:
            logger.error(f"❌ CRITICAL ERROR calculating revenue: {e}")
            raise

    def _save_result(self, result: RevenueResult):
        """Save revenue result to disk."""
        try:
            results_dir = Path("shadow_bidding/results/revenue")
            results_dir.mkdir(parents=True, exist_ok=True)

            filename = results_dir / f"revenue_{result.date.strftime('%Y%m%d')}_{result.battery_name}.json"

            with open(filename, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)

            logger.debug(f"💾 Saved revenue result to {filename}")

        except Exception as e:
            logger.error(f"⚠️ Failed to save revenue result: {e}")

    def generate_performance_report(self, days: int = 30) -> pd.DataFrame:
        """
        Generate performance report for last N days.

        Shows:
        - Average revenue per day
        - Revenue forecast accuracy
        - Bid clearing rates
        - Model performance trends
        """
        try:
            logger.info(f"Generating {days}-day performance report...")

            results_dir = Path("shadow_bidding/results/revenue")

            # Load all results
            results = []
            for result_file in sorted(results_dir.glob("*.json")):
                with open(result_file, 'r') as f:
                    results.append(json.load(f))

            if not results:
                logger.warning("⚠️ No results found")
                return pd.DataFrame()

            df = pd.DataFrame(results)
            df['date'] = pd.to_datetime(df['date'])

            # Summary statistics
            summary = {
                'Total Days': len(df),
                'Avg Daily Revenue': df['actual_total'].mean(),
                'Total Revenue': df['actual_total'].sum(),
                'Avg Forecast Error': df['revenue_error'].mean(),
                'Forecast Error %': df['revenue_error_pct'].mean(),
                'Avg DA Clearing Rate': df['da_clearing_rate'].mean(),
                'Best Day Revenue': df['actual_total'].max(),
                'Worst Day Revenue': df['actual_total'].min(),
            }

            logger.info("\n" + "="*80)
            logger.info(f"📊 {days}-DAY PERFORMANCE REPORT")
            logger.info("="*80)
            for key, value in summary.items():
                if 'Revenue' in key or 'Error' in key:
                    logger.info(f"   {key:.<30} ${value:,.0f}")
                elif '%' in key or 'Rate' in key:
                    logger.info(f"   {key:.<30} {value:.1%}")
                else:
                    logger.info(f"   {key:.<30} {value:.0f}")
            logger.info("="*80 + "\n")

            return df

        except Exception as e:
            logger.error(f"❌ Error generating performance report: {e}")
            return pd.DataFrame()


def main():
    """Test revenue calculator."""
    print("\n" + "="*80)
    print("TESTING REVENUE CALCULATOR")
    print("="*80 + "\n")

    calculator = RevenueCalculator()

    # Generate performance report
    report = calculator.generate_performance_report(days=30)

    if not report.empty:
        print("\n📊 PERFORMANCE SUMMARY:")
        print(report[['date', 'expected_total', 'actual_total', 'revenue_error_pct']].tail(10))

    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
