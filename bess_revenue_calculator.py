#!/usr/bin/env python3
"""
BESS Revenue Calculator - Historical Revenue Analysis

Calculates actual revenues for ERCOT BESS units from historical market data.
This is FORENSIC ACCOUNTING, not optimization - we reconstruct what happened.

Revenue Components:
1. DAM Discharge: Gen Resource energy awards × DAM prices
2. DAM AS (Gen): Ancillary service awards × MCPC for Gen Resource
3. DAM AS (Load): Ancillary service awards × MCPC for Load Resource
4. RT Net: (Discharge - Charging) × RT prices for 5-min intervals

Data Sources:
- DAM_Gen_Resources: DAM discharge awards and AS awards (Gen side)
- DAM_Load_Resources: DAM AS awards (Load side only - no energy)
- SCED_Gen_Resources: RT discharge BasePoints
- SCED_Load_Resources: RT charging BasePoints (NOW with ResourceName!)
- Prices: DAM SPP, RT SPP, AS MCPCs at Resource Nodes
"""

import os
import polars as pl
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BESSRevenueCalculator:
    """Calculate historical revenues for ERCOT BESS units"""

    def __init__(self, base_dir: str, year: int = 2024):
        """
        Initialize calculator

        Args:
            base_dir: Base directory for ERCOT data
            year: Year to analyze (default 2024)
        """
        self.base_dir = Path(base_dir)
        self.year = year
        self.rollup_dir = self.base_dir / "rollup_files"

        # Verify all required data exists
        self._verify_data_files()

    def _verify_data_files(self):
        """Verify all required parquet files exist"""
        required_files = [
            f"DAM_Gen_Resources/{self.year}.parquet",
            f"DAM_Load_Resources/{self.year}.parquet",
            f"SCED_Gen_Resources/{self.year}.parquet",
            f"SCED_Load_Resources/{self.year}.parquet",
        ]

        for file in required_files:
            path = self.rollup_dir / file
            if not path.exists():
                raise FileNotFoundError(f"Required file not found: {path}")

        logger.info(f"✅ All required data files found for {self.year}")

    def load_bess_mapping(self, mapping_file: str) -> pl.DataFrame:
        """
        Load BESS Gen↔Load resource mapping

        Args:
            mapping_file: Path to BESS mapping CSV

        Returns:
            Polars DataFrame with BESS mapping (filtered to operational with Load Resources)
        """
        logger.info(f"Loading BESS mapping from {mapping_file}")

        df = pl.read_csv(mapping_file)

        # Map actual column names
        expected_cols = {
            'BESS_Gen_Resource': 'Gen_Resource',
            'BESS_Load_Resource': 'Load_Resource',
            'Settlement_Point': 'Resource_Node',
            'IQ_Capacity_MW': 'Capacity_MW',
            'True_Operational_Status': 'Status'
        }

        # Verify columns exist
        for col in expected_cols.keys():
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Rename to standard names
        df = df.rename(expected_cols)

        # Filter to operational units with Load Resources
        df = df.filter(
            (pl.col('Load_Resource').is_not_null()) &
            (pl.col('Status') == 'Operational (has Load Resource)')
        )

        logger.info(f"✅ Loaded {len(df)} operational BESS units with Load Resources")
        return df

    def calculate_dam_discharge_revenue(self, gen_resource: str) -> float:
        """
        Calculate DAM discharge revenue for Gen Resource

        Revenue = Σ(AwardedQuantity × DAM_Price)

        Args:
            gen_resource: Gen Resource name

        Returns:
            Total DAM discharge revenue ($)
        """
        dam_gen_file = self.rollup_dir / f"DAM_Gen_Resources/{self.year}.parquet"

        df = pl.read_parquet(dam_gen_file).filter(
            pl.col("ResourceName") == gen_resource
        )

        if len(df) == 0:
            logger.warning(f"No DAM Gen data found for {gen_resource}")
            return 0.0

        # Calculate revenue: AwardedQuantity (MW) × Price ($/MWh)
        # DAM is hourly, so MW = MWh for 1-hour intervals
        df = df.with_columns([
            (pl.col("AwardedQuantity") * pl.col("EnergySettlementPointPrice")).alias("revenue")
        ])

        total_revenue = df.select(pl.col("revenue").sum()).item()
        return total_revenue

    def _load_da_price_series(self, resource_node: str) -> pl.DataFrame:
        """Load hourly DA price series for a single settlement point.

        Returns columns: local_date, local_hour, da_price
        """
        # Try flattened wide format first
        flat1 = self.rollup_dir / f"flattened/DA_prices_{self.year}.parquet"
        flat2 = self.rollup_dir / f"flattened/DA_prices_flat_{self.year}.parquet"
        longp = self.rollup_dir / f"DA_prices/{self.year}.parquet"

        if flat1.exists() or flat2.exists():
            path = flat1 if flat1.exists() else flat2
            df = pl.read_parquet(path)
            col = resource_node if resource_node in df.columns else ("HB_BUSAVG" if "HB_BUSAVG" in df.columns else None)
            if col is None:
                # Fall back to long path
                df = None
            else:
                df = df.select([
                    pl.col("datetime"),
                    pl.col(col).alias("da_price")
                ])
        else:
            df = None

        if df is None:
            # Long format: filter by settlement point
            if not longp.exists():
                return pl.DataFrame({"local_date": [], "local_hour": [], "da_price": []})
            df_long = pl.read_parquet(longp)
            # Column variants
            sp_col = "settlement_point" if "settlement_point" in df_long.columns else "SettlementPointName"
            price_col = "da_lmp" if "da_lmp" in df_long.columns else "SettlementPointPrice"
            df = df_long.filter(pl.col(sp_col) == resource_node).select([
                "datetime",
                pl.col(price_col).alias("da_price")
            ])

        # Convert to local date/hour for DAM alignment
        df = df.with_columns([
            pl.col("datetime").dt.replace_time_zone("UTC").dt.convert_time_zone("America/Chicago").dt.date().alias("local_date"),
            pl.col("datetime").dt.replace_time_zone("UTC").dt.convert_time_zone("America/Chicago").dt.hour().alias("local_hour")
        ]).select(["local_date", "local_hour", "da_price"]).group_by(["local_date", "local_hour"]).agg(pl.col("da_price").mean()).sort(["local_date", "local_hour"])

        return df

    def calculate_dam_charge_cost(self, gen_resource: str, load_resource: str, resource_node: str) -> float:
        """Calculate DAM charging cost from two sources:
        1) Negative awards (if any) in DAM_Gen_Resources for this Gen resource.
        2) Energy Bid Awards at the resource node (negative MW indicate load purchases).
        """
        total_cost = 0.0

        # 1) Negative awards in DAM Gen (rare but present for some BESS)
        dam_gen_file = self.rollup_dir / f"DAM_Gen_Resources/{self.year}.parquet"
        if dam_gen_file.exists():
            df = pl.read_parquet(dam_gen_file).filter(
                (pl.col("ResourceName") == gen_resource) & (pl.col("AwardedQuantity") < 0)
            )
            if len(df) > 0:
                df = df.with_columns([
                    (pl.col("AwardedQuantity") * pl.col("EnergySettlementPointPrice")).alias("cost")
                ])
                cost = df.select(pl.col("cost").abs().sum()).item()
                total_cost += cost if cost else 0.0

        # 2) Energy Bid Awards at the node
        eba_file = self.rollup_dir / f"DAM_Energy_Bid_Awards/{self.year}.parquet"
        if eba_file.exists():
            try:
                eba = pl.read_parquet(eba_file)
                # Column variants
                sp_col = "SettlementPoint" if "SettlementPoint" in eba.columns else ("settlement_point" if "settlement_point" in eba.columns else None)
                mw_col = "EnergyBidAwardMW" if "EnergyBidAwardMW" in eba.columns else ("energy_bid_award_mw" if "energy_bid_award_mw" in eba.columns else None)
                date_col = "DeliveryDate" if "DeliveryDate" in eba.columns else ("delivery_date" if "delivery_date" in eba.columns else None)
                hour_col = "hour" if "hour" in eba.columns else ("HourEnding" if "HourEnding" in eba.columns else None)

                if all(c is not None for c in [sp_col, mw_col, date_col, hour_col]):
                    eba = eba.filter(pl.col(sp_col) == resource_node)
                    # Normalize to local date/hour
                    if hour_col == "HourEnding":
                        # Convert HH:MM to int hour; HourEnding is end-of-hour, use HE as hour
                        eba = eba.with_columns(pl.col(hour_col).str.slice(0,2).cast(pl.Int32).alias("local_hour"))
                    else:
                        eba = eba.with_columns(pl.col(hour_col).cast(pl.Int32).alias("local_hour"))
                    eba = eba.with_columns(pl.col(date_col).cast(pl.Date).alias("local_date"))

                    # Negative MW = buying energy to charge
                    eba = eba.filter(pl.col(mw_col) < 0).select([
                        pl.col("local_date"), pl.col("local_hour"), pl.col(mw_col).alias("mw")
                    ])

                    if len(eba) > 0:
                        # Join DA prices
                        da_prices = self._load_da_price_series(resource_node)
                        if len(da_prices) > 0:
                            joined = eba.join(da_prices, on=["local_date", "local_hour"], how="left")
                            cost = joined.select((pl.col("mw").abs() * pl.col("da_price")).sum()).item()
                            if cost:
                                total_cost += cost
            except Exception as _:
                pass

        return total_cost

    def calculate_dam_as_revenue(self, resource: str, is_gen: bool) -> Dict[str, float]:
        """
        Calculate DAM Ancillary Service revenue

        FIXED VERSION:
        - Gen: Sums all RRS variants (PFR+FFR+UFR), uses ECRSSDAwarded, joins system MCPC
        - Load: Sums all variants, uses embedded resource-specific MCPC (no join!)

        Args:
            resource: Resource name (Gen or Load)
            is_gen: True for Gen Resource, False for Load Resource

        Returns:
            Dictionary of AS revenues by type
        """
        if is_gen:
            return self._calculate_gen_as_revenues(resource)
        else:
            return self._calculate_load_as_revenues(resource)

    def _calculate_gen_as_revenues(self, resource: str) -> Dict[str, float]:
        """
        Calculate AS revenues for Gen resources.

        Gen resources do NOT have embedded MCPC - must join with system-wide prices.

        CRITICAL FIXES:
        - RRS: Sum RRSPFRAwarded + RRSFFRAwarded + RRSUFRAwarded (not just RRSAwarded!)
        - ECRS: Use ECRSSDAwarded (not ECRSAwarded which is 0 in 2024)
        """
        dam_file = self.rollup_dir / f"DAM_Gen_Resources/{self.year}.parquet"
        as_price_file = self.rollup_dir / f"AS_prices/{self.year}.parquet"

        df = pl.read_parquet(dam_file).filter(pl.col("ResourceName") == resource)

        if len(df) == 0:
            logger.warning(f"No DAM AS data found for Gen {resource}")
            return {"RegUp": 0.0, "RegDown": 0.0, "RRS": 0.0, "ECRS": 0.0, "NonSpin": 0.0}

        df_prices = pl.read_parquet(as_price_file)

        # RegUp - straightforward
        regup_revenue = self._calc_gen_as_product(df, df_prices, "REGUP", ["RegUpAwarded"])

        # RegDown - straightforward
        regdown_revenue = self._calc_gen_as_product(df, df_prices, "REGDN", ["RegDownAwarded"])

        # RRS - SUM ALL THREE VARIANTS! (PFR + FFR + UFR)
        rrs_revenue = self._calc_gen_as_product(df, df_prices, "RRS",
                                                 ["RRSPFRAwarded", "RRSFFRAwarded", "RRSUFRAwarded", "RRSAwarded"])

        # ECRS - Use ECRSSDAwarded (Service Deployment), not base ECRSAwarded!
        ecrs_revenue = self._calc_gen_as_product(df, df_prices, "ECRS", ["ECRSSDAwarded", "ECRSAwarded"])

        # NonSpin - straightforward
        nonspin_revenue = self._calc_gen_as_product(df, df_prices, "NSPIN", ["NonSpinAwarded"])

        return {
            "RegUp": regup_revenue,
            "RegDown": regdown_revenue,
            "RRS": rrs_revenue,
            "ECRS": ecrs_revenue,
            "NonSpin": nonspin_revenue
        }

    def _calc_gen_as_product(self, df: pl.DataFrame, df_prices: pl.DataFrame,
                              price_type: str, award_cols: list) -> float:
        """
        Calculate revenue for one AS product for Gen resources.
        Sums across multiple award column variants (e.g., RRS PFR+FFR+UFR).
        """
        # Get system-wide MCPC prices for this AS type
        df_as_prices = df_prices.filter(
            pl.col("AncillaryType") == price_type
        ).select([
            pl.col("DeliveryDate").alias("date"),
            pl.col("hour").str.slice(0, 2).cast(pl.Int32).alias("hour"),
            "MCPC"
        ])

        total_revenue = 0.0

        # Try each award column variant and sum them
        for award_col in award_cols:
            if award_col not in df.columns:
                continue

            # Join awards with prices
            df_joined = df.select([
                pl.col("DeliveryDate").cast(pl.Date).alias("date"),
                pl.col("hour").cast(pl.Int32).alias("hour"),
                pl.col(award_col).alias("awarded")
            ]).join(df_as_prices, on=["date", "hour"], how="left")

            # Calculate revenue for this variant
            revenue = df_joined.select(
                (pl.col("awarded") * pl.col("MCPC")).sum()
            ).item()

            if revenue:
                total_revenue += revenue

        return total_revenue

    def _calculate_load_as_revenues(self, resource: str) -> Dict[str, float]:
        """
        Calculate AS revenues for Load resources.

        CRITICAL: Load resources have EMBEDDED resource-specific MCPC in the file!
        Do NOT join with AS_prices - use the MCPC columns that are already there!

        CRITICAL FIXES:
        - Use embedded MCPC columns (resource-specific prices, not system-wide)
        - RRS: Sum RRSFFR + RRSPFR + RRSUFR Awarded (not just FFR!)
        - ECRS: Sum ECRSSD + ECRSMD Awarded (both variants!)
        """
        dam_file = self.rollup_dir / f"DAM_Load_Resources/{self.year}.parquet"

        df = pl.read_parquet(dam_file).filter(pl.col("Load Resource Name") == resource)

        if len(df) == 0:
            logger.warning(f"No DAM AS data found for Load {resource}")
            return {"RegUp": 0.0, "RegDown": 0.0, "RRS": 0.0, "ECRS": 0.0, "NonSpin": 0.0}

        # Use EMBEDDED MCPC - prices are already in the file!
        # No join needed - just multiply awards by their MCPC columns

        # RegUp: award × embedded MCPC
        regup_revenue = self._calc_load_as_simple(df, "RegUp Awarded", "RegUp MCPC")

        # RegDown: award × embedded MCPC
        regdown_revenue = self._calc_load_as_simple(df, "RegDown Awarded", "RegDown MCPC")

        # RRS: SUM ALL THREE VARIANTS with embedded MCPC
        rrs_revenue = (
            self._calc_load_as_simple(df, "RRSFFR Awarded", "RRS MCPC") +
            self._calc_load_as_simple(df, "RRSPFR Awarded", "RRS MCPC") +
            self._calc_load_as_simple(df, "RRSUFR Awarded", "RRS MCPC")
        )

        # ECRS: SUM BOTH VARIANTS (SD + MD) with embedded MCPC
        ecrs_revenue = (
            self._calc_load_as_simple(df, "ECRSSD Awarded", "ECRS MCPC") +
            self._calc_load_as_simple(df, "ECRSMD Awarded", "ECRS MCPC")
        )

        # NonSpin: award × embedded MCPC
        nonspin_revenue = self._calc_load_as_simple(df, "NonSpin Awarded", "NonSpin MCPC")

        return {
            "RegUp": regup_revenue,
            "RegDown": regdown_revenue,
            "RRS": rrs_revenue,
            "ECRS": ecrs_revenue,
            "NonSpin": nonspin_revenue
        }

    def _calc_load_as_simple(self, df: pl.DataFrame, award_col: str, mcpc_col: str) -> float:
        """
        Calculate revenue for one Load AS product using embedded MCPC.
        Returns 0 if columns don't exist (for backwards compatibility).
        """
        if award_col not in df.columns or mcpc_col not in df.columns:
            return 0.0

        revenue = df.select((pl.col(award_col) * pl.col(mcpc_col)).sum()).item()
        return revenue if revenue else 0.0

    def calculate_rt_net_revenue(
        self,
        gen_resource: str,
        load_resource: str,
        resource_node: str
    ) -> Tuple[float, Dict[str, float], pl.DataFrame, pl.DataFrame]:
        """
        Calculate Real-Time net energy revenue

        Two‑settlement (deviation) approach:

        RT Net Revenue = Σ((Actual_Net_MW - DAM_Scheduled_MW) × RT_Price × 15/60)

        Where:
          Actual_Net_MW    = BasePoint_Gen - BasePoint_Load (averaged to 15‑min)
          DAM_Scheduled_MW = Hourly DAM Gen award (net; gen awards positive, negative if present)
          RT_Price         = Resource‑node SPP at 15‑min settlement timestamp
          15/60            = Converts 15‑min MW to MWh

        Args:
            gen_resource: Gen Resource name
            load_resource: Load Resource name
            resource_node: Resource Node for pricing

        Returns:
            Tuple of (total_revenue, stats_dict)
        """
        sced_gen_file = self.rollup_dir / f"SCED_Gen_Resources/{self.year}.parquet"
        sced_load_file = self.rollup_dir / f"SCED_Load_Resources/{self.year}.parquet"
        rt_price_file = self.rollup_dir / f"RT_prices/{self.year}.parquet"
        dam_gen_file = self.rollup_dir / f"DAM_Gen_Resources/{self.year}.parquet"

        # Load discharge data - filter to correct year only
        # NOTE: SCED timestamps are in Central Time (America/Chicago)
        # Use TelemeteredNetOutput (actual metered) if available, else BasePoint (instruction)
        df_gen_raw_full = pl.read_parquet(sced_gen_file)
        # Prefer ResourceName filter; if missing, attempt SettlementPointName
        if "ResourceName" in df_gen_raw_full.columns:
            df_gen_raw = df_gen_raw_full.filter(pl.col("ResourceName") == gen_resource)
        elif "SettlementPointName" in df_gen_raw_full.columns:
            df_gen_raw = df_gen_raw_full.filter(pl.col("SettlementPointName") == resource_node)
        else:
            df_gen_raw = df_gen_raw_full

        # Check which columns are available
        has_telemetry = "TelemeteredNetOutput" in df_gen_raw.columns
        discharge_col = "TelemeteredNetOutput" if has_telemetry else "BasePoint"

        df_gen = df_gen_raw.with_columns([
            pl.col("SCEDTimeStamp").str.strptime(pl.Datetime, "%m/%d/%Y %H:%M:%S")
                .dt.replace_time_zone("America/Chicago", ambiguous="earliest")
                .dt.convert_time_zone("UTC")
                .alias("sced_dt")
        ]).filter(
            pl.col("sced_dt").dt.year() == self.year
        ).select([
            "SCEDTimeStamp",
            pl.col(discharge_col).alias("discharge_mw"),
            pl.col("BasePoint").alias("bp_gen_mw")
        ])

        if has_telemetry:
            logger.info(f"Using TelemeteredNetOutput for {gen_resource} discharge (actual metered)")
        else:
            logger.warning(f"TelemeteredNetOutput not available for {gen_resource}, using BasePoint")

        # Load charging data (NOW POSSIBLE with fixed ResourceName!) - filter to correct year only
        # NOTE: SCED timestamps are in Central Time (America/Chicago)
        df_load_full = pl.read_parquet(sced_load_file)
        if "ResourceName" in df_load_full.columns:
            df_load_filt = df_load_full.filter(pl.col("ResourceName") == load_resource)
        else:
            # Fallback: try matching by SettlementPointName/Settlement Point Name
            if "SettlementPointName" in df_load_full.columns:
                df_load_filt = df_load_full.filter(pl.col("SettlementPointName") == resource_node)
            elif "Settlement Point Name" in df_load_full.columns:
                df_load_filt = df_load_full.filter(pl.col("Settlement Point Name") == resource_node)
            else:
                df_load_filt = df_load_full.head(0)  # empty

        df_load = df_load_filt.with_columns([
            pl.col("SCEDTimeStamp").str.strptime(pl.Datetime, "%m/%d/%Y %H:%M:%S")
                .dt.replace_time_zone("America/Chicago", ambiguous="earliest")
                .dt.convert_time_zone("UTC")
                .alias("sced_dt")
        ]).filter(
            pl.col("sced_dt").dt.year() == self.year
        ).select([
            "SCEDTimeStamp",
            pl.col("BasePoint").alias("charge_mw"),
            pl.col("BasePoint").alias("bp_load_mw")
        ])

        if len(df_gen) == 0 and len(df_load) == 0:
            logger.warning(f"No RT data found for {gen_resource}/{load_resource}")
            return 0.0, {
                "discharge_mwh": 0.0,
                "charge_mwh": 0.0,
                "efficiency": 0.0,
                "intervals": 0,
                "rt_gross_revenue": 0.0,
                "da_spread_revenue": 0.0,
                "active_days": 0
            }, pl.DataFrame(), pl.DataFrame()

        # Load RT prices strictly for this resource node (no hub fallback)
        df_rt_all = pl.read_parquet(rt_price_file)
        df_prices = df_rt_all.filter(pl.col("SettlementPointName") == resource_node).select([
            pl.col("datetime").alias("price_datetime"),
            pl.col("SettlementPointPrice").alias("rt_price")
        ])

        if len(df_prices) == 0:
            raise ValueError(
                f"NO RT PRICES FOUND for resource node '{resource_node}' in {rt_price_file}."
            )

        # Build 15‑min actual net output = Gen − Load
        # Convert SCED timestamps to UTC 15‑min bins and average within the bin
        def _to_15min(df: pl.DataFrame, value_col: str, alias: str) -> pl.DataFrame:
            if len(df) == 0:
                return pl.DataFrame({"rounded_dt": [], alias: []})
            return (
                df.with_columns([
                    pl.col("SCEDTimeStamp").str.strptime(pl.Datetime, "%m/%d/%Y %H:%M:%S")
                        .dt.replace_time_zone("America/Chicago", ambiguous="earliest")
                        .dt.convert_time_zone("UTC")
                        .dt.truncate("15m")
                        .alias("rounded_dt")
                ])
                .group_by("rounded_dt")
                .agg(pl.col(value_col).mean().alias(alias))
                .sort("rounded_dt")
            )

        # Build 15-min time series for actuals and basepoints
        gen_15 = _to_15min(df_gen, "discharge_mw", "gen_mw")
        gen_15_bp = _to_15min(df_gen, "bp_gen_mw", "bp_gen_mw") if "bp_gen_mw" in df_gen.columns else pl.DataFrame({"rounded_dt": [], "bp_gen_mw": []})
        if len(gen_15_bp) > 0:
            gen_15 = gen_15.join(gen_15_bp, on="rounded_dt", how="left")

        load_15 = _to_15min(df_load, "charge_mw", "load_mw")
        load_15_bp = _to_15min(df_load, "bp_load_mw", "bp_load_mw") if "bp_load_mw" in df_load.columns else pl.DataFrame({"rounded_dt": [], "bp_load_mw": []})
        if len(load_15_bp) > 0:
            load_15 = load_15.join(load_15_bp, on="rounded_dt", how="left")

        # Outer join and compute actual net MW
        actual_15 = (
            gen_15.join(load_15, on="rounded_dt", how="outer")
                  .with_columns([
                      pl.col("gen_mw").fill_null(0.0),
                      pl.col("load_mw").fill_null(0.0),
                      pl.col("bp_gen_mw").fill_null(0.0),
                      pl.col("bp_load_mw").fill_null(0.0),
                      (pl.col("gen_mw") - pl.col("load_mw")).alias("actual_mw")
                  ])
        )

        if len(actual_15) == 0:
            return 0.0, {"discharge_mwh": 0.0, "charge_mwh": 0.0, "efficiency": 0.0, "intervals": 0, "rt_gross_revenue": 0.0, "da_spread_revenue": 0.0, "active_days": 0}, pl.DataFrame(), pl.DataFrame()

        # Attach local date/hour to align with DAM hourly awards
        actual_15 = actual_15.with_columns([
            pl.col("rounded_dt").dt.convert_time_zone("America/Chicago").dt.date().alias("local_date"),
            pl.col("rounded_dt").dt.convert_time_zone("America/Chicago").dt.hour().alias("local_hour"),
            pl.col("rounded_dt").dt.epoch("ms").alias("rounded_epoch")
        ])

        # Build DAM hourly schedule for this Gen resource (net MW)
        df_dam = (
            pl.read_parquet(dam_gen_file)
              .filter(pl.col("ResourceName") == gen_resource)
              .select([
                  pl.col("DeliveryDate").cast(pl.Date).alias("local_date"),
                  pl.col("hour").cast(pl.Int32).alias("local_hour"),
                  pl.col("AwardedQuantity").alias("dam_award_mw")
              ])
        )

        if len(df_dam) == 0:
            # No DAM schedule -> entire actual settled at RT (should be rare)
            df_dam = pl.DataFrame({"local_date": [], "local_hour": [], "dam_award_mw": []})

        dev_15 = actual_15.join(df_dam, on=["local_date", "local_hour"], how="left").with_columns([
            pl.col("dam_award_mw").fill_null(0.0),
            (pl.col("actual_mw") - pl.col("dam_award_mw")).alias("deviation_mw")
        ])

        # Align strictly to price availability window for the resource node
        price_bounds = df_prices.select([
            pl.min("price_datetime").alias("min_ts"),
            pl.max("price_datetime").alias("max_ts"),
        ]).row(0)
        min_ts, max_ts = price_bounds
        dev_15 = dev_15.filter((pl.col("rounded_epoch") >= min_ts) & (pl.col("rounded_epoch") <= max_ts))

        # Join RT prices at 15‑min settlement timestamps
        dev_15 = dev_15.join(
            df_prices,
            left_on="rounded_epoch",
            right_on="price_datetime",
            how="left"
        )

        # Filter out rows without prices
        missing_prices = dev_15.filter(pl.col("rt_price").is_null()).select(pl.len()).item()
        dev_15 = dev_15.filter(pl.col("rt_price").is_not_null())

        # Split positive/negative deviations for reporting
        dev_pos = dev_15.with_columns(
            pl.when(pl.col("deviation_mw") > 0).then(pl.col("deviation_mw")).otherwise(0.0).alias("dev_pos")
        )
        dev_neg = dev_15.with_columns(
            pl.when(pl.col("deviation_mw") < 0).then(pl.col("deviation_mw")).otherwise(0.0).alias("dev_neg")
        )

        discharge_revenue = dev_pos.select((pl.col("dev_pos") * pl.col("rt_price") * (15.0/60.0)).sum()).item() or 0.0
        charge_cost = dev_neg.select((pl.col("dev_neg").abs() * pl.col("rt_price") * (15.0/60.0)).sum()).item() or 0.0

        discharge_mwh = dev_pos.select((pl.col("dev_pos") * (15.0/60.0)).sum()).item() or 0.0
        charge_mwh = dev_neg.select((pl.col("dev_neg").abs() * (15.0/60.0)).sum()).item() or 0.0

        total_revenue = discharge_revenue - charge_cost

        if missing_prices > 0:
            logger.warning(f"Excluded {missing_prices:,} 15‑min intervals with missing RT prices for {resource_node}")

        stats = {
            "discharge_mwh": discharge_mwh,  # positive deviations
            "charge_mwh": charge_mwh,        # negative deviations (abs)
            "intervals": len(dev_15),
            "discharge_revenue": discharge_revenue,
            "charge_cost": charge_cost
        }

        # Calculate round-trip efficiency
        if stats["charge_mwh"] > 0:
            stats["efficiency"] = stats["discharge_mwh"] / stats["charge_mwh"]
        else:
            stats["efficiency"] = 0.0

        # Build hourly dispatch aggregates for parquet export
        # Prepare DA price series for hourly/15-min revenue components
        try:
            da_price_series = self._load_da_price_series(resource_node)
            dev_15_da = dev_15.join(da_price_series, on=["local_date", "local_hour"], how="left")
        except Exception:
            dev_15_da = dev_15.with_columns([pl.lit(None).alias("da_price")])

        # 15-min settlement dataset to be exported as-is
        settlement_15min = dev_15_da.select([
            pl.col("rounded_dt").alias("ts_utc"),
            "local_date","local_hour",
            "gen_mw","load_mw","actual_mw","bp_gen_mw","bp_load_mw",
            "dam_award_mw","deviation_mw","rt_price","da_price",
            (pl.col("deviation_mw") * pl.col("rt_price") * (15.0/60.0)).alias("rt_net_revenue_15m"),
            (pl.col("actual_mw") * pl.col("rt_price") * (15.0/60.0)).alias("rt_gross_revenue_15m"),
            (pl.col("dam_award_mw") * (pl.col("da_price") - pl.col("rt_price")) * (15.0/60.0)).alias("da_spread_revenue_15m"),
            (pl.col("dam_award_mw") * pl.col("da_price") * (15.0/60.0)).alias("da_energy_revenue_15m")
        ])

        hourly_dispatch = (
            dev_15_da.group_by(["local_date", "local_hour"]).agg([
                pl.col("gen_mw").mean().alias("actual_gen_mw_avg"),
                (pl.col("gen_mw").sum() * (15.0/60.0)).alias("actual_gen_mwh"),
                pl.col("load_mw").mean().alias("actual_load_mw_avg"),
                (pl.col("load_mw").sum() * (15.0/60.0)).alias("actual_load_mwh"),
                pl.col("actual_mw").mean().alias("net_actual_mw_avg"),
                (pl.col("actual_mw").sum() * (15.0/60.0)).alias("net_actual_mwh"),
                pl.col("bp_gen_mw").mean().alias("basepoint_gen_mw_avg"),
                (pl.col("bp_gen_mw").sum() * (15.0/60.0)).alias("basepoint_gen_mwh"),
                pl.col("bp_load_mw").mean().alias("basepoint_load_mw_avg"),
                (pl.col("bp_load_mw").sum() * (15.0/60.0)).alias("basepoint_load_mwh"),
                pl.col("rt_price").mean().alias("rt_price_avg"),
                pl.col("da_price").mean().alias("da_price_hour"),
                # Exact hourly settlement components
                (pl.col("deviation_mw") * pl.col("rt_price") * (15.0/60.0)).sum().alias("rt_net_revenue_hour"),
                (pl.col("actual_mw") * pl.col("rt_price") * (15.0/60.0)).sum().alias("rt_gross_revenue_hour"),
                (pl.col("dam_award_mw") * (pl.col("da_price") - pl.col("rt_price")) * (15.0/60.0)).sum().alias("da_spread_revenue_hour"),
                (pl.col("dam_award_mw") * pl.col("da_price") * (15.0/60.0)).sum().alias("da_energy_revenue_hour")
            ])
            .with_columns([
                (pl.col("local_date").cast(pl.Utf8) + " " + pl.col("local_hour").cast(pl.Utf8) + ":00:00").str.strptime(pl.Datetime).alias("hour_start_local")
            ])
            .sort(["local_date", "local_hour"])
        )

        return total_revenue, stats, hourly_dispatch, settlement_15min

    def calculate_bess_revenue(
        self,
        bess_name: str,
        gen_resource: str,
        load_resource: str,
        resource_node: str,
        capacity_mw: float
    ) -> Dict:
        """
        Calculate complete revenue for a BESS unit

        Args:
            bess_name: BESS unit name
            gen_resource: Gen Resource name
            load_resource: Load Resource name
            resource_node: Resource Node
            capacity_mw: Nameplate capacity (MW)

        Returns:
            Dictionary with revenue breakdown and statistics
        """
        logger.info(f"Calculating revenue for {bess_name}")

        # 1. DAM Discharge Revenue and DAM Charging Cost
        dam_discharge = self.calculate_dam_discharge_revenue(gen_resource)
        dam_charge_cost = self.calculate_dam_charge_cost(gen_resource, load_resource, resource_node)

        # 2. DAM AS Revenue (Gen side)
        dam_as_gen = self.calculate_dam_as_revenue(gen_resource, is_gen=True)

        # 3. DAM AS Revenue (Load side)
        dam_as_load = self.calculate_dam_as_revenue(load_resource, is_gen=False)

        # 4. RT Net Revenue
        rt_revenue, rt_stats, hourly_dispatch, settlement_15 = self.calculate_rt_net_revenue(
            gen_resource,
            load_resource,
            resource_node
        )

        # Total revenues
        total_dam_as = sum(dam_as_gen.values()) + sum(dam_as_load.values())
        da_net_energy = dam_discharge - dam_charge_cost
        total_revenue = da_net_energy + total_dam_as + rt_revenue

        # Revenue per MW metrics
        revenue_per_mw_year = total_revenue / capacity_mw if capacity_mw > 0 else 0
        revenue_per_mw_month = revenue_per_mw_year / 12

        # Normalized $/kW-year using active days (RT price window)
        active_days = rt_stats.get("active_days", 0)
        if capacity_mw > 0 and active_days and active_days > 0:
            norm_scale = 365.0 / float(active_days)
            normalized_total_per_kw_year = (total_revenue / capacity_mw) * norm_scale
            normalized_energy_per_kw_year = ((da_net_energy + rt_revenue) / capacity_mw) * norm_scale
            normalized_da_per_kw_year = (da_net_energy / capacity_mw) * norm_scale
            normalized_rt_per_kw_year = (rt_revenue / capacity_mw) * norm_scale
            normalized_as_per_kw_year = (total_dam_as / capacity_mw) * norm_scale
        else:
            normalized_total_per_kw_year = 0.0
            normalized_energy_per_kw_year = 0.0
            normalized_da_per_kw_year = 0.0
            normalized_rt_per_kw_year = 0.0
            normalized_as_per_kw_year = 0.0

        # Export hourly dispatch parquet
        try:
            self._export_hourly_dispatch(bess_name, hourly_dispatch)
            self._export_hourly_awards(bess_name, gen_resource, load_resource, resource_node)
            self._export_settlement_15min(bess_name, settlement_15)
        except Exception:
            pass

        return {
            "bess_name": bess_name,
            "gen_resource": gen_resource,
            "load_resource": load_resource,
            "resource_node": resource_node,
            "capacity_mw": capacity_mw,
            "year": self.year,

            # Revenue breakdown
            "dam_discharge_revenue": dam_discharge,
            "dam_charge_cost": dam_charge_cost,
            "da_net_energy": da_net_energy,
            "dam_as_gen_revenue": sum(dam_as_gen.values()),
            "dam_as_load_revenue": sum(dam_as_load.values()),
            "rt_discharge_revenue": rt_stats.get("discharge_revenue", 0.0),
            "rt_charge_cost": rt_stats.get("charge_cost", 0.0),
            "rt_net_revenue": rt_revenue,
            "total_revenue": total_revenue,
            "rt_gross_energy_revenue": rt_stats.get("rt_gross_revenue", 0.0),
            "da_spread_revenue": rt_stats.get("da_spread_revenue", 0.0),

            # Per-MW metrics
            "revenue_per_mw_year": revenue_per_mw_year,
            "revenue_per_mw_month": revenue_per_mw_month,

            # Normalized metrics
            "normalized_total_per_kw_year": normalized_total_per_kw_year,
            "normalized_energy_per_kw_year": normalized_energy_per_kw_year,
            "normalized_da_per_kw_year": normalized_da_per_kw_year,
            "normalized_rt_per_kw_year": normalized_rt_per_kw_year,
            "normalized_as_per_kw_year": normalized_as_per_kw_year,

            # AS breakdown
            **{f"dam_as_gen_{k.lower()}": v for k, v in dam_as_gen.items()},
            **{f"dam_as_load_{k.lower()}": v for k, v in dam_as_load.items()},

            # RT statistics
            "rt_discharge_mwh": rt_stats["discharge_mwh"],
            "rt_charge_mwh": rt_stats["charge_mwh"],
            "rt_efficiency": rt_stats["efficiency"],
            "rt_intervals": rt_stats["intervals"],
            "active_days": active_days
        }

    def _export_hourly_dispatch(self, bess_name: str, hourly_df: pl.DataFrame):
        """Write hourly dispatch parquet for a single BESS."""
        if hourly_df is None or len(hourly_df) == 0:
            return
        out_dir = self.base_dir / "bess_analysis" / "hourly" / "dispatch"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{bess_name}_{self.year}_dispatch.parquet"
        hourly_df.write_parquet(str(out_path))

    def _export_hourly_awards(self, bess_name: str, gen_resource: str, load_resource: str, resource_node: str):
        """Aggregate and write hourly market awards (energy + AS) to parquet."""
        dam_gen_fp = self.rollup_dir / f"DAM_Gen_Resources/{self.year}.parquet"
        dam_load_fp = self.rollup_dir / f"DAM_Load_Resources/{self.year}.parquet"
        as_prices_fp = self.rollup_dir / f"AS_prices/{self.year}.parquet"

        try:
            gen = pl.read_parquet(dam_gen_fp).filter(pl.col("ResourceName") == gen_resource)
        except Exception:
            gen = pl.DataFrame()
        try:
            load = pl.read_parquet(dam_load_fp).filter(pl.col("Load Resource Name") == load_resource)
        except Exception:
            load = pl.DataFrame()
        try:
            as_prices = pl.read_parquet(as_prices_fp)
        except Exception:
            as_prices = pl.DataFrame()

        def col_exists(df: pl.DataFrame, name: str) -> bool:
            return name in df.columns

        # Build gen awards hourly
        parts = []
        if len(gen) > 0:
            g = gen.select([
                pl.col("DeliveryDate").cast(pl.Date).alias("local_date"),
                pl.col("hour").cast(pl.Int32).alias("local_hour"),
                pl.col("AwardedQuantity").alias("da_energy_award_mw"),
                pl.col("RegUpAwarded").fill_null(0.0).alias("regup_mw"),
                pl.col("RegDownAwarded").fill_null(0.0).alias("regdown_mw"),
                (pl.sum_horizontal([
                    pl.col(c).fill_null(0.0) for c in [
                        "RRSPFRAwarded","RRSFFRAwarded","RRSUFRAwarded","RRSAwarded"
                    ] if c in gen.columns
                ])).alias("rrs_mw"),
                (pl.sum_horizontal([
                    pl.col(c).fill_null(0.0) for c in [
                        "ECRSSDAwarded","ECRSMDAwarded","ECRSAwarded"
                    ] if c in gen.columns
                ])).alias("ecrs_mw"),
                pl.col("NonSpinAwarded").fill_null(0.0).alias("nonspin_mw")
            ]).group_by(["local_date","local_hour"]).agg(pl.all().sum()).with_columns([pl.lit("gen").alias("side")])
            parts.append(g)

        if len(load) > 0:
            l = load.select([
                pl.col("DeliveryDate").cast(pl.Date).alias("local_date"),
                pl.col("hour").cast(pl.Int32).alias("local_hour"),
                pl.lit(0.0).alias("da_energy_award_mw"),
                pl.col("RegUp Awarded").fill_null(0.0).alias("regup_mw"),
                pl.col("RegDown Awarded").fill_null(0.0).alias("regdown_mw"),
                (pl.sum_horizontal([
                    pl.col(c).fill_null(0.0) for c in [
                        "RRSPFR Awarded","RRSFFR Awarded","RRSUFR Awarded"
                    ] if c in load.columns
                ])).alias("rrs_mw"),
                (pl.sum_horizontal([
                    pl.col(c).fill_null(0.0) for c in [
                        "ECRSSD Awarded","ECRSMD Awarded"
                    ] if c in load.columns
                ])).alias("ecrs_mw"),
                pl.col("NonSpin Awarded").fill_null(0.0).alias("nonspin_mw")
            ]).group_by(["local_date","local_hour"]).agg(pl.all().sum()).with_columns([pl.lit("load").alias("side")])
            parts.append(l)

        if not parts:
            return

        awards = pl.concat(parts, how="diagonal_relaxed").group_by(["local_date","local_hour"]).agg([
            pl.col("da_energy_award_mw").sum(),
            pl.col("regup_mw").sum(),
            pl.col("regdown_mw").sum(),
            pl.col("rrs_mw").sum(),
            pl.col("ecrs_mw").sum(),
            pl.col("nonspin_mw").sum()
        ]).sort(["local_date","local_hour"]) 

        # Attach AS MCPC hourly (system) if available
        if len(as_prices) > 0:
            mcpc = (as_prices.select([
                pl.col("DeliveryDate").cast(pl.Date).alias("local_date"),
                pl.col("hour").str.slice(0,2).cast(pl.Int32).alias("local_hour"),
                "AncillaryType",
                pl.col("MCPC")
            ]).group_by(["local_date","local_hour","AncillaryType"]).agg(pl.col("MCPC").mean())
               .pivot(values="MCPC", index=["local_date","local_hour"], columns="AncillaryType")
               .rename({"REGUP":"regup_mcpc","REGDN":"regdown_mcpc","RRS":"rrs_mcpc","ECRS":"ecrs_mcpc","NSPIN":"nonspin_mcpc"})
            )
            awards = awards.join(mcpc, on=["local_date","local_hour"], how="left")

        out_dir = self.base_dir / "bess_analysis" / "hourly" / "awards"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{bess_name}_{self.year}_awards.parquet"
        awards.write_parquet(str(out_path))

    def _export_settlement_15min(self, bess_name: str, df_15: pl.DataFrame):
        if df_15 is None or len(df_15) == 0:
            return
        out_dir = self.base_dir / "bess_analysis" / "settlement_15min"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{bess_name}_{self.year}_settlement_15min.parquet"
        df_15.write_parquet(str(out_path))

    def calculate_all_bess(self, mapping_file: str, output_file: str = None) -> pd.DataFrame:
        """
        Calculate revenues for all BESS units in mapping file

        Args:
            mapping_file: Path to BESS mapping CSV
            output_file: Optional output CSV path

        Returns:
            Pandas DataFrame with all BESS revenues
        """
        logger.info(f"Starting revenue calculation for all BESS units ({self.year})")

        # Load mapping
        bess_mapping = self.load_bess_mapping(mapping_file)

        # Calculate revenue for each BESS
        results = []
        total_units = len(bess_mapping)

        for idx, row in enumerate(bess_mapping.iter_rows(named=True), 1):
            logger.info(f"Processing {idx}/{total_units}: {row['Gen_Resource']}")

            try:
                # Validate required data - NO DEFAULTS
                if row['Capacity_MW'] is None or row['Capacity_MW'] == 0:
                    raise ValueError(
                        f"Missing capacity for {row['Gen_Resource']} - cannot calculate per-MW metrics"
                    )

                revenue_data = self.calculate_bess_revenue(
                    bess_name=row['Gen_Resource'],  # Use Gen Resource as name
                    gen_resource=row['Gen_Resource'],
                    load_resource=row['Load_Resource'],
                    resource_node=row['Resource_Node'],
                    capacity_mw=float(row['Capacity_MW'])
                )
                results.append(revenue_data)

            except Exception as e:
                logger.error(f"Error processing {row['Gen_Resource']}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                # Add placeholder with error
                results.append({
                    "bess_name": row['Gen_Resource'],
                    "gen_resource": row['Gen_Resource'],
                    "load_resource": row['Load_Resource'],
                    "error": str(e),
                    "total_revenue": 0.0
                })

        # Convert to DataFrame
        df = pd.DataFrame(results)

        # Sort by total revenue
        df = df.sort_values('total_revenue', ascending=False)

        # Save if output file specified
        if output_file:
            df.to_csv(output_file, index=False)
            logger.info(f"✅ Results saved to {output_file}")

        # Print summary
        self._print_summary(df)

        return df

    def _print_summary(self, df: pd.DataFrame):
        """Print revenue summary statistics"""
        print("\n" + "="*80)
        print(f"BESS Revenue Analysis Summary - {self.year}")
        print("="*80)

        print(f"\nTotal BESS Units Analyzed: {len(df)}")
        print(f"Total Revenue: ${df['total_revenue'].sum():,.2f}")
        print(f"Average Revenue per BESS: ${df['total_revenue'].mean():,.2f}")
        print(f"Median Revenue per BESS: ${df['total_revenue'].median():,.2f}")

        print("\n--- Revenue Breakdown ---")
        print(f"DAM Discharge: ${df['dam_discharge_revenue'].sum():,.2f}")
        print(f"DAM AS (Gen): ${df['dam_as_gen_revenue'].sum():,.2f}")
        print(f"DAM AS (Load): ${df['dam_as_load_revenue'].sum():,.2f}")
        print(f"RT Net: ${df['rt_net_revenue'].sum():,.2f}")

        print("\n--- Top 10 BESS by Revenue ---")
        print(df[['bess_name', 'capacity_mw', 'total_revenue', 'revenue_per_mw_year']].head(10).to_string(index=False))

        print("\n--- Bottom 10 BESS by Revenue ---")
        print(df[['bess_name', 'capacity_mw', 'total_revenue', 'revenue_per_mw_year']].tail(10).to_string(index=False))

        print("\n" + "="*80)


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Calculate historical BESS revenues from ERCOT market data"
    )
    parser.add_argument(
        '--year',
        type=int,
        default=2024,
        help='Year to analyze (default: 2024)'
    )
    parser.add_argument(
        '--mapping',
        type=str,
        default='bess_mapping/BESS_UNIFIED_MAPPING_V3_CLARIFIED.csv',
        help='Path to BESS mapping file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV file path (default: bess_revenue_{year}.csv)'
    )
    parser.add_argument(
        '--base-dir',
        type=str,
        default=os.getenv('ERCOT_DATA_DIR', '/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data'),
        help='Base directory for ERCOT data'
    )

    args = parser.parse_args()

    # Set default output filename
    if args.output is None:
        args.output = f"bess_revenue_{args.year}.csv"

    # Create calculator
    calc = BESSRevenueCalculator(
        base_dir=args.base_dir,
        year=args.year
    )

    # Run calculation
    results = calc.calculate_all_bess(
        mapping_file=args.mapping,
        output_file=args.output
    )

    logger.info("✅ Revenue calculation complete!")


if __name__ == "__main__":
    main()
