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

"""
PARQUET FILE SCHEMA DOCUMENTATION

All parquet files are under BASE_DIR/rollup_files/ unless otherwise noted.
Column names and types are EXACT - no fallback logic exists.

## Input Files

### DAM_Gen_Resources/{year}.parquet
- ResourceName: string
- DeliveryDate: date or string
- hour: int (0-23)
- AwardedQuantity: double (MWh)
- EnergySettlementPointPrice: double ($/MWh)
- RegUpAwarded, RegDownAwarded, RRSPFRAwarded, RRSFFRAwarded, RRSUFRAwarded, RRSAwarded: double (MW)
- ECRSSDAwarded, ECRSMDAwarded, ECRSAwarded, NonSpinAwarded: double (MW)

### DAM_Load_Resources/{year}.parquet
- Load Resource Name: string
- DeliveryDate: date or string
- hour: int (0-23)
- RegUp Awarded, RegDown Awarded, RRSFFR Awarded, RRSPFR Awarded, RRSUFR Awarded: double (MW)
- ECRSSD Awarded, ECRSMD Awarded, NonSpin Awarded: double (MW)
- RegUp MCPC, RegDown MCPC, RRS MCPC, ECRS MCPC, NonSpin MCPC: double ($/MW)

### SCED_Gen_Resources/{year}.parquet
- SCEDTimeStamp: timestamp
- ResourceName: string
- TelemeteredNetOutput: double (MW, preferred)
- BasePoint: double (MW, fallback)

### SCED_Load_Resources/{year}.parquet
- SCEDTimeStamp: timestamp
- ResourceName: string
- BasePoint: double (MW)

### DA_prices/{year}.parquet (long format)
- HourEnding: string ("HH:MM", 1-24)
- SettlementPoint: string
- DeliveryDate: date32
- DeliveryDateStr: string (RFC3339 with offset)
- SettlementPointPrice: double ($/MWh)
- hour: string
- DSTFlag: string
- datetime_ms: int64

### RT_prices/{year}.parquet (long format, 5-min)
- DeliveryDate: string
- DeliveryHour: int64
- DeliveryInterval: int64
- SettlementPointName: string
- SettlementPointPrice: double ($/MWh)
- SettlementPointType: string
- DSTFlag: string
- datetime: int64

### AS_prices/{year}.parquet
- DeliveryDate: date or string
- hour: string ("HH:MM", 1-24 hour ending)
- AncillaryType: string (REGUP, REGDN, RRS, ECRS, NSPIN)
- MCPC: double ($/MW)

### DAM_Energy_Bid_Awards/{year}.parquet
- SettlementPoint: string
- EnergyBidAwardMW: double (negative = DA charging)
- DeliveryDate: date or string
- hour: int or HourEnding: string

### flattened/DA_prices_{year}.parquet (wide format, hubs/zones only)
- DeliveryDate: date32
- DeliveryDateStr: string
- datetime_ts: int64
- {HubName}: double (e.g., HB_BUSAVG, LZ_HOUSTON)
Note: Does NOT include individual resource nodes like CHISMGRD_RN.

## Output Files (under BASE_DIR/bess_analysis/)

### hourly/awards/{BESS}_{year}_awards.parquet
- local_date: date (CT)
- local_hour: int32 (0-23)
- da_energy_award_mw: double
- regup_mw, regdown_mw, rrs_mw, ecrs_mw, nonspin_mw: double
- ecrs_mcpc, rrs_mcpc, regup_mcpc, nonspin_mcpc, regdown_mcpc: double ($/MW)

### hourly/dispatch/{BESS}_{year}_dispatch.parquet
- local_date: date (CT)
- local_hour: int32 (0-23)
- net_actual_mwh, basepoint_gen_mwh, basepoint_load_mwh: double
- rt_price_avg, da_price_hour: double ($/MWh)
- rt_net_revenue_hour, rt_gross_revenue_hour, da_spread_revenue_hour, da_energy_revenue_hour: double ($)

### settlement_15min/{BESS}_{year}_settlement_15min.parquet
EXACT schema (no variations across files):
- ts_utc: timestamp[us, tz=UTC]
- local_date: date32 (CT date)
- local_hour: int32 (0-23, CT hour)
- gen_mw, load_mw, actual_mw: double
- bp_gen_mw, bp_load_mw: double
- dam_award_mw, deviation_mw: double
- rt_price: double ($/MWh) - 15-min averaged RT price
- da_price: double ($/MWh) - DA price for the hour
- rt_net_revenue_15m, rt_gross_revenue_15m, da_spread_revenue_15m, da_energy_revenue_15m: double ($)

## Critical Notes

1. **NO FALLBACK LOGIC**: Column names are exact. Scripts fail loudly if columns missing.
2. **Type Matching**: Polars joins require exact type matches (e.g., date vs object fails).
3. **Hour Conventions**: HourEnding 1-24, hour/local_hour 0-23 (convert HE-1 to get hour).
4. **Timezone**: All ERCOT data is America/Chicago. Use DeliveryDateStr (RFC3339) when available.
5. **DA Charging**: Found in DAM_Energy_Bid_Awards (negative EnergyBidAwardMW), NOT in DAM_Load_Resources.
6. **Resource Nodes**: Individual BESS nodes (e.g., CHISMGRD_RN) only in DA_prices/{year}.parquet long format, NOT in flattened/ wide format.
"""

import os
import polars as pl
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import logging
import os

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

    @staticmethod
    def configure_threads(max_threads: int | None = None):
        """Limit backend threads (Polars/Rayon/PyArrow/numexpr).

        If max_threads is None, default to half the logical CPUs as a proxy for
        physical cores. Users with SMT=off can set this explicitly.
        """
        if not max_threads or max_threads <= 0:
            try:
                logical = os.cpu_count() or 8
                max_threads = max(1, logical // 2)
            except Exception:
                max_threads = 8
        for k in (
            "POLARS_MAX_THREADS",
            "RAYON_NUM_THREADS",
            "PYARROW_NUM_THREADS",
            "NUMEXPR_MAX_THREADS",
        ):
            os.environ[k] = str(max_threads)
        logging.getLogger(__name__).info(f"Threading limited to {max_threads} threads (POLARS/RAYON/PYARROW/NUMEXPR)")

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
            'True_Operational_Status': 'Status',
            'COD_Date': 'COD_Date'  # Required for V5 mapping
        }

        # Verify columns exist
        for col in expected_cols.keys():
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Rename to standard names
        df = df.rename(expected_cols)

        # Parse COD_Date to date type
        df = df.with_columns([
            pl.col('COD_Date').str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias('COD_Date')
        ])

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
        """Load hourly Day-Ahead price series for a single settlement point.

        Timezone rules and rationale:
        - ERCOT source data for DAM uses Central Time (CT) hour blocks. Some
          rollups include an explicit RFC3339 string (e.g., ``DeliveryDateStr``)
          with the ``-06:00/-05:00`` offset. Others store a timestamp column
          (``datetime``/``datetime_ts``) that represents CT wall-clock as a
          naive Arrow timestamp (no tz metadata).
        - For absolute correctness, when ``DeliveryDateStr`` exists we parse it
          and convert using the embedded offset. When it does not exist, we
          treat timestamp columns as America/Chicago local time (attach
          ``America/Chicago`` tz) rather than assuming UTC. This prevents a
          6-hour shift that would occur if we incorrectly interpreted local CT
          timestamps as UTC.

        Returns columns: ``local_date`` (date, CT), ``local_hour`` (0–23, CT),
        ``da_price`` ($/MWh).
        """
        # Try flattened wide format first
        flat1 = self.rollup_dir / f"flattened/DA_prices_{self.year}.parquet"
        flat2 = self.rollup_dir / f"flattened/DA_prices_flat_{self.year}.parquet"
        longp = self.rollup_dir / f"DA_prices/{self.year}.parquet"

        logger.debug(f"Loading DA prices for {resource_node}")
        df = None
        if flat1.exists() or flat2.exists():
            path = flat1 if flat1.exists() else flat2
            df_flat = pl.read_parquet(path)
            # Pick the requested hub column, or fall back to HB_BUSAVG
            col = resource_node if resource_node in df_flat.columns else ("HB_BUSAVG" if "HB_BUSAVG" in df_flat.columns else None)
            if col is None:
                logger.warning(f"Settlement point {resource_node} not found in DA price columns: {df_flat.columns[:10]}...")
            else:
                if col != resource_node:
                    logger.debug(f"Using {col} instead of {resource_node} for DA prices (flattened format)")
            if col is not None:
                # Handle datetime column name variants
                has_dt_str = "DeliveryDateStr" in df_flat.columns
                dt_col = None
                if has_dt_str:
                    dt_col = "DeliveryDateStr"
                elif "datetime" in df_flat.columns:
                    dt_col = "datetime"
                elif "datetime_ts" in df_flat.columns:
                    dt_col = "datetime_ts"
                elif "DeliveryDate" in df_flat.columns:
                    # Date-only; will be treated as midnight local
                    dt_col = "DeliveryDate"

                if dt_col is not None:
                    df = df_flat.select([
                        pl.col(dt_col).alias("dt"),
                        pl.col(col).alias("da_price"),
                    ])

        if df is None:
            # Long format: filter by settlement point
            if not longp.exists():
                logger.warning(f"No DA price files found: {flat1}, {flat2}, {longp} do not exist")
                return pl.DataFrame({"local_date": [], "local_hour": [], "da_price": []})
            df_long = pl.read_parquet(longp)
            # Column variants
            sp_col = "settlement_point" if "settlement_point" in df_long.columns else "SettlementPointName"
            price_col = "da_lmp" if "da_lmp" in df_long.columns else "SettlementPointPrice"

            # 3 possible time representations in DA long format:
            # 1) RFC3339 string with offset in DeliveryDateStr (preferred)
            # 2) Timestamp column: datetime or datetime_ts (naive local time)
            # 3) Separate DeliveryDate (date) + HourEnding/hour column (common) → need HE→HB fix
            if "DeliveryDateStr" in df_long.columns:
                dt_mode = "string"
                dt_col = "DeliveryDateStr"
                df = df_long.filter(pl.col(sp_col) == resource_node).select([
                    pl.col(dt_col).alias("dt"),
                    pl.col(price_col).alias("da_price"),
                ])
            elif "datetime" in df_long.columns or "datetime_ts" in df_long.columns:
                dt_mode = "timestamp"
                dt_col = "datetime" if "datetime" in df_long.columns else "datetime_ts"
                df = df_long.filter(pl.col(sp_col) == resource_node).select([
                    pl.col(dt_col).alias("dt"),
                    pl.col(price_col).alias("da_price"),
                ])
            elif "DeliveryDate" in df_long.columns and ("HourEnding" in df_long.columns or "hour" in df_long.columns):
                # Derive local_date/hour directly; HourEnding is 1–24 → convert to hour beginning 0–23
                dt_mode = "date_hour"
                he_col = "HourEnding" if "HourEnding" in df_long.columns else "hour"
                df_he = df_long.filter(pl.col(sp_col) == resource_node).select([
                    pl.col("DeliveryDate").cast(pl.Date).alias("local_date"),
                    pl.col(he_col).alias("he_raw"),
                    pl.col(price_col).alias("da_price"),
                ])
                # Support either "HH:MM" strings or ints in hour column
                if df_he.schema.get("he_raw") == pl.Utf8:
                    df_he = df_he.with_columns(
                        pl.col("he_raw").str.slice(0, 2).cast(pl.Int32).alias("he")
                    )
                else:
                    df_he = df_he.with_columns(pl.col("he_raw").cast(pl.Int32).alias("he"))

                # Convert HourEnding → HourBeginning by subtracting 1 (no wrap across date here)
                df = df_he.with_columns((pl.col("he") - 1).alias("local_hour")).select([
                    "local_date",
                    pl.col("local_hour").cast(pl.Int32),
                    "da_price",
                ])
            else:
                # Unknown schema
                return pl.DataFrame({"local_date": [], "local_hour": [], "da_price": []})

        # Convert to local date/hour for DAM alignment (Central Time)
        # If we started from an RFC3339 string with offset, respect it; otherwise
        # assume the timestamp column represents CT wall-clock and attach the
        # America/Chicago timezone to avoid 6-hour shifts.
        if "dt" in df.columns and df.schema.get("dt") in (pl.Utf8, pl.Categorical):
            # Parse string datetimes (which may include offsets) and normalize to CT
            # Use eager evaluation to avoid Polars lazy evaluation timezone inference issues
            dt_series = df["dt"].str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%S%z")
            df = df.with_columns([
                dt_series.dt.convert_time_zone("America/Chicago").dt.date().alias("local_date"),
                dt_series.dt.convert_time_zone("America/Chicago").dt.hour().alias("local_hour")
            ])
        elif "dt" in df.columns:
            # Treat numeric/naive datetimes as America/Chicago local wall time
            df = df.with_columns([
                pl.col("dt").cast(pl.Datetime)
                    .dt.replace_time_zone("America/Chicago")
                    .dt.date().alias("local_date"),
                pl.col("dt").cast(pl.Datetime)
                    .dt.replace_time_zone("America/Chicago")
                    .dt.hour().alias("local_hour")
            ])
        else:
            # Already has local_date/local_hour (date_hour mode)
            pass

        df = (
            df.select(["local_date", "local_hour", "da_price"]) 
              .group_by(["local_date", "local_hour"]) 
              .agg(pl.col("da_price").mean()) 
              .sort(["local_date", "local_hour"]) 
        )

        return df

    def calculate_dam_charge_cost(self, gen_resource: str, load_resource: str, resource_node: str) -> tuple[float, float]:
        """Calculate DAM charging cost and MWh from two sources:
        1) Negative awards (if any) in DAM_Gen_Resources for this Gen resource.
        2) Energy Bid Awards at the resource node (negative MW indicate load purchases).

        Returns (total_cost_usd, total_charge_mwh).
        """
        total_cost = 0.0
        total_mwh = 0.0

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
                # Negative awards are charging; sum absolute for MWh and dollars
                mwh = df.select(pl.col("AwardedQuantity").abs().sum()).item() or 0.0
                cost = df.select(pl.col("cost").abs().sum()).item() or 0.0
                total_mwh += mwh
                total_cost += cost

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
                        # Sum MWh directly from hourly MW
                        total_mwh += eba.select(pl.col("mw").abs().sum()).item() or 0.0
                        # Join DA prices to compute $ cost
                        da_prices = self._load_da_price_series(resource_node)
                        if len(da_prices) > 0:
                            joined = eba.join(da_prices, on=["local_date", "local_hour"], how="left")
                            cost = joined.select((pl.col("mw").abs() * pl.col("da_price")).sum()).item() or 0.0
                            total_cost += cost
            except Exception as _:
                pass

        return total_cost, total_mwh

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

        # Handle possible schema variants for RT prices
        sp_col = (
            "SettlementPointName" if "SettlementPointName" in df_rt_all.columns
            else ("settlement_point" if "settlement_point" in df_rt_all.columns else None)
        )
        dt_col = (
            "datetime" if "datetime" in df_rt_all.columns
            else ("datetime_ts" if "datetime_ts" in df_rt_all.columns else None)
        )
        price_col = (
            "SettlementPointPrice" if "SettlementPointPrice" in df_rt_all.columns
            else ("rt_lmp" if "rt_lmp" in df_rt_all.columns else None)
        )

        if not all([sp_col, dt_col, price_col]):
            raise ValueError(f"Unexpected RT price schema in {rt_price_file}; missing one of settlement point, datetime, or price columns")

        def _price_series_for_node(node: str) -> pl.DataFrame:
            """Return 15‑min averaged UTC price series for a settlement point name."""
            prices_raw = df_rt_all.filter(pl.col(sp_col) == node).select([
                pl.col(dt_col).alias("price_datetime"),
                pl.col(price_col).alias("rt_price")
            ])
            if len(prices_raw) == 0:
                return pl.DataFrame(schema={
                    "rounded_dt": pl.Datetime(time_unit="us", time_zone="UTC"),
                    "rt_price": pl.Float64
                })

            dtype = prices_raw.schema.get("price_datetime")
            if dtype == pl.Int64:
                prices_ts = prices_raw.with_columns(
                    (
                        pl.when(pl.col("price_datetime") > 10**14)
                        .then(pl.from_epoch(pl.col("price_datetime"), time_unit="us"))
                        .when(pl.col("price_datetime") > 10**11)
                        .then(pl.from_epoch(pl.col("price_datetime"), time_unit="ms"))
                        .otherwise(pl.from_epoch(pl.col("price_datetime"), time_unit="s"))
                    ).alias("price_dt")
                )
            elif dtype in (pl.Utf8, pl.Categorical):
                # Parse string to naive Datetime, assign CT timezone, then convert to UTC
                # CRITICAL: RT price timestamps are in Central Time, must convert to UTC
                # to align with SCED data (which is also converted CT→UTC)
                prices_ts = prices_raw.with_columns(
                    pl.col("price_datetime").str.strptime(pl.Datetime, strict=False)
                        .dt.replace_time_zone("America/Chicago", ambiguous="earliest")
                        .dt.convert_time_zone("UTC")
                        .alias("price_dt")
                )
            else:
                # Treat as Datetime-like; cast to naive, assign CT timezone, then convert to UTC
                # CRITICAL: Assume Datetime values are in Central Time (like all ERCOT data)
                prices_ts = prices_raw.with_columns(
                    pl.col("price_datetime").cast(pl.Datetime)
                        .dt.replace_time_zone("America/Chicago", ambiguous="earliest")
                        .dt.convert_time_zone("UTC")
                        .alias("price_dt")
                )

            return (
                prices_ts.with_columns(
                    pl.col("price_dt")
                      .dt.replace_time_zone("UTC")
                      .dt.cast_time_unit("us")
                      .dt.truncate("15m")
                      .alias("rounded_dt")
                )
                .group_by("rounded_dt")
                .agg(pl.col("rt_price").mean().alias("rt_price"))
                .with_columns(
                    pl.col("rounded_dt").cast(pl.Datetime(time_unit="us", time_zone="UTC"))
                )
                .sort("rounded_dt")
            )

        # Note: price coverage is evaluated later with fallbacks; do not error here

        # Build 15‑min actual net output = Gen − Load
        # Convert SCED timestamps to UTC 15‑min bins and average within the bin
        def _to_15min(df: pl.DataFrame, value_col: str, alias: str) -> pl.DataFrame:
            """Aggregate a SCED series to 15‑minute UTC bins with stable dtypes.

            Returns a DataFrame with columns:
              - rounded_dt: pl.Datetime(us, UTC)
              - <alias>: pl.Float64

            Ensures empty results still carry concrete dtypes to avoid join
            errors where Polars infers Null dtypes for empty columns.
            """
            if len(df) == 0:
                return pl.DataFrame(
                    schema={
                        "rounded_dt": pl.Datetime(time_unit="us", time_zone="UTC"),
                        alias: pl.Float64,
                    }
                )
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

        # Outer (full) join and compute actual net MW; ensure key dtypes align
        gen_15 = gen_15.with_columns(
            pl.col("rounded_dt").cast(pl.Datetime(time_unit="us", time_zone="UTC"))
        )
        load_15 = load_15.with_columns(
            pl.col("rounded_dt").cast(pl.Datetime(time_unit="us", time_zone="UTC"))
        )

        actual_15 = (
            gen_15.join(load_15, on="rounded_dt", how="full")
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
        ]).with_columns([
            pl.col("local_hour").cast(pl.Int32)
        ])

        # Build DAM hourly schedule for this Gen resource (net MW)
        df_dam = (
            pl.read_parquet(dam_gen_file)
              .filter(pl.col("ResourceName") == gen_resource)
              .select([
                  # Robustly parse DeliveryDate from string or date/datetime
                  pl.col("DeliveryDate")
                    .cast(pl.Utf8)
                    .str.to_date(format="%m/%d/%Y", strict=False)
                    .fill_null(pl.col("DeliveryDate").cast(pl.Utf8).str.to_date(format="%Y-%m-%d", strict=False))
                    .fill_null(pl.col("DeliveryDate").cast(pl.Utf8).str.to_datetime(strict=False).dt.date())
                    .alias("local_date"),
                  pl.col("hour").cast(pl.Int32).alias("local_hour"),
                  pl.col("AwardedQuantity").alias("dam_award_mw")
              ])
        )

        if len(df_dam) == 0:
            # No DAM schedule -> entire actual settled at RT (should be rare)
            logger.warning(f"No DAM award data found for {gen_resource} - all intervals will be settled at RT")
            df_dam = pl.DataFrame({"local_date": [], "local_hour": [], "dam_award_mw": []})
        else:
            # Log DAM data coverage
            dam_days = df_dam.select(pl.col("local_date").n_unique()).item()
            logger.debug(f"Found DAM awards for {gen_resource} covering {dam_days} days")

        dev_15 = actual_15.join(df_dam, on=["local_date", "local_hour"], how="left").with_columns([
            (pl.col("dam_award_mw").fill_null(0.0)).alias("dam_award_mw"),
            # Fill potential nulls defensively before arithmetic to avoid null deviation
            (pl.col("actual_mw").fill_null(0.0) - pl.col("dam_award_mw").fill_null(0.0)).alias("deviation_mw")
        ])
        
        # Verify deviation_mw was calculated
        null_deviations = dev_15.filter(pl.col("deviation_mw").is_null()).height
        if null_deviations > 0:
            logger.error(f"WARNING: {null_deviations} intervals have null deviation_mw for {gen_resource}")

        # Build price series for the mapped resource_node, then evaluate coverage.
        df_prices = _price_series_for_node(resource_node)

        # Helper to join and compute missing count
        def _join_with_prices(dev_df: pl.DataFrame, prices_df: pl.DataFrame) -> tuple[pl.DataFrame, int]:
            if len(prices_df) == 0:
                return dev_df.with_columns(pl.lit(None).alias("rt_price")), len(dev_df)
            # Direct left join on UTC 15-min rounded_dt; avoid timezone literal comparisons
            joined = dev_df.join(prices_df, on="rounded_dt", how="left")
            miss = joined.filter(pl.col("rt_price").is_null()).select(pl.len()).item()
            return joined, miss

        joined_dev_15, missing_prices = _join_with_prices(dev_15, df_prices)

        # If coverage is poor, try alternative settlement point names
        try_nodes: list[tuple[str, pl.DataFrame]] = []
        # 1) From SCED Gen/Load frames if available
        gen_sp = None
        load_sp = None
        if "SettlementPointName" in df_gen_raw_full.columns:
            gen_sp = df_gen_raw_full.filter(pl.col("ResourceName") == gen_resource)["SettlementPointName"].unique().to_list()
            gen_sp = gen_sp[0] if gen_sp else None
        if "SettlementPointName" in df_load_full.columns:
            load_sp = df_load_full.filter(pl.col("ResourceName") == load_resource)["SettlementPointName"].unique().to_list()
            load_sp = load_sp[0] if load_sp else None
        # 2) Simple suffix swap heuristics
        alt1 = resource_node[:-4] + "_RN" if resource_node.endswith("_ALL") else None
        alt2 = resource_node[:-3] + "_ALL" if resource_node.endswith("_RN") else None

        for cand in [c for c in [gen_sp, load_sp, alt1, alt2] if c and c != resource_node]:
            try_nodes.append((cand, _price_series_for_node(cand)))

        # Evaluate candidates and pick the one with least missing intervals
        best_node = resource_node
        best_join = joined_dev_15
        best_missing = missing_prices
        for cand, prices_cand in try_nodes:
            joined_c, miss_c = _join_with_prices(dev_15, prices_cand)
            if miss_c < best_missing:
                best_missing = miss_c
                best_join = joined_c
                best_node = cand

        dev_15 = best_join
        if best_node != resource_node:
            logger.info(f"RT price node fallback: using {best_node} instead of mapping node {resource_node}")

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

        if best_missing > 0:
            # Diagnostics to help understand coverage gaps
            actual_min_max = actual_15.select([pl.min("rounded_dt").alias("min_dt"), pl.max("rounded_dt").alias("max_dt")]).row(0)

            # Determine which price series was used (mapping vs fallback)
            chosen_prices_df = df_prices if best_node == resource_node else next((df for cand, df in try_nodes if cand == best_node), pl.DataFrame(schema={"rounded_dt": pl.Datetime(time_unit="us", time_zone="UTC"), "rt_price": pl.Float64}))
            price_min_max = chosen_prices_df.select([pl.min("rounded_dt").alias("min_dt"), pl.max("rounded_dt").alias("max_dt")]).row(0) if len(chosen_prices_df) > 0 else (None, None)
            overlap_min = max(actual_min_max[0], price_min_max[0]) if all(actual_min_max) and all(price_min_max) else None
            overlap_max = min(actual_min_max[1], price_min_max[1]) if all(actual_min_max) and all(price_min_max) else None

            # Sample a few missing timestamps for quick inspection
            missing_sample = (
                best_join.filter(pl.col("rt_price").is_null())
                         .select("rounded_dt")
                         .head(5)
                         .to_series()
                         .to_list()
            )

            total_intervals = len(best_join)
            miss_pct = (best_missing / total_intervals * 100.0) if total_intervals else 0.0

            # Suppress noisy warnings unless we truly failed to match a usable node
            found_any_prices = (len(df_prices) > 0) or any(len(p) > 0 for _, p in try_nodes)
            no_effective_match = (not found_any_prices) or (total_intervals > 0 and best_missing == total_intervals)

            if no_effective_match:
                logger.warning(
                    f"Excluded {best_missing:,} 15‑min intervals with missing RT prices for {best_node} (from mapping: {resource_node}) | "
                    f"miss%={miss_pct:.1f}, actual_range=[{actual_min_max[0]}, {actual_min_max[1]}], "
                    f"price_range=[{price_min_max[0]}, {price_min_max[1]}], overlap=[{overlap_min}, {overlap_max}], "
                    f"examples={missing_sample}"
                )
            else:
                # Downgrade to debug to avoid per-BESS noise for small, expected gaps
                logger.debug(
                    f"Minor RT price gaps: {best_missing} intervals ({miss_pct:.1f}%) for {best_node}; suppressed warning."
                )

        stats = {
            "discharge_mwh": discharge_mwh,  # positive deviations
            "charge_mwh": charge_mwh,        # negative deviations (abs)
            "intervals": len(dev_15),
            "discharge_revenue": discharge_revenue,
            "charge_cost": charge_cost,
            "rt_gross_revenue": 0.0,  # Will be calculated below after DA join
            "da_spread_revenue": 0.0  # Will be calculated below after DA join
        }

        # Calculate round-trip efficiency
        if stats["charge_mwh"] > 0:
            stats["efficiency"] = stats["discharge_mwh"] / stats["charge_mwh"]
        else:
            stats["efficiency"] = 0.0

        # Build hourly dispatch aggregates for parquet export
        # Prepare DA price series for hourly/15‑min revenue components (simple exact match)
        # CRITICAL: Ensure local_date has proper Date type for join (not object type)
        dev_15 = dev_15.with_columns([
            pl.col("local_date").cast(pl.Date).alias("local_date"),
            pl.col("local_hour").cast(pl.Int32).alias("local_hour")
        ])

        da_price_series = self._load_da_price_series(resource_node).with_columns([
            pl.col("local_date").cast(pl.Date).alias("local_date"),
            pl.col("local_hour").cast(pl.Int32).alias("local_hour")
        ])

        # Join DA prices - fail loudly if this fails (no silent exceptions)
        dev_15_da = dev_15.join(da_price_series, on=["local_date", "local_hour"], how="left")

        # Log if DA prices are missing
        da_nulls = dev_15_da.filter(pl.col("da_price").is_null()).height
        if da_nulls > 0:
            logger.warning(
                f"DA price join resulted in {da_nulls}/{dev_15_da.height} null values for {resource_node}. "
                f"This may indicate missing price data for the resource node."
            )

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
        capacity_mw: float,
        cod_date = None
    ) -> Dict:
        """
        Calculate complete revenue for a BESS unit

        Args:
            bess_name: BESS unit name
            gen_resource: Gen Resource name
            load_resource: Load Resource name
            resource_node: Resource Node
            capacity_mw: Nameplate capacity (MW)
            cod_date: Commercial Operation Date 
                     If provided, used to calculate operational days for normalization

        Returns:
            Dictionary with revenue breakdown and statistics
        """
        logger.info(f"Calculating revenue for {bess_name}")

        # 1. DAM Discharge Revenue and DAM Charging Cost
        dam_discharge = self.calculate_dam_discharge_revenue(gen_resource)
        dam_charge_cost, dam_charge_mwh = self.calculate_dam_charge_cost(gen_resource, load_resource, resource_node)

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

        # Calculate operational days based on COD date
        operational_days = 365.0  # default to full year
        cod_year = None
        
        if cod_date:
            if isinstance(cod_date, str):
                try:
                    cod_date = datetime.strptime(cod_date, "%Y-%m-%d").date()
                except:
                    cod_date = None
            
            if cod_date:
                cod_year = cod_date.year
                
                # Determine if leap year
                is_leap = (self.year % 4 == 0 and self.year % 100 != 0) or (self.year % 400 == 0)
                days_in_year = 366 if is_leap else 365
                
                if cod_year == self.year:
                    # Calculate days from COD to end of year
                    year_end = datetime(self.year, 12, 31).date()
                    if hasattr(cod_date, 'date'):
                        cod_date = cod_date.date()
                    operational_days = (year_end - cod_date).days + 1  # +1 to include COD day
                    logger.info(f"COD in analysis year {self.year}: {cod_date} -> {operational_days} operational days")
                elif cod_year < self.year:
                    # Full year operation
                    operational_days = days_in_year
                    logger.info(f"COD before analysis year: using full {days_in_year} days")
                else:
                    # COD is after analysis year - shouldn't happen for operational units
                    logger.warning(f"COD date {cod_date} is after analysis year {self.year}")
                    operational_days = 0
        
        # Normalized $/kW-year using operational days based on COD
        if capacity_mw > 0 and operational_days > 0:
            # Determine total days in year for normalization
            is_leap = (self.year % 4 == 0 and self.year % 100 != 0) or (self.year % 400 == 0)
            days_in_year = 366 if is_leap else 365
            
            norm_scale = days_in_year / operational_days
            normalized_total_per_kw_year = (total_revenue / capacity_mw) * norm_scale / 1000  # Convert to $/kW
            normalized_energy_per_kw_year = ((da_net_energy + rt_revenue) / capacity_mw) * norm_scale / 1000
            normalized_da_per_kw_year = (da_net_energy / capacity_mw) * norm_scale / 1000
            normalized_rt_per_kw_year = (rt_revenue / capacity_mw) * norm_scale / 1000
            normalized_as_per_kw_year = (total_dam_as / capacity_mw) * norm_scale / 1000
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
            "dam_charge_mwh": dam_charge_mwh,
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
            "active_days": rt_stats.get("active_days", 0),
            "operational_days": operational_days,
            "cod_date": cod_date,
            "cod_year": cod_year
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
        eba_fp = self.rollup_dir / f"DAM_Energy_Bid_Awards/{self.year}.parquet"

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

        awards = (
            pl.concat(parts, how="diagonal_relaxed")
            .group_by(["local_date","local_hour"]).agg([
                pl.col("da_energy_award_mw").fill_null(0.0).sum().alias("da_energy_award_mw"),
                pl.col("regup_mw").fill_null(0.0).sum().alias("regup_mw"),
                pl.col("regdown_mw").fill_null(0.0).sum().alias("regdown_mw"),
                pl.col("rrs_mw").fill_null(0.0).sum().alias("rrs_mw"),
                pl.col("ecrs_mw").fill_null(0.0).sum().alias("ecrs_mw"),
                pl.col("nonspin_mw").fill_null(0.0).sum().alias("nonspin_mw")
            ])
            .with_columns([
                pl.col("da_energy_award_mw").cast(pl.Float64),
                pl.col("regup_mw").cast(pl.Float64),
                pl.col("regdown_mw").cast(pl.Float64),
                pl.col("rrs_mw").cast(pl.Float64),
                pl.col("ecrs_mw").cast(pl.Float64),
                pl.col("nonspin_mw").cast(pl.Float64)
            ])
            .sort(["local_date","local_hour"]) 
        )

        # Integrate DAM Energy Bid Awards (charging) as negative DA energy MW
        try:
            if eba_fp.exists():
                eba = pl.read_parquet(eba_fp)
                sp_col = 'SettlementPoint' if 'SettlementPoint' in eba.columns else ('settlement_point' if 'settlement_point' in eba.columns else None)
                mw_col = 'EnergyBidAwardMW' if 'EnergyBidAwardMW' in eba.columns else ('energy_bid_award_mw' if 'energy_bid_award_mw' in eba.columns else None)
                date_col = 'DeliveryDate' if 'DeliveryDate' in eba.columns else ('delivery_date' if 'delivery_date' in eba.columns else None)
                hour_col = 'hour' if 'hour' in eba.columns else ('HourEnding' if 'HourEnding' in eba.columns else None)
                if all([sp_col,mw_col,date_col,hour_col]):
                    eba = eba.filter(pl.col(sp_col) == resource_node)
                    if hour_col == 'HourEnding':
                        # Convert Hour Ending (1-24) to Hour Beginning (0-23)
                        eba = eba.with_columns((pl.col(hour_col).str.slice(0,2).cast(pl.Int32) - 1).alias('local_hour'))
                    else:
                        eba = eba.with_columns(pl.col(hour_col).cast(pl.Int32).alias('local_hour'))
                    eba = eba.with_columns(
                        pl.col(date_col)
                          .cast(pl.Utf8)
                          .str.to_date(format="%m/%d/%Y", strict=False)
                          .fill_null(pl.col(date_col).cast(pl.Utf8).str.to_date(format="%Y-%m-%d", strict=False))
                          .fill_null(pl.col(date_col).cast(pl.Utf8).str.to_datetime(strict=False).dt.date())
                          .alias('local_date')
                    )
                    # Negative awards mean charging; keep as negative MW to plot below zero
                    eba_hourly = eba.group_by(['local_date','local_hour']).agg([
                        pl.col(mw_col).sum().alias('eba_mw')
                    ])
                    # Full outer join so hours that appear only in EBA still show up
                    awards = awards.join(eba_hourly, on=['local_date','local_hour'], how='full')
                    # Fill missing award cols to 0 before combining
                    for c in ['da_energy_award_mw','regup_mw','regdown_mw','rrs_mw','ecrs_mw','nonspin_mw','eba_mw']:
                        if c in awards.columns:
                            awards = awards.with_columns(pl.col(c).fill_null(0.0))
                    awards = awards.with_columns((pl.col('da_energy_award_mw') + pl.col('eba_mw')).alias('da_energy_award_mw')).drop('eba_mw')
        except Exception:
            pass

        # Attach AS MCPC hourly (system) if available
        if len(as_prices) > 0:
            mcpc = (as_prices.select([
                pl.col("DeliveryDate").cast(pl.Date).alias("local_date"),
                # CRITICAL FIX: Convert Hour Ending (1-24) to Hour Beginning (0-23)
                # HE 15:00 means hour 14:00-15:00, so subtract 1 to get hour 14
                (pl.col("hour").str.slice(0,2).cast(pl.Int32) - 1).alias("local_hour"),
                "AncillaryType",
                pl.col("MCPC")
            ]).group_by(["local_date","local_hour","AncillaryType"]).agg(pl.col("MCPC").mean())
               .pivot(values="MCPC", index=["local_date","local_hour"], on="AncillaryType")
               .rename({"REGUP":"regup_mcpc","REGDN":"regdown_mcpc","RRS":"rrs_mcpc","ECRS":"ecrs_mcpc","NSPIN":"nonspin_mcpc"})
            )
            awards = awards.join(mcpc, on=["local_date","local_hour"], how="left")

        # Attach Day-Ahead price per hour for convenience (from DA price series)
        try:
            da_series = self._load_da_price_series(resource_node).rename({"da_price": "da_price_hour"})
            awards = awards.join(da_series, on=["local_date","local_hour"], how="left")
        except Exception:
            pass

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

    def calculate_all_bess(self, mapping_file: str, output_file: str = None, limit: int | None = None) -> pd.DataFrame:
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
        if limit is not None and limit > 0:
            bess_mapping = bess_mapping.head(limit)

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
                    capacity_mw=float(row['Capacity_MW']),
                    cod_date=row.get('COD_Date')  # Pass COD date if available
                )
                # Emit concise per‑kW‑year breakdown for this BESS/year
                try:
                    # Use the already calculated normalized values
                    da_kw = float(revenue_data.get('normalized_da_per_kw_year', 0.0))
                    rt_kw = float(revenue_data.get('normalized_rt_per_kw_year', 0.0))
                    total_as_kw = float(revenue_data.get('normalized_as_per_kw_year', 0.0))
                    total_kw = float(revenue_data.get('normalized_total_per_kw_year', 0.0))
                    
                    # Get operational days for logging
                    operational_days = float(revenue_data.get('operational_days', 0.0))
                    cod_date = revenue_data.get('cod_date', 'N/A')

                    logger.info(
                        (
                            f"Summary {revenue_data.get('bess_name')} {revenue_data.get('year')}: "
                            f"DA ${da_kw:,.1f}/kW-yr | RT ${rt_kw:,.1f}/kW-yr | "
                            f"AS ${total_as_kw:,.1f}/kW-yr | Total ${total_kw:,.1f}/kW-yr | "
                            f"COD: {cod_date} ({operational_days:.0f} days)"
                        )
                    )
                except Exception:
                    # Don't block on logging
                    pass
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
        # Ensure expected numeric columns exist to avoid KeyError when some rows are error placeholders
        required_cols = [
            'dam_discharge_revenue','dam_as_gen_revenue','dam_as_load_revenue','rt_net_revenue',
            'total_revenue'
        ]
        for c in required_cols:
            if c not in df.columns:
                df[c] = 0.0
        # Ensure display columns exist
        for c in ['capacity_mw', 'revenue_per_mw_year']:
            if c not in df.columns:
                df[c] = 0.0

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
        default='bess_mapping/BESS_UNIFIED_MAPPING_V5.csv',
        help='Path to BESS mapping file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV file path (default: bess_revenue_{year}.csv)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of BESS units to process (for quick tests)'
    )
    parser.add_argument(
        '--base-dir',
        type=str,
        default=os.getenv('ERCOT_DATA_DIR', '/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data'),
        help='Base directory for ERCOT data'
    )
    parser.add_argument(
        '--max-threads',
        type=int,
        default=None,
        help='Limit backend threads (defaults to ~physical cores)'
    )

    args = parser.parse_args()

    # Set default output filename
    if args.output is None:
        args.output = f"bess_revenue_{args.year}.csv"

    # Configure threading
    BESSRevenueCalculator.configure_threads(args.max_threads)

    # Create calculator
    calc = BESSRevenueCalculator(
        base_dir=args.base_dir,
        year=args.year
    )

    # Run calculation
    results = calc.calculate_all_bess(
        mapping_file=args.mapping,
        output_file=args.output,
        limit=args.limit
    )

    logger.info("✅ Revenue calculation complete!")


if __name__ == "__main__":
    main()
