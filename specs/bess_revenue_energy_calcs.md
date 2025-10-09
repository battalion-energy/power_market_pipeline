Here’s how I distinguish DA schedule vs RT deviation, where it comes from in your data, the certainty, and what I changed.
  
  Data sources and signals
  
  - DA schedule (hourly): rollup_files/DAM_Gen_Resources/<year>.parquet
      - Key fields: ResourceName, DeliveryDate (local), hour (HE, local), AwardedQuantity (MW), EnergySettlementPointPrice ($/MWh).
      - Interpretation: Hourly DAM energy award (positive = discharge; negative, if present, = charge). This is the baseline schedule used in the
  two‑settlement system.
  - RT actuals (5‑min → 15‑min settlement):
      - Discharge: rollup_files/SCED_Gen_Resources/<year>.parquet with TelemeteredNetOutput (preferred if present) or BasePoint (MW) and SCEDTimeStamp (CPT).
      - Charge: rollup_files/SCED_Load_Resources/<year>.parquet with BasePoint (MW) and SCEDTimeStamp.
      - Handling: Convert SCEDTimeStamp from America/Chicago to UTC, truncate to 15‑minute, then average the three 5‑minute values to a single 15‑minute
  actual.
  - RT prices (15‑minute): rollup_files/RT_prices/<year>.parquet
      - Key fields: SettlementPointName, datetime (UTC), SettlementPointPrice ($/MWh).
      - Join at the 15‑minute settlement timestamp.
  - DA prices (hourly): rollup_files/flattened/DA_prices_{year}.parquet (wide) or rollup_files/DA_prices/{year}.parquet (long)
      - Used to value DAM charging from Energy Bid Awards when price isn’t embedded.
  - DAM charging via load bids: rollup_files/DAM_Energy_Bid_Awards/<year>.parquet
      - Key fields (variants): SettlementPoint, DeliveryDate, hour/HourEnding, EnergyBidAwardMW.
      - Interpretation: Negative MW = buying energy at the node (battery charging in DA). This is a cost at DA price.
  
  Certainty and caveats
  
  - DAM schedule certainty: High. AwardedQuantity is the DAM energy award at the Gen resource for each local HE; ERCOT two‑settlement uses this as the
  baseline schedule.
  - RT deviations certainty: High for settlement mechanics; settlement is on deviations (actual − DA schedule) at 15‑min prices. The main nuance is “actual”:
  ERCOT settles on telemetry; if telemetry is missing, BasePoint is a reasonable proxy. I use TelemeteredNetOutput when available, else BasePoint.
  - Time alignment: High, with care. I convert SCED times (CPT) → UTC 15‑minute; DA awards keyed by local date/hour; I stamp each 15‑min interval with its
  local hour to subtract the correct hourly DAM schedule. DST transitions are handled via pytz rules.
  - DAM charging certainty: Medium‑high. Charging awards originate from load resources via Energy Bid Awards at the same node; occasionally Gen file may show
  negative awards. I include both sources. A rare unit could have a load node different from its gen node; if that happens, you’d need the explicit Gen↔Load
  node mapping (we already have Gen resource and the node from the mapping file; if a few assets deviate, we can special‑case them).
  - Missing price intervals: We drop 15‑min rows without RT prices and log counts. This slightly understates RT if there are data gaps.
  - RT offer curves: Not needed for settlement math (revenues don’t depend on the offer stack once prices and actuals are known). They’re useful for
  diagnostics but not required to compute settlement revenue.
  
  Cross‑referencing with RT bid curves
  
  - Not required to compute correct revenues. Useful to sanity‑check dispatch vs offers (e.g., if actuals exceed offered capabilities), but it won’t change
  settlement revenue numbers.
  - If you want, I can add a one‑off diagnostic that overlays accepted BasePoint against offer curve breakpoints for a few days, yet this is optional.
  
  Ancillary Services verification
  
  - Gen AS: DAM Gen file does not embed MCPC; join with rollup_files/AS_prices/<year>.parquet. I sum correct award variants:
      - RegUp: RegUpAwarded
      - RegDown: RegDownAwarded
      - RRS: RRSPFRAwarded + RRSFFRAwarded + RRSUFRAwarded (plus RRSAwarded if present)
      - ECRS: use ECRSSDAwarded (and ECRSAwarded if present; 2024 commonly uses SD)
      - Non-Spin: NonSpinAwarded
  - Load AS: DAM Load file embeds resource‑specific MCPC columns; do not re‑join system prices. I multiply awards by the in‑file MCPC:
      - RegUp: RegUp Awarded × RegUp MCPC
      - RegDown: RegDown Awarded × RegDown MCPC
      - RRS: sum each award variant × RRS MCPC
      - ECRS: ECRSSD Awarded + ECRSMD Awarded × ECRS MCPC
      - Non‑Spin: NonSpin Awarded × NonSpin MCPC
  
  Changes implemented now
  
  - RT two‑settlement deviation fix (done):
      - In bess_revenue_calculator.py::calculate_rt_net_revenue, RT net = Σ((Actual_Net − DAM_Award) × RT_Price × 15/60).
      - Actual_Net = 15‑min average of SCED Gen − SCED Load; DAM award joined by local date/hour.
  - DA charging cost (done):
      - New calculate_dam_charge_cost: sums (i) negative AwardedQuantity in DAM Gen (rare), and (ii) negative EnergyBidAwardMW at the resource node (Energy
  Bid Awards) × DA price.
      - Robust DA price loader handles flattened wide or long formats; falls back to HB_BUSAVG if node column is missing.
  - Output fields extended (done):
      - Adds dam_charge_cost and da_net_energy to the per‑unit result; total_revenue = da_net_energy + DAM AS + RT net.
  
  What I suggest next
  
  - Run 2024 to regenerate: python bess_revenue_calculator.py --year 2024 --output bess_revenue_2024.csv
  - Spot‑check a few hours for 2–3 batteries:
      - Confirm: Total energy = (DAM award × DAM price) + Σ((Actual − DAM award) × RT price × 0.25).
      - Check DA charge existence via Energy Bid Awards at the node.
  - If discrepancies remain for specific assets, we can verify their Load Resource node equals the Gen node; if not, I’ll route Energy Bid Awards to the
  correct node per mapping.
  
