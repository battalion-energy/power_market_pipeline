Here’s how to read “RT Basepoint” vs “Output,” and what the gap means operationally and financially.
  
  What the series are
  
  - RT Basepoint: The 5‑minute dispatch instruction from SCED (converted to hourly MWh on the chart). For storage we plot
  net basepoint = BasePointGen − BasePointLoad. It reflects your RT offer curve, operating limits, AS deployments, and system
  constraints.
  - Output: Actual telemetered net energy (TelemeteredNetOutput), likewise netted Gen − Load and aggregated.
  
  Why they differ
  
  - Ramping/latency: Unit is moving toward a new basepoint and lags/leads during ramps.
  - Regulation/ECRS action: AGC/reg signals move the unit around the basepoint; output will oscillate above/below.
  - Physical limits/SOC: Hitting charge/discharge limits or SOC constraints caps output while basepoint requests more/less.
  - Controls/curtailments: Availability derates, trips, manual overrides, EMS issues.
  - Data gaps: If telemetry is missing we may fall back to basepoint (we prefer telemetry; gaps are rare in these charts).
  
  Where this shows up in markets/settlement
  
  - Day‑Ahead (financial schedule):
      - DA energy award MWh × DA price is your DA energy revenue/cost.
      - This schedule is financial; it is not the physical basepoint you must follow.
  - Real‑Time (two‑settlement energy):
      - Deviations from your DA schedule settle at RT price:
      RT deviation revenue ≈ Σ((Actual net − DA scheduled) × RT price × interval_hours).
  - Basepoint itself is not a settlement quantity; it’s the instruction you’re expected to follow.
  - Basepoint compliance/penalties:
      - ERCOT tracks Base Point Deviation (BPD). Sustained deviations beyond allowed tolerance can trigger non‑compliance exposure,
  affect AS qualification, and draw attention from control/operations. (We don’t book a dollar charge on basepoint deviation in
  these outputs; the financials come from DA+RT energy and AS payments.)
  - Ancillary Services:
      - AS revenue is capacity-based (award MW × MCPC). Deployments to provide regulation/ECRS change basepoint and explain Output–
  Basepoint spreads without changing AS revenue calculation.
  
  How to interpret the gap on the right plot
  
  - Output ≈ Basepoint: On instruction and tracking; deviations from DA schedule (if any) settle at RT price but the resource is
  following RT dispatch.
  - Output < Basepoint (discharge case) or Output > Basepoint (charge case): Under‑delivering relative to instruction; check for
  ramping, SOC/limits, derates, or control issues.
  - Output > Basepoint (discharge) or Output < Basepoint (charge): Over‑delivering; can come from AGC/reg swings or control
  overshoot.
  - Large persistent gaps with little AS activity: Likely constraints or unavailable capacity; worth checking telemetry quality and
  the RT bids depth chart to see if the unit was economically dispatched to those levels.
  
  Quick diagnostics to apply on any day
  
  - Check DA awards (left plot). If DA awards are small/none but RT prices spike and basepoint rises, RT offers drove the dispatch.
  - Overlay AS MCPCs (right plot legend) and look for times with Reg/ECRS awards — output should move around the basepoint then.
  - Open the bids‑depth charts:
      - RT bids depth: shows where your SCED curves would economically land the unit across prices.
      - DA bids depth: shows DA Energy‑Only Offers and the node’s DA price.
  - Compute deviation vs basepoint (operational) and vs DA schedule (settlement):
      - Operational: Σ(Output − Basepoint) × 0.25 h per 15‑min gives a daily “tracking MWh.”
      - Settlement: Already in our outputs as RT net = Σ((Actual − DA schedule) × RT price × interval_hours).
