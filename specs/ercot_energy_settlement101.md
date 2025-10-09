<pre><font color="#999997">I don&apos;t understand the thinking behind this: rt_gross_energy_revenue = Σ(Actual × RT × 0.25).</font>
<font color="#1C6B77">▌</font><font color="#999997">      - da_spread_revenue = Σ(DA_Award × (DA − RT) × 0.25).</font>

&gt; Great question. Those two terms are just a different way to “slice” the same two‑settlement outcome, not a different settlement method.
  
  Two‑settlement 101
  
  - ERCOT energy settlement for an interval uses:
  Total = DA_Award × DA_Price + (Actual − DA_Award) × RT_Price
  where
      - DA_Award: the scheduled MW for the hour (we apply it across the 4 × 15‑min intervals),
      - Actual: the realized net MW for the interval (Gen − Load),
      - 0.25 converts 15‑min MW to MWh.
  
  Algebraic decomposition
  Rearrange the standard formula:
  Total
  = DA_Award × DA + (Actual − DA_Award) × RT
  = Actual × RT + DA_Award × (DA − RT)
  
  Interpretation of the two parts
  
  - RT gross energy revenue = Σ(Actual × RT × 0.25)
      - “What would I earn if every MWh produced/consumed were settled at the RT price?”
      - Positive when net generating, negative when net charging.
  - DA spread revenue = Σ(DA_Award × (DA − RT) × 0.25)
      - Adjustment for the portion of energy that was scheduled in DA.
      - If DA &gt; RT, the spread is positive (you locked in a better DA price for the scheduled MW).
      - If DA &lt; RT, it’s negative (you gave up upside by scheduling at DA).
  - Sum of both = the actual settlement total for energy. Nothing is double‑counted or omitted.
  
  Why show this view?
  
  - Presentation: Many market reports show a big “RT” bar (Actual × RT) and a separate “DA spread” bar. It’s the same total as the canonical DA +
  RT‑deviation formula, just decomposed differently.
  - Intuition: RT gross reflects the value of real‑time operations; DA spread shows the benefit/cost of your DA position relative to RT.
  
  What the 0.25 factor is
  
  - Each settlement interval is 15 minutes; MW × 0.25 = MWh. We multiply every 15‑min term by 0.25 before summing.
  
  Edge cases
  
  - No DA award: DA spread is zero; total reduces to Actual × RT (as expected).
  - Net charging intervals: “RT gross” is negative (buying power); spread still adjusts only the DA‑scheduled portion.
  - DA charging: We handle DA charging cost via Energy Bid Awards (and any negative Gen awards) in the DA bucket. The “DA spread” term above applies to
  DA Gen awards only; it doesn’t double‑count DA charging.
  
  Bottom line
  
  - Using “RT gross + DA spread” or “DA award + RT deviation” yields identical totals. The first is often easier to compare to published charts (big RT
</pre>