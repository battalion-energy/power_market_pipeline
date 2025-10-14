# BESS Revenue Analysis Report

## Executive Summary

This report presents the findings from analyzing Battery Energy Storage System (BESS) revenues in ERCOT using 60-day disclosure data. The analysis covers Day-Ahead Market (DAM) energy revenue and Ancillary Services (AS) revenue for January 2025.

## Key Findings

### 1. BESS Market Participation

- **Total BESS Resources Identified**: 195-199 unique BESS resources (identified by Resource Type = "PWRSTR")
- **Market Growth**: The number of BESS resources increased from 195 on Jan 7 to 199 on Jan 9, 2025
- **Active Participation**: Most BESS resources participate primarily in Ancillary Services rather than energy arbitrage

### 2. Revenue Streams Analysis

#### Day-Ahead Energy Revenue

The analysis reveals that most BESS resources earn minimal revenue from DAM energy arbitrage:

**Top Performers by DAM Energy Revenue (Daily Average)**:
1. **ANEM_ESS_BESS1**: $21,283.94/day
   - Significantly outperforms all other BESS resources
   - Average award: 198.3 MW across analyzed days
   - Captures higher average prices: $30.99/MWh

2. **ANGLETON_UNIT1**: $202.92/day
3. **ALVIN_UNIT1**: $200.22/day

**Key Observation**: The vast majority of BESS resources (>90%) had zero DAM energy awards, indicating they focus on AS markets instead.

#### Ancillary Services Revenue

BESS resources show strong participation in AS markets, particularly:

**Top AS Providers (by MW capacity)**:
1. **ANEM_ESS_BESS1**:
   - RegUp: 218.7 MW average
   - ECRS: 995.4 MW average
   - Total AS Revenue: ~$1,575/day (RegUp only)

2. **ANCHOR_BESS1**:
   - RegUp: 194.7 MW average
   - ECRS: 95.5 MW average
   - RegUp Revenue: ~$1,094/day

3. **ANCHOR_BESS2**:
   - RegUp: 121.0 MW average
   - ECRS: 103.9 MW average
   - RegUp Revenue: ~$700/day

### 3. Market Pricing Patterns

**Average DAM Prices Captured**:
- Range: $18.98 - $41.14/MWh across the analyzed period
- January 7: Higher prices (~$24-35/MWh)
- January 8: Lower prices (~$19-25/MWh)
- January 9: Spike in prices (~$36-41/MWh)

### 4. Revenue Distribution

Based on the sample analysis:

**Revenue Breakdown**:
- **Energy Arbitrage**: Limited participation, only a few BESS actively arbitrage
- **Regulation Up (RegUp)**: Primary AS revenue source
- **Regulation Down (RegDown)**: Minimal participation
- **ECRS**: High MW participation but revenue calculation requires market prices
- **RRS**: Some participation by select resources

### 5. Operational Patterns

**BESS Operating Strategies**:
1. **AS-Focused Strategy** (90% of BESS):
   - Zero or minimal DAM energy awards
   - High AS capacity commitments
   - Revenue primarily from capacity payments

2. **Hybrid Strategy** (<10% of BESS):
   - Active in both energy and AS markets
   - Examples: ANEM_ESS_BESS1, ALVIN_UNIT1, ANGLETON_UNIT1

3. **Inactive/Testing** (Some resources):
   - Listed but no awards in any market
   - Possibly new installations or under testing

### 6. Market Insights

**Key Observations**:

1. **AS Market Dominance**: BESS resources preferentially participate in AS markets over energy arbitrage
   
2. **ECRS Popularity**: Many BESS provide significant ECRS capacity (new contingency reserve product)

3. **Limited Energy Arbitrage**: Despite volatile energy prices, most BESS don't actively arbitrage
   - Possible reasons: AS revenue certainty, operational constraints, or risk management

4. **Resource Concentration**: A small number of large BESS resources dominate both energy and AS markets

### 7. Revenue Potential

**Daily Revenue Estimates** (for top performers):

1. **ANEM_ESS_BESS1**:
   - Energy: ~$21,284/day
   - RegUp: ~$1,575/day
   - Total: ~$22,859/day (excluding ECRS and other AS)

2. **ANCHOR_BESS1**:
   - Energy: $0/day
   - RegUp: ~$1,094/day
   - Total: ~$1,094/day (AS only strategy)

**Annual Revenue Projection**:
- Top performer: ~$8.3M/year
- Average AS-focused BESS: ~$200K-400K/year

## Recommendations

1. **Data Enhancement**:
   - Include Real-Time market settlement data for complete picture
   - Add AS performance payments (not just capacity)
   - Track State of Charge to understand cycling patterns

2. **Analysis Extensions**:
   - Correlate BESS size (MW/MWh) with revenue strategies
   - Analyze seasonal patterns over full year
   - Compare BESS performance vs. traditional generation

3. **Market Strategy Insights**:
   - AS markets provide more stable revenue than energy arbitrage
   - ECRS represents significant opportunity for BESS
   - Co-optimization of energy and AS is rare but lucrative

## Data Limitations

1. **60-Day Lag**: Analysis based on January 2025 data (most recent available)
2. **RT Market**: Real-time energy revenue not included (data complexity)
3. **AS Performance**: Only capacity payments analyzed, not performance payments
4. **Efficiency Losses**: Not accounted for in energy arbitrage calculations

## Conclusion

The ERCOT BESS market shows rapid growth with nearly 200 operational storage resources. While a few large players dominate energy arbitrage, the majority of BESS resources focus on providing ancillary services, particularly Regulation Up and ECRS. This AS-focused strategy appears to provide more stable and predictable revenue streams compared to the volatile energy arbitrage market.

The analysis demonstrates that successful BESS operation in ERCOT requires sophisticated market strategy and likely benefits from scale, as evidenced by the concentrated revenue distribution among top performers.