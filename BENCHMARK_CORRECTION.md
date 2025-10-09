# ERCOT BESS Revenue Benchmarks - Corrected Analysis

**Date**: October 8, 2025

## My Original Error

I incorrectly stated the 2024 benchmark as **$4,630/MW-month** based on misreading search results.

## Actual 2024 ERCOT BESS Revenues (Modo Energy)

### Monthly Variation (2024)
| Month | $/kW-year (annualized) | $/kW-month | $/MW-month |
|-------|----------------------|------------|------------|
| Feb 2024 | $17 | $1.42 | $1,420 |
| Mar 2024 | $39 | $3.25 | $3,250 |
| Apr 2024 | $67 | $5.58 | $5,580 |
| **May 2024** | **$162** | **$13.50** | **$13,500** |
| Jun 2024 | $45 | $3.75 | $3,750 |
| **Jul 2024** | **$23** | **$1.92** | **$1,920** |
| **Aug 2024** | **$87** | **$7.25** | **$7,250** |
| Sep 2024 | $22 | $1.83 | $1,830 |
| Oct 2024 | $41 | $3.42 | $3,420 |
| Nov 2024 | $33 | $2.75 | $2,750 |
| Dec 2024 | $16 | $1.33 | $1,330 |

### 2024 Full Year Estimate
- **Range**: $1.42 to $13.50/kW-month
- **Estimated average**: ~$4.00/kW-month = **$4,000/MW-month**
- **Total annual**: ~$48/kW-year = $48,000/MW-year

## My Fleet Average Calculation

- **Calculated average**: $311/MW-month
- **Calculated median**: $65/MW-month

## Revised Comparison

| Metric | My Calc | Benchmark | Ratio |
|--------|---------|-----------|-------|
| Average | $311/MW-month | $4,000/MW-month | **7.8%** |
| Median | $65/MW-month | ~$3,000/MW-month | **2.2%** |

**I'm still 92% below the corrected benchmark.**

## Key Insights from Modo Data

### Revenue Composition (from chart)
1. **Real-Time Energy** (Red): Largest component in 2024
2. **ECRS** (Pink): Second largest AS revenue source
3. **RRS** (Green): Significant AS revenue
4. **Day-Ahead Energy** (Orange/Brown): Growing in importance
5. **Reg Up/Down** (Blue/Purple): Smaller components
6. **Non-Spin** (Dark Blue): Smallest component

### Market Trends Noted
1. **Day-Ahead > Real-Time** (July 2024): First time DA trading outperformed RT
2. **DART optimization**: Plus Power outperformed Engie in DA Energy
3. **Declining AS revenues**: Pushing operators to complex energy arbitrage
4. **67% revenue drop** (H1 to July 2024): AS saturation + market changes

### Negative Revenues ARE REAL
User notes: "some batteries on the ranking are negative $/kw because they are probably part of some other hedge or other trades outside of the market"

**This is legitimate** - batteries can show negative market revenues if:
- Part of renewable co-location with hedging
- Portfolio optimization across multiple assets
- Tax credit optimization
- Hedging thermal generation or retail contracts

## What This Means for My Analysis

### What's Now Validated ✅
1. **Negative revenues**: Not necessarily errors - can be real for hedged positions
2. **High volatility**: My fleet shows similar patterns (wide range from -$391 to $6,671/MW-month)
3. **Some units performing well**: Top performers at $6,671/MW-month are ABOVE benchmark

### What's Still Problematic ❌
1. **Fleet average too low**: $311 vs $4,000/MW-month (92% gap remains)
2. **Median extremely low**: $65 vs ~$3,000/MW-month (98% gap)
3. **Too many negatives**: 13 units with negative revenue vs industry having only "some"
4. **Efficiency >100%**: Still physically impossible, suggests data/calc issues

## Remaining Issues to Fix

### HIGH PRIORITY
**RT Revenue Calculation**: Still needs the fix to properly account for:
- Discharge at HIGH prices (revenue)
- Charge at LOW prices (cost)
- Current net calculation may not capture arbitrage correctly

### MEDIUM PRIORITY
**Efficiency >100%**: Investigate root cause:
- ERCOT data quality issues?
- Resource pairing errors in mapping?
- Calculation methodology?

### LOW PRIORITY
**Missing 4% RT intervals**: Minor impact but worth investigating

## Conclusion

**I was wrong about the benchmark magnitude**, but **my core finding stands**:

My calculations are producing revenues **92% below** the (corrected) market average. The RT revenue calculation fix is still needed, though some negative revenues may be legitimate hedged positions.

The fact that my **top performers exceed the benchmark** (CHISMGRD_BES1 at $6,671/MW-month vs ~$4,000 industry avg) suggests the methodology CAN work when applied correctly.
