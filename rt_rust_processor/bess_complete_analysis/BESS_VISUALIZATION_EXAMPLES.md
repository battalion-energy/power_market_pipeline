# BESS Data Visualization Examples

## 1. Revenue Leaderboard Table

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ ERCOT BESS Performance Leaderboard - 2024 YTD                                      │
├─────┬────────────────────┬─────────┬──────────┬──────────┬────────┬───────────────┤
│ Rank│ BESS Name          │ Total   │ $/MW-yr  │ Energy % │ AS %   │ vs Median     │
├─────┼────────────────────┼─────────┼──────────┼──────────┼────────┼───────────────┤
│ 1   │ RRANCHES_UNIT1     │ $4.09M  │ $163,654 │ 45%      │ 55%    │ +187% 📈      │
│ 2   │ RRANCHES_UNIT2     │ $3.92M  │ $156,640 │ 43%      │ 57%    │ +175% 📈      │
│ 3   │ EBNY_ESS_BESS1     │ $3.40M  │ $136,000 │ 52%      │ 48%    │ +139% 📈      │
│ ... │                    │         │          │          │        │               │
│ 50  │ MEDIAN_PERFORMER   │ $1.42M  │ $56,800  │ 48%      │ 52%    │ MEDIAN        │
│ ... │                    │         │          │          │        │               │
│ 163 │ SMALL_BESS_1       │ $0.12M  │ $24,000  │ 65%      │ 35%    │ -58% 📉       │
└─────┴────────────────────┴─────────┴──────────┴──────────┴────────┴───────────────┘
```

## 2. Revenue Mix Evolution Chart

```
BESS Revenue Mix Evolution (2022-2025)
100% ┤                                                         
     │ ████ RegDown                                           
  90%│ ████ RegUp      ████                                   
     │ ████            ████ ECRS                              
  80%│ ████            ████           ████                    
     │ ████ RRS        ████           ████ NonSpin           
  70%│ ████            ████ RRS       ████                    
     │ ████            ████           ████ ECRS              
  60%│ ████            ████           ████                    
AS % │ ████            ████           ████ RRS               
  50%│ ████            ████           ████━━━━━━━━━━━━━━━━━━ 
     │ ████            ████           ████ Energy Arbitrage   
  40%│ ████            ████ Energy    ████                    
     │ ████            ████           ████                    
  30%│ ████            ████           ████                    
     │ ████ Energy     ████           ████                    
  20%│ ████            ████           ████                    
     │ ████            ████           ████                    
  10%│ ████            ████           ████                    
     │ ████            ████           ████                    
   0%└─────────────────────────────────────────────────────
      2022         2023         2024         2025
```

## 3. Nodal Price Impact Scatter Plot

```
Revenue vs Nodal Price Premium
Annual Revenue ($M)
   5│                                    • RRANCHES_UNIT1
    │                                  •   
   4│                              • •     
    │                          • •         
   3│                      • • • •         
    │                  • • • • •           
   2│              • • • • • •             
    │      • • • • • • • •                 
   1│  • • • • • • • •                     
    │• • • • • •                           
   0└────────────────────────────────────
    -10    -5     0     +5    +10   +15
         Avg Nodal Premium vs Hub ($/MWh)
    
Legend: • = 1 BESS  Size = Capacity (MW)
```

## 4. Bidding Strategy Clusters

```
BESS Operating Strategy Clusters (2024)

         High AS Participation
                │
    ┌───────────┼───────────┐
    │     AS    │   Hybrid  │
    │Specialist │ Optimizer │
    │  (23%)    │   (31%)   │
    │           │           │
────┼───────────┼───────────┼──── # Daily Cycles
    │           │           │
    │Conservative│ Arbitrage │
    │   (18%)   │  Master   │
    │           │   (28%)   │
    └───────────┼───────────┘
                │
         Low AS Participation

Cluster Characteristics:
• AS Specialist: >70% revenue from AS, 0.5-1 cycles/day
• Hybrid Optimizer: 40-60% AS, 1-2 cycles/day
• Conservative: <30% utilization, <0.5 cycles/day
• Arbitrage Master: >60% energy, 2-3 cycles/day
```

## 5. State of Charge Heatmap

```
Typical BESS State of Charge Pattern - Top Performer
Hour  00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23
Mon   ██ ██ ██ ██ ██ ░░ ░░ ░░ ▒▒ ▒▒ ▒▒ ▒▒ ▒▒ ▒▒ ░░ ░░ ░░ ░░ ██ ██ ██ ██ ██ ██
Tue   ██ ██ ██ ██ ██ ░░ ░░ ░░ ▒▒ ▒▒ ▒▒ ▒▒ ▒▒ ▒▒ ░░ ░░ ░░ ░░ ██ ██ ██ ██ ██ ██
Wed   ██ ██ ██ ██ ██ ░░ ░░ ░░ ▒▒ ▒▒ ▒▒ ▒▒ ▒▒ ▒▒ ░░ ░░ ░░ ░░ ██ ██ ██ ██ ██ ██
Thu   ██ ██ ██ ██ ██ ░░ ░░ ░░ ▒▒ ▒▒ ▒▒ ▒▒ ▒▒ ▒▒ ░░ ░░ ░░ ░░ ██ ██ ██ ██ ██ ██
Fri   ██ ██ ██ ██ ██ ░░ ░░ ░░ ▒▒ ▒▒ ▒▒ ▒▒ ▒▒ ▒▒ ░░ ░░ ░░ ░░ ██ ██ ██ ██ ██ ██
Sat   ██ ██ ██ ██ ██ ██ ██ ▒▒ ▒▒ ▒▒ ▒▒ ▒▒ ▒▒ ▒▒ ▒▒ ▒▒ ░░ ░░ ░░ ██ ██ ██ ██ ██
Sun   ██ ██ ██ ██ ██ ██ ██ ▒▒ ▒▒ ▒▒ ▒▒ ▒▒ ▒▒ ▒▒ ▒▒ ▒▒ ░░ ░░ ░░ ██ ██ ██ ██ ██

Legend: ██ Charged (>80%) ▒▒ Medium (40-80%) ░░ Discharged (<40%)
```

## 6. Location Value Analysis

```
BESS Performance by Location Type

        Revenue ($/MW-yr)
$200k ┤  ┌──┐
      │  │██│ Premium Nodes
$150k ┤  │██│ ┌──┐
      │  │██│ │▒▒│ Load Zones
$100k ┤  │██│ │▒▒│ ┌──┐
      │  │██│ │▒▒│ │░░│ Hubs
 $50k ┤  │██│ │▒▒│ │░░│
      │  │██│ │▒▒│ │░░│
   $0 └──┴──┴─┴──┴─┴──┴───
         Top    Avg   Bottom
         25%          25%

Key Insights:
• Premium nodes average 35% higher revenue
• Load zones show highest variability
• Hub locations most consistent but lower ceiling
```

## 7. What-If Scenario Dashboard

```
┌─────────────────────────────────────────────────────────┐
│ What-If Analysis: MEDIOCRE_BESS1                       │
├─────────────────────────────────────────────────────────┤
│ Current Performance (2024)                              │
│ • Revenue: $1.2M ($48k/MW-yr)                          │
│ • Location: RURAL_NODE (-$3/MWh avg basis)            │
│ • Strategy: Conservative (0.7 cycles/day)              │
├─────────────────────────────────────────────────────────┤
│ Improvement Scenarios                      Δ Revenue    │
│ ┌─────────────────────────────────────┬──────────────┐ │
│ │ 1. Move to HB_NORTH hub            │ +$180k (+15%)│ │
│ │ 2. Adopt median bidding strategy    │ +$360k (+30%)│ │
│ │ 3. Top quartile operations          │ +$540k (+45%)│ │
│ │ 4. Move to premium node + better ops│ +$840k (+70%)│ │
│ └─────────────────────────────────────┴──────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## 8. Monthly Performance Tracking

```
BESS Monthly Revenue Trend with Events
$/MW
$20k ┤                    Storm Uri
     │                       ↓      ECRS Launch
$18k ┤                    ┌──┐           ↓
     │                    │██│        ┌──┐
$16k ┤              ┌──┐  │██│  ┌──┐  │██│
     │        ┌──┐  │██│  │██│  │██│  │██│
$14k ┤  ┌──┐  │██│  │██│  │██│  │██│  │██│
     │  │▒▒│  │▒▒│  │▒▒│  │▒▒│  │▒▒│  │▒▒│ ← Your BESS
$12k ┤  │▒▒│  │▒▒│  │▒▒│  │▒▒│  │▒▒│  │▒▒│
     │  │░░│  │░░│  │░░│  │░░│  │░░│  │░░│ ← Market Avg
$10k ┤  │░░│  │░░│  │░░│  │░░│  │░░│  │░░│
     │  │░░│  │░░│  │░░│  │░░│  │░░│  │░░│
 $8k └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──
      J  F  M  A  M  J  J  A  S  O  N  D
                      2024
```

## 9. Peer Comparison Radar Chart

```
BESS Performance vs Peers (100 = Best in Class)

        Revenue/MW
           100
            │
     Cycles │     AS Revenue
         80 ├──●──┐
            │ ╱ ╲ │
         60 ├●   ●┤ 80
            ╱     ╲
         40 ●  You ● 60
           ╱   ●   ╲
        20 ●───┼───● 40
          ╱    │    ╲
    Efficiency │  Location  20
              20  Premium

    ─── Your BESS
    ─── Peer Average (similar size/duration)
    ─── Top Performer
```

## 10. Revenue Attribution Waterfall

```
Revenue Build-Up Analysis - EXAMPLE_BESS (2024)

$200k ┤                                          ┌────┐
      │                              ┌────┐      │████│ Total
$180k ┤                    ┌────┐    │████│      │████│ $187k
      │          ┌────┐    │████│    │████│      │████│
$160k ┤          │████│    │████│    │████│      │████│
      │ ┌────┐   │████│    │████│    │████│      │████│
$140k ┤ │████│   │████│    │████│    │████│      │████│
      │ │████│   │████│    │████│    │████│      │████│
$120k ┤ │████│   │████│    │████│    │████│      │████│
      │ │████│   │████│    │████│    │████│ ┌────┤████│
$100k ┤ │████│   │████│    │████│    │████│ │-$8k│████│
      │ │████│   │████│    │████│    │████│ │    │████│
      │ │████│   │████│    │████│    │████│ │    │████│
      │ │$95k│   │+$42k│   │+$38k│   │+$20k│ │    │████│
   $0 └─┴────┴───┴────┴────┴────┴────┴────┴─┴────┴────┘
       Energy    RegUp     RRS      ECRS   Location  Final
       Arbitrage                           Discount
```

## Implementation Notes

1. **Color Coding**: 
   - Green (█): Above average performance
   - Yellow (▒): Average performance  
   - Red (░): Below average performance

2. **Interactive Elements**:
   - All charts should be clickable for drill-down
   - Hover tooltips with detailed breakdowns
   - Export functionality for reports

3. **Real-time Updates**:
   - Dashboard refreshes every 15 minutes
   - Historical data updates daily
   - Forward curves update hourly

4. **Mobile Responsiveness**:
   - Simplified views for mobile
   - Swipeable charts
   - Essential metrics prioritized

## 11. TBX Capture Rate Comparison

```
TB2 Revenue Capture Analysis - 2024
Resource Name         Actual    TB2 Potential  Capture Rate
─────────────────────────────────────────────────────────────
RRANCHES_UNIT1        $4.09M    $4.35M         94.0% ████████████████████
RRANCHES_UNIT2        $3.92M    $4.30M         91.2% ██████████████████
EBNY_ESS_BESS1        $3.40M    $4.10M         82.9% ████████████████
MEDIOCRE_BESS         $1.42M    $2.84M         50.0% ██████████
POOR_PERFORMER        $0.84M    $2.80M         30.0% ██████

Legend: █ = 5% capture rate
Top Quartile: >75% | Median: 50% | Bottom Quartile: <25%
```

## 12. Geographic TB2 Revenue Heat Map

```
Texas TB2 Revenue Potential ($/MW-yr) - Interactive Map

                    Dallas ●
                      │
         Fort Worth ● │  TB2: $145k
                      │  BESS: 12
           ┌──────────┴──────────┐
           │    North Zone        │
           │  ████████████████    │
           │    TB2: $180k       │
           └─────────────────────┘
                      │
    Austin ●──────────┼──────────● Houston
    TB2: $165k        │          TB2: $195k
    BESS: 8          │          BESS: 18
           ┌─────────┴──────────┐
           │    South Zone      │
           │  ██████████████    │
           │    TB2: $150k      │
           └────────────────────┘
                      │
              San Antonio ●
              TB2: $155k
              BESS: 5

Color Scale: ██ >$200k  ██ $150-200k  ██ $100-150k  ██ <$100k
Click any node for detailed time series and BESS performance
```

## 13. TBX Missed Opportunities Calendar

```
Missed TB2 Arbitrage Opportunities - MEDIOCRE_BESS (January 2024)

Week  Mon    Tue    Wed    Thu    Fri    Sat    Sun    Total Missed
────────────────────────────────────────────────────────────────────
1     $2.1k  $1.8k  $3.2k  $0.5k  $1.2k  $0.8k  $0.3k  $9.9k
      ████   ███    █████  █      ██     █      
2     $1.5k  $2.8k  $1.1k  $4.5k  $3.8k  $1.2k  $0.9k  $15.8k
      ███    ████   ██     ██████ █████  ██     █
3     $0.8k  $1.2k  $2.5k  $1.8k  $2.2k  $0.5k  $0.7k  $9.7k
      █      ██     ████   ███    ████   █      █
4     $3.1k  $2.4k  $1.9k  $2.8k  $3.5k  $1.1k  $0.8k  $15.6k
      █████  ████   ███    ████   █████  ██     █

Total Monthly Missed: $51.0k (Capture Rate: 50%)
Click any day for hourly breakdown of missed opportunities
```

## 14. TBX Performance Gauge Dashboard

```
BESS TBX Capture Rate Dashboard

┌─────────────────┬─────────────────┬─────────────────┐
│  Fleet Average  │ Your BESS       │ Best Performer  │
│                 │                 │                 │
│      73.5%      │     65.2%       │     94.2%       │
│    ╭─────╮      │    ╭─────╮      │    ╭─────╮      │
│   ╱       ╲     │   ╱       ╲     │   ╱       ╲     │
│  │    ●    │    │  │    ●    │    │  │      ● │    │
│  │ GOOD    │    │  │  FAIR   │    │  │EXCELLENT│    │
│  ╰─────────╯    │  ╰─────────╯    │  ╰─────────╯    │
│                 │                 │                 │
│ Trend: ↑ +2.3%  │ Trend: ↓ -1.5% │ Trend: → 0.0%  │
└─────────────────┴─────────────────┴─────────────────┘

Performance Tiers:
Excellent: >90% | Good: 70-90% | Fair: 50-70% | Poor: <50%
```

## 15. TBX vs Actual Revenue Time Series

```
Daily TB2 vs Actual Revenue Comparison - EXAMPLE_BESS
$/day
$12k ┤     TB2 Potential ----
     │     Actual Revenue ████
$10k ┤   ┌────┐
     │   │    │╱╲    ┌────┐
 $8k ┤   │    ╱  ╲   │    │     ┌────┐
     │ ┌─┤   ╱    ╲  │    │   ╱─┤    │
 $6k ┤ │█│  ╱      ╲_│    │  ╱  │████│
     │ │█│ ╱         │████│ ╱   │████│
 $4k ┤ │█│╱          │████│╱    │████│
     │ │█│           │████│     │████│
 $2k ┤ │█│           │████│     │████│
     │ │█│           │████│     │████│
 $0k └─┴─┴───────────┴────┴─────┴────┴──
      1  5  10  15  20  25  30
              Day of Month

Average Capture: 65% | Best Day: 92% | Worst Day: 23%
```

## 16. TBX Opportunity Heat Map by Hour

```
TB2 Missed Revenue by Hour of Day - EXAMPLE_BESS (2024)

Hour  00  01  02  03  04  05  06  07  08  09  10  11  12  13  14  15  16  17  18  19  20  21  22  23
────────────────────────────────────────────────────────────────────────────────────────────────────
Avg$  12  8   5   3   2   15  45  89  76  54  48  52  58  64  78  95  120 145 132 98  76  54  32  18

      ░░  ░░  ░░  ░░  ░░  ▒▒  ▒▒  ███ ███ ██  ██  ██  ██  ██  ███ ███ ███ ███ ███ ███ ███ ██  ▒▒  ▒▒

Legend: ███ >$100/hr missed  ██ $50-100/hr  ▒▒ $20-50/hr  ░░ <$20/hr

Key Insights:
• Peak hours (16-19) show highest missed opportunities
• Morning ramp (06-08) consistently underutilized
• AS commitments during peak hours main cause of gaps
```

## 17. Performance Improvement Roadmap

```
Path to 90% TB2 Capture Rate - MEDIOCRE_BESS

Current State (50%)          Target State (90%)
────────────────────────────────────────────────
                   +15%        +10%        +10%         +5%
Current ──────► Optimize ──► Better ──► Reduce ──► Advanced ──► Target
50%            Charging     Peak      AS Lock    Forecast     90%
               Schedule    Discharge              Models

Implementation Steps:
1. Optimize Charging (Month 1-2)
   - Charge during true bottom hours
   - Avoid mid-day charging
   - Expected gain: +15% capture

2. Better Peak Discharge (Month 2-3)
   - Focus on 16:00-20:00 window
   - Reduce AS during peaks
   - Expected gain: +10% capture

3. Reduce AS Lock-in (Month 3-4)
   - Selective AS participation
   - Day-ahead optimization
   - Expected gain: +10% capture

4. Advanced Forecasting (Month 4-6)
   - ML price prediction
   - Weather integration
   - Expected gain: +5% capture
```

## 18. Zone-Level TBX Analysis

```
TB2 Performance by ERCOT Zone (2024 Average)

Zone         TB2 Potential  Avg Capture  Best BESS   Worst BESS
────────────────────────────────────────────────────────────────
Houston      $195k/MW-yr    72%         HOUS_BESS1  HOUS_OLD
             ████████████   ███████     (91%)       (45%)

North        $180k/MW-yr    68%         NRTH_NEW    NRTH_SMALL
             ███████████    ██████      (88%)       (38%)

West         $165k/MW-yr    81%         WEST_MEGA   WEST_TINY
             ██████████     ████████    (94%)       (52%)

South        $150k/MW-yr    64%         STH_PRIME   STH_LEGACY
             █████████      ██████      (85%)       (41%)

Key Findings:
• West zone shows highest average capture despite lower TB2
• Houston has highest potential but moderate capture
• Wide performance spread within each zone
```