# PG&E Capacity Bidding Program (CBP) Research Summary

**Research Date:** October 11, 2025  
**Research Duration:** 30 minutes  
**Data Quality Score:** 7/10  
**Program URL:** https://www.pge.com/en/save-energy-and-money/energy-saving-programs/demand-response-programs/business-programs.html#cbp

## Critical Findings

### IMPORTANT CLARIFICATION: $6.00/kWh Rate
**The $6.00/kWh figure is a PENALTY, not a performance payment.**
- Participants are penalized at $6.00/kWh for energy usage OVER their Firm Service Level (FSL) during curtailment events
- This is NOT money participants earn - it's money they pay for non-compliance
- Energy payments are based on natural gas prices (city gate price with 15,000 BTU/kWh heat rate trigger)

### Payment Structure Verified
1. **Capacity Payments:** Monthly payments for enrolled capacity commitment
   - CPUC-authorized rates through 2027
   - Varies by month (highest in July-September)
   - Historical data (2006): August peak at $21.57/kW-month
   - Web sources indicate current range: $4-$12/kW-month varying by season
   - **SPECIFIC CURRENT RATES NOT ACCESSIBLE** (requires Schedule E-CBP tariff)

2. **Energy Payments:** Per-event payments when curtailment called
   - Based on natural gas price indexing
   - Only paid when events occur
   - Paid up to 150% of nominated reduction

3. **Performance Thresholds:**
   - 95%+ delivery = 100% capacity payment (full payment)
   - <95% delivery = penalties based on kWh shortfall
   - Larger shortfalls = larger penalties

## Program Parameters

### Operational Details
- **Season:** May 1 - October 31
- **Notification:** Day-ahead by 5 PM for next day
- **Call Windows (2024):**
  - May: 5:00 PM - 10:00 PM PT
  - June-October: 4:00 PM - 9:00 PM PT
- **Event Duration:** Up to 4 hours maximum
- **Event Frequency:** Max 1 per day, can exceed 6 per month

### Eligibility
- **Customer Classes:** Residential, Commercial, Industrial, Agricultural
- **Minimum Capacity:** Not specified by PG&E (but CAISO PDR requires 100 kW minimum)
- **Meter Requirement:** 15-minute interval meter with remote reading capability
- **Resource Types:** Load curtailment, battery storage (BESS), BTM resources
- **Aggregator:** Required (third-party managed)

### Battery Storage Specific
- **Explicitly Eligible:** Yes, battery energy storage systems can participate
- **Behind-the-Meter:** Yes, BTM resources eligible
- **Stacking Allowed:** Compatible with E-OBMC, ELRP-B2, SGIP programs
- **PG&E Territory BTM Storage:** 1,100 MW (35% of national BTM capacity!)

## Market Integration

### CAISO Integration
- Program integrated with CAISO wholesale market via Proxy Demand Resources (PDR)
- Aggregators bid PDR resources into day-ahead market
- Events triggered when CAISO accepts bids (market awards)
- Uses CAISO Demand Response Registration System (DRRS)
- Net benefits test determines monthly price threshold for bids

## Data Gaps

### Critical Missing Information
1. **Current capacity payment rates** ($/kW-month by month for 2024-2025)
   - Contained in Schedule E-CBP tariff PDF
   - PDF could not be fully read due to encoding issues
   - This is the PRIMARY data gap

2. **Current energy payment calculation details**
   - Natural gas price formula specifics
   - Current heat rate values

3. **Historical event data**
   - No public record of when/how often events called
   - No participant count or enrolled capacity statistics

4. **Detailed operational parameters**
   - Response time requirements
   - Minimum rest between events
   - Maximum consecutive events
   - Annual/seasonal event limits

### Information Sources
**Verified:**
- CPUC Resolution E-4020 (2006) - Program authorization and historical rates
- PG&E Program Webpage - Current eligibility, structure, triggers
- CAISO PDR Documentation - Market integration details
- Multiple web searches - 2024 call windows, performance thresholds

**Not Accessible:**
- Schedule E-CBP Tariff PDF - Current rates (encoding issues)
- Aggregator contracts - Customer-facing terms
- Historical event logs - Actual dispatch history

## Optimizer Integration Assessment

### Strengths for Battery Optimization
1. ✅ Day-ahead notification allows planning
2. ✅ Predictable call windows (4-10 PM peak hours)
3. ✅ Battery storage explicitly eligible
4. ✅ Stacking allowed with other programs
5. ✅ CAISO price signals available for optimization
6. ✅ May-Oct season leaves Nov-Apr for other revenue

### Challenges
1. ⚠️ 95% performance threshold with penalties for shortfalls
2. ⚠️ 100 kW CAISO minimum (limits small systems)
3. ⚠️ Aggregator-managed (requires third-party relationship)
4. ⚠️ Payment rates not publicly available (requires tariff access)
5. ⚠️ Energy payment economics unclear (gas price indexed)

## Revenue Accounting Implications

For **historical revenue analysis** (not optimization):
- Can reconstruct actual events from aggregator notifications
- Can calculate capacity payments from enrolled MW and tariff rates
- Can calculate energy payments from actual performance and gas prices
- Need aggregator data for actual dispatch history
- $6/kWh penalty applies to over-performance (using more than allowed)

**This is revenue accounting, not optimization.**

## Recommended Next Steps

1. **Obtain Schedule E-CBP Tariff** - Contact PG&E customer service for current rates
2. **Contact Aggregators** - Research Enel, OhmConnect, others for customer terms
3. **CAISO DRRS Documentation** - Get PDR registration and bidding procedures
4. **Historical Data Request** - Contact aggregators for actual event history
5. **Participant Interviews** - Talk to current participants for real-world experience
6. **DRAM Program Review** - Similar program with comparable structure

## Bottom Line

**Program is well-suited for battery participation** but specific revenue potential cannot be calculated without:
1. Current capacity payment rates (Schedule E-CBP)
2. Historical event frequency data
3. Energy payment calculation details

**Data quality is good for structure/eligibility (7/10) but limited for financial modeling.**

The clarification that $6/kWh is a penalty (not payment) significantly changes revenue expectations. Actual revenue comes from monthly capacity payments ($4-12+/kW-month range) plus event-based energy payments (gas-price indexed).

---
**Research completed:** 2025-10-11  
**Researcher:** Claude Code  
**Output file:** program_batch2_007_pge_cbp_enriched.json
