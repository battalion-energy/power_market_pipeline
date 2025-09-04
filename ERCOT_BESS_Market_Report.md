# ERCOT BESS Market Operations: Comprehensive Report for Battalion Platform

## Executive Summary

This report examines how Battery Energy Storage Systems (BESS) operate in ERCOT's complex market structure. A critical finding is that BESS projects CANNOT choose their settlement zone - it's determined by their physical interconnection point at the substation. However, BESS operators can use various strategies including Congestion Revenue Rights (CRRs), strategic siting decisions, and sophisticated optimization to maximize revenues within their assigned settlement zones.

## 1. ERCOT Market Structure for BESS

### 1.1 Nodal Market Design

ERCOT operates a nodal market where prices are calculated at over 4,000 nodes across the grid. Each node has a Locational Marginal Price (LMP) that reflects:
- Energy price
- Congestion costs  
- Marginal losses

BESS projects settle at specific **Resource Nodes**, which are determined by their physical point of interconnection.

### 1.2 Settlement Points are Determined by Physical Location

**Critical Correction**: A BESS's settlement point is NOT a strategic choice - it's determined by the physical interconnection location at the substation. The Resource Node is either:
- The electrical bus at the substation (for radially connected resources)
- The resource's side of the electrical bus (for other connections)

**Crossett BESS Clarification**: Initial confusion arose from missing coordinate data. Crossett BESS is actually physically located in Crane County, West Texas (in the Permian Basin, ~32 miles from Odessa), NOT in Houston. Its LZ_WEST settlement zone correctly reflects its actual physical location.

## 2. Congestion Revenue Rights (CRRs) for BESS

### 2.1 What are CRRs?

CRRs are financial instruments that:
- Hedge against congestion costs between two points on the grid
- Can result in payments OR charges to the holder
- Are settled based on Day-Ahead Market congestion prices

### 2.2 Types of CRRs

**Point-to-Point (PTP) Options**:
- Only result in payments to holder (never charges)
- Lower risk but typically more expensive to acquire

**Point-to-Point (PTP) Obligations**:
- Can result in payments OR charges
- More complex but potentially more profitable
- Used by sophisticated market participants

### 2.3 CRR Auction Process

ERCOT conducts two types of auctions:

**Monthly Auctions**:
- Held once per month for the upcoming month
- One-month terms

**Long-term Auctions**:
- Held twice yearly
- Cover six-month periods
- Offer strips of 1-6 consecutive months

### 2.4 BESS CRR Strategy

A BESS can use CRRs to:
1. **Hedge congestion between charging and discharging locations** - If charging from cheap West Texas wind and discharging to serve Houston load
2. **Lock in revenue certainty** - Secure predictable congestion payments between settlement points
3. **Arbitrage auction vs real-time prices** - Purchase undervalued CRRs in auctions

## 3. Transmission Infrastructure & Zone Connections

### 3.1 Current Infrastructure

ERCOT's transmission system primarily uses 345kV lines, with new 765kV lines being developed:
- 345kV has been standard since 1960s
- First 765kV line approved in 2025 (Howard-Solstice, 300 miles)
- $5 billion/year transmission investment planned over 6 years

### 3.2 Houston-West Zone Connection

**Physical Distance**: Houston and West Texas are approximately 400-500 miles apart

**Transmission Challenges**:
- West Texas peak demand increased 10x in past decade
- Permian Basin oil/gas activity drives massive load growth
- Transmission capacity hasn't kept pace with demand

**Key Constraints**:
- **Westex GTC** (Generic Transmission Constraint): Group of lines monitored for stability
- Active congestion 51% of the time in some areas
- 2018 Yucca Drive line congestion costs: $257 million

### 3.3 Understanding the Data Confusion

Our initial data showed some BESS projects with settlement zones that didn't match their expected physical locations. After investigation:

1. **Data Quality Issues**: Missing coordinate data led to incorrect physical zone assignments
2. **Crossett Example**: Actually located in Crane County (West Texas), not Houston
3. **Key Learning**: Settlement zones MUST match physical interconnection points
4. **Data Validation Need**: Critical to verify substation locations with actual coordinates

## 4. Revenue Opportunities by Strategy

### 4.1 Energy Arbitrage

**2024 Market Shift**: Energy arbitrage now represents 28-41% of BESS revenues (up from 11-15% in 2023)

**West Zone Advantages**:
- Highest intra-day price spreads in ERCOT
- Strong evening peaks (7-8pm prices 42% higher than South Zone)
- Industrial demand from oil/gas creates predictable patterns

### 4.2 Ancillary Services

**System-Wide Market** (Corrected Understanding):
- ERCOT procures ancillary services system-wide, NOT zonally
- All BESS access same AS markets regardless of location
- Products include: Regulation Up/Down, Non-Spinning Reserve, ECRS

**2024 Market Saturation**:
- AS revenues declined 71% year-over-year
- Average BESS earned $55/kW in 2024 vs $192/kW in 2023
- Saturation driving shift to energy arbitrage

### 4.3 Strategic Siting Decisions

**Why BESS Developers Choose Specific Locations**:

Since settlement zones are determined by physical interconnection, developers must carefully choose WHERE to build. Considerations include:

1. **Zone Price Characteristics**: West Zone has 2x higher daily spreads than other zones
2. **Local Load Requirements**: Proximity to industrial customers (e.g., Permian Basin oil/gas)
3. **Transmission Availability**: Available interconnection capacity at substations
4. **Renewable Co-location**: Access to wind/solar congestion patterns
5. **Land and Permitting**: Practical considerations like land cost and local regulations

## 5. Risk Management Strategies

### 5.1 Hedging Instruments

**Tolling Agreements**:
- Third party operates battery for fixed fee
- Removes revenue volatility
- Growing market as BESS matures

**Shorter Tenor Hedges**:
- Monthly or quarterly contracts
- More flexibility than long-term agreements
- Emerging risk products

### 5.2 Operational Strategies

**Multi-Zone Participation**:
- Some BESS projects split capacity across zones
- Requires sophisticated optimization
- 48% revenue uplift possible with AI optimization

**Location Arbitrage**:
- Physical location in one zone, settlement in another
- Requires deep understanding of congestion patterns
- CRRs critical for risk management

## 6. Key Findings for Battalion Platform Software

### 6.1 Critical Software Capabilities Needed

1. **CRR Valuation Engine**: Model expected congestion values across all node pairs
2. **Settlement Point Optimizer**: Recommend optimal settlement points based on historical/forecast data
3. **Multi-Zone Portfolio Management**: Track positions across different zones
4. **Congestion Pattern Analytics**: Identify predictable congestion for CRR strategies
5. **Real-time Price Monitoring**: Track nodal prices for dispatch decisions

### 6.2 Data Requirements

- Historical nodal prices (all 4,000+ nodes)
- CRR auction results and clearing prices
- Transmission outage schedules
- Load forecasts by zone
- Renewable generation forecasts

### 6.3 Risk Analytics

- CRR portfolio value-at-risk
- Congestion exposure by settlement point
- Basis risk between physical and financial positions
- AS saturation indicators

## 7. Market Evolution & Future Considerations

### 7.1 Real-Time Co-Optimization (RTC+B)

**Launch Date**: December 5, 2025

**Impact on BESS**:
- Co-optimizes energy and AS in real-time
- New block products for batteries
- Better price convergence between DA and RT
- May reduce arbitrage opportunities

### 7.2 Transmission Expansion

**Major Projects**:
- 765kV lines will triple capacity vs 345kV
- $13+ billion in transmission investment
- Focus on West Texas congestion relief

### 7.3 Market Saturation Trends

- AS saturation expected by end of 2024
- Energy arbitrage becoming dominant revenue source
- Location and optimization increasingly critical
- CRRs more important as congestion increases

## 8. Recommendations for Battalion Platform

### 8.1 Immediate Priorities

1. **Develop CRR Analytics Module**: Essential for sophisticated BESS operators
2. **Build Settlement Point Database**: Map all possible settlement points with historical performance
3. **Create Zone Arbitrage Calculator**: Quantify benefits of different settlement strategies

### 8.2 Strategic Considerations

1. **Partner with CRR Trading Desks**: Access expertise in congestion trading
2. **Integrate Weather Data**: Renewable generation drives congestion patterns
3. **Monitor Regulatory Changes**: NPRR 1186 (state-of-charge) and other rules

### 8.3 Competitive Advantages

1. **Cross-Zone Optimization**: Few platforms optimize across settlement zones
2. **CRR Integration**: Most BESS software ignores CRR opportunities
3. **Congestion Forecasting**: AI/ML models for predicting congestion value

## Conclusion

BESS projects in ERCOT must settle at their physical interconnection points - they cannot choose different settlement zones. However, this constraint makes the initial siting decision critically important. The Crossett BESS example (actually located in Crane County, West Texas) shows how developers strategically choose locations in high-value zones like West Texas to capture superior economics.

As ERCOT's market evolves with RTC+B implementation and massive transmission expansion, the complexity of these strategies will increase. Battalion Platform should prioritize development of tools that help BESS operators navigate this complexity, particularly around CRR valuation, settlement point optimization, and cross-zone arbitrage.

The 71% revenue decline in 2024 makes clear that simple strategies no longer suffice. Success requires sophisticated software that can optimize across multiple revenue streams, manage complex hedging strategies, and adapt to rapidly changing market conditions.