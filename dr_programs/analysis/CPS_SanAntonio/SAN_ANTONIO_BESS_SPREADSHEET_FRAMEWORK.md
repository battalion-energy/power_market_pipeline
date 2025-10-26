# San Antonio BESS Analysis - Spreadsheet Framework
## Excel/Google Sheets Implementation Guide

**Purpose:** This document provides a complete framework for building the San Antonio 38-building BESS analysis in Excel or Google Sheets.

**File Structure:** Create a workbook with the following tabs:
1. Dashboard
2. Building Inventory
3. Value Calculations
4. CPS Energy Data
5. Financial Model
6. Sensitivity Analysis
7. Prioritization Scoring
8. Phase Planning

---

## TAB 1: Dashboard (Executive Summary)

### Section A: Key Metrics (Single Page View)

**Layout:**

```
╔════════════════════════════════════════════════════════════════════╗
║           SAN ANTONIO BESS DEPLOYMENT - EXECUTIVE DASHBOARD        ║
║                           38 Buildings                              ║
╠════════════════════════════════════════════════════════════════════╣
║                                                                     ║
║  FLEET SUMMARY                    │  FINANCIAL SUMMARY              ║
║  ─────────────────────            │  ───────────────────            ║
║  Total Capacity: [____] MW        │  Total CAPEX: $[_____]M        ║
║  Total Buildings: 38              │  Annual Revenue: $[_____]M     ║
║  Avg System Size: [___] kW        │  20-Year NPV: $[_____]M        ║
║  Total Battery: [____] MWh        │  IRR: [___]%                   ║
║                                   │  Payback: [___] years          ║
║───────────────────────────────────┼─────────────────────────────────║
║                                                                     ║
║  VALUE STACK ($/kW-year)          │  REVENUE BY SOURCE (Annual)     ║
║  ────────────────────────          │  ───────────────────────────   ║
║  [Bar Chart]                       │  [Pie Chart]                   ║
║  - Demand Charges: $___            │  - Demand Charges: $_____      ║
║  - CPS DR: $___                    │  - CPS DR: $_____              ║
║  - ERCOT Call: $___                │  - ERCOT Call: $_____          ║
║  - Infrastructure: $___            │  - Infrastructure: $_____      ║
║  - Resilience: $___                │  - Resilience: $_____          ║
║  - Arbitrage: $___                 │  - Arbitrage: $_____           ║
║  TOTAL: $___/kW-yr                 │  TOTAL: $_____/year            ║
║                                                                     ║
║───────────────────────────────────┼─────────────────────────────────║
║                                                                     ║
║  TOP 10 PRIORITY BUILDINGS         │  DEPLOYMENT PHASES              ║
║  ──────────────────────            │  ──────────────────            ║
║  [Table with scores]               │  Phase 1 (Yr 1): __ buildings  ║
║  1. [Building Name]: ___ pts       │  Phase 2 (Yr 2): __ buildings  ║
║  2. [Building Name]: ___ pts       │  Phase 3 (Yr 3): __ buildings  ║
║  ...                               │                                ║
║                                                                     ║
╚════════════════════════════════════════════════════════════════════╝
```

### Excel Formulas for Dashboard:

**Total Capacity:**
```excel
=SUM('Building Inventory'!E2:E39)
```

**Total CAPEX:**
```excel
='Financial Model'!B5
```

**Annual Revenue:**
```excel
='Financial Model'!B10
```

**Average $/kW-year:**
```excel
='Value Calculations'!B50
```

---

## TAB 2: Building Inventory

### Column Structure:

| Column | Header | Data Type | Formula/Source |
|--------|--------|-----------|----------------|
| A | Building ID | Text | B001-B038 |
| B | Building Name | Text | Manual entry |
| C | Address | Text | Manual entry |
| D | Building Type | Dropdown | Emergency Ops / Police / Fire / City Hall / Standard |
| E | Proposed BESS Size (kW) | Number | Manual or calculated |
| F | Battery Capacity (kWh) | Number | =E2*2 (assume 2-hour duration) |
| G | CPS Account Number | Text | From CPS Energy bills |
| H | Rate Schedule | Text | From CPS Energy (LGS, GS, etc.) |
| I | Peak Demand (kW) | Number | From interval data |
| J | Annual Energy (kWh) | Number | From CPS bills |
| K | Current Demand Charge ($/month) | Number | From CPS bills |
| L | Distribution Feeder ID | Text | From CPS Energy Grid Planning |
| M | Critical Facility | Yes/No | =IF(D2="Emergency Ops","Yes","No") |
| N | Priority Score | Number | =Link to Scoring tab |
| O | Phase Assignment | Number | 1, 2, or 3 |
| P | Notes | Text | Manual |

### Sample Data Row:

```excel
A2: B001
B2: Emergency Operations Center
C2: 8009 Ahern Dr, San Antonio, TX 78216
D2: Emergency Ops
E2: 150
F2: =E2*2  → 300
G2: 123456789
H2: LGS
I2: 185
J2: 850000
K2: 3500
L2: FEEDER-47A
M2: =IF(D2="Emergency Ops","Yes","No") → Yes
N2: =VLOOKUP(A2,'Prioritization Scoring'!A:C,3,FALSE)
O2: 1
P2: Critical facility, constrained feeder
```

### Conditional Formatting:

**Priority Score (Column N):**
- Red: 0-50 (Low priority)
- Yellow: 51-75 (Medium priority)
- Green: 76-100 (High priority)

**Phase Assignment (Column O):**
- Phase 1: Green fill
- Phase 2: Yellow fill
- Phase 3: Orange fill

---

## TAB 3: Value Calculations

### Section A: Building-Specific Revenue Streams

**Column Headers:**

| Column | Header | Formula/Calculation |
|--------|--------|---------------------|
| A | Building ID | =Link to Inventory |
| B | BESS Capacity (kW) | =Link to Inventory!E |
| C | **1. Demand Charge Reduction** | |
| D | - Summer Demand (kW) | From interval data |
| E | - Winter Demand (kW) | From interval data |
| F | - Summer Rate ($/kW-month) | From rate schedule |
| G | - Winter Rate ($/kW-month) | From rate schedule |
| H | - Peak Reduction (%) | Conservative: 80% |
| I | - Annual Savings ($) | =(D*F*4 + E*G*8)*H |
| J | - $/kW-year | =I/B |
| K | **2. CPS Energy DR** | |
| L | - Expected Events | Conservative: 20 / Typical: 25 / Aggressive: 30 |
| M | - Summer Rate ($/kW-season) | $73 (30-min) or $70 (2-hour) |
| N | - Winter Rate ($/kW-total) | $45 (30-min) or $40 (60-min) |
| O | - Annual Revenue ($) | =B*(M+(N*L/25)) |
| P | - $/kW-year | =O/B |
| Q | **3. ERCOT RT Call Option** | |
| R | - Call Option Value ($/kW-month) | $8.50 |
| S | - Annual Revenue ($) | =B*R*12 |
| T | - $/kW-year | =S/B |
| U | **4. Infrastructure Deferral** | |
| V | - Feeder Upgrade Cost ($) | From CPS Energy data |
| W | - Feeder Capacity (kW) | From CPS Energy data |
| X | - Years Deferred | 5-10 years |
| Y | - Proportional Deferral ($) | =V*(B/W) |
| Z | - Annualized Value ($) | =PMT(5%,20,Y) |
| AA | - $/kW-year | =Z/B |
| AB | **5. Resilience Value** | |
| AC | - VOLL ($/kWh) | From lookup table |
| AD | - Expected Outage Hours/year | 5-20 hrs |
| AE | - Critical Load (kW) | 50-80% of capacity |
| AF | - Annual Value ($) | =AC*AD*AE |
| AG | - $/kW-year | =AF/B |
| AH | **6. Energy Arbitrage** | |
| AI | - Annual Cycles | 200-250 |
| AJ | - Margin per Cycle ($) | Battery kWh * $0.058/kWh |
| AK | - Annual Value ($) | =AI*AJ |
| AL | - $/kW-year | =AK/B |
| AM | **TOTAL VALUE** | |
| AN | - Total Annual Revenue ($) | =SUM(I,O,S,Z,AF,AK) |
| AO | - $/kW-year | =AN/B |

### Sample Formulas (Building B001, 150 kW):

```excel
# 1. Demand Charge Reduction
D2: 185  # Summer peak demand from data
E2: 160  # Winter peak demand
F2: 22   # Summer rate $/kW-month
G2: 12   # Winter rate $/kW-month
H2: 0.80 # 80% reduction capability
I2: =(D2*F2*4 + E2*G2*8)*H2
   = (185*22*4 + 160*12*8)*0.80 = $25,152

# 2. CPS Energy DR
L2: 25   # Expected events
M2: 73   # Summer $/kW-season
N2: 45   # Winter $/kW-total
O2: =150*(73 + (45*25/25)) = 150*118 = $17,700

# 3. ERCOT Call Option
R2: 8.50
S2: =150*8.50*12 = $15,300

# 4. Infrastructure Deferral
V2: 2500000  # $2.5M transformer upgrade
W2: 500      # 500 kW feeder capacity
X2: 7        # Years deferred
Y2: =2500000*(150/500) = $750,000
Z2: =PMT(0.05,20,750000) = $60,165
   (Use Excel PMT function)

# 5. Resilience Value
AC2: 60      # $60/kWh VOLL (Emergency Ops)
AD2: 10      # 10 hours/year
AE2: 120     # 80% of 150 kW
AF2: =60*10*120 = $72,000

# 6. Energy Arbitrage
AI2: 220     # Annual cycles
AJ2: =(150*2)*0.058 = $17.40 per cycle (300 kWh battery, $0.058/kWh margin)
AK2: =220*17.40 = $3,828

# TOTAL
AN2: =SUM(25152,17700,15300,60165,72000,3828) = $194,145
AO2: =194145/150 = $1,294/kW-year
```

### Section B: Fleet Summary

**Row 50-60: Aggregate Metrics**

```excel
A50: "FLEET TOTALS"
B50: =SUM(B2:B39)  # Total kW
C50: "Average $/kW-year"
D50: =SUM(AN2:AN39)/SUM(B2:B39)  # Weighted average

A52: "REVENUE BY STREAM"
B52: "Demand Charges"
C52: =SUM(I2:I39)
B53: "CPS DR"
C53: =SUM(O2:O39)
B54: "ERCOT Call"
C54: =SUM(S2:S39)
B55: "Infrastructure"
C55: =SUM(Z2:Z39)
B56: "Resilience"
C56: =SUM(AF2:AF39)
B57: "Arbitrage"
C57: =SUM(AK2:AK39)
B58: "TOTAL"
C58: =SUM(C52:C57)
```

---

## TAB 4: CPS Energy Data

### Section A: Distribution System Data (From CPS Energy)

**Table 1: Feeder Analysis**

| Column | Header | Data Source |
|--------|--------|-------------|
| A | Feeder ID | CPS Energy Grid Planning |
| B | Substation | CPS Energy |
| C | Feeder Capacity (kVA) | CPS Energy |
| D | Current Peak Load (kW) | CPS Energy |
| E | Loading % | =D/C |
| F | Constraint Status | =IF(E>0.9,"Critical",IF(E>0.8,"Constrained","OK")) |
| G | Planned Upgrade | Yes/No from CPS Energy |
| H | Upgrade Cost ($) | CPS Energy estimate |
| I | Upgrade Timeline | Years (from CPS Energy) |
| J | Buildings on Feeder | Count from Inventory |
| K | Total BESS on Feeder (kW) | =SUMIF(Inventory!L:L,A2,Inventory!E:E) |
| L | Deferral Potential ($) | =H*(K/C) |

**Table 2: Rate Schedules**

| Column | Header | Value |
|--------|--------|-------|
| A | Rate Class | LGS, GS, MLG, etc. |
| B | Summer Demand ($/kW-month) | From CPS tariffs |
| C | Winter Demand ($/kW-month) | From CPS tariffs |
| D | Energy On-Peak ($/kWh) | From CPS tariffs |
| E | Energy Off-Peak ($/kWh) | From CPS tariffs |
| F | Fixed Charge ($/month) | From CPS tariffs |

**Sample Data:**

```excel
# Rate Schedule: LGS (Large General Service)
A2: LGS
B2: 22.50  # Summer demand
C2: 12.00  # Winter demand
D2: 0.072  # On-peak energy
E2: 0.045  # Off-peak energy
F2: 150    # Fixed charge
```

### Section B: Historical Event Data

**Table 3: DR Event History (Request from CPS Energy)**

| Column | Header | Notes |
|--------|--------|-------|
| A | Date | MM/DD/YYYY |
| B | Event Type | Summer / Winter / Bonus |
| C | Start Time | HH:MM |
| D | End Time | HH:MM |
| E | Duration (hours) | =HOURS(D-C) |
| F | Trigger | Heat wave / ERCOT emergency / etc. |
| G | ERCOT RT Price ($/MWh) | From ERCOT data |

**Analysis:**
```excel
# Summary Stats
"Average Events per Year": =COUNTIF(A:A,YEAR(TODAY()))/5
"Average Summer Events": =COUNTIFS(B:B,"Summer")
"Average Duration": =AVERAGE(E:E)
```

---

## TAB 5: Financial Model

### Section A: Project Costs

**Row Structure:**

```excel
A1: "CAPITAL EXPENDITURES (CAPEX)"

A3: "BESS Equipment & Installation"
A4: "Total Capacity (kW)"
B4: =SUM('Building Inventory'!E2:E39)

A5: "Unit Cost ($/kW)"
B5: 1200  # Input assumption
C5: "Range: $1,100-1,300/kW"

A6: "Total Equipment & Install"
B6: =B4*B5
C6: "Includes: batteries, inverters, BMS, installation, commissioning"

A8: "Soft Costs"
A9: "Engineering & Design (5%)"
B9: =B6*0.05

A10: "Permits & Interconnection (2%)"
B10: =B6*0.02

A11: "Project Management (3%)"
B11: =B6*0.03

A12: "Contingency (10%)"
B12: =B6*0.10

A14: "TOTAL CAPEX"
B14: =SUM(B6,B9:B12)
```

### Section B: Operating Costs (Annual)

```excel
A16: "ANNUAL OPERATING EXPENSES"

A18: "Operations & Maintenance"
A19: "O&M (% of CAPEX)"
B19: 0.015  # 1.5% assumption
C19: =B14*B19

A20: "Monitoring & Software"
B20: 15  # $/kW-year
C20: =B4*B20

A21: "Insurance"
B21: 0.005  # 0.5% of CAPEX
C21: =B14*B21

A22: "Property Tax (if applicable)"
B22: 0.000  # Municipal property - exempt
C22: =B14*B22

A24: "Total Annual O&M"
B24: =SUM(C19:C22)

A26: "Battery Replacement (Year 15)"
A27: "Replacement Cost (50% of CAPEX)"
B27: =B14*0.50
```

### Section C: Revenue Streams (Annual)

```excel
A30: "ANNUAL REVENUES"

A32: "Demand Charge Reduction"
B32: ='Value Calculations'!C52

A33: "CPS Energy DR"
B33: ='Value Calculations'!C53

A34: "ERCOT Call Option"
B34: ='Value Calculations'!C54

A35: "Infrastructure Deferral (Annualized)"
B35: ='Value Calculations'!C55

A36: "Resilience Value"
B36: ='Value Calculations'!C56

A37: "Energy Arbitrage"
B37: ='Value Calculations'!C57

A39: "TOTAL ANNUAL REVENUE"
B39: =SUM(B32:B37)
```

### Section D: Cash Flow Analysis (20-Year)

**Column Structure:**

| Column | Header | Formula |
|--------|--------|---------|
| A | Year | 0, 1, 2, ... 20 |
| B | CAPEX | =IF(A45=0,-$B$14,0) |
| C | Demand Charge Savings | =IF(A45>0,$B$32*1.02^A45,0) |
| D | CPS DR Revenue | =IF(A45>0,$B$33,0) |
| E | ERCOT Call | =IF(A45>0,$B$34,0) |
| F | Infrastructure | =IF(A45>0,$B$35,0) |
| G | Resilience | =IF(A45>0,$B$36,0) |
| H | Arbitrage | =IF(A45>0,$B$37,0) |
| I | Total Revenue | =SUM(C45:H45) |
| J | O&M Costs | =IF(A45>0,-$B$24,0) |
| K | Battery Replacement | =IF(A45=15,-$B$27,0) |
| L | Net Cash Flow | =B45+I45+J45+K45 |
| M | Cumulative Cash Flow | =L45+M44 |

**Example Row (Year 1):**
```excel
A46: 1
B46: 0
C46: =$B$32*1.02^1  # 2% annual escalation
D46: =$B$33
E46: =$B$34
F46: =$B$35
G46: =$B$36
H46: =$B$37
I46: =SUM(C46:H46)
J46: =-$B$24
K46: 0
L46: =SUM(B46:K46)
M46: =L46+M45
```

### Section E: Financial Metrics

```excel
A68: "KEY FINANCIAL METRICS"

A70: "Net Present Value (5% discount)"
B70: =NPV(0.05,L46:L65)+L45

A71: "Internal Rate of Return (IRR)"
B71: =IRR(L45:L65)

A72: "Modified IRR (MIRR, 5% reinvest)"
B72: =MIRR(L45:L65,0.05,0.05)

A73: "Simple Payback (years)"
B73: =MATCH(0,M45:M65,1)

A74: "Discounted Payback (years)"
B74: [Calculate using NPV of cash flows]

A76: "Benefit-Cost Ratio"
B76: =NPV(0.05,I46:I65)/(-B45)

A77: "Levelized Cost of Storage ($/kWh)"
B77: =(B14+NPV(0.05,J46:J65))/(B4*2*250*20)
    # CAPEX + PV(O&M) / (kW * 2hr * 250 cycles/yr * 20 years)
```

---

## TAB 6: Sensitivity Analysis

### One-Way Sensitivity Table

**Setup:**

```excel
A1: "SENSITIVITY ANALYSIS - NPV Impact"

A3: "Base Case NPV"
B3: ='Financial Model'!B70

A5: "Variable: Demand Charge Savings ($/kW-year)"
A6: "Base Case:"
B6: ='Value Calculations'!D50

A8: "Sensitivity Range:"
     -50%   -25%   Base   +25%   +50%
    ─────────────────────────────────
B9:  $110  $165   $220   $275   $330
```

**NPV Formula (cell B10):**
```excel
=NPV(0.05, [Modified cash flows with demand charge = B9]) + [CAPEX]
```

**Create Data Table:**
1. Enter sensitivity values in row 9
2. Reference NPV formula in A10
3. Select A9:F10
4. Data → What-If Analysis → Data Table
5. Column input cell: [Demand charge $/kW cell]

### Two-Way Sensitivity Table

```excel
A15: "TWO-WAY SENSITIVITY: NPV ($M)"
A16: "Demand Charge Savings ($/kW-yr) vs CAPEX ($/kW)"

        CAPEX →
Demand  $1,000  $1,100  $1,200  $1,300  $1,400
  ↓
$150
$180
$220 ← Base Case
$260
$300
```

**Setup:**
1. Row headers (A17:A21): Demand charge values
2. Column headers (B16:F16): CAPEX values
3. Reference formula in A16
4. Select A16:F21
5. Data Table with Row input = Demand charge, Column input = CAPEX

### Scenario Manager

**Create Named Scenarios:**

1. **Conservative:**
   - Demand Charges: $180/kW-yr
   - DR Events: 15/year
   - Infrastructure: $10/kW-yr
   - CAPEX: $1,300/kW

2. **Base Case:**
   - Demand Charges: $220/kW-yr
   - DR Events: 25/year
   - Infrastructure: $40/kW-yr
   - CAPEX: $1,200/kW

3. **Aggressive:**
   - Demand Charges: $280/kW-yr
   - DR Events: 30/year
   - Infrastructure: $100/kW-yr
   - CAPEX: $1,100/kW

**Scenario Summary Table:**

| Metric | Conservative | Base | Aggressive |
|--------|--------------|------|------------|
| Annual Revenue | $1.52M | $1.98M | $2.68M |
| CAPEX | $4.88M | $4.47M | $4.10M |
| NPV (20yr) | $12.5M | $22.3M | $35.2M |
| IRR | 28% | 48% | 67% |
| Payback | 3.2 yr | 2.1 yr | 1.5 yr |

---

## TAB 7: Prioritization Scoring

### Scoring Matrix

**Column Structure:**

| Col | Header | Weight | Formula |
|-----|--------|--------|---------|
| A | Building ID | - | Link to Inventory |
| B | Building Name | - | Link to Inventory |
| C | **1. Demand Charge Score** | 30% | |
| D | Annual Demand Savings ($) | - | Link to Value Calc |
| E | Score (0-30) | - | =MIN(30, D/(30000)*30) |
| F | **2. Infrastructure Score** | 25% | |
| G | Feeder Loading % | - | From CPS Data |
| H | Upgrade Timeline | - | From CPS Data |
| I | Constraint Multiplier | - | =IF(G>0.9,1,IF(G>0.8,0.7,IF(G>0.7,0.4,0.1))) |
| J | Timeline Multiplier | - | =IF(H<3,1,IF(H<7,0.7,0.4)) |
| K | Score (0-25) | - | =I*J*25 |
| L | **3. DR + ERCOT Score** | 20% | |
| M | Annual DR+ERCOT ($) | - | Link to Value Calc |
| N | Score (0-20) | - | =MIN(20, M/(15000)*20) |
| O | **4. Resilience Score** | 15% | |
| P | Critical Facility? | - | Link to Inventory |
| Q | Facility Type | - | Link to Inventory |
| R | Multiplier | - | Lookup table |
| S | Score (0-15) | - | =R*15 |
| T | **5. Site Readiness** | 10% | |
| U | Electrical Score | - | Manual (0-1) |
| V | Space Score | - | Manual (0-1) |
| W | Score (0-10) | - | =(U+V)/2*10 |
| X | **TOTAL SCORE** | 100% | =SUM(E,K,N,S,W) |
| Y | Rank | - | =RANK(X2,X$2:X$39,0) |
| Z | Phase | - | =IF(Y<=10,1,IF(Y<=25,2,3)) |

### Lookup Tables

**Resilience Multiplier:**

| Facility Type | Multiplier | Score at 15% |
|---------------|------------|--------------|
| Emergency Ops | 1.00 | 15.0 |
| 911 Center | 1.00 | 15.0 |
| Police/Fire | 0.80 | 12.0 |
| Water Control | 0.60 | 9.0 |
| City Hall | 0.40 | 6.0 |
| Standard | 0.10 | 1.5 |

**Scoring Example (Building B001: Emergency Ops Center):**

```excel
# 1. Demand Charge Score
D2: 25152  # Annual savings
E2: =MIN(30, 25152/30000*30) = 25.2

# 2. Infrastructure Score
G2: 0.92  # Feeder loading %
H2: 3     # Years to upgrade
I2: =IF(0.92>0.9,1,IF(0.92>0.8,0.7,0.4)) = 1.0
J2: =IF(3<3,1,IF(3<7,0.7,0.4)) = 0.7
K2: =1.0*0.7*25 = 17.5

# 3. DR + ERCOT Score
M2: 33000  # DR + ERCOT annual
N2: =MIN(20, 33000/15000*20) = 20.0

# 4. Resilience Score
P2: Yes
Q2: Emergency Ops
R2: 1.0 (from lookup)
S2: =1.0*15 = 15.0

# 5. Site Readiness
U2: 0.9  # Good electrical
V2: 0.8  # Adequate space
W2: =(0.9+0.8)/2*10 = 8.5

# TOTAL
X2: =SUM(25.2, 17.5, 20.0, 15.0, 8.5) = 86.2
Y2: =RANK(86.2, X$2:X$39, 0) = 1
Z2: =IF(1<=10, 1, IF(1<=25, 2, 3)) = 1
```

### Priority Ranking Output

**Format as Table:**

| Rank | Building | Score | Demand$ | Infra | DR/ERCOT | Resilience | Phase |
|------|----------|-------|---------|-------|----------|------------|-------|
| 1 | Emergency Ops | 86.2 | 25.2 | 17.5 | 20.0 | 15.0 | 1 |
| 2 | 911 Center | 84.1 | 23.5 | 18.0 | 19.6 | 15.0 | 1 |
| 3 | Fire Station #3 | 78.9 | 22.1 | 16.2 | 18.4 | 12.0 | 1 |
| ... | ... | ... | ... | ... | ... | ... | ... |

---

## TAB 8: Phase Planning

### Phase 1: Detailed Plan

**Table Structure:**

| Col | Header | Formula/Data |
|-----|--------|--------------|
| A | Rank | From Prioritization |
| B | Building | Phase 1 only (rank 1-10) |
| C | BESS Size (kW) | From Inventory |
| D | Battery (kWh) | =C*2 |
| E | CAPEX ($) | =C*$1200 |
| F | Annual Revenue ($) | From Value Calc |
| G | Simple Payback (yrs) | =E/F |
| H | Deployment Quarter | Q1, Q2, Q3, Q4 |
| I | Feeder | From Inventory |
| J | CPS Coordination Needed | Yes/No |
| K | Site Survey Status | Not Started / Complete |
| L | Design Status | Not Started / In Progress / Complete |
| M | Permit Status | Not Started / Submitted / Approved |
| N | Expected Commissioning | MM/DD/YYYY |

### Gantt Chart (Visual Timeline)

**Setup:**

```
        Q1-2026    Q2-2026    Q3-2026    Q4-2026
        ─────────────────────────────────────────
B001    ██████████
B002         ██████████
B003              ██████████
...

Legend:
██ Design & Engineering
██ Procurement
██ Construction
██ Commissioning
```

**Conditional Formatting:**
- Use color scales to show progress
- Red = Not started
- Yellow = In progress
- Green = Complete

### Phase Summaries

```excel
A50: "PHASE SUMMARY"

        Phase 1   Phase 2   Phase 3   TOTAL
        ─────────────────────────────────────
Buildings   10       15        13       38
Capacity    1.5 MW   1.9 MW    1.4 MW   4.8 MW
CAPEX       $1.8M    $2.3M     $1.7M    $5.8M
Annual Rev  $1.2M    $1.4M     $0.9M    $3.5M
Payback     1.5 yrs  1.6 yrs   1.9 yrs  1.7 yrs

Timeline
Start       Q1-26    Q1-27     Q1-28
Complete    Q4-26    Q4-27     Q4-28
```

---

## IMPLEMENTATION CHECKLIST

### Step 1: Setup (Day 1)
- [ ] Create Excel workbook with 8 tabs
- [ ] Format all column headers
- [ ] Set up named ranges for key inputs
- [ ] Create dropdown lists (building types, phases, etc.)
- [ ] Apply conditional formatting

### Step 2: Data Entry (Week 1)
- [ ] Enter 38 building addresses and names
- [ ] Input current peak demands (from interval data)
- [ ] Input annual energy usage (from bills)
- [ ] Input current demand charges (from bills)
- [ ] Assign building types (Emergency Ops, Police, etc.)

### Step 3: CPS Energy Data (Week 2)
- [ ] Request and input feeder IDs
- [ ] Input feeder loading data
- [ ] Input planned upgrades and costs
- [ ] Input rate schedules
- [ ] Input historical DR event data (if available)

### Step 4: Calculations (Week 3)
- [ ] Complete all value stream formulas
- [ ] Verify demand charge calculations
- [ ] Calculate infrastructure deferral values
- [ ] Build cash flow model
- [ ] Calculate NPV, IRR, payback

### Step 5: Analysis (Week 4)
- [ ] Run sensitivity analyses
- [ ] Complete prioritization scoring
- [ ] Rank all 38 buildings
- [ ] Assign to phases
- [ ] Create dashboard visuals

### Step 6: Validation (Week 4-5)
- [ ] Cross-check all formulas
- [ ] Verify against manual calculations
- [ ] Spot-check 5-10 buildings in detail
- [ ] Review with finance team
- [ ] Get sign-off on methodology

---

## ADVANCED FEATURES

### Macros & Automation

**1. Data Refresh Button**
```vba
Sub RefreshAllData()
    ' Recalculates all sheets
    Application.CalculateFullRebuild

    ' Updates dashboard charts
    Sheets("Dashboard").ChartObjects.Refresh

    MsgBox "Data refreshed successfully!"
End Sub
```

**2. Export Summary Report**
```vba
Sub ExportToWord()
    ' Exports dashboard and top 10 buildings to Word document
    ' [Detailed VBA code]
End Sub
```

### Power Query Integration

**Connect to External Data Sources:**
1. CPS Energy rate schedules (web scraping)
2. ERCOT RT prices (API or CSV import)
3. Building interval data (CSV import)

### Power BI Dashboard (Optional)

**For executive presentations:**
- Connect Excel model to Power BI
- Create interactive visuals
- Enable drill-down by building/phase
- Mobile-friendly dashboard

---

## FORMULAS LIBRARY

### Key Calculations Reference

**Net Present Value:**
```excel
=NPV(discount_rate, cashflow_range) + initial_investment
```

**Internal Rate of Return:**
```excel
=IRR(cashflow_range_including_year_0)
```

**Payback Period:**
```excel
=MATCH(0, cumulative_cashflow_range, 1)
```

**Annualized Infrastructure Value:**
```excel
=PMT(discount_rate, years, -deferred_cost)
```

**Weighted Average:**
```excel
=SUMPRODUCT(values_range, weights_range) / SUM(weights_range)
```

**Percentile Ranking:**
```excel
=PERCENTRANK(score_range, this_score)
```

---

## DATA VALIDATION RULES

### Input Cells (Yellow Background)

**BESS Capacity (kW):**
- Data Type: Whole number
- Minimum: 50
- Maximum: 500
- Alert: "Typical range: 75-200 kW"

**Building Type:**
- List: Emergency Ops, 911 Center, Police, Fire, Water, City Hall, Standard

**Phase Assignment:**
- List: 1, 2, 3

**Rate Schedule:**
- List: LGS, GS, MLG, SPS (from CPS Energy tariffs)

---

## VISUAL FORMATTING

### Color Scheme

**Primary Colors:**
- Header rows: Dark blue (#2E75B5)
- Input cells: Light yellow (#FFF2CC)
- Formula cells: Light blue (#DAEEF3)
- Output cells: Light green (#E2EFDA)

**Conditional Formatting:**
- **Priority Score:**
  - 0-50: Red scale
  - 51-75: Yellow scale
  - 76-100: Green scale

- **Financial Metrics:**
  - NPV: Green if >$10M
  - IRR: Green if >20%
  - Payback: Green if <3 years

---

## ERROR CHECKING

### Common Issues & Solutions

**#DIV/0! Error:**
- Cause: Division by zero (usually empty BESS capacity)
- Fix: =IFERROR(formula, 0)

**#VALUE! Error:**
- Cause: Text in number field
- Fix: Data validation + input checks

**#REF! Error:**
- Cause: Deleted cell reference
- Fix: Restore reference or use named ranges

**Circular Reference:**
- Cause: Formula references itself
- Fix: Break circular logic or use iterative calculation

---

## DOCUMENTATION

### Create a "Instructions" Tab

**Include:**
1. Model overview and purpose
2. Tab-by-tab description
3. Key assumptions list
4. Data sources and update frequency
5. Contact information for questions
6. Change log (version history)

**Example:**
```
═══════════════════════════════════════════════════════
SAN ANTONIO BESS ANALYSIS MODEL - USER GUIDE
═══════════════════════════════════════════════════════

PURPOSE:
This model calculates the financial viability of deploying
battery energy storage systems (BESS) across 38 City of
San Antonio buildings.

QUICK START:
1. Review Dashboard tab for executive summary
2. Update Building Inventory with your specific data
3. Input CPS Energy data when received
4. Review Prioritization Scoring for deployment order

KEY ASSUMPTIONS:
- BESS Cost: $1,200/kW (range: $1,100-1,300)
- Battery Duration: 2 hours
- Project Life: 20 years
- Discount Rate: 5%
- Demand Charge Escalation: 2%/year

CONTACT:
[Your Name]
[Email]
[Phone]

VERSION HISTORY:
v1.0 - 2025-10-25 - Initial model creation
```

---

## NEXT STEPS AFTER BUILDING MODEL

### 1. Collect Real Data
- [ ] Request 12 months interval data from CPS Energy
- [ ] Request distribution system analysis
- [ ] Obtain actual rate schedules for each building
- [ ] Conduct site visits for electrical assessment

### 2. Refine Assumptions
- [ ] Get vendor quotes for BESS costs
- [ ] Validate demand charge reduction % with utility
- [ ] Confirm DR event frequency with CPS Energy
- [ ] Update resilience VOLL for specific facilities

### 3. Stakeholder Review
- [ ] Present to City Energy Manager
- [ ] Review with CFO/Finance team
- [ ] Present to CPS Energy (infrastructure deferral)
- [ ] Briefing for City Council (if needed)

### 4. Scenario Planning
- [ ] Run "worst case" scenario (low events, high cost)
- [ ] Run "best case" scenario (enhanced CPS terms)
- [ ] Identify break-even conditions
- [ ] Determine go/no-go decision criteria

---

**This framework provides everything needed to build a comprehensive, data-driven analysis of the 38-building BESS deployment. Implement in Excel or Google Sheets following the structure above.**

**Questions or need help with specific formulas? Contact the analysis team.**
