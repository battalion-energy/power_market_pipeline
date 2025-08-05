# ERCOT 60-Day Disclosure Data File Structures

This document provides a comprehensive overview of all file types in ERCOT's 60-day disclosure data, including file naming conventions, column headers, and example data entries. The data is released with a 60-day lag for confidentiality reasons.

## Data Directory Structure

The ERCOT 60-day disclosure data is organized into five main directories:

1. **60-Day_COP_Adjustment_Period_Snapshot** - Current Operating Plan snapshots
2. **60-Day_COP_All_Updates** - All iterative COP submissions with changes
3. **60-Day_DAM_Disclosure_Reports** - Day-Ahead Market disclosure data
4. **60-Day_SASM_Disclosure_Reports** - Supplemental Ancillary Services Market data
5. **60-Day_SCED_Disclosure_Reports** - Security-Constrained Economic Dispatch data

## 1. 60-Day COP Adjustment Period Snapshot

### File Naming Convention
`60d_COP_Adjustment_Period_Snapshot-DD-MMM-YY.csv`

Example: `60d_COP_Adjustment_Period_Snapshot-01-JUL-25.csv`

### Column Headers
```
"Delivery Date","QSE Name","Resource Name","Hour Ending","Status","High Sustained Limit","Low Sustained Limit","High Emergency Limit","Low Emergency Limit","Reg Up","Reg Down","RRSPFR","RRSFFR","RRSUFR","NSPIN","ECRS","Minimum SOC","Maximum SOC","Hour Beginning Planned SOC"
```

### Example Data Entries
```
"05/02/2025","QAEN","ARAGORN_UNIT1","01:00","ON","0","0","0","0","0","0","0","0","0","0","0","0","0","0"
"05/02/2025","QAEN","ARAGORN_UNIT1","02:00","ON","0","0","0","0","0","0","0","0","0","0","0","0","0","0"
```

## 2. 60-Day COP All Updates

### File Naming Convention
`60d_COP_All_Updates-DD-MMM-YY.csv`

Example: `60d_COP_All_Updates-28-FEB-25.csv`

### Column Headers
```
"Delivery Date","QSE Name","Resource Name","Hour Ending","Status","High Sustained Limit","Low Sustained Limit","High Emergency Limit","Low Emergency Limit","Reg Up","Reg Down","RRSPFR","RRSFFR","RRSUFR","NSPIN","ECRS","Minimum SOC","Maximum SOC","Hour Beginning Planned SOC","Cancel Flag","Update Time","Submit Time"
```

### Example Data Entries
```
"12/30/2024","QAEN","ARAGORN_UNIT1","1:00","ON","0.00","0.00","187.20","0.00","0.00","0.00","0.00","0.00","0.00","0.00","0.00","0.00","0.00","0.00","N","12/22/2024 09:01:33","12/22/2024 09:01:32"
"12/30/2024","QAEN","ARAGORN_UNIT1","1:00","ON","0.00","0.00","0.00","0.00","0.00","0.00","0.00","0.00","0.00","0.00","0.00","0.00","0.00","0.00","N","12/23/2024 01:51:14","12/22/2024 09:01:32"
```

## 3. 60-Day DAM Disclosure Reports

This directory contains 13 different file types related to the Day-Ahead Market:

### 3.1 DAM Energy Bids

**File Naming**: `60d_DAM_EnergyBids-DD-MMM-YY.csv`

**Column Headers**:
```
"Delivery Date","Hour Ending","Settlement Point","QSE Name","Energy Only Bid MW1","Energy Only Bid Price1","Energy Only Bid MW2","Energy Only Bid Price2","Energy Only Bid MW3","Energy Only Bid Price3","Energy Only Bid MW4","Energy Only Bid Price4","Energy Only Bid MW5","Energy Only Bid Price5","Energy Only Bid MW6","Energy Only Bid Price6","Energy Only Bid MW7","Energy Only Bid Price7","Energy Only Bid MW8","Energy Only Bid Price8","Energy Only Bid MW9","Energy Only Bid Price9","Energy Only Bid MW10","Energy Only Bid Price10","Energy Only Bid ID","Multi-Hour Block Indicator","Block/Curve indicator"
```

**Example Data**:
```
"12/23/2024","1","ADL_RN","QCONO3","1","15","","","","","","","","","","","","","","","","","","","233223533","N","V"
"12/23/2024","1","ADL_RN","QCONO3","1","18","","","","","","","","","","","","","","","","","","","233223532","N","V"
```

### 3.2 DAM Generation Resource Data

**File Naming**: `60d_DAM_Gen_Resource_Data-DD-MMM-YY.csv`

**Column Headers**:
```
"Delivery Date","Hour Ending","QSE","DME","Resource Name","Resource Type","QSE submitted Curve-MW1","QSE submitted Curve-Price1","QSE submitted Curve-MW2","QSE submitted Curve-Price2","QSE submitted Curve-MW3","QSE submitted Curve-Price3","QSE submitted Curve-MW4","QSE submitted Curve-Price4","QSE submitted Curve-MW5","QSE submitted Curve-Price5","QSE submitted Curve-MW6","QSE submitted Curve-Price6","QSE submitted Curve-MW7","QSE submitted Curve-Price7","QSE submitted Curve-MW8","QSE submitted Curve-Price8","QSE submitted Curve-MW9","QSE submitted Curve-Price9","QSE submitted Curve-MW10","QSE submitted Curve-Price10","Start Up Hot","Start Up Inter","Start Up Cold","Min Gen Cost","HSL","LSL","Resource Status","Awarded Quantity","Settlement Point Name","Energy Settlement Point Price","RegUp Awarded","RegUp MCPC","RegDown Awarded","RegDown MCPC","RRSPFR Awarded","RRSFFR Awarded","RRSUFR Awarded","RRS MCPC","ECRSSD Awarded","ECRS MCPC","NonSpin Awarded","NonSpin MCPC"
```
Note: that Resource Type = "PWRSTR" for BESS energy storage systems!

**Example Data**:
```
"12/23/2024","1","QSUE66","Y7VSOL","7RNCHSLR_UNIT1","PVGR","","","","","","","","","","","","","","","","","","","","","","","","","0","0","ON","","7RNCHSLR_ALL","14","","","","","","","","","","","",""
"12/23/2024","1","QSUE66","Y7VSOL","7RNCHSLR_UNIT2","PVGR","","","","","","","","","","","","","","","","","","","","","","","","","0","0","ON","","7RNCHSLR_ALL","14","","","","","","","","","","","",""
```

### Other DAM File Types (677 files each):
- `60d_DAM_EnergyBidAwards`
- `60d_DAM_EnergyOnlyOffers`
- `60d_DAM_EnergyOnlyOfferAwards`
- `60d_DAM_Generation_Resource_ASOffers`
- `60d_DAM_Load_Resource_Data`
- `60d_DAM_Load_Resource_ASOffers`
- `60d_DAM_PTPObligationBids`
- `60d_DAM_PTPObligationBidAwards`
- `60d_DAM_QSE_Self_Arranged_AS`

### PTP Option Files (656 files each):
- `60d_DAM_PTP_Obligation_Option`
- `60d_DAM_PTP_Obligation_OptionAwards`

### CRR Files (21 files each):
- `60d_DAM_CRROffers`
- `60d_DAM_CRROfferAwards`

## 4. 60-Day SASM Disclosure Reports

The SASM directory contains 4 file types (502 files each):

### 4.1 SASM Generation Resource AS Offers

**File Naming**: `60d_SASM_Generation_Resource_AS_Offers-DD-MMM-YY.csv`

**Column Headers**:
```
"Delivery Date","SASM ID","Hour Ending","QSE NAME","DME","Resource Name","Multi-Hour Block Flag","BLOCK INDICATOR1","PRICE1 RRSPFR","PRICE1 RRSFFR","PRICE1 RRSUFR","PRICE1 ECRS","PRICE1 OFFEC","PRICE1 ONLINE NONSPIN","PRICE1 REGUP","PRICE1 REGDOWN","PRICE1 OFFLINE NONSPIN","QUANTITY MW1","BLOCK INDICATOR2","PRICE2 RRSPFR","PRICE2 RRSFFR","PRICE2 RRSUFR","PRICE2 ECRS","PRICE2 OFFEC","PRICE2 ONLINE NONSPIN","PRICE2 REGUP","PRICE2 REGDOWN","PRICE2 OFFLINE NONSPIN","QUANTITY MW2","BLOCK INDICATOR3","PRICE3 RRSPFR","PRICE3 RRSFFR","PRICE3 RRSUFR","PRICE3 ECRS","PRICE3 OFFEC","PRICE3 ONLINE NONSPIN","PRICE3 REGUP","PRICE3 REGDOWN","PRICE3 OFFLINE NONSPIN","QUANTITY MW3","BLOCK INDICATOR4","PRICE4 RRSPFR","PRICE4 RRSFFR","PRICE4 RRSUFR","PRICE4 ECRS","PRICE4 OFFEC","PRICE4 ONLINE NONSPIN","PRICE4 REGUP","PRICE4 REGDOWN","PRICE4 OFFLINE NONSPIN","QUANTITY MW4","BLOCK INDICATOR5","PRICE5 RRSPFR","PRICE5 RRSFFR","PRICE5 RRSUFR","PRICE5 ECRS","PRICE5 OFFEC","PRICE5 ONLINE NONSPIN","PRICE5 REGUP","PRICE5 REGDOWN","PRICE5 OFFLINE NONSPIN","QUANTITY MW5"
```

**Example Data**:
```
"05/02/2025","05/02/2025 09:40:13","01","QSUE53","YPLUSP","ANEM_ESS_BESS1","N","V","3430","","","3430","","3430","3430","","","19.5","V","2450","","","2450","","2450","2450","","","29.2","V","343","","","343","","343","343","","","58.5","V","1470","","","1470","","1470","1470","","","39","V","735","","","735","","735","735","","","48.8"
"05/02/2025","05/02/2025 09:40:13","01","QBTU","YBRYN","ATKINS_ATKINSG7","N","V","","","","","","500","","","","12","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","",""
```

### Other SASM File Types:
- `60d_SASM_Generation_Resource_AS_Offer_Awards`
- `60d_SASM_Load_Resource_AS_Offers`
- `60d_SASM_Load_Resource_AS_Offer_Awards`

## 5. 60-Day SCED Disclosure Reports

The SCED directory contains the most diverse set of files:

### 5.1 SCED Generation Resource Data (549 files)

**File Naming**: `60d_SCED_Gen_Resource_Data-DD-MMM-YY.csv`

**Column Headers** (abbreviated due to length - contains 187 columns):
```
"SCED Time Stamp","Repeated Hour Flag","QSE","DME","Resource Name","Resource Type","SCED1 Curve-MW1","SCED1 Curve-Price1"...[up to 35 MW/Price pairs for SCED1 and SCED2]..."Output Schedule","HSL","HASL","HDL","LSL","LASL","LDL","Telemetered Resource Status","Base Point","Telemetered Net Output","Ancillary Service REGUP","Ancillary Service REGDN","Ancillary Service RRS","Ancillary Service RRSFFR","Ancillary Service NSRS","Ancillary Service ECRS","Bid_Type","Start Up Cold Offer","Start Up Hot Offer","Start Up Inter Offer","Min Gen Cost"...[TPO offers]..."Proxy Extension"
```

**Example Data**:
```
"06/04/2025 00:00:25","N","QSUE66","Y7VSOL","7RNCHSLR_UNIT1","PVGR","0","-250","0","-3.00999999046326","139.199996948242","-3","139.199996948242","-3"...
"06/04/2025 00:00:25","N","QSUE66","Y7VSOL","7RNCHSLR_UNIT2","PVGR","0","-250","0","-3.00999999046326","95.1999969482422","-3","95.1999969482422","-3"...
```

### 5.2 SCED Load Resource Data (546 files)

**File Naming**: `60d_Load_Resource_Data_in_SCED-DD-MMM-YY.csv`

**Column Headers**:
```
"SCED Time Stamp","Repeated Hour Flag","QSE","DME","Resource Name","Telemetered Resource Status","Max Power Consumption","Low Power Consumption","Real Power Consumption","AS Responsibility for RRS","AS Responsibility for RRSFFR","AS Responsibility for NonSpin","AS Responsibility for RegUp","AS Responsibility for RegDown","AS Responsibility for ECRS","SCED Bid to Buy Curve-MW1","SCED Bid to Buy Curve-Price1","SCED Bid to Buy Curve-MW2","SCED Bid to Buy Curve-Price2","SCED Bid to Buy Curve-MW3","SCED Bid to Buy Curve-Price3","SCED Bid to Buy Curve-MW4","SCED Bid to Buy Curve-Price4","SCED Bid to Buy Curve-MW5","SCED Bid to Buy Curve-Price5","SCED Bid to Buy Curve-MW6","SCED Bid to Buy Curve-Price6","SCED Bid to Buy Curve-MW7","SCED Bid to Buy Curve-Price7","SCED Bid to Buy Curve-MW8","SCED Bid to Buy Curve-Price8","SCED Bid to Buy Curve-MW9","SCED Bid to Buy Curve-Price9","SCED Bid to Buy Curve-MW10","SCED Bid to Buy Curve-Price10","HASL","HDL","LASL","LDL","Base Point"
```

**Example Data**:
```
"06/04/2025 00:00:25","N","QEDE20","","AAPIPLNC_LD3","ONRL","24",".1","14.1","14","0","0","0","0","0","","","","","","","","","","","","","","","","","","","","","0","0","0","0",""
"06/04/2025 00:00:25","N","QEDE20","","AAPIPLNC_LD4","ONRL","15","2.6","10.6","8","0","0","0","0","0","","","","","","","","","","","","","","","","","","","","","0","0","0","0",""
```

### 5.3 SCED SMNE Generation Resources (549 files)

**File Naming**: `60d_SCED_SMNE_GEN_RES-DD-MMM-YY.csv`

**Column Headers**:
```
"Interval Time","Interval Number","Resource Code","Interval Value"
```

**Example Data**:
```
"06/04/2025 00:14:59","1","19599_1_PV_A1","0"
"06/04/2025 00:14:59","1","19599_2_PV_2","0"
```

### 5.4 SCED QSE Self-Arranged Ancillary Services (549 files)

**File Naming**: `60d_SCED_QSE_Self_Arranged_AS-DD-MMM-YY.csv`

**Column Headers**:
```
"SCED Time Stamp","Repeated Hour Flag","QSE","REGUP","REGDN","NSPIN","NSPNM","RRSPFR","RRSFFR","RRSUFR","ECRSS","ECRSM"
```

**Example Data**:
```
"06/04/2025 00:00:25","N","QAEN","0","17.5","105","","0","0","0","41","0"
"06/04/2025 00:00:25","N","QALTRE","","","22","","15","0","14.5","",""
```

### 5.5 SCED DSR Load Data (549 files)

**File Naming**: `60d_SCED_DSR_Load_Data-DD-MMM-YY.csv`

**Column Headers**:
```
"SCED Time Stamp","Repeated Hour Flag","QSE Name","Total Telemetered DSR Loads"
```

**Example Data**:
```
"06/04/2025 00:00:25","N","ERCCRE","0"
"06/04/2025 00:00:25","N","QAEN","0"
```

### 5.6 HDL/LDL Manual Override (549 files)

**File Naming**: `60d_HDL_LDL_ManOverride-DD-MMM-YY.csv`

**Column Headers**:
```
"SCED Timestamp","Repeated Hour Flag","Participant Name","Resource Name","HDL Original","HDL Manual","HDL Final","LDL Original","LDL Manual","LDL Final","Reason Code"
```

**Note**: This file typically contains only headers when no manual overrides were performed.

### 5.7 SCED EOC Updates in Operating Hour (286 files)

**File Naming**: `60d_SCED_EOC_Updates_in_OpHour-DD-MMM-YY.csv`

**Column Headers**:
```
"Delivery Date","Delivery Hour","Resource Name","Reason","Count of Updates During Operating Hour"
```

**Example Data**:
```
"06/04/2025","1","AE_BESS","SOC Management; SOC Management; SOC Management; SOC Management; SOC Management; SOC Management","6"
"06/04/2025","2","AE_BESS","SOC Management; SOC Management; SOC Management; SOC Management; SOC Management; SOC Management","6"
```

### Aggregated Data Files (56 files each):
- `2d_Agg_Output_Sched_[Region]` (West, South, North, Houston, Total)
- `2d_Agg_Load_Summary_[Region]` (West, South, North, Houston, Total)
- `2d_Agg_Gen_Summary_[Region]` (West, South, North, Houston, Total)
- `2d_Agg_DSR_Loads`

### 48-Hour Aggregated Data (46 files each):
- `48h_Agg_Output_Sched_[Region]`
- `48h_Agg_Load_Summary_[Region]`
- `48h_Agg_Gen_Summary_[Region]`
- `48h_Agg_DSR_Loads`

