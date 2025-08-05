# ERCOT Price Data File Structures

This document provides a comprehensive overview of all ERCOT price data file types, including file naming conventions, column headers, and example data entries. These files contain critical pricing information for energy and ancillary services in the ERCOT market.

## Key Distinction: Settlement Point Prices vs LMPs

**Settlement Point Prices (SPP)** include scarcity adders and are the correct prices to use for revenue calculations for Battery Energy Storage Systems (BESS). **Locational Marginal Prices (LMPs)** represent the marginal cost of energy at specific locations without scarcity adders.

## Data Directory Structure

The ERCOT price data is organized into the following directories:

### Energy Prices
1. **DAM_Settlement_Point_Prices** - Day-Ahead Market settlement prices (includes scarcity)
2. **Settlement_Point_Prices_at_Resource_Nodes,_Hubs_and_Load_Zones** - Real-time settlement prices
3. **DAM_Hourly_LMPs** - Day-Ahead Market locational marginal prices

### Ancillary Services Prices
4. **DAM_Clearing_Prices_for_Capacity** - Day-Ahead ancillary services clearing prices

### Congestion and Shadow Prices
5. **SCED_Shadow_Prices_and_Binding_Transmission_Constraints** - Real-time congestion data
6. **DAM_Shadow_Prices** - Day-Ahead congestion data

### System Information
7. **SCED_System_Lambda** - System-wide marginal price
8. **Settlement_Points_List_and_Electrical_Buses_Mapping** - Reference data
9. **SASM_Aggregated_Ancillary_Service_Offer_Curve** - AS supply curves

## 1. DAM Settlement Point Prices

### File Naming Convention
`cdr.00012331.0000000000000000.YYYYMMDD.HHMMSS.DAMSPNP4190.csv`

Example: `cdr.00012331.0000000000000000.20250804.123222.DAMSPNP4190.csv`

### Column Headers
```
DeliveryDate,HourEnding,SettlementPoint,SettlementPointPrice,DSTFlag
```

### Example Data Entries
```
08/05/2025,01:00,7RNCHSLR_ALL, 23.94,N
08/05/2025,01:00,A4_DGR1_RN, 24.67,N
```

### Key Fields
- **DeliveryDate**: Operating day (MM/DD/YYYY)
- **HourEnding**: Hour ending time (01:00 - 24:00)
- **SettlementPoint**: Settlement point name
- **SettlementPointPrice**: Price in $/MWh (includes scarcity adders)
- **DSTFlag**: Daylight Saving Time flag (Y/N)

## 2. Real-Time Settlement Point Prices

### File Naming Convention
`cdr.00012301.0000000000000000.YYYYMMDD.HHMMSS.SPPHLZNP6905_YYYYMMDD_HHMM.csv`

Example: `cdr.00012301.0000000000000000.20250730.084701.SPPHLZNP6905_20250730_0845.csv`

### Column Headers
```
DeliveryDate,DeliveryHour,DeliveryInterval,SettlementPointName,SettlementPointType,SettlementPointPrice,DSTFlag
```

### Example Data Entries
```
07/30/2025,9,3,7RNCHSLR_ALL,RN,19.27,N
07/30/2025,9,3,A4_DGR1_RN,RN,19.27,N
```

### Key Fields
- **DeliveryInterval**: 15-minute interval within the hour (1-4)
- **SettlementPointType**: Type of settlement point (RN=Resource Node, HB=Hub, LZ=Load Zone)
- **SettlementPointPrice**: Real-time price in $/MWh (includes scarcity)

## 3. DAM Hourly LMPs

### File Naming Convention
`cdr.00012328.0000000000000000.YYYYMMDD.HHMMSS.DAMHRLMPNP4183.csv`

Example: `cdr.00012328.0000000000000000.20250804.123224.DAMHRLMPNP4183.csv`

### Column Headers
```
DeliveryDate,HourEnding,BusName,LMP,DSTFlag
```

### Example Data Entries
```
08/05/2025,01:00,CADICKS_804V,23.2,N
08/05/2025,01:00,ADICKS__138C,23.2,N
```

### Key Fields
- **BusName**: Electrical bus identifier
- **LMP**: Locational Marginal Price in $/MWh (without scarcity adders)

## 4. DAM Clearing Prices for Capacity (Ancillary Services)

### File Naming Convention
`cdr.00012329.0000000000000000.YYYYMMDD.HHMMSS.DAMCPCNP4188.csv`

Example: `cdr.00012329.0000000000000000.20250804.123221.DAMCPCNP4188.csv`

### Column Headers
```
DeliveryDate,HourEnding,AncillaryType,MCPC,DSTFlag
```

### Example Data Entries
```
08/05/2025,01:00,ECRS,0.07,N
08/05/2025,01:00,REGDN,0.89,N
```

### Key Fields
- **AncillaryType**: Type of ancillary service
  - REGUP: Regulation Up
  - REGDN: Regulation Down
  - RRS: Responsive Reserve Service
  - ECRS: ERCOT Contingency Reserve Service
  - NSPIN: Non-Spinning Reserve
- **MCPC**: Market Clearing Price for Capacity in $/MW

## 5. SCED Shadow Prices and Binding Transmission Constraints

### File Naming Convention
`cdr.00012302.0000000000000000.YYYYMMDD.HHMMSS.SCEDBTCNP686.csv`

Example: `cdr.00012302.0000000000000000.20250804.200500.SCEDBTCNP686.csv`

### Column Headers
```
SCEDTimeStamp,RepeatedHourFlag,ConstraintID,ConstraintName,ContingencyName,ShadowPrice,MaxShadowPrice,Limit,Value,ViolatedMW,FromStation,ToStation,FromStationkV,ToStationkV,CCTStatus
```

### Example Data Entries
```
08/04/2025 19:55:10,N,1,LARDVN_LASCRU1_1,MFOAVLO5,74.53611,3500,265.4,265.4,0,LARDVNTH,LASCRUCE,138,138,COMP
08/04/2025 19:55:10,N,3,BRUNI_69_1,SLAQLOB8,161.60629,3500,34.7,34.7,0,BRUNI,BRUNI,138,69,NONCOMP
```

### Key Fields
- **ConstraintName**: Name of the transmission constraint
- **ShadowPrice**: Marginal cost of the constraint in $/MW
- **Limit**: Constraint limit in MW
- **Value**: Actual flow in MW
- **CCTStatus**: Competitive Constraint Test status (COMP/NONCOMP)

## 6. DAM Shadow Prices

### File Naming Convention
`cdr.00012332.0000000000000000.YYYYMMDD.HHMMSS.DASPBCNP4191.csv`

Example: `cdr.00012332.0000000000000000.20250804.123423.DASPBCNP4191.csv`

### Column Headers
```
DeliveryDate,HourEnding,ConstraintID,ConstraintName,ContingencyName,ConstraintLimit,ConstraintValue,ViolationAmount,ShadowPrice,FromStation,ToStation,FromStationkV,ToStationkV,DeliveryTime,DSTFlag
```

### Example Data Entries
```
08/05/2025,01:00,4, 6345__L, DWLFMET5, 206, 206, 0, 19.127, SNDHT, WLFSW, 138, 138, 08/05/2025 00:00:00,N
08/05/2025,01:00,1, MENPHT_YELWJC1_1, SPHMMAS9, 60, 60, 0, 0.463, YELWJCKT, MENPHTAP, 69, 69, 08/05/2025 00:00:00,N
```

## 7. SCED System Lambda

### File Naming Convention
`cdr.00013114.0000000000000000.YYYYMMDD.HHMMSS.SCEDSYSLAMBDANP6322_YYYYMMDD_HHMMSS.csv`

Example: `cdr.00013114.0000000000000000.20250804.205512.SCEDSYSLAMBDANP6322_20250804_205511.csv`

### Column Headers
```
SCEDTimeStamp,RepeatedHourFlag,SystemLambda
```

### Example Data Entries
```
08/04/2025 20:55:11,N,33.3174743652344
```

### Key Fields
- **SystemLambda**: System-wide marginal price in $/MWh

## 8. Settlement Points List and Electrical Buses Mapping

### Primary File: Settlement_Points_MMDDYYYY_HHMMSS.csv

### Column Headers
```
ELECTRICAL_BUS,NODE_NAME,PSSE_BUS_NAME,VOLTAGE_LEVEL,SUBSTATION,SETTLEMENT_LOAD_ZONE,RESOURCE_NODE,HUB_BUS_NAME,HUB,PSSE_BUS_NUMBER
```

### Example Data Entries
```
0001,0001,L_BUCKRA8_1Y,138,BUBORA,LZ_SOUTH,,,,7103
0001DUPV1_,0001,DUP_DUPV1_G1,13.8,DUPV1,LZ_SOUTH,,,,110681
```

### Additional Reference Files
- **CCP_Resource_Names**: Combined Cycle Plant resource mappings
- **Hub_Name_AND_DC_Ties**: Hub definitions and DC tie points
- **NOIE_Mapping**: Non-Opt-In Entity mappings
- **Resource_Node_to_Unit**: Resource node to unit mappings

## 9. SASM Aggregated Ancillary Service Offer Curve

### File Naming Convention
`cdr.00012350.0000000000000000.YYYYMMDD.HHMMSS.SASMASAGGNP6913.csv`

Example: `cdr.00012350.0000000000000000.20250801.230316.SASMASAGGNP6913.csv`

### Column Headers
```
DeliveryDate,HourEnding,SASMID,AncillaryType,Price,Quantity,DSTFlag
```

### Example Data Entries
```
08/02/2025, 01:00, 08/01/2025 22:54:02, ECRSS, 850, 212.2,N
08/02/2025, 01:00, 08/01/2025 22:54:02, ECRSS, 650, 192.2,N
```

### Key Fields
- **SASMID**: Supplemental AS Market run timestamp
- **Price**: Offer price in $/MW
- **Quantity**: Cumulative quantity available at or below this price (MW)

## BESS Revenue Calculation Using Price Files

### Energy Revenue

**Day-Ahead Market**:
- Use **DAM_Settlement_Point_Prices** files
- Revenue = MW × SettlementPointPrice × Hours
- Charge during low-price hours, discharge during high-price hours

**Real-Time Market**:
- Use **Settlement_Point_Prices_at_Resource_Nodes,_Hubs_and_Load_Zones** files
- Revenue based on deviations from DAM schedule
- Settlement at 15-minute intervals

### Ancillary Services Revenue

**Capacity Payments**:
- Use **DAM_Clearing_Prices_for_Capacity** files
- Revenue = MW_awarded × MCPC × Hours
- Services: REGUP, REGDN, RRS, ECRS, NSPIN

### Important Considerations

1. **Always use Settlement Point Prices (not LMPs) for revenue calculations**
   - SPPs include scarcity adders that affect actual payments
   - LMPs are useful for understanding congestion but not for settlements

2. **Time Granularity**:
   - DAM: Hourly prices
   - Real-Time: 15-minute intervals (4 per hour)

3. **Price Volatility**:
   - Monitor shadow prices for congestion impacts
   - System lambda indicates overall market conditions

4. **Location Matters**:
   - Prices vary significantly by settlement point
   - Use Settlement Points List to map resources to correct pricing nodes

5. **File Update Frequency**:
   - DAM files: Published daily after day-ahead market clears (~12:30 PM)
   - Real-time files: Published every 5-15 minutes
   - Reference files: Updated as needed (check timestamps)