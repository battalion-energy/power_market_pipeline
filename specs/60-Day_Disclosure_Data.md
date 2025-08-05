
ERCOT 60 Day Disclosure Data:

ERCOT releases some data with a 60-day lag. These include:
- The current operating plan
- The day-ahead market disclosure reports
- The SASM (which our ancillary services)
- The SCED disclosure reports  I download all the zip files into /data/ERCOT_data into the following directories. I then run a script to extract all those zip files into a subfolder called CSV in each of those directories.


Data Sourced from: https://data.ercot.com/


60-Day COP Adjustment Period Snapshot 
Complete Current Operating Plan (COP) data for each QSE snapshot on each hour; confidentiality expired 60 day data. Previously named Complete Current Operating Plan Data.

60-Day COP All Updates
This report will contain all iterative Current Operating Plan (COP) submissions where a change has occurred for the operating day. Previously named 60-Day Current Operating Plan.

60-Day DAM Disclosure Reports 
This report will contain all 60 day disclosure data related to DAM. The following individual files are included in the report: 60d_DAM_EnergyBidAwards; 60d_DAM_EnergyBids; 60d_DAM_EnergyOnlyOfferAwards; 60d_DAM_EnergyOnlyOffers; 60d_DAM_Generation_Resource_ASOffers; 60d_DAM_Gen_Resource_Data; 60d_DAM_Load_Resource_ASOffers; 60d_DAM_Load_Resource_Data; 60d_DAM_PTPObligationBidAwards; 60d_DAM_PTPObligationBids; 60d_DAM_PTP_Obligation_Option; 60d_DAM_PTP_Obligation_OptionAwards; and 60d_DAM_QSE_Self_Arranged_AS.

60-Day SASM Disclosure Reports 
This report will contain all 60 day disclosure data related to SASM for Generation and Load Resources. The following individual files are included in the report: 60d_SASM_Generation_Resource_AS_Offers-YY-MMM-DD.csv60d_SASM_Load_Resource_AS_Offers-YY-MMM-DD.csv60d_SASM_Generation_Resource_AS_Offer Awards-YY-MMM-DD.csv60d_SASM_Load_Resource_AS_Offer_Awards-YY-MMM-DD.csv

60-Day SCED Disclosure Reports 
This report will contain all 60 day disclosure data related to SCED. The following individual files are included in the report: 60d_HDL_LDL_ManOverride; 60d_Load_Resource_Data_in_SCED; 60d_SCED_DSR_Load_Data; 60d_SCED_EOC_Updates_in_OpHour; 60d_SCED_Gen_Resource_Data; 60d_SCED_QSE_Self_Arranged; and 60d_SCED_SMNE_GEN_RES.


These are the 5 folders included in the 60 Day (delayed) Disclosure Data from ERCOT.

60-Day_COP_Adjustment_Period_Snapshot
60-Day_COP_All_Updates
60-Day_DAM_Disclosure_Reports
60-Day_SASM_Disclosure_Reports
60-Day_SCED_Disclosure_Reports



Based on the ERCOT Nodal Protocols document and information from ERCOT's data structure, I can help explain the column headers in the ERCOT 60-day SCED disclosure Load Resource data file. Here are the key column definitions:
Core Resource Information

Resource Name: The unique identifier for the Load Resource
QSE: Qualified Scheduling Entity - A Market Participant that is qualified by ERCOT for communication with ERCOT for Resource Entities and LSEs and for settling payments and charges with ERCOT
DME: Decision Making Entity - The entity designated by Controllable Load Resources for operational control

Operational Limits (in MW)

HDL (High Dispatch Limit): For Load Resources, the limit calculated by ERCOT, using the QSE-established Maximum Power Consumption (MPC)
LDL (Low Dispatch Limit): For Load Resources, the limit calculated by ERCOT, using the QSE-established Low Power Consumption (LPC)
HASL (High Ancillary Service Limit): A dynamically calculated MW upper limit on a Resource to reserve the part of the Resource's capacity committed for Ancillary Service
LASL (Low Ancillary Service Limit): A dynamically calculated MW lower limit on a Resource to maintain the ability of the Resource to provide committed Ancillary Service

Power Consumption Parameters

MPC (Maximum Power Consumption): For a Load Resource, the limit established by the QSE, continuously updated in Real-Time, that describes the maximum sustained power consumption of a Load Resource
LPC (Low Power Consumption): For a Load Resource, the limit established by the QSE, continuously updated in Real-Time, that describes the minimum sustained power consumption of a Load Resource
Real Power Consumption: The actual power consumption of the Load Resource at the time

Dispatch Information

Base Point: The MW output level for a Resource produced by the Security-Constrained Economic Dispatch (SCED) process
Telemetered Resource Status: The real-time operational status of the resource as reported to ERCOT
SCED Bid to Buy Curve: The bid curve submitted by the Load Resource indicating willingness to purchase power at various price levels

Ancillary Service Responsibilities
The file includes columns for various ancillary service responsibilities in MW:

RRS: Responsive Reserve - An Ancillary Service that provides operating reserves
RRSFFR: Responsive Reserve from Fast Frequency Response
NonSpin: Non-Spinning Reserve - An Ancillary Service that is provided through use of the part of Off-Line Generation Resources that can be synchronized and ramped to a specified output level within 30 minutes
RegUp: Regulation Up Service - An Ancillary Service that provides capacity that can respond to signals from ERCOT within three to five seconds
RegDown: Regulation Down Service - An Ancillary Service that provides capacity that can respond to signals from ERCOT within three to five seconds
ECRS: ERCOT Contingency Reserve Service

Time Information

Interval Start/End: The beginning and end times of the SCED interval
SCED Time Stamp: The exact time when the SCED calculation was performed

These columns provide a comprehensive view of how Load Resources participate in ERCOT's real-time market, including their operational limits, dispatch levels, and ancillary service obligations.