#!/bin/bash
for month in JAN FEB MAR APR MAY JUN JUL AUG; do 
  count=$(ls /pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/60-Day_DAM_Disclosure_Reports/csv/ | grep "Gen_Resource_Data.*-${month}-25\.csv" | wc -l)
  echo "$month 2025: $count files"
done
