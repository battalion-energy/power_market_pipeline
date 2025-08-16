// COMPREHENSIVE LIST OF ALL ERCOT NUMERIC COLUMNS THAT NEED FLOAT64 OVERRIDE
// Based on official ERCOT documentation

pub fn get_all_ercot_float_columns() -> Vec<&'static str> {
    vec![
        // ============ PRICE COLUMNS ============
        // DAM Settlement Point Prices
        "SettlementPointPrice",
        
        // Real-Time Settlement Point Prices  
        "SettlementPointPrice",
        
        // DAM Hourly LMPs
        "LMP",
        
        // DAM Clearing Prices for Capacity (Ancillary Services)
        "MCPC",
        
        // SCED Shadow Prices
        "ShadowPrice",
        "MaxShadowPrice",
        "Limit",
        "Value",
        "ViolatedMW",
        
        // DAM Shadow Prices
        "ConstraintLimit",
        "ConstraintValue",
        "ViolationAmount",
        "ShadowPrice",
        
        // SCED System Lambda
        "SystemLambda",
        
        // SASM AS Offer Curve
        "Price",
        "Quantity",
        
        // ============ DISCLOSURE REPORT COLUMNS ============
        // SCED Gen Resource Data
        "LSL",
        "HSL",
        "Base Point",
        "Telemetered Net Output",
        "Output Schedule",
        "Output Schedule 2",
        
        // All Ancillary Service columns (multiple naming variants)
        "Ancillary Service RRS",
        "Ancillary Service RRSPFR", 
        "Ancillary Service RRSFFR",
        "Ancillary Service RRSUFR",
        "Ancillary Service Reg-Up",
        "Ancillary Service Reg-Down",
        "Ancillary Service REGUP",
        "Ancillary Service REGDN",
        "Ancillary Service Non-Spin",
        "Ancillary Service NSRS",
        "Ancillary Service NSPIN",
        "Ancillary Service ECRS",
        "Ancillary Service ECRSSD",
        
        // SCED Curve Points - ALL possible points (1-35 seen in data)
        // SCED1 Curve MW and Price
        "SCED1 Curve-MW1", "SCED1 Curve-Price1",
        "SCED1 Curve-MW2", "SCED1 Curve-Price2",
        "SCED1 Curve-MW3", "SCED1 Curve-Price3",
        "SCED1 Curve-MW4", "SCED1 Curve-Price4",
        "SCED1 Curve-MW5", "SCED1 Curve-Price5",
        "SCED1 Curve-MW6", "SCED1 Curve-Price6",
        "SCED1 Curve-MW7", "SCED1 Curve-Price7",
        "SCED1 Curve-MW8", "SCED1 Curve-Price8",
        "SCED1 Curve-MW9", "SCED1 Curve-Price9",
        "SCED1 Curve-MW10", "SCED1 Curve-Price10",
        "SCED1 Curve-MW11", "SCED1 Curve-Price11",
        "SCED1 Curve-MW12", "SCED1 Curve-Price12",
        "SCED1 Curve-MW13", "SCED1 Curve-Price13",
        "SCED1 Curve-MW14", "SCED1 Curve-Price14",
        "SCED1 Curve-MW15", "SCED1 Curve-Price15",
        "SCED1 Curve-MW16", "SCED1 Curve-Price16",
        "SCED1 Curve-MW17", "SCED1 Curve-Price17",
        "SCED1 Curve-MW18", "SCED1 Curve-Price18",
        "SCED1 Curve-MW19", "SCED1 Curve-Price19",
        "SCED1 Curve-MW20", "SCED1 Curve-Price20",
        "SCED1 Curve-MW21", "SCED1 Curve-Price21",
        "SCED1 Curve-MW22", "SCED1 Curve-Price22",
        "SCED1 Curve-MW23", "SCED1 Curve-Price23",
        "SCED1 Curve-MW24", "SCED1 Curve-Price24",
        "SCED1 Curve-MW25", "SCED1 Curve-Price25",
        "SCED1 Curve-MW26", "SCED1 Curve-Price26",
        "SCED1 Curve-MW27", "SCED1 Curve-Price27",
        "SCED1 Curve-MW28", "SCED1 Curve-Price28",
        "SCED1 Curve-MW29", "SCED1 Curve-Price29",
        "SCED1 Curve-MW30", "SCED1 Curve-Price30",
        "SCED1 Curve-MW31", "SCED1 Curve-Price31",
        "SCED1 Curve-MW32", "SCED1 Curve-Price32",
        "SCED1 Curve-MW33", "SCED1 Curve-Price33",
        "SCED1 Curve-MW34", "SCED1 Curve-Price34",
        "SCED1 Curve-MW35", "SCED1 Curve-Price35",
        
        // SCED2 Curve MW and Price
        "SCED2 Curve-MW1", "SCED2 Curve-Price1",
        "SCED2 Curve-MW2", "SCED2 Curve-Price2",
        "SCED2 Curve-MW3", "SCED2 Curve-Price3",
        "SCED2 Curve-MW4", "SCED2 Curve-Price4",
        "SCED2 Curve-MW5", "SCED2 Curve-Price5",
        "SCED2 Curve-MW6", "SCED2 Curve-Price6",
        "SCED2 Curve-MW7", "SCED2 Curve-Price7",
        "SCED2 Curve-MW8", "SCED2 Curve-Price8",
        "SCED2 Curve-MW9", "SCED2 Curve-Price9",
        "SCED2 Curve-MW10", "SCED2 Curve-Price10",
        "SCED2 Curve-MW11", "SCED2 Curve-Price11",
        "SCED2 Curve-MW12", "SCED2 Curve-Price12",
        "SCED2 Curve-MW13", "SCED2 Curve-Price13",
        "SCED2 Curve-MW14", "SCED2 Curve-Price14",
        "SCED2 Curve-MW15", "SCED2 Curve-Price15",
        "SCED2 Curve-MW16", "SCED2 Curve-Price16",
        "SCED2 Curve-MW17", "SCED2 Curve-Price17",
        "SCED2 Curve-MW18", "SCED2 Curve-Price18",
        "SCED2 Curve-MW19", "SCED2 Curve-Price19",
        "SCED2 Curve-MW20", "SCED2 Curve-Price20",
        "SCED2 Curve-MW21", "SCED2 Curve-Price21",
        "SCED2 Curve-MW22", "SCED2 Curve-Price22",
        "SCED2 Curve-MW23", "SCED2 Curve-Price23",
        "SCED2 Curve-MW24", "SCED2 Curve-Price24",
        "SCED2 Curve-MW25", "SCED2 Curve-Price25",
        "SCED2 Curve-MW26", "SCED2 Curve-Price26",
        "SCED2 Curve-MW27", "SCED2 Curve-Price27",
        "SCED2 Curve-MW28", "SCED2 Curve-Price28",
        "SCED2 Curve-MW29", "SCED2 Curve-Price29",
        "SCED2 Curve-MW30", "SCED2 Curve-Price30",
        "SCED2 Curve-MW31", "SCED2 Curve-Price31",
        "SCED2 Curve-MW32", "SCED2 Curve-Price32",
        "SCED2 Curve-MW33", "SCED2 Curve-Price33",
        "SCED2 Curve-MW34", "SCED2 Curve-Price34",
        "SCED2 Curve-MW35", "SCED2 Curve-Price35",
        
        // DAM Gen Resource columns
        "LSL",
        "HSL",
        "Awarded Quantity",
        "Energy Settlement Point Price",
        "RegUp Award",
        "RegDown Award",
        "RRS Award",
        "RRSPFR Award",
        "RRSFFR Award",
        "RRSUFR Award",
        "ECRS Award",
        "Non-Spin Award",
        "RegUp Settlement Point Price",
        "RegDown Settlement Point Price",
        "RRS Settlement Point Price",
        "RRSPFR Settlement Point Price",
        "RRSFFR Settlement Point Price",
        "RRSUFR Settlement Point Price",
        "ECRS Settlement Point Price",
        "Non-Spin Settlement Point Price",
        
        // COP Adjustment Period Snapshot columns
        "High Sustained Limit",
        "Low Sustained Limit",
        "High Emergency Limit",
        "Low Emergency Limit",
        "Minimum SOC",
        "Maximum SOC",
        "Hour Beginning Planned SOC",
        "Reg Up",
        "Reg Down",
        "RRSPFR",
        "RRSFFR",
        "RRSUFR",
        "NSPIN",
        "ECRS",
        
        // Additional numeric columns that might appear
        "MW",
        "MWh",
        "Capacity",
        "Load",
        "Generation",
        "Net Output",
        "Telemetered Output",
        "Scheduled Output",
        "Base Load",
        "Peak Load",
        "Energy",
        "AS Award",
        "AS Price",
        "Settlement Price",
        "Clearing Price",
        "Marginal Price",
        "Energy Price",
        "Capacity Price",
        
        // Voltage levels (sometimes stored as numbers)
        "FromStationkV",
        "ToStationkV",
        "VOLTAGE_LEVEL",
        
        // Interval and hour columns (when stored as numbers)
        "DeliveryHour",
        "DeliveryInterval",
        "HourEnding",  // Sometimes numeric like "1" instead of "01:00"
        "Hour Ending", // Space variant
        
        // Bus numbers
        "PSSE_BUS_NUMBER",
        
        // Any column with these patterns should be Float64
        // Patterns to match: *Price*, *MW*, *Limit*, *Value*, *Quantity*, *Award*
    ]
}

// Function to check if a column name should be Float64
pub fn should_be_float64(column_name: &str) -> bool {
    // Exact matches
    if get_all_ercot_float_columns().contains(&column_name) {
        return true;
    }
    
    // Pattern matches (case-insensitive)
    let lower = column_name.to_lowercase();
    
    // Any column containing these words should be Float64
    let float_patterns = [
        "price", "mw", "limit", "value", "quantity", "award",
        "capacity", "load", "generation", "output", "energy",
        "lambda", "shadow", "mcpc", "lmp", "spp", "settlement",
        "curve", "point", "lsl", "hsl", "soc", "schedule",
        "telemetered", "net", "base", "reg", "rrs", "ecrs",
        "nspin", "nsrs", "ancillary", "as_", "constraint",
        "violation", "kv", "voltage", "bus", "hour", "interval"
    ];
    
    for pattern in &float_patterns {
        if lower.contains(pattern) {
            return true;
        }
    }
    
    false
}