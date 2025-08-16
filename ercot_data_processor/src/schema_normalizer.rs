use anyhow::Result;
use polars::prelude::*;

/// Normalizes DataFrames to have consistent schema, handling schema evolution
pub fn normalize_dataframe(df: DataFrame, expected_columns: &[&str]) -> Result<DataFrame> {
    let mut df = df;
    
    // Get current columns
    let current_cols: Vec<String> = df.get_column_names()
        .iter()
        .map(|s| s.to_string())
        .collect();
    
    // Add missing columns with default values
    for expected_col in expected_columns {
        if !current_cols.contains(&expected_col.to_string()) {
            // Add missing column with appropriate default
            let default_series = match *expected_col {
                "DSTFlag" => {
                    // DST flag was added in 2011, default to "N" for earlier data
                    Series::new("DSTFlag", vec!["N"; df.height()])
                },
                _ => {
                    // For other columns, use null values
                    Series::new_null(expected_col, df.height())
                }
            };
            df.with_column(default_series)?;
        }
    }
    
    // Select only the expected columns in the correct order
    let selected = df.select(expected_columns)?;
    
    Ok(selected)
}

/// Handles different CSV schemas for RT Settlement Point Prices
pub fn normalize_rt_prices(df: DataFrame) -> Result<DataFrame> {
    // Expected columns for RT prices
    let expected_columns = vec![
        "DeliveryDate",
        "DeliveryHour", 
        "DeliveryInterval",
        "SettlementPointName",
        "SettlementPointType",
        "SettlementPointPrice",
        "DSTFlag",
    ];
    
    normalize_dataframe(df, &expected_columns)
}

/// Handles different CSV schemas for DAM prices
pub fn normalize_dam_prices(df: DataFrame) -> Result<DataFrame> {
    let expected_columns = vec![
        "DeliveryDate",
        "HourEnding",
        "SettlementPoint",
        "SettlementPointPrice",
        "DSTFlag",
    ];
    
    normalize_dataframe(df, &expected_columns)
}

/// Handles different CSV schemas for AS prices
pub fn normalize_as_prices(df: DataFrame) -> Result<DataFrame> {
    let expected_columns = vec![
        "DeliveryDate",
        "HourEnding",
        "AncillaryType",
        "MCPC",
        "DSTFlag",
    ];
    
    normalize_dataframe(df, &expected_columns)
}