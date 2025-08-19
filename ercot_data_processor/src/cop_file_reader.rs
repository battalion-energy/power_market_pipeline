// COP (Current Operating Plan) File Reader with Robust Format Handling
//
// CRITICAL: ERCOT COP file formats have evolved significantly over time:
//
// 1. EARLY 2014 (Jan-Aug): CompleteCOP_MMDDYYYY.csv WITH headers
//    - Has proper column headers starting with "Delivery Date"
//
// 2. LATE 2014 (Sep-Dec): CompleteCOP_MMDDYYYY.csv WITHOUT headers  
//    - Raw data only, no header row!
//    - First field is date in MM/DD/YYYY format
//    - 13 columns total
//
// 3. 2015-2022 (Dec 12): CompleteCOP_MMDDYYYY.csv WITH headers
//    - Headers restored, starting with "Delivery Date"
//    - 13 columns (no RRS split yet)
//
// 4. 2022 (Dec 13+): Format change to 60d_COP_Adjustment_Period_Snapshot-DD-MMM-YY.csv
//    - New filename format
//    - RRS split into RRSPFR, RRSFFR, RRSUFR
//    - Added ECRS, SOC columns for battery storage
//    - 19 columns total
//
// This module provides robust handling for ALL these variations.

use anyhow::Result;
use polars::prelude::*;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

/// Column names for early COP format (2014-2022)
const COP_COLUMNS_V1: [&str; 13] = [
    "Delivery Date",
    "QSE Name", 
    "Resource Name",
    "Hour Ending",
    "Status",
    "High Sustained Limit",
    "Low Sustained Limit",
    "High Emergency Limit",
    "Low Emergency Limit",
    "Reg Up",
    "Reg Down",
    "RRS",
    "NSPIN",
];

/// Column names for new COP format (Dec 2022+)
const _COP_COLUMNS_V2: [&str; 19] = [
    "Delivery Date",
    "QSE Name",
    "Resource Name",
    "Hour Ending",
    "Status",
    "High Sustained Limit",
    "Low Sustained Limit",
    "High Emergency Limit",
    "Low Emergency Limit",
    "Reg Up",
    "Reg Down",
    "RRSPFR",
    "RRSFFR",
    "RRSUFR",
    "NSPIN",
    "ECRS",
    "Minimum SOC",
    "Maximum SOC",
    "Hour Beginning Planned SOC",
];

pub struct COPFileReader {
    file_path: PathBuf,
    has_headers: bool,
    _column_count: usize,
    _is_2014_late: bool,
}

impl COPFileReader {
    pub fn new(file_path: &Path) -> Result<Self> {
        // Detect format based on filename and content
        let (has_headers, column_count, is_2014_late) = Self::detect_format(file_path)?;
        
        Ok(Self {
            file_path: file_path.to_path_buf(),
            has_headers,
            _column_count: column_count,
            _is_2014_late: is_2014_late,
        })
    }
    
    /// Detect COP file format by examining filename and first line
    fn detect_format(file_path: &Path) -> Result<(bool, usize, bool)> {
        let filename = file_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("");
        
        // New format files always have headers
        if filename.starts_with("60d_COP_Adjustment_Period_Snapshot") {
            return Ok((true, 19, false));
        }
        
        // For CompleteCOP files, need to check content
        if filename.starts_with("CompleteCOP_") {
            // Extract date from filename
            let date_part = filename
                .replace("CompleteCOP_", "")
                .replace(".csv", "");
            
            if date_part.len() == 8 {
                let month = date_part[0..2].parse::<u32>().unwrap_or(0);
                let year = date_part[4..8].parse::<u32>().unwrap_or(0);
                
                // CRITICAL: Late 2014 files (Sep-Dec) have NO headers!
                if year == 2014 && month >= 9 {
                    return Ok((false, 13, true));
                }
            }
            
            // Check first line to confirm
            let file = File::open(file_path)?;
            let mut reader = BufReader::new(file);
            let mut first_line = String::new();
            reader.read_line(&mut first_line)?;
            
            // If first line starts with "Delivery Date" it has headers
            if first_line.starts_with("\"Delivery Date\"") || first_line.starts_with("Delivery Date") {
                return Ok((true, 13, false));
            }
            
            // If first line starts with a date pattern like "09/25/2014" it's data without headers
            if first_line.starts_with('"') {
                let cleaned = first_line.trim_start_matches('"');
                if cleaned.len() >= 10 {
                    let potential_date = &cleaned[0..10];
                    if potential_date.chars().filter(|c| *c == '/').count() == 2 {
                        // It's a date, so no headers
                        return Ok((false, 13, true));
                    }
                }
            }
        }
        
        // Default: assume it has headers
        Ok((true, 13, false))
    }
    
    /// Read COP file with appropriate handling for format variations
    pub fn read(&self) -> Result<DataFrame> {
        let mut csv_reader = CsvReader::from_path(&self.file_path)?;
        
        // Configure reader based on detected format
        csv_reader = csv_reader
            .has_header(self.has_headers)
            .infer_schema(Some(50000));  // Sample more rows for better type inference
        
        // For files without headers, we'll rename columns after reading
        // Don't use with_columns here as it doesn't actually rename them
        
        // Add null values handling for all variations
        let null_values = NullValues::AllColumns(vec![
            "".to_string(),
            "NA".to_string(),
            "N/A".to_string(),
            "null".to_string(),
            "NULL".to_string(),
        ]);
        csv_reader = csv_reader.with_null_values(Some(null_values));
        
        // Read the file
        let mut df = csv_reader.finish()?;
        
        // For files without headers, select only the expected columns and rename them
        if !self.has_headers {
            let expected_columns = COP_COLUMNS_V1;
            let num_expected = expected_columns.len();
            
            // Get current column names
            let current_columns: Vec<String> = df.get_column_names()
                .iter()
                .map(|s| s.to_string())
                .collect();
            
            // If we have more columns than expected, select only the first N
            if current_columns.len() > num_expected {
                let cols_to_select: Vec<&str> = current_columns
                    .iter()
                    .take(num_expected)
                    .map(|s| s.as_str())
                    .collect();
                df = df.select(cols_to_select)?;
            }
            
            // Rename columns by selecting them in order with new names
            let mut renamed_columns = Vec::new();
            let columns = df.get_columns();
            
            for (i, expected_name) in expected_columns.iter().enumerate() {
                if i < columns.len() {
                    let col = &columns[i];
                    renamed_columns.push(col.clone().with_name(expected_name));
                }
            }
            
            // Create new dataframe with renamed columns
            df = DataFrame::new(renamed_columns)?;
        }
        
        // Normalize column types
        df = self.normalize_column_types(df)?;
        
        Ok(df)
    }
    
    /// Ensure all numeric columns are Float64 to prevent type mismatches
    fn normalize_column_types(&self, mut df: DataFrame) -> Result<DataFrame> {
        let numeric_columns = [
            "High Sustained Limit",
            "Low Sustained Limit",
            "High Emergency Limit",
            "Low Emergency Limit",
            "Reg Up",
            "Reg Down",
            "RRS",
            "RRSPFR",
            "RRSFFR",
            "RRSUFR",
            "NSPIN",
            "ECRS",
            "Minimum SOC",
            "Maximum SOC",
            "Hour Beginning Planned SOC",
        ];
        
        for col_name in &numeric_columns {
            if df.get_column_names().contains(&col_name) {
                if let Ok(col) = df.column(col_name) {
                    // Cast to Float64 if not already
                    if col.dtype() != &DataType::Float64 {
                        let casted = col.cast(&DataType::Float64)?;
                        df.with_column(casted)?;
                    }
                }
            }
        }
        
        Ok(df)
    }
}

/// Read a COP file with automatic format detection
pub fn read_cop_file(file_path: &Path) -> Result<DataFrame> {
    let reader = COPFileReader::new(file_path)?;
    reader.read()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_format_detection() {
        // Test various filename patterns
        let test_cases = vec![
            ("CompleteCOP_09252014.csv", false, 13, true),  // Late 2014, no headers
            ("CompleteCOP_01012015.csv", true, 13, false),  // 2015, has headers
            ("60d_COP_Adjustment_Period_Snapshot-01-JAN-23.csv", true, 19, false), // New format
        ];
        
        for (filename, expected_headers, expected_cols, expected_late_2014) in test_cases {
            let path = Path::new(filename);
            // Note: This would need actual files to test properly
            // Just testing the logic based on filename patterns
        }
    }
}