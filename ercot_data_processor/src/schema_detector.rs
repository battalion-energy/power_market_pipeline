use anyhow::Result;
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use rayon::prelude::*;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "PascalCase")]
pub enum ErcotDataType {
    Float64,
    Int64,
    Utf8,
    Date,
    Datetime,
    Boolean,
}

impl<'de> serde::Deserialize<'de> for ErcotDataType {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        match s.as_str() {
            "Float64" => Ok(ErcotDataType::Float64),
            "Int64" => Ok(ErcotDataType::Int64),
            "Utf8" => Ok(ErcotDataType::Utf8),
            "Date" => Ok(ErcotDataType::Date),
            "Datetime" => Ok(ErcotDataType::Datetime),
            "Boolean" => Ok(ErcotDataType::Boolean),
            _ => Err(serde::de::Error::custom(format!("Unknown data type: {}", s))),
        }
    }
}

impl From<&ErcotDataType> for DataType {
    fn from(ercot_type: &ErcotDataType) -> Self {
        match ercot_type {
            ErcotDataType::Float64 => DataType::Float64,
            ErcotDataType::Int64 => DataType::Int64,
            ErcotDataType::Utf8 => DataType::Utf8,
            ErcotDataType::Date => DataType::Date,
            ErcotDataType::Datetime => DataType::Datetime(TimeUnit::Milliseconds, None),
            ErcotDataType::Boolean => DataType::Boolean,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileSchema {
    pub file_pattern: String,  // e.g., "60d_COP_Adjustment_Period_Snapshot*.csv"
    pub columns: HashMap<String, ErcotDataType>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SchemaRegistry {
    pub schemas: Vec<FileSchema>,
}

impl SchemaRegistry {
    pub fn new() -> Self {
        Self {
            schemas: Vec::new(),
        }
    }

    pub fn load_from_file(path: &Path) -> Result<Self> {
        let content = fs::read_to_string(path)?;
        let registry: SchemaRegistry = serde_json::from_str(&content)?;
        Ok(registry)
    }

    pub fn save_to_file(&self, path: &Path) -> Result<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, self)?;
        Ok(())
    }

    pub fn get_schema_for_file(&self, file_path: &Path) -> Option<&FileSchema> {
        let file_name = file_path.file_name()?.to_str()?;
        
        for schema in &self.schemas {
            // Improved pattern matching to handle wildcards properly
            if schema.file_pattern.contains("*") {
                let pattern_parts: Vec<&str> = schema.file_pattern.split("*").collect();
                if pattern_parts.len() == 2 {
                    let prefix = pattern_parts[0];
                    let suffix = pattern_parts[1];
                    if file_name.starts_with(prefix) && file_name.ends_with(suffix) {
                        return Some(schema);
                    }
                }
            } else {
                // Exact match for patterns without wildcards
                if file_name == schema.file_pattern {
                    return Some(schema);
                }
            }
        }
        None
    }

    pub fn _build_polars_schema(&self, file_path: &Path) -> Option<Schema> {
        let file_schema = self.get_schema_for_file(file_path)?;
        let mut schema = Schema::new();
        
        for (col_name, data_type) in &file_schema.columns {
            schema.with_column(col_name.clone().into(), data_type.into());
        }
        
        Some(schema)
    }
}

pub struct SchemaDetector {
    base_dir: PathBuf,
    output_file: PathBuf,
}

impl SchemaDetector {
    pub fn new(base_dir: PathBuf) -> Self {
        Self {
            output_file: base_dir.join("ercot_schema_registry.json"),
            base_dir,
        }
    }

    pub fn auto_detect_schemas(&self) -> Result<SchemaRegistry> {
        let mut registry = SchemaRegistry::new();
        
        // Define patterns to scan
        let patterns = vec![
            ("60-Day_COP_Adjustment_Period_Snapshot/csv", "60d_COP_Adjustment_Period_Snapshot*.csv"),
            ("60-Day_SCED_Disclosure_Reports/csv", "60d_SCED_Gen_Resource_Data*.csv"),
            ("60-Day_DAM_Disclosure_Reports/csv", "60d_DAM_Gen_Resource_Data*.csv"),
            ("60-Day_SCED_Disclosure_Reports/csv", "60d_SCED_Load_Resource_Data*.csv"),
            ("60-Day_DAM_Disclosure_Reports/csv", "60d_DAM_Load_Resource_Data*.csv"),
            ("DA_prices", "*.csv"),
            ("AS_prices", "*.csv"),
            ("Settlement_Point_Prices_at_Resource_Nodes,_Hubs_and_Load_Zones", "*.csv"),
        ];

        // Process patterns in parallel
        let schemas: Vec<FileSchema> = patterns
            .par_iter()
            .filter_map(|(dir_path, pattern)| {
                let full_dir = self.base_dir.join(dir_path);
                if !full_dir.exists() {
                    println!("Skipping non-existent directory: {:?}", full_dir);
                    return None;
                }

                println!("Scanning directory: {:?}", full_dir);
                let sample_file = self.find_sample_file(&full_dir, pattern).ok()?;
                
                if let Some(file_path) = sample_file {
                    println!("  Analyzing sample file: {:?}", file_path);
                    match self.detect_file_schema(&file_path, pattern) {
                        Ok(schema) => Some(schema),
                        Err(e) => {
                            eprintln!("  Error detecting schema: {}", e);
                            None
                        }
                    }
                } else {
                    None
                }
            })
            .collect();
        
        registry.schemas = schemas;
        Ok(registry)
    }

    fn find_sample_file(&self, dir: &Path, pattern: &str) -> Result<Option<PathBuf>> {
        let entries = fs::read_dir(dir)?;
        let pattern_prefix = pattern.replace("*.csv", "");
        
        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            let file_name = path.file_name().unwrap_or_default().to_str().unwrap_or("");
            
            if file_name.contains(&pattern_prefix) && file_name.ends_with(".csv") {
                return Ok(Some(path));
            }
        }
        
        Ok(None)
    }

    fn detect_file_schema(&self, file_path: &Path, pattern: &str) -> Result<FileSchema> {
        // Read CSV with large sample size to detect types
        let df = CsvReader::from_path(file_path)?
            .has_header(true)
            .infer_schema(Some(50000))  // Look at many rows
            .finish()?;

        let mut columns = HashMap::new();
        
        for col_name in df.get_column_names() {
            let col = df.column(col_name)?;
            let dtype = self.determine_ercot_type(col_name, col)?;
            columns.insert(col_name.to_string(), dtype);
        }

        Ok(FileSchema {
            file_pattern: pattern.to_string(),
            columns,
        })
    }

    fn determine_ercot_type(&self, col_name: &str, col: &Series) -> Result<ErcotDataType> {
        let lower_name = col_name.to_lowercase();
        
        // Rule-based type determination
        
        // Date/time columns
        if lower_name.contains("date") && !lower_name.contains("update") {
            return Ok(ErcotDataType::Date);
        }
        if lower_name.contains("timestamp") || lower_name.contains("time_stamp") {
            return Ok(ErcotDataType::Datetime);
        }
        
        // String columns
        if lower_name.contains("name") || 
           lower_name.contains("type") ||
           lower_name.contains("status") ||
           lower_name.contains("flag") ||
           lower_name.contains("constraint") ||
           lower_name.contains("contingency") ||
           lower_name.contains("station") ||
           lower_name.contains("qse") {
            return Ok(ErcotDataType::Utf8);
        }
        
        // Special case: Hour Ending might be "01:00" format
        if lower_name == "hour ending" || lower_name == "hourending" {
            // Sample some values to check format
            if let Ok(first_val) = col.get(0) {
                if let Some(val_str) = first_val.get_str() {
                    if val_str.contains(':') {
                        return Ok(ErcotDataType::Utf8);
                    }
                }
            }
            // If numeric, treat as Float64
            return Ok(ErcotDataType::Float64);
        }
        
        // Boolean columns
        if lower_name.contains("dst") && lower_name.contains("flag") {
            return Ok(ErcotDataType::Utf8);  // DST Flag is Y/N string
        }
        
        // CRITICAL: Any numeric-looking column should be Float64!
        // This prevents i64 inference issues
        if lower_name.contains("price") ||
           lower_name.contains("mw") ||
           lower_name.contains("limit") ||
           lower_name.contains("value") ||
           lower_name.contains("quantity") ||
           lower_name.contains("award") ||
           lower_name.contains("capacity") ||
           lower_name.contains("load") ||
           lower_name.contains("generation") ||
           lower_name.contains("output") ||
           lower_name.contains("soc") ||
           lower_name.contains("lsl") ||
           lower_name.contains("hsl") ||
           lower_name.contains("base") ||
           lower_name.contains("point") ||
           lower_name.contains("telemetered") ||
           lower_name.contains("schedule") ||
           lower_name.contains("reg") ||
           lower_name.contains("rrs") ||
           lower_name.contains("ecrs") ||
           lower_name.contains("nspin") ||
           lower_name.contains("nsrs") ||
           lower_name.contains("spin") ||
           lower_name.contains("mcpc") ||
           lower_name.contains("lambda") ||
           lower_name.contains("shadow") ||
           lower_name.contains("curve") ||
           lower_name.contains("sustained") ||
           lower_name.contains("emergency") ||
           lower_name.contains("minimum") ||
           lower_name.contains("maximum") ||
           lower_name.contains("beginning") ||
           lower_name.contains("ending") && !lower_name.contains("hour") ||
           lower_name.contains("interval") ||
           lower_name.contains("kv") ||
           lower_name.contains("voltage") {
            return Ok(ErcotDataType::Float64);
        }
        
        // Default based on Polars detection
        match col.dtype() {
            DataType::Float32 | DataType::Float64 => Ok(ErcotDataType::Float64),
            DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 |
            DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64 => {
                // FORCE numeric columns to Float64 to prevent i64 issues!
                Ok(ErcotDataType::Float64)
            },
            DataType::Utf8 => Ok(ErcotDataType::Utf8),
            DataType::Date => Ok(ErcotDataType::Date),
            DataType::Datetime(_, _) => Ok(ErcotDataType::Datetime),
            DataType::Boolean => Ok(ErcotDataType::Boolean),
            _ => Ok(ErcotDataType::Utf8),  // Default to string
        }
    }

    pub fn generate_and_save_schema(&self) -> Result<()> {
        println!("🔍 Auto-detecting ERCOT file schemas...");
        let registry = self.auto_detect_schemas()?;
        
        println!("💾 Saving schema registry to: {:?}", self.output_file);
        registry.save_to_file(&self.output_file)?;
        
        println!("✅ Schema registry created with {} file patterns", registry.schemas.len());
        for schema in &registry.schemas {
            println!("  - {}: {} columns", schema.file_pattern, schema.columns.len());
        }
        
        Ok(())
    }
}

// Usage in main processor:
pub fn read_csv_with_schema_registry(
    file_path: &Path,
    registry: &SchemaRegistry,
) -> Result<DataFrame> {
    // Special handling for CompleteCOP files - use dedicated reader
    if file_path.to_string_lossy().contains("CompleteCOP_") {
        // Use the robust COP file reader for these special cases
        return crate::cop_file_reader::read_cop_file(file_path);
    }
    
    // For all other files, normal reading with headers
    let mut df = CsvReader::from_path(file_path)?
        .has_header(true)
        .infer_schema(Some(10000))
        .finish()
        .map_err(|e| anyhow::anyhow!("Failed to read CSV: {}", e))?;
    
    // If we have a schema for this file type, apply type corrections
    if let Some(file_schema) = registry.get_schema_for_file(file_path) {
        // Apply type corrections for columns that exist in both schema and dataframe
        for (col_name, expected_type) in &file_schema.columns {
            if df.get_column_names().contains(&col_name.as_str()) {
                if let Ok(column) = df.column(col_name) {
                    // Convert to expected type if needed
                    let target_dtype: DataType = expected_type.into();
                    if column.dtype() != &target_dtype {
                        // Try to cast the column to the expected type
                        match column.cast(&target_dtype) {
                            Ok(casted) => {
                                let _ = df.with_column(casted);
                            }
                            Err(_) => {
                                // If cast fails, keep original (this handles string values in numeric columns)
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Apply Float64 conversion for known problematic columns
    force_numeric_columns_to_float64(df)
}

fn force_numeric_columns_to_float64(mut df: DataFrame) -> Result<DataFrame> {
    // List of columns that are known to cause i64/f64 type mismatches
    let problematic_columns = [
        "RRSFFR", "RRSPFR", "RRSUFR", "NSPIN", "ECRS", "ECRSM", "ECRSS",
        "Reg Up", "Reg Down", "RegUp", "RegDown", "REGUP", "REGDN",
        "Low Sustained Limit", "High Sustained Limit", "Maximum SOC", "Minimum SOC",
        "High Emergency Limit", "Low Emergency Limit", "Hour Beginning Planned SOC"
    ];
    
    // Cast numeric columns to Float64 if they exist
    for col_name in &problematic_columns {
        if df.get_column_names().contains(&col_name) {
            if let Ok(column) = df.column(col_name) {
                match column.dtype() {
                    DataType::Int64 => {
                        let casted_column = column.cast(&DataType::Float64).map_err(|e| {
                            anyhow::anyhow!("Failed to cast {} to Float64: {}", col_name, e)
                        })?;
                        df.with_column(casted_column).map_err(|e| anyhow::anyhow!("with_column error: {}", e))?;
                    }
                    DataType::Int32 => {
                        let casted_column = column.cast(&DataType::Float64).map_err(|e| {
                            anyhow::anyhow!("Failed to cast {} to Float64: {}", col_name, e)
                        })?;
                        df.with_column(casted_column).map_err(|e| anyhow::anyhow!("with_column error: {}", e))?;
                    }
                    _ => {} // Already correct type or not numeric
                }
            }
        }
    }
    
    Ok(df)
}