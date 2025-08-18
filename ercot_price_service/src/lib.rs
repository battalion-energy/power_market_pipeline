pub mod data_loader;
pub mod flight_service;
pub mod json_api;
pub mod models;

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use dashmap::DashMap;

pub struct PriceDataCache {
    pub cache: Arc<DashMap<String, arrow::record_batch::RecordBatch>>,
    pub data_dir: PathBuf,
}

impl PriceDataCache {
    pub fn new(data_dir: PathBuf) -> Self {
        Self {
            cache: Arc::new(DashMap::new()),
            data_dir,
        }
    }

    pub async fn load_data(&self, key: &str) -> Result<arrow::record_batch::RecordBatch> {
        if let Some(batch) = self.cache.get(key) {
            return Ok(batch.clone());
        }

        let batch = data_loader::load_parquet_file(&self.data_dir, key).await?;
        self.cache.insert(key.to_string(), batch.clone());
        Ok(batch)
    }
}
