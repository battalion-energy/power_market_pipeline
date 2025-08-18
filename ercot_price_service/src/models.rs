use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceQuery {
    pub start_date: DateTime<Utc>,
    pub end_date: DateTime<Utc>,
    pub hubs: Vec<String>,
    pub price_type: PriceType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PriceType {
    DayAhead,
    RealTime,
    AncillaryServices,
    Combined,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceResponse {
    pub timestamps: Vec<DateTime<Utc>>,
    pub data: Vec<HubPrices>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HubPrices {
    pub hub: String,
    pub prices: Vec<Option<f64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: String,
    pub details: Option<String>,
}