use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::Json,
    routing::get,
    Router,
};
use chrono::{DateTime, Utc};
use serde::Deserialize;
use std::sync::Arc;
use tower_http::cors::CorsLayer;

use crate::data_loader;
use crate::models::{ErrorResponse, HubPrices, PriceQuery, PriceResponse, PriceType};
use crate::PriceDataCache;

#[derive(Debug, Deserialize)]
pub struct QueryParams {
    pub start_date: DateTime<Utc>,
    pub end_date: DateTime<Utc>,
    pub hubs: String, // comma-separated
    pub price_type: String,
}

pub fn create_router(cache: Arc<PriceDataCache>) -> Router {
    Router::new()
        .route("/api/prices", get(get_prices))
        .route("/api/health", get(health_check))
        .route("/api/available_hubs", get(get_available_hubs))
        .layer(CorsLayer::permissive())
        .with_state(cache)
}

async fn get_prices(
    Query(params): Query<QueryParams>,
    State(cache): State<Arc<PriceDataCache>>,
) -> Result<Json<PriceResponse>, (StatusCode, Json<ErrorResponse>)> {
    let hubs: Vec<String> = params
        .hubs
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();

    let price_type = match params.price_type.to_lowercase().as_str() {
        "day_ahead" | "da" => PriceType::DayAhead,
        "real_time" | "rt" => PriceType::RealTime,
        "ancillary_services" | "as" => PriceType::AncillaryServices,
        "combined" => PriceType::Combined,
        _ => {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: "Invalid price type".to_string(),
                    details: Some("Valid types: day_ahead, real_time, ancillary_services, combined".to_string()),
                }),
            ))
        }
    };

    let query = PriceQuery {
        start_date: params.start_date,
        end_date: params.end_date,
        hubs,
        price_type: price_type.clone(),
    };

    // Load data for the date range
    // Use monthly combined files (combined=true) which have complete data
    let key = data_loader::get_file_key(&query.price_type, &query.start_date, true);
    
    let batch = cache
        .load_data(&key)
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "Failed to load data".to_string(),
                    details: Some(e.to_string()),
                }),
            )
        })?;

    // Filter by date range
    let filtered_batch = data_loader::filter_batch_by_date_range(
        &batch,
        &query.start_date,
        &query.end_date,
    )
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "Failed to filter data".to_string(),
                details: Some(e.to_string()),
            }),
        )
    })?;

    // Extract timestamps
    let timestamps = data_loader::extract_timestamps(&filtered_batch)
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "Failed to extract timestamps".to_string(),
                    details: Some(e.to_string()),
                }),
            )
        })?;

    // Extract hub prices
    let hub_data = data_loader::extract_hub_prices(&filtered_batch, &query.hubs, &price_type)
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "Failed to extract hub prices".to_string(),
                    details: Some(e.to_string()),
                }),
            )
        })?;

    let data: Vec<HubPrices> = hub_data
        .into_iter()
        .map(|(hub, prices)| HubPrices { hub, prices })
        .collect();

    Ok(Json(PriceResponse { timestamps, data }))
}

async fn health_check() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "healthy",
        "service": "ercot_price_service",
        "timestamp": Utc::now(),
    }))
}

async fn get_available_hubs() -> Json<Vec<String>> {
    Json(vec![
        "HB_BUSAVG".to_string(),
        "HB_HOUSTON".to_string(),
        "HB_HUBAVG".to_string(),
        "HB_NORTH".to_string(),
        "HB_PAN".to_string(),
        "HB_SOUTH".to_string(),
        "HB_WEST".to_string(),
        "LZ_AEN".to_string(),
        "LZ_CPS".to_string(),
        "LZ_HOUSTON".to_string(),
        "LZ_LCRA".to_string(),
        "LZ_NORTH".to_string(),
        "LZ_RAYBN".to_string(),
        "LZ_SOUTH".to_string(),
        "LZ_WEST".to_string(),
        "DC_E".to_string(),
        "DC_L".to_string(),
        "DC_N".to_string(),
        "DC_R".to_string(),
        "DC_S".to_string(),
    ])
}