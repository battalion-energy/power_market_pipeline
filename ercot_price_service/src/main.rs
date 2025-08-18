use anyhow::Result;
use arrow_flight::flight_service_server::FlightServiceServer;
use clap::Parser;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tonic::transport::Server;
use tracing::{info, warn};

use ercot_price_service::{
    flight_service::ErcotFlightService, json_api, PriceDataCache,
};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, default_value = "/home/enrico/data/ERCOT_data/rollup_files")]
    data_dir: PathBuf,

    #[arg(long, default_value = "0.0.0.0:8080")]
    json_addr: String,

    #[arg(long, default_value = "0.0.0.0:50051")]
    flight_addr: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("ercot_price_service=info".parse()?),
        )
        .init();

    let args = Args::parse();

    info!("Starting ERCOT Price Service");
    info!("Data directory: {:?}", args.data_dir);

    if !args.data_dir.exists() {
        warn!("Data directory does not exist: {:?}", args.data_dir);
    }

    let cache = Arc::new(PriceDataCache::new(args.data_dir));

    // Start Arrow Flight service
    let flight_service = ErcotFlightService::new(cache.clone());
    let flight_addr: SocketAddr = args.flight_addr.parse()?;
    
    let flight_handle = tokio::spawn(async move {
        info!("Starting Arrow Flight service on {}", flight_addr);
        Server::builder()
            .add_service(FlightServiceServer::new(flight_service))
            .serve(flight_addr)
            .await
            .expect("Failed to start Flight service");
    });

    // Start JSON API service
    let json_addr: SocketAddr = args.json_addr.parse()?;
    let json_app = json_api::create_router(cache);
    
    let json_handle = tokio::spawn(async move {
        info!("Starting JSON API on {}", json_addr);
        let listener = tokio::net::TcpListener::bind(json_addr)
            .await
            .expect("Failed to bind JSON API");
        axum::serve(listener, json_app)
            .await
            .expect("Failed to start JSON API");
    });

    info!("Services started successfully");
    info!("JSON API: http://{}", args.json_addr);
    info!("Arrow Flight: grpc://{}", args.flight_addr);

    // Wait for both services
    tokio::select! {
        _ = flight_handle => warn!("Flight service stopped"),
        _ = json_handle => warn!("JSON API stopped"),
    }

    Ok(())
}