use anyhow::Result;
use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use arrow_flight::flight_service_server::FlightService;
use arrow_flight::{
    Action, ActionType, Criteria, Empty, FlightData, FlightDescriptor, FlightEndpoint,
    FlightInfo, HandshakeRequest, HandshakeResponse, PutResult, SchemaAsIpc, SchemaResult,
    Ticket,
};
use futures::stream::{self, BoxStream};
use std::sync::Arc;
use tonic::{Request, Response, Status, Streaming};

use crate::data_loader;
use crate::models::PriceQuery;
use crate::PriceDataCache;

pub struct ErcotFlightService {
    cache: Arc<PriceDataCache>,
}

impl ErcotFlightService {
    pub fn new(cache: Arc<PriceDataCache>) -> Self {
        Self { cache }
    }
}

#[tonic::async_trait]
impl FlightService for ErcotFlightService {
    type HandshakeStream = BoxStream<'static, Result<HandshakeResponse, Status>>;
    type ListFlightsStream = BoxStream<'static, Result<FlightInfo, Status>>;
    type DoGetStream = BoxStream<'static, Result<FlightData, Status>>;
    type DoPutStream = BoxStream<'static, Result<PutResult, Status>>;
    type DoActionStream = BoxStream<'static, Result<arrow_flight::Result, Status>>;
    type ListActionsStream = BoxStream<'static, Result<ActionType, Status>>;
    type DoExchangeStream = BoxStream<'static, Result<FlightData, Status>>;

    async fn handshake(
        &self,
        _request: Request<Streaming<HandshakeRequest>>,
    ) -> Result<Response<Self::HandshakeStream>, Status> {
        Err(Status::unimplemented("Handshake not implemented"))
    }

    async fn list_flights(
        &self,
        _request: Request<Criteria>,
    ) -> Result<Response<Self::ListFlightsStream>, Status> {
        let flight_info = FlightInfo::new()
            .try_with_schema(&Schema::new(vec![
                Field::new("datetime", DataType::Timestamp(TimeUnit::Millisecond, None), false),
                Field::new("hub", DataType::Utf8, false),
                Field::new("price", DataType::Float64, true),
            ]))
            .map_err(|e| Status::internal(format!("Schema error: {}", e)))?
            .with_descriptor(FlightDescriptor::new_path(vec!["ercot_prices".to_string()]))
            .with_endpoint(FlightEndpoint::new().with_ticket(Ticket::new("ercot_prices")));

        let stream = stream::once(async { Ok(flight_info) });
        Ok(Response::new(Box::pin(stream)))
    }

    async fn get_flight_info(
        &self,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        let descriptor = request.into_inner();
        
        let schema = Schema::new(vec![
            Field::new("datetime", DataType::Timestamp(TimeUnit::Millisecond, None), false),
            Field::new("hub", DataType::Utf8, false),
            Field::new("price", DataType::Float64, true),
        ]);
        
        let flight_info = FlightInfo::new()
            .try_with_schema(&schema)
            .map_err(|e| Status::internal(format!("Schema error: {}", e)))?
            .with_descriptor(descriptor.clone())
            .with_endpoint(FlightEndpoint::new().with_ticket(Ticket::new("ercot_prices")));

        Ok(Response::new(flight_info))
    }

    async fn get_schema(
        &self,
        _request: Request<FlightDescriptor>,
    ) -> Result<Response<SchemaResult>, Status> {
        let schema = Schema::new(vec![
            Field::new("datetime", DataType::Timestamp(TimeUnit::Millisecond, None), false),
            Field::new("hub", DataType::Utf8, false),
            Field::new("price", DataType::Float64, true),
        ]);
        
        let schema_result = SchemaAsIpc::new(&schema, &Default::default())
            .try_into()
            .map_err(|e| Status::internal(format!("Schema conversion error: {}", e)))?;

        Ok(Response::new(schema_result))
    }

    async fn do_get(
        &self,
        request: Request<Ticket>,
    ) -> Result<Response<Self::DoGetStream>, Status> {
        let ticket = request.into_inner();
        let query: PriceQuery = serde_json::from_slice(&ticket.ticket)
            .map_err(|e| Status::invalid_argument(format!("Invalid query: {}", e)))?;

        let cache = self.cache.clone();
        
        let stream = stream::try_unfold(Some(query), move |state| {
            let cache = cache.clone();
            async move {
                if let Some(query) = state {
                    let key = data_loader::get_file_key(&query.price_type, &query.start_date, true);
                    
                    match cache.load_data(&key).await {
                        Ok(batch) => {
                            let filtered = data_loader::filter_batch_by_date_range(
                                &batch,
                                &query.start_date,
                                &query.end_date,
                            )
                            .map_err(|e| Status::internal(format!("Filter error: {}", e)))?;
                            
                            let ipc_options = arrow::ipc::writer::IpcWriteOptions::default();
                            let (_dicts, batch_flight_data) = arrow_flight::utils::flight_data_from_arrow_batch(
                                &filtered,
                                &ipc_options,
                            );
                            
                            // Return the batch flight data
                            Ok(Some((batch_flight_data, None)))
                        }
                        Err(e) => Err(Status::internal(format!("Data loading error: {}", e))),
                    }
                } else {
                    Ok(None)
                }
            }
        });

        Ok(Response::new(Box::pin(stream)))
    }

    async fn do_put(
        &self,
        _request: Request<Streaming<FlightData>>,
    ) -> Result<Response<Self::DoPutStream>, Status> {
        Err(Status::unimplemented("Put not supported"))
    }

    async fn do_action(
        &self,
        _request: Request<Action>,
    ) -> Result<Response<Self::DoActionStream>, Status> {
        Err(Status::unimplemented("Actions not implemented"))
    }

    async fn list_actions(
        &self,
        _request: Request<Empty>,
    ) -> Result<Response<Self::ListActionsStream>, Status> {
        let actions = vec![
            ActionType {
                r#type: "HealthCheck".to_string(),
                description: "Check service health".to_string(),
            },
        ];
        
        let stream = stream::iter(actions.into_iter().map(Ok));
        Ok(Response::new(Box::pin(stream)))
    }

    async fn do_exchange(
        &self,
        _request: Request<Streaming<FlightData>>,
    ) -> Result<Response<Self::DoExchangeStream>, Status> {
        Err(Status::unimplemented("Exchange not implemented"))
    }

    async fn poll_flight_info(
        &self,
        _request: Request<FlightDescriptor>,
    ) -> Result<Response<arrow_flight::PollInfo>, Status> {
        Err(Status::unimplemented("Poll not implemented"))
    }
}