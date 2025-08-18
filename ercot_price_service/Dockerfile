# Build stage
FROM rust:1.75 as builder

WORKDIR /app

# Copy Cargo files first for better caching
COPY Cargo.toml Cargo.lock ./

# Create dummy main to cache dependencies
RUN mkdir src && \
    echo "fn main() {}" > src/main.rs && \
    echo "pub fn dummy() {}" > src/lib.rs

# Build dependencies only
RUN cargo build --release && \
    rm -rf src

# Copy actual source code
COPY src ./src

# Build the application
RUN touch src/main.rs src/lib.rs && \
    cargo build --release

# Runtime stage
FROM debian:bookworm-slim

RUN apt-get update && \
    apt-get install -y ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the binary from builder
COPY --from=builder /app/target/release/ercot-price-server /usr/local/bin/

# Create data directory
RUN mkdir -p /data

# Expose ports
EXPOSE 8080 50051

# Set environment variable for data directory
ENV DATA_DIR=/data

# Run the service
CMD ["ercot-price-server", "--data-dir", "/data", "--json-addr", "0.0.0.0:8080", "--flight-addr", "0.0.0.0:50051"]