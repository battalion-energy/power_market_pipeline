.PHONY: build release test run clean install docker

# Build debug version
build:
	cargo build

# Build optimized release version
release:
	cargo build --release

# Run tests
test:
	cargo test

# Run the service (debug mode)
run:
	cargo run -- --data-dir /home/enrico/data/ERCOT_data/rollup_files

# Run the service (release mode)
run-release: release
	./target/release/ercot-price-server --data-dir /home/enrico/data/ERCOT_data/rollup_files

# Clean build artifacts
clean:
	cargo clean

# Install the service binary to system
install: release
	sudo cp target/release/ercot-price-server /usr/local/bin/

# Build Docker image
docker:
	docker build -t ercot-price-service:latest .

# Format code
fmt:
	cargo fmt

# Lint code
lint:
	cargo clippy -- -D warnings

# Check code without building
check:
	cargo check

# Watch for changes and rebuild
watch:
	cargo watch -x build -x test

# Generate documentation
docs:
	cargo doc --open

# Benchmark
bench:
	cargo bench

# Full CI check (format, lint, test, build)
ci: fmt lint test build

# Development mode with auto-reload
dev:
	cargo watch -x 'run -- --data-dir /home/enrico/data/ERCOT_data/rollup_files'