# ERCOT Power Market Pipeline Makefile
# Comprehensive build and data processing automation

.PHONY: help build test clean install run-all extract rollup bess verify docker lint format check

# Default target - show help
help:
	@echo "ERCOT Power Market Pipeline - Make Targets"
	@echo "==========================================="
	@echo ""
	@echo "Setup & Build:"
	@echo "  make install          Install Python and Rust dependencies"
	@echo "  make build           Build Rust processor (debug mode)"
	@echo "  make build-release   Build Rust processor (release mode)"
	@echo "  make clean           Clean build artifacts and temp files"
	@echo ""
	@echo "Data Processing - Full:"
	@echo "  make extract         Extract all ERCOT CSV files from zips"
	@echo "  make rollup          Run annual rollup with gap tracking (debug)"
	@echo "  make rollup-release  Run annual rollup (ALL datasets, optimized)"
	@echo "  make rollup-test     Test rollup on 2011 data (DST flag test)"
	@echo ""
	@echo "Data Processing - Individual Datasets (FAST):"
	@echo "  make rollup-da-prices    Day-Ahead Settlement Point Prices"
	@echo "  make rollup-as-prices    Ancillary Services Clearing Prices"
	@echo "  make rollup-dam-gen      60-Day DAM Generation Resources (discharge)"
	@echo "  make rollup-dam-load     60-Day DAM Load Resources (charging)"
	@echo "  make rollup-sced-gen     60-Day SCED Generation Resources (discharge)"
	@echo "  make rollup-sced-load    60-Day SCED Load Resources (charging)"
	@echo "  make rollup-cop          60-Day COP Adjustment Snapshots"
	@echo "  make rollup-rt-prices    Real-Time Prices (WARNING: Large!)"
	@echo ""
	@echo "Price Data Processing:"
	@echo "  make flatten-prices      Flatten DA/RT/AS prices (RT keeps 5-min)"
	@echo "  make aggregate-rt-hourly Create hourly RT from 5-min data"
	@echo "  make combine-prices      Combine prices and create monthly files"
	@echo "  make process-prices      Run full price processing pipeline"
	@echo ""
	@echo "ERCOT Price Service:"
	@echo "  make build-price-service Build the Rust price service"
	@echo "  make run-price-service   Run the price service (ports 8080, 50051)"
	@echo "  make price-service-docker Build Docker image for price service"
	@echo ""
	@echo "Analysis:"
	@echo "  make bess            Run BESS revenue analysis"
	@echo "  make bess-leaderboard Run BESS daily revenue leaderboard"
	@echo "  make bess-match      Create BESS resource matching file"
	@echo "  make verify          Verify data quality of processed files"
	@echo "  make verify-parquet  Check parquet files (Rust version)"
	@echo "  make verify-all-parquet  Comprehensive parallel verification (Python)"
	@echo ""
	@echo "Python Pipeline:"
	@echo "  make download        Download recent ERCOT data (7 days)"
	@echo "  make backfill        Backfill historical data from 2019"
	@echo "  make realtime        Start real-time data updater"
	@echo "  make catalog         View data catalog"
	@echo "  make db-init         Initialize database schema"
	@echo ""
	@echo "Development:"
	@echo "  make test            Run all tests (Python and Rust)"
	@echo "  make test-python     Run Python tests only"
	@echo "  make test-rust       Run Rust tests only"
	@echo "  make lint            Run linters (ruff + clippy)"
	@echo "  make format          Format code (ruff + rustfmt)"
	@echo "  make check           Type checking (mypy + cargo check)"
	@echo ""
	@echo "Utilities:"
	@echo "  make parquet-stats   Show statistics for parquet files"
	@echo "  make gaps-report     Display gaps in processed data"
	@echo "  make disk-usage      Show disk usage of data directories"
	@echo "  make logs            Tail processing logs"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build    Build Docker image"
	@echo "  make docker-run      Run pipeline in Docker"
	@echo ""

# ============= Setup & Build =============

install: install-python install-rust
	@echo "✅ All dependencies installed"

install-python:
	@echo "📦 Installing Python dependencies with uv..."
	uv sync

install-rust:
	@echo "📦 Installing Rust dependencies..."
	cd ercot_data_processor && cargo fetch

build:
	@echo "🔨 Building Rust processor (debug) with 24 cores..."
	cd ercot_data_processor && \
		PATH="$$HOME/.cargo/bin:$$PATH" \
		CARGO_BUILD_JOBS=24 \
		RUSTFLAGS="-C codegen-units=256" \
		cargo build --jobs 24
	@echo "✅ Debug build complete: ercot_data_processor/target/debug/ercot_data_processor"

build-release:
	@echo "🔨 Building Rust processor (release) with 24 cores..."
	@echo "🚀 Using maximum parallelism for compilation..."
	cd ercot_data_processor && \
		PATH="$$HOME/.cargo/bin:$$PATH" \
		CARGO_BUILD_JOBS=24 \
		RUSTFLAGS="-C codegen-units=256 -C target-cpu=native" \
		cargo build --release --jobs 24
	@echo "✅ Release build complete: ercot_data_processor/target/release/ercot_data_processor"

clean:
	@echo "🧹 Cleaning build artifacts..."
	cd ercot_data_processor && cargo clean
	rm -rf __pycache__ .pytest_cache .mypy_cache
	rm -rf /tmp/test_2011_ercot /tmp/test_2011_rt_prices
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	@echo "✅ Clean complete"

# ============= Schema Detection & Type Validation =============

detect-schema:
	@echo "🔍 Running first pass type detection on ERCOT files..."
	@echo "📂 Scanning: $(DATA_DIR)"
	cd ercot_data_processor && ./target/release/ercot_data_processor --detect-schema $(DATA_DIR)
	@echo "✅ Schema registry created: $(DATA_DIR)/ercot_schema_registry.json"

test-schema:
	@echo "🧪 Testing schema registry with sample files..."
	cd ercot_data_processor && ./target/release/ercot_data_processor --test-schema $(DATA_DIR)
	@echo "✅ Schema validation complete"

validate-rollup:
	@echo "🚀 Running rollup with schema registry validation..."
	cd ercot_data_processor && SKIP_CSV=1 ./target/release/ercot_data_processor --annual-rollup-validated $(DATA_DIR) 2>&1 | tee /tmp/validated_rollup.log
	@echo "📊 Checking for type errors..."
	@grep -c "Could not parse" /tmp/validated_rollup.log || echo "✅ NO TYPE ERRORS FOUND!"

rollup-validated: build-release
	@echo "🚀 Running validated annual rollup with schema registry..."
	@echo "📂 Using schema: $(DATA_DIR)/ercot_schema_registry.json"
	cd ercot_data_processor && ./target/release/ercot_data_processor --annual-rollup-validated $(DATA_DIR)
	@echo "✅ Validated rollup complete"

# ============= Data Processing =============

extract:
	@echo "📂 Extracting all ERCOT CSV files from zips..."
	cd ercot_data_processor && ./target/debug/ercot_data_processor --extract-all-ercot $(DATA_DIR)

# ============= Price Data Flattening & Combining =============

flatten-prices:
	@echo "📊 Flattening ERCOT price data to wide format..."
	@echo "  • DA prices: Each row = 1 hour, columns = HBs, LZs, DCs"
	@echo "  • RT prices: 15-minute intervals preserved (4 per hour)"
	@echo "  • AS prices: Each row = 1 hour, columns = AS types"
	uv run python flatten_ercot_prices.py
	@echo "✅ Flattened files saved to: $(ROLLUP_DIR)/flattened/"
	@echo "  • RT files: RT_prices_15min_YYYY.parquet"

aggregate-rt-hourly:
	@echo "📊 Creating hourly aggregated RT prices from 15-min data..."
	uv run python flatten_ercot_prices_hourly.py
	@echo "✅ Hourly RT files saved to: $(ROLLUP_DIR)/flattened/"
	@echo "  • Files: RT_prices_hourly_YYYY.parquet"

combine-prices:
	@echo "🔀 Combining and splitting price files..."
	@echo "  • Creating DA+AS combined files"
	@echo "  • Creating DA+AS+RT combined files"
	@echo "  • Splitting into monthly files"
	uv run python combine_ercot_prices.py
	@echo "✅ Combined files saved to: $(ROLLUP_DIR)/combined/"
	@echo "✅ Monthly files saved to: $(ROLLUP_DIR)/combined/monthly/"

process-prices: flatten-prices aggregate-rt-hourly combine-prices
	@echo "✅ Price processing pipeline complete!"
	@echo "📁 Output locations:"
	@echo "  • Flattened: $(ROLLUP_DIR)/flattened/"
	@echo "    - DA_prices_YYYY.parquet (hourly)"
	@echo "    - RT_prices_15min_YYYY.parquet (15-minute intervals, 4 per hour)"
	@echo "    - RT_prices_hourly_YYYY.parquet (hourly avg)"
	@echo "    - AS_prices_YYYY.parquet (hourly)"
	@echo "  • Combined:  $(ROLLUP_DIR)/combined/"
	@echo "    - DA_AS_combined_YYYY.parquet"
	@echo "    - DA_AS_RT_combined_YYYY.parquet (hourly aligned)"
	@echo "    - DA_AS_RT_15min_combined_YYYY.parquet (15-min with DA/AS repeated)"
	@echo "  • Monthly:   $(ROLLUP_DIR)/combined/monthly/"

# ============= ERCOT Price Service (Rust) =============

build-price-service:
	@echo "🔨 Building ERCOT Price Service..."
	cd ercot_price_service && cargo build --release
	@echo "✅ Binary: ercot_price_service/target/release/ercot-price-server"

run-price-service: build-price-service
	@echo "🚀 Starting ERCOT Price Service..."
	@echo "  • JSON API: http://localhost:8080"
	@echo "  • Arrow Flight: grpc://localhost:50051"
	cd ercot_price_service && ./target/release/ercot-price-server \
		--data-dir $(ROLLUP_DIR) \
		--json-addr 0.0.0.0:8080 \
		--flight-addr 0.0.0.0:50051

price-service-docker:
	@echo "🐳 Building Price Service Docker image..."
	cd ercot_price_service && docker build -t ercot-price-service:latest .

price-service-compose:
	@echo "🐳 Starting Price Service with Docker Compose..."
	cd ercot_price_service && docker-compose up -d
	@echo "✅ Services running:"
	@echo "  • JSON API: http://localhost:8080/api/health"
	@echo "  • Arrow Flight: grpc://localhost:50051"

rollup: build
	@echo "📊 Running annual rollup with gap tracking..."
	cd ercot_data_processor && ./target/debug/ercot_data_processor --annual-rollup

rollup-release: build-release
	@echo "📊 Running annual rollup with release build..."
	@echo "🚀 Optimizing for 24 cores / 32 threads..."
	cd ercot_data_processor && \
		RAYON_NUM_THREADS=32 \
		POLARS_MAX_THREADS=24 \
		RUST_MIN_STACK=8388608 \
		./target/release/ercot_data_processor --annual-rollup

# Individual dataset rollup targets for faster iteration
rollup-da-prices: build-release
	@echo "📊 Processing Day-Ahead Prices only..."
	cd ercot_data_processor && ./target/release/ercot_data_processor --annual-rollup $(DATA_DIR) --dataset DA_prices

rollup-as-prices: build-release
	@echo "📊 Processing Ancillary Services Prices only..."
	cd ercot_data_processor && ./target/release/ercot_data_processor --annual-rollup $(DATA_DIR) --dataset AS_prices

rollup-dam-gen: build-release
	@echo "📊 Processing DAM Generation Resources only..."
	cd ercot_data_processor && ./target/release/ercot_data_processor --annual-rollup $(DATA_DIR) --dataset DAM_Gen_Resources

rollup-sced-gen: build-release
	@echo "📊 Processing SCED Generation Resources only..."
	cd ercot_data_processor && ./target/release/ercot_data_processor --annual-rollup $(DATA_DIR) --dataset SCED_Gen_Resources

rollup-cop: build-release
	@echo "📊 Processing COP Snapshots only..."
	cd ercot_data_processor && ./target/release/ercot_data_processor --annual-rollup $(DATA_DIR) --dataset COP_Snapshots

rollup-rt-prices: build-release
	@echo "📊 Processing Real-Time Prices only (WARNING: 500K+ files!)..."
	@echo "🚀 Using controlled parallelism to avoid file descriptor exhaustion..."
	@echo "📁 Files to process: ~515,814"
	cd ercot_data_processor && \
		RAYON_NUM_THREADS=16 \
		POLARS_MAX_THREADS=24 \
		RUST_MIN_STACK=8388608 \
		./target/release/ercot_data_processor --annual-rollup $(DATA_DIR) --dataset RT_prices

rollup-dam-load: build-release
	@echo "📊 Processing DAM Load Resources only..."
	cd ercot_data_processor && ./target/release/ercot_data_processor --annual-rollup $(DATA_DIR) --dataset DAM_Load_Resources

rollup-sced-load: build-release
	@echo "📊 Processing SCED Load Resources only..."
	cd ercot_data_processor && ./target/release/ercot_data_processor --annual-rollup $(DATA_DIR) --dataset SCED_Load_Resources

rollup-cop-old: build-release
	@echo "📊 Processing COP files only (old method - specific directory)..."
	cd ercot_data_processor && ./target/release/ercot_data_processor --annual-rollup $(DATA_DIR)/60-Day_COP_Adjustment_Period_Snapshot

rollup-test: build
	@echo "🧪 Testing rollup on 2011 data (DST flag evolution)..."
	cd ercot_data_processor && ./test_2011_rollup.sh

bess: build
	@echo "💰 Running BESS revenue analysis..."
	cd ercot_data_processor && ./target/debug/ercot_data_processor --bess-parquet

bess-leaderboard: build-release
	@echo "🏆 Running BESS daily revenue leaderboard analysis..."
	cd ercot_data_processor && ./target/release/ercot_data_processor --bess-daily-revenue

bess-parquet-revenue: build-release
	@echo "💰 Running high-performance BESS revenue processor (parallel)..."
	@echo "🚀 Using 24 cores / 32 threads for maximum performance..."
	cd ercot_data_processor && \
		RAYON_NUM_THREADS=32 \
		POLARS_MAX_THREADS=24 \
		./target/release/ercot_data_processor --bess-parquet-revenue

bess-match:
	@echo "🔗 Creating BESS resource matching file..."
	uv run python create_bess_match_file.py
	@echo "✅ Created: bess_match_file.csv and bess_match_rules.json"

verify: build
	@echo "✔️ Verifying data quality..."
	cd ercot_data_processor && ./target/debug/ercot_data_processor --verify-results

verify-parquet: build-release
	@echo "🔍 Verifying parquet files for data integrity..."
	@echo "📊 Checking for duplicates, gaps, and corruption..."
	cd ercot_data_processor && \
		PATH="$$HOME/.cargo/bin:$$PATH" \
		./target/release/ercot_data_processor --verify-parquet $(DATA_DIR)
	@echo "📝 Check verification_report.md for detailed results"

verify-all-parquet:
	@echo "🔍 Comprehensive Parquet Verification"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "📊 Checking all parquet files in rollup directory..."
	@echo "   • Data integrity"
	@echo "   • Duplicate detection"
	@echo "   • Time series gaps"
	@echo "   • Schema consistency"
	@echo ""
	@uv run python verify_parquet_files.py $(DATA_DIR)
	@echo ""
	@echo "📝 Reports generated:"
	@echo "   • $(DATA_DIR)/rollup_files/verification_report.md"
	@echo "   • $(DATA_DIR)/rollup_files/verification_report.json"

# ============= Python Pipeline =============

download:
	@echo "⬇️ Downloading recent ERCOT data (7 days)..."
	uv run pmp download --iso ERCOT --days 7

backfill:
	@echo "📅 Backfilling historical data from 2019..."
	uv run pmp backfill --iso ERCOT --start 2019-01-01

realtime:
	@echo "🔄 Starting real-time data updater..."
	uv run pmp realtime --iso ERCOT

catalog:
	@echo "📚 Viewing data catalog..."
	uv run pmp catalog

db-init:
	@echo "🗄️ Initializing database..."
	createdb power_market || true
	uv run pmp init
	uv run pmp-db

# ============= Development =============

test: test-python test-rust
	@echo "✅ All tests passed"

test-python:
	@echo "🧪 Running Python tests..."
	uv run pytest -v

test-rust:
	@echo "🧪 Running Rust tests..."
	cd ercot_data_processor && cargo test

test-integration:
	@echo "🧪 Running integration tests..."
	uv run pytest power_market_pipeline/tests/test_integration.py -v

lint: lint-python lint-rust
	@echo "✅ Linting complete"

lint-python:
	@echo "🔍 Linting Python code..."
	uv run ruff check .

lint-rust:
	@echo "🔍 Linting Rust code..."
	cd ercot_data_processor && cargo clippy -- -D warnings

format: format-python format-rust
	@echo "✅ Formatting complete"

format-python:
	@echo "✨ Formatting Python code..."
	uv run ruff format .

format-rust:
	@echo "✨ Formatting Rust code..."
	cd ercot_data_processor && cargo fmt

check: check-python check-rust
	@echo "✅ Type checking complete"

check-python:
	@echo "🔎 Type checking Python..."
	uv run mypy .

check-rust:
	@echo "🔎 Checking Rust code..."
	cd ercot_data_processor && cargo check

# ============= Utilities =============

list-datasets:
	@echo "Available datasets for selective processing:"
	@echo "============================================"
	@echo ""
	@echo "Dataset Name         | Description"
	@echo "--------------------|------------------------------------------"
	@echo "DA_prices           | Day-Ahead Settlement Point Prices"
	@echo "AS_prices           | Ancillary Services Clearing Prices"
	@echo "DAM_Gen_Resources   | 60-Day DAM Generation Resources"
	@echo "SCED_Gen_Resources  | 60-Day SCED Generation Resources"
	@echo "COP_Snapshots       | 60-Day COP Adjustment Period Snapshots"
	@echo "RT_prices           | Real-Time Settlement Point Prices (LARGE!)"
	@echo ""
	@echo "Usage examples:"
	@echo "  make rollup-da-prices     # Process only Day-Ahead prices"
	@echo "  make rollup-cop           # Process only COP snapshots"
	@echo "  make rollup-release       # Process ALL datasets"

parquet-stats:
	@echo "📊 Parquet file statistics:"
	@find $(DATA_DIR)/rollup_files -name "*.parquet" -exec du -h {} \; | sort -hr | head -20

gaps-report:
	@echo "📋 Data gaps report:"
	@find $(DATA_DIR)/rollup_files -name "gaps_report.md" -exec echo "=== {} ===" \; -exec cat {} \; | head -100

disk-usage:
	@echo "💾 Disk usage by directory:"
	@du -sh $(DATA_DIR)/* | sort -hr | head -20

logs:
	@echo "📜 Recent processing logs:"
	@tail -f /tmp/ercot_processor.log 2>/dev/null || echo "No logs found"

count-files:
	@echo "📁 File counts:"
	@echo -n "CSV files: " && find $(DATA_DIR) -name "*.csv" 2>/dev/null | wc -l
	@echo -n "Parquet files: " && find $(DATA_DIR) -name "*.parquet" 2>/dev/null | wc -l
	@echo -n "ZIP files: " && find $(DATA_DIR) -name "*.zip" 2>/dev/null | wc -l

# ============= Quick Processing Chains =============

process-all: extract rollup bess verify
	@echo "✅ Full processing pipeline complete"

process-daily: download rollup-release bess
	@echo "✅ Daily processing complete"

process-test: rollup-test
	@echo "✅ Test processing complete"

# ============= Docker =============

docker-build:
	@echo "🐳 Building Docker image..."
	docker build -t ercot-pipeline .

docker-run:
	@echo "🐳 Running pipeline in Docker..."
	docker run -v $(dir $(DATA_DIR)):/data -e DATABASE_URL=${DATABASE_URL} ercot-pipeline

docker-shell:
	@echo "🐳 Opening shell in Docker container..."
	docker run -it -v $(dir $(DATA_DIR)):/data -e DATABASE_URL=${DATABASE_URL} ercot-pipeline /bin/bash

# ============= Database =============

db-backup:
	@echo "💾 Backing up database..."
	pg_dump power_market | gzip > backups/power_market_$(shell date +%Y%m%d_%H%M%S).sql.gz

db-restore:
	@echo "♻️ Restoring database from latest backup..."
	@LATEST=$$(ls -t backups/*.sql.gz | head -1); \
	if [ -n "$$LATEST" ]; then \
		echo "Restoring from $$LATEST"; \
		gunzip -c $$LATEST | psql power_market; \
	else \
		echo "No backup found"; \
	fi

db-reset:
	@echo "⚠️ Resetting database (this will delete all data)..."
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		dropdb power_market || true; \
		createdb power_market; \
		uv run pmp init; \
	fi

# ============= Advanced Analysis =============

analyze-2024:
	@echo "📈 Analyzing 2024 BESS performance..."
	cd ercot_data_processor && ./target/debug/ercot_data_processor --bess-yearly

market-report:
	@echo "📊 Generating market report..."
	cd ercot_data_processor && ./target/debug/ercot_data_processor --bess-report

visualize:
	@echo "📉 Generating visualizations..."
	cd ercot_data_processor && ./target/debug/ercot_data_processor --bess-viz

# ============= Development Shortcuts =============

dev: format lint test
	@echo "✅ Development checks complete"

ci: install build test lint check
	@echo "✅ CI pipeline complete"

release: clean build-release test
	@echo "✅ Release build complete"

# ============= Environment Setup =============

env-check:
	@echo "🔍 Checking environment variables..."
	@echo -n "DATABASE_URL: "; [ -n "${DATABASE_URL}" ] && echo "✅ Set" || echo "❌ Not set"
	@echo -n "ERCOT_USERNAME: "; [ -n "${ERCOT_USERNAME}" ] && echo "✅ Set" || echo "❌ Not set"
	@echo -n "ERCOT_PASSWORD: "; [ -n "${ERCOT_PASSWORD}" ] && echo "✅ Set" || echo "❌ Not set"
	@echo -n "ERCOT_SUBSCRIPTION_KEY: "; [ -n "${ERCOT_SUBSCRIPTION_KEY}" ] && echo "✅ Set" || echo "❌ Not set"

env-template:
	@echo "📝 Creating .env template..."
	@echo "# ERCOT Power Market Pipeline Environment Variables" > .env.template
	@echo "DATABASE_URL=postgresql://user:pass@localhost/power_market" >> .env.template
	@echo "ERCOT_USERNAME=" >> .env.template
	@echo "ERCOT_PASSWORD=" >> .env.template
	@echo "ERCOT_SUBSCRIPTION_KEY=" >> .env.template
	@echo "CAISO_USERNAME=" >> .env.template
	@echo "CAISO_PASSWORD=" >> .env.template
	@echo "✅ Created .env.template"

# ============= Performance Monitoring =============

benchmark:
	@echo "⚡ Running performance benchmarks..."
	cd ercot_data_processor && cargo bench

profile:
	@echo "📊 Profiling data processing..."
	cd ercot_data_processor && cargo build --release --features profiling
	cd ercot_data_processor && valgrind --tool=callgrind ./target/release/ercot_data_processor --annual-rollup

monitor:
	@echo "📈 Monitoring system resources..."
	@watch -n 1 "ps aux | grep ercot_data_processor | grep -v grep; echo '---'; df -h $(dir $(DATA_DIR)); echo '---'; top -l 1 | head -10"

# ============= Maintenance =============

update-deps:
	@echo "⬆️ Updating dependencies..."
	uv sync --upgrade
	cd ercot_data_processor && cargo update

security-audit:
	@echo "🔒 Running security audit..."
	cd ercot_data_processor && cargo audit

cleanup-old-data:
	@echo "🗑️ Cleaning up old data files..."
	find $(DATA_DIR) -name "*.csv" -mtime +30 -delete
	find /tmp -name "test_*" -mtime +7 -delete

# Variables for common paths
RUST_BIN := ercot_data_processor/target/debug/ercot_data_processor
RUST_BIN_RELEASE := ercot_data_processor/target/release/ercot_data_processor
DATA_DIR ?= $(shell echo $${ERCOT_DATA_DIR:-/home/enrico/data/ERCOT_data})
ROLLUP_DIR := $(DATA_DIR)/rollup_files