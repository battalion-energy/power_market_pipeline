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
	@echo "  make rollup-dam-gen      60-Day DAM Generation Resources"
	@echo "  make rollup-sced-gen     60-Day SCED Generation Resources"
	@echo "  make rollup-cop          60-Day COP Adjustment Snapshots"
	@echo "  make rollup-rt-prices    Real-Time Prices (WARNING: Large!)"
	@echo ""
	@echo "Analysis:"
	@echo "  make bess            Run BESS revenue analysis"
	@echo "  make bess-leaderboard Run BESS daily revenue leaderboard"
	@echo "  make verify          Verify data quality of processed files"
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
	@echo "🔨 Building Rust processor (debug)..."
	cd ercot_data_processor && cargo build
	@echo "✅ Debug build complete: ercot_data_processor/target/debug/ercot_data_processor"

build-release:
	@echo "🔨 Building Rust processor (release)..."
	cd ercot_data_processor && cargo build --release
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
	@echo "📂 Scanning: /Users/enrico/data/ERCOT_data"
	cd ercot_data_processor && ./target/release/ercot_data_processor --detect-schema /Users/enrico/data/ERCOT_data
	@echo "✅ Schema registry created: /Users/enrico/data/ERCOT_data/ercot_schema_registry.json"

test-schema:
	@echo "🧪 Testing schema registry with sample files..."
	cd ercot_data_processor && ./target/release/ercot_data_processor --test-schema /Users/enrico/data/ERCOT_data
	@echo "✅ Schema validation complete"

validate-rollup:
	@echo "🚀 Running rollup with schema registry validation..."
	cd ercot_data_processor && SKIP_CSV=1 ./target/release/ercot_data_processor --annual-rollup-validated /Users/enrico/data/ERCOT_data 2>&1 | tee /tmp/validated_rollup.log
	@echo "📊 Checking for type errors..."
	@grep -c "Could not parse" /tmp/validated_rollup.log || echo "✅ NO TYPE ERRORS FOUND!"

rollup-validated: build-release
	@echo "🚀 Running validated annual rollup with schema registry..."
	@echo "📂 Using schema: /Users/enrico/data/ERCOT_data/ercot_schema_registry.json"
	cd ercot_data_processor && ./target/release/ercot_data_processor --annual-rollup-validated /Users/enrico/data/ERCOT_data
	@echo "✅ Validated rollup complete"

# ============= Data Processing =============

extract:
	@echo "📂 Extracting all ERCOT CSV files from zips..."
	cd ercot_data_processor && ./target/debug/ercot_data_processor --extract-all-ercot /Users/enrico/data/ERCOT_data

rollup: build
	@echo "📊 Running annual rollup with gap tracking..."
	cd ercot_data_processor && ./target/debug/ercot_data_processor --annual-rollup /Users/enrico/data/ERCOT_data

rollup-release: build-release
	@echo "📊 Running annual rollup with release build..."
	cd ercot_data_processor && ./target/release/ercot_data_processor --annual-rollup /Users/enrico/data/ERCOT_data

# Individual dataset rollup targets for faster iteration
rollup-da-prices: build-release
	@echo "📊 Processing Day-Ahead Prices only..."
	cd ercot_data_processor && ./target/release/ercot_data_processor --annual-rollup /Users/enrico/data/ERCOT_data --dataset DA_prices

rollup-as-prices: build-release
	@echo "📊 Processing Ancillary Services Prices only..."
	cd ercot_data_processor && ./target/release/ercot_data_processor --annual-rollup /Users/enrico/data/ERCOT_data --dataset AS_prices

rollup-dam-gen: build-release
	@echo "📊 Processing DAM Generation Resources only..."
	cd ercot_data_processor && ./target/release/ercot_data_processor --annual-rollup /Users/enrico/data/ERCOT_data --dataset DAM_Gen_Resources

rollup-sced-gen: build-release
	@echo "📊 Processing SCED Generation Resources only..."
	cd ercot_data_processor && ./target/release/ercot_data_processor --annual-rollup /Users/enrico/data/ERCOT_data --dataset SCED_Gen_Resources

rollup-cop: build-release
	@echo "📊 Processing COP Snapshots only..."
	cd ercot_data_processor && ./target/release/ercot_data_processor --annual-rollup /Users/enrico/data/ERCOT_data --dataset COP_Snapshots

rollup-rt-prices: build-release
	@echo "📊 Processing Real-Time Prices only (WARNING: Large dataset!)..."
	cd ercot_data_processor && ./target/release/ercot_data_processor --annual-rollup /Users/enrico/data/ERCOT_data --dataset RT_prices

rollup-cop-old: build-release
	@echo "📊 Processing COP files only (old method - specific directory)..."
	cd ercot_data_processor && ./target/release/ercot_data_processor --annual-rollup /Users/enrico/data/ERCOT_data/60-Day_COP_Adjustment_Period_Snapshot

rollup-test: build
	@echo "🧪 Testing rollup on 2011 data (DST flag evolution)..."
	cd ercot_data_processor && ./test_2011_rollup.sh

bess: build
	@echo "💰 Running BESS revenue analysis..."
	cd ercot_data_processor && ./target/debug/ercot_data_processor --bess-parquet

bess-leaderboard: build-release
	@echo "🏆 Running BESS daily revenue leaderboard analysis..."
	cd ercot_data_processor && ./target/release/ercot_data_processor --bess-daily-revenue

verify: build
	@echo "✔️ Verifying data quality..."
	cd ercot_data_processor && ./target/debug/ercot_data_processor --verify-results

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
	@find /Users/enrico/data/ERCOT_data/rollup_files -name "*.parquet" -exec du -h {} \; | sort -hr | head -20

gaps-report:
	@echo "📋 Data gaps report:"
	@find /Users/enrico/data/ERCOT_data/rollup_files -name "gaps_report.md" -exec echo "=== {} ===" \; -exec cat {} \; | head -100

disk-usage:
	@echo "💾 Disk usage by directory:"
	@du -sh /Users/enrico/data/ERCOT_data/* | sort -hr | head -20

logs:
	@echo "📜 Recent processing logs:"
	@tail -f /tmp/ercot_processor.log 2>/dev/null || echo "No logs found"

count-files:
	@echo "📁 File counts:"
	@echo -n "CSV files: " && find /Users/enrico/data/ERCOT_data -name "*.csv" 2>/dev/null | wc -l
	@echo -n "Parquet files: " && find /Users/enrico/data/ERCOT_data -name "*.parquet" 2>/dev/null | wc -l
	@echo -n "ZIP files: " && find /Users/enrico/data/ERCOT_data -name "*.zip" 2>/dev/null | wc -l

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
	docker run -v /Users/enrico/data:/data -e DATABASE_URL=${DATABASE_URL} ercot-pipeline

docker-shell:
	@echo "🐳 Opening shell in Docker container..."
	docker run -it -v /Users/enrico/data:/data -e DATABASE_URL=${DATABASE_URL} ercot-pipeline /bin/bash

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
	@watch -n 1 "ps aux | grep ercot_data_processor | grep -v grep; echo '---'; df -h /Users/enrico/data; echo '---'; top -l 1 | head -10"

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
	find /Users/enrico/data/ERCOT_data -name "*.csv" -mtime +30 -delete
	find /tmp -name "test_*" -mtime +7 -delete

# Variables for common paths
RUST_BIN := ercot_data_processor/target/debug/ercot_data_processor
RUST_BIN_RELEASE := ercot_data_processor/target/release/ercot_data_processor
DATA_DIR := /Users/enrico/data/ERCOT_data
ROLLUP_DIR := $(DATA_DIR)/rollup_files