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
	@echo "Analysis:"
	@echo "  make bess            Run BESS revenue analysis"
	@echo "  make bess-leaderboard Run BESS daily revenue leaderboard"
	@echo "  make bess-match      Create BESS resource matching file"
	@echo "  make tbx             Calculate TB2/TB4 battery arbitrage values (Python)"
	@echo "  make tbx-rust        Calculate TBX with Rust (limited nodes)"
	@echo "  make tbx-all-nodes   Calculate TBX for ALL 1,098 nodes (Rust V2)"
	@echo "  make tbx-reports     Generate monthly/quarterly TBX reports"
	@echo "  make tbx-custom      Calculate TBX with custom parameters
  make tbx-comprehensive  Calculate all 6 TB variants (DA, RT, DA+RT) for all years"
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

bess:
	@echo "💰 Running Unified BESS Revenue Calculator (High-Performance Rust Version)..."
	@echo "📊 Processing all revenue streams: DA, RT, AS (RegUp, RegDn, RRS, NonSpin, ECRS)..."
	@echo "⚡ Using parallel processing for maximum speed..."
	cd ercot_data_processor && \
		RAYON_NUM_THREADS=32 \
		POLARS_MAX_THREADS=24 \
		cargo run --release --bin ercot_data_processor -- --bess-unified
	@echo "✅ BESS analysis complete. Check database_export/ for results."

bess-leaderboard:
	@echo "🏆 Running Unified BESS Revenue Calculator (Python Version)..."
	@echo "📊 Processing all revenue streams: DA, RT, AS..."
	@echo "📉 Generating visualizations and leaderboard..."
	uv run python unified_bess_revenue_calculator.py
	@echo "✅ Leaderboard generated. Check database_export/ for results."

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

bess-compare: build-release
	@echo "🔬 Running BESS Revenue Comparison: Python vs Rust"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	@echo "1️⃣  Running Python version (expected: slower)..."
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@/usr/bin/time -v uv run python unified_bess_revenue_calculator.py 2>&1 | grep -E "User time|System time|Elapsed|Maximum resident"
	@echo ""
	@echo "2️⃣  Running Rust version (expected: faster)..."
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@cd ercot_data_processor && \
		RAYON_NUM_THREADS=32 \
		POLARS_MAX_THREADS=24 \
		/usr/bin/time -v ./target/release/ercot_data_processor --bess-unified 2>&1 | grep -E "User time|System time|Elapsed|Maximum resident"
	@echo ""
	@echo "📊 Comparing results..."
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "Python output: $(DATA_DIR)/bess_analysis/database_export/"
	@ls -lh $(DATA_DIR)/bess_analysis/database_export/*.parquet 2>/dev/null || echo "No Python output found"
	@echo ""
	@echo "Rust output: $(DATA_DIR)/bess_analysis/database_export/"
	@ls -lh $(DATA_DIR)/bess_analysis/database_export/*.parquet 2>/dev/null || echo "No Rust output found"
	@echo ""
	@echo "✅ Comparison complete!"

verify:
	@echo "✔️ Verifying data quality..."
	@echo "🔍 Checking settlement point mappings..."
	uv run python verify_settlement_mapping.py
	@echo "✅ Verification complete. Check output for any issues."

verify-parquet:
	@echo "🔍 Verifying parquet files for data integrity..."
	@echo "📊 Checking for duplicates, gaps, and corruption..."
	uv run python verify_parquet_files.py $(DATA_DIR)
	@echo "📝 Check verification output for detailed results"

verify-all-parquet:
	@echo "🔍 Comprehensive Parquet Verification"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "📊 Checking all parquet files in rollup directory..."
	@echo "   • Data integrity"
	@echo "   • Duplicate detection"
	@echo "   • Time series gaps"
	@echo "   • Schema consistency"
	@echo ""
	@echo "Running parallel verification on all datasets..."
	uv run python verify_parquet_files.py $(DATA_DIR)
	@echo ""
	@echo "✅ Comprehensive verification complete"
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
	@echo ""
	@echo "=== Largest Parquet Files ==="
	@find $(DATA_DIR)/rollup_files -name "*.parquet" -exec du -h {} \; | sort -hr | head -20
	@echo ""
	@echo "=== Total Size by Dataset Type ==="
	@for dir in $(DATA_DIR)/rollup_files/*/; do \
		if [ -d "$$dir" ]; then \
			size=$$(du -sh "$$dir" 2>/dev/null | cut -f1); \
			name=$$(basename "$$dir"); \
			printf "%-40s %s\n" "$$name:" "$$size"; \
		fi; \
	done | sort -k2 -hr
	@echo ""
	@echo "=== Flattened Price Files ==="
	@ls -lh $(DATA_DIR)/rollup_files/flattened/*.parquet 2>/dev/null | tail -n +2 | awk '{print $$9 ": " $$5}' | sort -k2 -hr || echo "No flattened files found"
	@echo ""
	@echo "=== Combined Price Files ==="
	@ls -lh $(DATA_DIR)/rollup_files/combined/*.parquet 2>/dev/null | tail -n +2 | awk '{print $$9 ": " $$5}' | sort -k2 -hr || echo "No combined files found"

gaps-report:
	@echo "📋 Data gaps report:"
	@echo ""
	@if [ -f "$(DATA_DIR)/rollup_files/gaps_summary.txt" ]; then \
		cat "$(DATA_DIR)/rollup_files/gaps_summary.txt"; \
	else \
		echo "No gaps summary found. Searching for individual reports..."; \
		echo ""; \
		for report in $$(find $(DATA_DIR)/rollup_files -name "gaps_report.md" 2>/dev/null); do \
			dataset=$$(dirname "$$report" | xargs basename); \
			echo "=== $$dataset ==="; \
			grep -E "^## Year|gaps detected|Gap:|Missing" "$$report" | head -20; \
			echo ""; \
		done; \
	fi

disk-usage:
	@echo "💾 Disk usage summary:"
	@echo ""
	@echo "=== Total ERCOT Data ==="
	@du -sh $(DATA_DIR) 2>/dev/null || echo "Data directory not found"
	@echo ""
	@echo "=== By Category ==="
	@echo "Raw Downloads:"
	@du -sh $(DATA_DIR)/60-Day* $(DATA_DIR)/Settlement* $(DATA_DIR)/DAM_* 2>/dev/null | awk '{sum+=$$1} {print $$0} END {print "Total Raw: " sum "G"}' | sort -hr
	@echo ""
	@echo "Processed Data:"
	@du -sh $(DATA_DIR)/rollup_files 2>/dev/null || echo "No rollup files"
	@du -sh $(DATA_DIR)/csv_files 2>/dev/null || echo "No CSV files"
	@echo ""
	@echo "=== Top 10 Directories ==="
	@du -sh $(DATA_DIR)/* 2>/dev/null | sort -hr | head -10

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

# TBX Battery Arbitrage Calculator
tbx:
	@echo "⚡ Calculating TBX (TB2/TB4) battery arbitrage values..."
	@echo "🔋 TB2 = 2-hour battery arbitrage revenue"
	@echo "🔋 TB4 = 4-hour battery arbitrage revenue"
	@echo "📊 Processing all nodes for years: 2021-2025 (through July)"
	@echo "⚙️  Efficiency: 90% (10% losses on charge/discharge)"
	uv run python calculate_tbx_v2.py
	@echo "✅ TBX calculation complete. Results in: $(DATA_DIR)/tbx_results/"
	@echo ""
	@echo "📊 Generating monthly and quarterly reports..."
	uv run python generate_tbx_reports.py
	@echo "✅ Reports generated in: $(DATA_DIR)/tbx_results/reports/"

tbx-comprehensive:
	@echo "⚡ Calculating Comprehensive TBX (6 variants per node)..."
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "🔋 Computing 6 TB variants for each settlement point:"
	@echo "   • TB2_DA:   Day-Ahead only (2-hour arbitrage)"
	@echo "   • TB2_RT:   Real-Time only (15-min intervals)"
	@echo "   • TB2_DART: DA charge + RT discharge (hybrid)"
	@echo "   • TB4_DA:   Day-Ahead only (4-hour arbitrage)"
	@echo "   • TB4_RT:   Real-Time only (15-min intervals)"
	@echo "   • TB4_DART: DA charge + RT discharge (hybrid)"
	@echo ""
	@echo "📅 Processing years: 2021-2024"
	@echo "⚙️  Round-trip efficiency: 90%"
	@echo ""
	@for year in 2021 2022 2023 2024; do \
		echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; \
		echo "📅 Processing year $$year..."; \
		python calculate_tbx_comprehensive.py --year $$year || true; \
		echo ""; \
	done
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "✅ Comprehensive TBX calculation complete!"
	@echo "📁 Results saved in: /pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/tbx_results/"
	@ls -lh /pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/tbx_results/tbx_comprehensive_*.parquet 2>/dev/null || echo "No output files found"

tbx-rust: build-release
	@echo "⚡ Running TBX Calculator (High-Performance Rust Version)..."
	@echo "🔋 TB2 = 2-hour battery arbitrage revenue"
	@echo "🔋 TB4 = 4-hour battery arbitrage revenue"
	@echo "📊 Processing limited nodes (flattened files) for years: 2021-2025"
	@echo "⚙️  Efficiency: 90% (10% losses on charge/discharge)"
	@echo "🚀 Using parallel processing for maximum speed..."
	cd ercot_data_processor && \
		RAYON_NUM_THREADS=32 \
		POLARS_MAX_THREADS=24 \
		./target/release/ercot_data_processor --calculate-tbx
	@echo "✅ TBX calculation complete. Results in: $(DATA_DIR)/tbx_results/"

tbx-all-nodes: build-release
	@echo "⚡ Running TBX Calculator for ALL NODES (Rust V2)..."
	@echo "🔋 TB2 = 2-hour battery arbitrage revenue"
	@echo "🔋 TB4 = 4-hour battery arbitrage revenue"
	@echo "📊 Processing ALL 1,098 settlement points for years: 2021-2025"
	@echo "📁 Reading from raw DA price files (34M+ rows)"
	@echo "⚙️  Efficiency: 90% (10% losses on charge/discharge)"
	@echo "🚀 High-performance Rust implementation..."
	cd ercot_data_processor && \
		./target/release/ercot_data_processor --calculate-tbx-all-nodes
	@echo "✅ TBX calculation complete for ALL nodes!"
	@echo "📊 Results in: $(DATA_DIR)/tbx_results_all_nodes/"
	@echo "  • Daily results: tbx_daily_YYYY_all_nodes.parquet"
	@echo "  • Monthly results: tbx_monthly_YYYY_all_nodes.parquet" 
	@echo "  • Annual results: tbx_annual_YYYY_all_nodes.parquet"
	@echo "  • Leaderboard: tbx_leaderboard_all_nodes.csv"

tbx-reports:
	@echo "📊 Generating TBX monthly and quarterly reports..."
	@echo "📅 Creating reports for all available data..."
	uv run python generate_tbx_reports.py \
		--data-dir $(DATA_DIR)/tbx_results \
		--output-dir $(DATA_DIR)/tbx_results
	@echo "✅ Reports generated in: $(DATA_DIR)/tbx_results/reports/"
	@echo "  • Monthly reports: $(DATA_DIR)/tbx_results/reports/monthly/"
	@echo "  • Quarterly reports: $(DATA_DIR)/tbx_results/reports/quarterly/"
	@echo "  • Formats: JSON (for API) and Markdown (for display)"

tbx-custom:
	@echo "⚡ Calculating TBX with custom parameters..."
	uv run python calculate_tbx_v2.py \
		--efficiency $(EFFICIENCY) \
		--years $(YEARS) \
		--data-dir $(DATA_DIR)/rollup_files/flattened \
		--output-dir $(OUTPUT_DIR)

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