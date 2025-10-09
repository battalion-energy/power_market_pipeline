# Repository Guidelines

## Project Structure & Module Organization
- `power_market_pipeline/`: CLI (`cli.py`) and Python package entrypoints.
- `downloaders/`, `services/`, `processors/`: ISO-specific fetchers, registry, real‑time updater, and data processing utilities.
- `src/` + `Cargo.toml`: Rust accelerator (`ercot_rt_processor`) for high‑volume processing.
- `test_*.py`: Pytest tests at repo root.
- `output/`, `target/`: Python/Rust build and data artifacts (ignored from VCS).

## Build, Test, and Development Commands
- Install deps (uv): `pip install uv && uv sync`
- Run CLI help: `uv run pmp --help`
- Initialize DB: `uv run pmp init` (requires `DATABASE_URL` in `.env`)
- Download sample data: `uv run pmp download --iso ERCOT --days 3`
- Real‑time updates: `uv run pmp realtime --iso ERCOT`
- Pytests: `uv run pytest -q`
- Lint: `uv run ruff check .`; Format: `uv run ruff format .`
- Type check: `uv run mypy .`
- Rust build: `cargo build --release`; run: `cargo run --release -- <args>`

## Coding Style & Naming Conventions
- Python 3.11, 4‑space indents, max line length 100 (see `pyproject.toml`).
- Use type hints and docstrings for public functions.
- Names: modules/files `snake_case.py`, classes `PascalCase`, functions/vars `snake_case`, constants `UPPER_SNAKE`.
- Keep CLI commands cohesive and idempotent; prefer `click` options over env flags.

## Testing Guidelines
- Framework: `pytest`; add tests as `test_<topic>.py` (root or `tests/` if added).
- Write deterministic tests; avoid network and live credentials—mock I/O and external APIs.
- Run with coverage when changing logic: `uv run pytest --cov=power_market_pipeline --cov-report=term-missing`.

## Commit & Pull Request Guidelines
- Commits: imperative tone, concise scope prefix when helpful (e.g., `fix:`, `refactor:`, `feat:`), e.g., `fix: correct ERCOT price column name`.
- PRs: clear summary, rationale, and scope; link issues; include run instructions, sample commands, and before/after notes; attach screenshots for CLI output or performance diffs when relevant.

## Security & Configuration Tips
- Copy `.env.example` → `.env`; never commit secrets. Key vars: `DATABASE_URL`, optional ISO creds (ERCOT/CAISO/ISONE/NYISO).
- Validate configuration with `uv run pmp catalog` and a small `download` run before long backfills.

## Architecture Overview
- Python orchestrates data ingestion and scheduling; Rust accelerates heavy ERCOT processing. Use the Python CLI for orchestration, and Rust binaries for large batch transforms when needed.
