#!/bin/bash

set -e

echo "Building ERCOT Price Service..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Rust is installed
if ! command -v cargo &> /dev/null; then
    echo -e "${RED}Error: Cargo is not installed${NC}"
    echo "Please install Rust from https://rustup.rs/"
    exit 1
fi

# Parse arguments
BUILD_TYPE="release"
RUN_AFTER_BUILD=false
INSTALL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            BUILD_TYPE="debug"
            shift
            ;;
        --run)
            RUN_AFTER_BUILD=true
            shift
            ;;
        --install)
            INSTALL=true
            shift
            ;;
        --help)
            echo "Usage: ./build.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --debug    Build in debug mode (default: release)"
            echo "  --run      Run the service after building"
            echo "  --install  Install the binary to /usr/local/bin"
            echo "  --help     Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Clean previous builds
echo -e "${YELLOW}Cleaning previous builds...${NC}"
cargo clean

# Format code
echo -e "${YELLOW}Formatting code...${NC}"
cargo fmt

# Run clippy linter
echo -e "${YELLOW}Running linter...${NC}"
cargo clippy -- -D warnings || {
    echo -e "${RED}Linting failed. Please fix the warnings above.${NC}"
    exit 1
}

# Run tests
echo -e "${YELLOW}Running tests...${NC}"
cargo test || {
    echo -e "${RED}Tests failed. Please fix the failing tests.${NC}"
    exit 1
}

# Build the project
if [ "$BUILD_TYPE" = "release" ]; then
    echo -e "${YELLOW}Building release version...${NC}"
    cargo build --release
    BINARY_PATH="target/release/ercot-price-server"
else
    echo -e "${YELLOW}Building debug version...${NC}"
    cargo build
    BINARY_PATH="target/debug/ercot-price-server"
fi

echo -e "${GREEN}Build successful!${NC}"
echo "Binary location: $BINARY_PATH"

# Get binary size
BINARY_SIZE=$(du -h "$BINARY_PATH" | cut -f1)
echo "Binary size: $BINARY_SIZE"

# Install if requested
if [ "$INSTALL" = true ]; then
    echo -e "${YELLOW}Installing to /usr/local/bin...${NC}"
    sudo cp "$BINARY_PATH" /usr/local/bin/
    echo -e "${GREEN}Installation complete!${NC}"
fi

# Run if requested
if [ "$RUN_AFTER_BUILD" = true ]; then
    echo -e "${YELLOW}Starting service...${NC}"
    DATA_DIR="${ERCOT_DATA_DIR:-/home/enrico/data/ERCOT_data/rollup_files}"
    echo "Using data directory: $DATA_DIR"
    "$BINARY_PATH" --data-dir "$DATA_DIR"
fi