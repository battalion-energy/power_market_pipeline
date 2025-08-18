#!/bin/bash

echo "Fast Linker Installation Script"
echo "================================"
echo ""
echo "This will help you install a fast linker for Rust compilation."
echo "Fast linkers can speed up compilation by 5-10x!"
echo ""

# Check if running with sudo
if [ "$EUID" -ne 0 ]; then 
    echo "Please run with sudo for system-wide installation:"
    echo "  sudo ./install_fast_linker.sh"
    echo ""
    echo "Or install in user space (no sudo):"
    echo ""
fi

echo "Option 1: Install LLD (recommended, fastest)"
echo "  sudo apt-get install lld clang"
echo ""

echo "Option 2: Install mold (newer, very fast)"
echo "  git clone https://github.com/rui314/mold.git"
echo "  cd mold"
echo "  ./install-build-deps.sh"
echo "  cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=c++ -B build"
echo "  cmake --build build -j $(nproc)"
echo "  sudo cmake --install build"
echo ""

echo "Option 3: Use gold linker (already installed, moderate speed)"
echo "  Add to .cargo/config.toml:"
echo '  [target.x86_64-unknown-linux-gnu]'
echo '  linker = "cc"'
echo '  rustflags = ["-C", "link-arg=-fuse-ld=gold"]'
echo ""

echo "After installing a fast linker, update .cargo/config.toml"
echo ""

# Check current linker
echo "Current linker setup:"
rustc --print link-args 2>/dev/null | grep -o 'fuse-ld=[^ ]*' || echo "  Using default linker (slow)"
echo ""

echo "Compilation Speed Tips:"
echo "======================="
echo "1. Use 'cargo build --jobs 24' to use all cores"
echo "2. Set CARGO_BUILD_JOBS=24 environment variable"
echo "3. Use codegen-units=256 for maximum parallelism"
echo "4. Disable LTO for development builds"
echo "5. Use incremental compilation"
echo "6. Consider using 'cargo check' for syntax checking"
echo ""
echo "Current settings in .cargo/config.toml will give you:"
echo "  - Maximum parallel compilation (24 cores)"
echo "  - Incremental compilation (faster rebuilds)"
echo "  - Native CPU optimizations"
echo ""