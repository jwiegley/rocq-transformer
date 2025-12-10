#!/usr/bin/env bash
# Test script for Rocq build system verification
# This script should be run from within `nix develop` shell

set -e  # Exit on error

echo "=== Rocq Build System Test ==="
echo ""

# Check Coq tools availability
echo "1. Verifying Coq/Rocq tools..."
if ! command -v coqc &> /dev/null; then
    echo "ERROR: coqc not found. Please run from within 'nix develop' shell"
    exit 1
fi

echo "   coqc version: $(coqc --version | head -n1)"
echo "   coq_makefile available: $(command -v coq_makefile)"
echo ""

# Navigate to rocq directory
cd "$(dirname "$0")"

# Test 1: Full build
echo "2. Testing full build (make)..."
make all
echo "   ✓ Full build successful"
echo ""

# Verify .vo files exist
echo "3. Verifying compiled files..."
if [ -f "RocqTransformer/Tensor.vo" ] && [ -f "RocqTransformer/Config.vo" ]; then
    echo "   ✓ Tensor.vo exists"
    echo "   ✓ Config.vo exists"
else
    echo "   ERROR: Expected .vo files not found"
    exit 1
fi
echo ""

# Test 2: Clean
echo "4. Testing clean target..."
make clean
if [ ! -f "RocqTransformer/Tensor.vo" ]; then
    echo "   ✓ Clean successful"
else
    echo "   ERROR: .vo files still exist after clean"
    exit 1
fi
echo ""

# Test 3: Individual module compilation
echo "5. Testing individual module compilation..."
make RocqTransformer/Tensor.vo
if [ -f "RocqTransformer/Tensor.vo" ]; then
    echo "   ✓ Individual compilation successful"
else
    echo "   ERROR: Individual compilation failed"
    exit 1
fi
echo ""

# Test 4: Rebuild all
echo "6. Testing rebuild..."
make all
echo "   ✓ Rebuild successful"
echo ""

echo "=== All Tests Passed ==="
echo ""
echo "The Rocq build system is working correctly!"
