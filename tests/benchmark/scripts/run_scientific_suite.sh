#!/bin/bash
set -e

RESULTS_DIR="tests/benchmark/results/scientific_suite"
mkdir -p $RESULTS_DIR

echo "=== Running Scientific Benchmark Suite ==="

# 1. Scalar (All Scientific Tests: Baseline, Batch Scale, Result Scale, Complexity)
echo "[1/3] Running Scalar Scientific Matrix..."
PYTHONPATH=. pytest tests/benchmark/search_perf/scalar/test_scalar_bench.py \
    --benchmark-only \
    --benchmark-json="$RESULTS_DIR/scalar_scientific.json" \
    -q

# 2. Vector (All Scientific Tests: Baseline, Batch Scale, Result Scale, Dim Cost)
echo "[2/3] Running Vector Scientific Matrix..."
PYTHONPATH=. pytest tests/benchmark/search_perf/vector/test_vector_bench.py \
    --benchmark-only \
    --benchmark-json="$RESULTS_DIR/vector_scientific.json" \
    -q

# 3. Insert (Standard)
echo "[3/3] Running Insert Benchmarks..."
PYTHONPATH=. pytest tests/benchmark/insert_perf/test_insert_bench.py \
    --benchmark-only \
    --benchmark-json="$RESULTS_DIR/insert_scientific.json" \
    -q

echo "Scientific Suite completed. Results in $RESULTS_DIR"
