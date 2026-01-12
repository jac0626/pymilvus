#!/bin/bash
set -e

# =============================================================================
# Milvus Client Benchmark CI Script
# =============================================================================
# This script runs the full suite of benchmarks and profiling scenarios.
# It is designed to be run in a CI environment (Github Actions).
#
# Pre-requisites:
#   - Python 3.8+
#   - pip install ".[dev]" (includes pytest-benchmark, memray, py-spy)
#
# Outputs:
#   - tests/benchmark/results/scientific_suite/*.json
#   - .benchmarks/*.stats (CPU profiles)
#   - .benchmarks/*.bin (Memory profiles)
# =============================================================================

# Setup Directories
RESULTS_DIR=".benchmarks"
PROFILE_DIR=".benchmarks"

mkdir -p "$RESULTS_DIR"
mkdir -p "$PROFILE_DIR"

echo "================================================================="
echo "Starting Benchmark & Profiling Suite"
echo "Date: $(date)"
echo "Commit: $(git rev-parse --short HEAD)"
echo "================================================================="

# -----------------------------------------------------------------------------
# 1. Scientific Benchmarks (Latency Metrics)
# -----------------------------------------------------------------------------
echo ""
echo "[1/2] Running Scientific Benchmarks..."

echo ">> Running Scalar Benchmarks..."
PYTHONPATH=. pytest tests/benchmark/search_perf/scalar/test_scalar_bench.py \
    --benchmark-only \
    --benchmark-json="$RESULTS_DIR/scalar_benchmark.json" \
    -q || echo "⚠️ Scalar benchmarks failed!"

echo ">> Running Vector Benchmarks..."
PYTHONPATH=. pytest tests/benchmark/search_perf/vector/test_vector_bench.py \
    --benchmark-only \
    --benchmark-json="$RESULTS_DIR/vector_benchmark.json" \
    -q || echo "⚠️ Vector benchmarks failed!"

echo ">> Running Proto Overhead Benchmarks..."
PYTHONPATH=. pytest tests/benchmark/search_perf/test_proto_overhead_bench.py \
    --benchmark-only \
    --benchmark-json="$RESULTS_DIR/proto_overhead.json" \
    -q || echo "⚠️ Proto benchmarks failed!"

echo ">> Running Insert Benchmarks..."
PYTHONPATH=. pytest tests/benchmark/insert_perf/test_insert_bench.py \
    --benchmark-only \
    --benchmark-json="$RESULTS_DIR/insert_benchmark.json" \
    -q || echo "⚠️ Insert benchmarks failed!"

# -----------------------------------------------------------------------------
# 2. Profiling Scenarios (Deep Analysis)
# -----------------------------------------------------------------------------
echo ""
echo "[2/2] Running Profiling Scenarios..."

echo ">> Running Vector Profiling (CPU)..."
python3 tests/benchmark/search_perf/vector/profile_vector_perf.py || echo "⚠️ vector CPU profile failed"

echo ">> Running Vector Profiling (Memory)..."
python3 tests/benchmark/search_perf/vector/profile_vector_perf.py --memory || echo "⚠️ vector Memory profile failed"

echo ">> Running Scalar Profiling (CPU)..."
python3 tests/benchmark/search_perf/scalar/profile_scalar_perf.py || echo "⚠️ scalar CPU profile failed"

echo ">> Running Scalar Profiling (Memory)..."
python3 tests/benchmark/search_perf/scalar/profile_scalar_perf.py --memory || echo "⚠️ scalar Memory profile failed"


# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo ""
echo "================================================================="

# -----------------------------------------------------------------------------
# 3. Generate Readable Reports
# -----------------------------------------------------------------------------
echo ""
echo "[3/3] Generating Human-Readable Reports..."
python3 tests/benchmark/scripts/generate_report.py

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo ""
echo "================================================================="
echo "Benchmark Suite Completed"
echo "================================================================="
echo "Reports generated in .benchmarks/reports/ :"
echo "  ├── summary/ (Speedup Tables)"
echo "  ├── cpu/     (Top Functions)"
echo "  └── memory/  (Allocation Stats)"
echo ""
ls -R .benchmarks/reports
echo ""
echo "Done."
