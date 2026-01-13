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

# Setup Directories (use absolute paths to avoid sudo/cwd issues)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
RESULTS_DIR="$PROJECT_ROOT/.benchmarks"

mkdir -p "$RESULTS_DIR"
echo "Output directory: $RESULTS_DIR"

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
# 3. Validation & Advanced Scenarios
# -----------------------------------------------------------------------------
echo ""
echo "[3/3] Running Validation & Advanced Scenarios..."

echo ">> Running Arrow IPC Validation Benchmark (E.7.4)..."
PYTHONPATH=. pytest tests/benchmark/search_perf/test_arrow_ipc_bench.py \
    --benchmark-only \
    --benchmark-json="$RESULTS_DIR/arrow_ipc_benchmark.json" \
    -q || echo "⚠️ Arrow IPC benchmark failed!"

echo ">> Running Columnar Memory Profiling (E.11)..."
# Create output dir for memray
mkdir -p "$RESULTS_DIR/memory_profiles"

# Legacy Mode
echo "   - Profiling Legacy Mode..."
python3 -m memray run -o "$RESULTS_DIR/memory_profiles/legacy.bin" -f \
    tests/benchmark/search_perf/profile_columnar_memory.py legacy > /dev/null

# Columnar Init
echo "   - Profiling Columnar Init..."
python3 -m memray run -o "$RESULTS_DIR/memory_profiles/columnar_init.bin" -f \
    tests/benchmark/search_perf/profile_columnar_memory.py columnar_init > /dev/null

# Columnar Partial Access
echo "   - Profiling Columnar Access..."
python3 -m memray run -o "$RESULTS_DIR/memory_profiles/columnar_access.bin" -f \
    tests/benchmark/search_perf/profile_columnar_memory.py columnar_access > /dev/null

# Columnar Full Iteration
echo "   - Profiling Columnar Iteration..."
python3 -m memray run -o "$RESULTS_DIR/memory_profiles/columnar_iter.bin" -f \
    tests/benchmark/search_perf/profile_columnar_memory.py columnar_iter > /dev/null

echo ">> Generating Memory Summaries..."
# Generate simple text stats for quick inspection in logs
python3 -m memray stats "$RESULTS_DIR/memory_profiles/legacy.bin" | head -n 20
echo "---"
python3 -m memray stats "$RESULTS_DIR/memory_profiles/columnar_init.bin" | head -n 20


# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo ""
echo "================================================================="

# -----------------------------------------------------------------------------
# 3. Generate Readable Reports
# -----------------------------------------------------------------------------
echo ""
echo "[4/4] Generating Human-Readable Reports..."
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
