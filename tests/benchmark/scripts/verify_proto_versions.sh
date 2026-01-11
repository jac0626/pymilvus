#!/bin/bash
set -e

RESULTS_DIR="tests/benchmark/results/proto_verification"
mkdir -p $RESULTS_DIR

echo "=== Protobuf Version Verification (v5 vs v6) ==="
CURRENT_VER=$(pip show protobuf | grep Version | awk '{print $2}')
echo "Current Version: $CURRENT_VER"

# Function to run benchmark and capture latency
run_bench() {
    LABEL=$1
    echo "Running Benchmark for $LABEL..."
    # Run standard vector search (NQ=10, TopK=1000)
    # We focus on the 'mean' time for Legacy vs Columnar
    PYTHONPATH=. .venv/bin/python -m pytest tests/benchmark/search_perf/vector/test_vector_bench.py \
        -k "test_float_vector_matrix_legacy[768-1000-10] or test_float_vector_matrix_columnar[768-1000-10]" \
        --benchmark-only \
        --benchmark-json="$RESULTS_DIR/res_vector_$LABEL.json" \
        -q
    
    echo "Running Scalar Benchmark (JSON) for $LABEL..."
    PYTHONPATH=. .venv/bin/python -m pytest tests/benchmark/search_perf/scalar/test_scalar_bench.py \
        -k "test_full_matrix_legacy[JSON-json_field-COMPLEX-1000-10] or test_full_matrix_columnar[JSON-json_field-COMPLEX-1000-10]" \
        --benchmark-only \
        --benchmark-json="$RESULTS_DIR/res_scalar_$LABEL.json" \
        -q
}

# 1. Test Current (v6)
run_bench "v6"

# 2. Downgrade to v5 (Baseline)
echo "Downgrading to Protobuf v5..."
pip install "protobuf<6" --quiet
V5_VER=$(pip show protobuf | grep Version | awk '{print $2}')
echo "Downgraded to: $V5_VER"

run_bench "v5"

# 3. Restore v6
echo "Restoring Protobuf v6 ($CURRENT_VER)..."
pip install "protobuf==$CURRENT_VER" --quiet

echo "Verification Complete. Results in $RESULTS_DIR"
