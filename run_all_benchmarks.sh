#!/bin/bash
set -e

# =============================================================================
# Milvus Benchmark Runner
# Usage: ./run_all_benchmarks.sh [small|medium|large]
#
# Sizes:
#   small  : NQ=100, TopK=1000  (Default, safe for 8GB RAM)
#   medium : NQ=500, TopK=1000  (Requires ~16GB RAM)
#   large  : NQ=1000, TopK=1000 (Requires ~32GB RAM for full profiling)
# =============================================================================

SIZE=${1:-small}
OUTPUT_DIR="benchmark_results"
mkdir -p "$OUTPUT_DIR"

echo "Using size: $SIZE"

if [ "$SIZE" == "small" ]; then
    NQ=100
    TOPK=1000
    ACCESS_COUNT=10000
    LOOPS=10
elif [ "$SIZE" == "medium" ]; then
    NQ=500
    TOPK=1000
    ACCESS_COUNT=50000
    LOOPS=10
elif [ "$SIZE" == "large" ]; then
    NQ=1000
    TOPK=1000
    ACCESS_COUNT=100000
    LOOPS=20
else
    echo "Invalid size. Use small, medium, or large."
    exit 1
fi

echo "========================================================"
echo "Running Pytest Benchmarks (Latency & Ops/s)"
echo "========================================================"
# Run fast benchmarks (smaller subset for quick metrics)
python -m pytest tests/benchmark/test_perf_search.py tests/benchmark/test_perf_insert.py \
    --benchmark-json="$OUTPUT_DIR/pytest_results.json" 

echo "Pytest results saved to $OUTPUT_DIR/pytest_results.json"


echo "========================================================"
echo "Running CPU Profiling (Py-Spy)"
echo "========================================================"

# Helper function for profiling
run_profile() {
    NAME=$1
    shift
    echo "Profiling: $NAME"
    py-spy record -o "$OUTPUT_DIR/${NAME}.svg" --format speedscope -- \
        python tests/benchmark/profile_runner.py "$@" || echo "Profiling $NAME failed"
}

# 1. Search Iteration (Legacy)
run_profile "cpu_search_legacy" --scenario search_iteration \
    --nq $NQ --topk $TOPK --result-type legacy --loops $LOOPS

# 2. Search Iteration (Columnar)
run_profile "cpu_search_columnar" --scenario search_iteration \
    --nq $NQ --topk $TOPK --result-type columnar --loops $LOOPS

# 3. Search Random (Legacy)
run_profile "cpu_random_legacy" --scenario search_random \
    --nq $NQ --topk $TOPK --access-count $ACCESS_COUNT --result-type legacy --loops $LOOPS

# 4. Search Random (Columnar)
run_profile "cpu_random_columnar" --scenario search_random \
    --nq $NQ --topk $TOPK --access-count $ACCESS_COUNT --result-type columnar --loops $LOOPS

# 5. Search Batch (Columnar)
run_profile "cpu_batch_columnar" --scenario search_columnar \
    --nq $NQ --topk $TOPK --result-type columnar --loops $LOOPS

# 6. Insert Preparation
BATCH_SIZE=$((NQ * TOPK / 10)) # Scale insert batch with NQ roughly
if [ $BATCH_SIZE -lt 1000 ]; then BATCH_SIZE=1000; fi
run_profile "cpu_insert_prep" --scenario insert \
    --batch-size $BATCH_SIZE --loops $LOOPS

echo "CPU profiles saved to $OUTPUT_DIR/*.svg"


echo "========================================================"
echo "Running Memory Profiling (Memray)"
echo "========================================================"

run_mem_profile() {
    NAME=$1
    shift
    echo "Mem Profiling: $NAME"
    memray run -o "$OUTPUT_DIR/${NAME}.bin" --Force \
        tests/benchmark/profile_runner.py "$@"
    memray flamegraph "$OUTPUT_DIR/${NAME}.bin" -o "$OUTPUT_DIR/${NAME}.html" || echo "Flamegraph $NAME failed"
    rm "$OUTPUT_DIR/${NAME}.bin"
}

# Reduce loops for memory profiling to save time/space
MEM_LOOPS=3

# 1. Search Iteration (Legacy)
run_mem_profile "mem_search_legacy" --scenario search_iteration \
    --nq $NQ --topk $TOPK --result-type legacy --loops $MEM_LOOPS

# 2. Search Iteration (Columnar)
run_mem_profile "mem_search_columnar" --scenario search_iteration \
    --nq $NQ --topk $TOPK --result-type columnar --loops $MEM_LOOPS

# 3. Insert Preparation
run_mem_profile "mem_insert_prep" --scenario insert \
    --batch-size $BATCH_SIZE --loops $MEM_LOOPS

echo "Memory profiles saved to $OUTPUT_DIR/*.html"

echo "========================================================"
echo "All Done! Results in $OUTPUT_DIR/"
echo "========================================================"
