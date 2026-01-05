
# Vector Performance Benchmarks

This directory contains benchmarks and profiling tools specifically designed to measure and analyze the performance of vector search results parsing in PyMilvus.

## Key Files

*   `test_vector_bench.py`: Standard `pytest-benchmark` suite. Used for Regression Testing and CI.
*   `profile_vector_performance.py`: Standalone script for CPU/Memory profiling. Used for deep dive analysis.

## 1. Running Benchmarks (Regression Testing)

Use `pytest` to run the scientific matrix benchmarks. This will output min/max/mean/stddev execution times.

```bash
# Run all vector benchmarks
python -m pytest tests/benchmark/vector_perf/test_vector_bench.py

# Generate histogram
python -m pytest tests/benchmark/vector_perf/test_vector_bench.py --benchmark-histogram
```

## 2. Profiling (Deep Dive Analysis)

Use the standalone script to perform in-depth analysis of specific scenarios without pytest overhead.

### CPU Profiling (Standard)

Generates `.stats` files compatible with `snakeviz` or `gprof2dot`.

```bash
# Profile NQ=10, TopK=1000, Dim=768
python tests/benchmark/vector_perf/profile_vector_performance.py --nq 10 --topk 1000 --dim 768
```

### Memory Profiling (memray)

Requires `memray` installed (`pip install memray`).

```bash
# Profile Legacy Mode (High Memory Overhead)
memray run -o mem_legacy.bin tests/benchmark/vector_perf/profile_vector_performance.py --mode legacy

# Profile Columnar Mode (Zero-Copy)
memray run -o mem_columnar.bin tests/benchmark/vector_perf/profile_vector_performance.py --mode columnar

# Generate Report
memray summary mem_legacy.bin
```
