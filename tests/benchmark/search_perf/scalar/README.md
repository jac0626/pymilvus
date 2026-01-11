# Scalar Field Performance Testing

This directory contains benchmarks and profiling for scalar field access performance in PyMilvus.

## Structure

- `test_scalar_bench.py` - pytest-benchmark suite for scalar fields (INT, VARCHAR, JSON)
- `profile_scalar_performance.py` - CPU profiling script
- `profile_scalar_memory.py` - Memory profiling script using memray

## Running Tests

```bash
# Run benchmarks
pytest tests/benchmark/search_perf/scalar/test_scalar_bench.py --benchmark-json=scalar_benchmark.json

# Run CPU profiling
python tests/benchmark/search_perf/scalar/profile_scalar_performance.py

# Run memory profiling  
python -m memray run tests/benchmark/search_perf/scalar/profile_scalar_memory.py
```

## Key Findings

- **Performance Gap**: 5-8x slower for Legacy vs Columnar
- **Root Cause**: Eager Loading + Mass Object Creation
- **Memory Overhead**: TBD (pending memray profiling)
