# Insert Performance Benchmarks

This directory contains benchmarks and profiling for insert/upsert operations in PyMilvus.

## Test Scenarios

Based on the performance report:
- **Basic**: 1,000 records, 6 fields
- **Complex**: 10,000 records, 6 fields

## Schema

```python
schema = [
    ("id", "INT64"),           # Primary Key
    ("embedding", "FLOAT_VECTOR", 128),
    ("name", "VARCHAR", 100),
    ("age", "INT32"),
    ("json_field", "JSON"),
    ("varchar_field", "VARCHAR", 100),
]
```

## Running Tests

```bash
# Run insert benchmarks
pytest tests/benchmark/insert_perf/test_insert_bench.py --benchmark-only

# CPU profiling
python tests/benchmark/insert_perf/profile_insert_performance.py

# Memory profiling
memray run -o insert_mem.bin tests/benchmark/insert_perf/profile_insert_memory.py
memray summary insert_mem.bin
```
