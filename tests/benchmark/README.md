# pymilvus Benchmark Suite

This benchmark suite measures client-side performance of pymilvus API operations without requiring a running Milvus server.

## Directory Structure

```
tests/benchmark/
├── kernels/                  # Shared utilities
│   ├── data_gen.py          # Mock data generation
│   ├── result_ops.py        # Result iteration helpers
│   └── profiling.py         # CPU/Memory profiling
├── search_perf/             # Search/Query/HybridSearch benchmarks
│   ├── scalar/              # Scalar field access
│   ├── vector/              # Vector field access
│   └── access_patterns/     # Random/slice access patterns
├── insert_perf/             # Insert/Upsert benchmarks
├── conftest.py              # Mock gRPC stubs & fixtures
├── mock_responses.py        # Fake protobuf builders
└── scripts/                 # Profiling helper scripts
```

## Quick Start

```bash
# Install dependencies
pip install -e ".[dev]"

# Run all benchmarks
pytest tests/benchmark/ --benchmark-only

# Run specific category
pytest tests/benchmark/search_perf/ --benchmark-only
pytest tests/benchmark/insert_perf/ --benchmark-only
```

## Benchmark Categories

### Search Performance (`search_perf/`)
- **scalar/**: INT64, VARCHAR, JSON, ARRAY field access
- **vector/**: FLOAT_VECTOR, FLOAT16, BINARY vector access
- **access_patterns/**: Random `res[i][j]`, slice `res[0][0:100]`

### Insert Performance (`insert_perf/`)
- Data generation benchmarks
- Columnar conversion benchmarks
- Field-specific serialization (VARCHAR, JSON, FLOAT_VECTOR)

## Profiling

```bash
# CPU profiling
pytest tests/benchmark/search_perf/scalar/ --benchmark-only
python tests/benchmark/search_perf/scalar/profile_scalar_performance.py

# Memory profiling
memray run -o mem.bin tests/benchmark/search_perf/scalar/profile_scalar_memory.py
memray summary mem.bin
```
