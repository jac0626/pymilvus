# Search Performance Benchmarks

This directory contains benchmarks for search/query/hybrid_search operations.

## Subdirectories

- `scalar/` - Scalar field access performance
- `vector/` - Vector field access performance  
- `access_patterns/` - Random access, slice, batch access patterns

## Running Tests

```bash
# Run all search benchmarks
pytest tests/benchmark/search_perf/ --benchmark-only

# Run specific category
pytest tests/benchmark/search_perf/scalar/ -v
pytest tests/benchmark/search_perf/vector/ -v
pytest tests/benchmark/search_perf/access_patterns/ -v
```
