#!/usr/bin/env python3
"""
Batch Access Benchmarks for Vector Fields

Compares iteration-based access vs batch column access (get_column API).
"""

import pytest
from pymilvus.client.columnar_search_result import ColumnarSearchResult

from tests.benchmark.kernels import build_search_result, iterate_result, get_column_batch


# Test configurations
VECTOR_CONFIGS = [
    ("FLOAT_VECTOR", 128),
    ("FLOAT16_VECTOR", 128),
    ("BINARY_VECTOR", 1024),
]


@pytest.mark.parametrize("dtype, dim", VECTOR_CONFIGS)
def test_vector_access_iteration(benchmark, dtype, dim):
    """Benchmark iteration-based vector access."""
    nq, topk = 10, 1000
    res_data = build_search_result(
        nq=nq, topk=topk,
        vector_fields=[("vector", dtype, dim)]
    )
    
    def run_iteration():
        result = ColumnarSearchResult(res_data)
        return iterate_result(result, "vector")
    
    count = benchmark(run_iteration)
    assert count == nq * topk


@pytest.mark.parametrize("dtype, dim", VECTOR_CONFIGS)
def test_vector_access_batch(benchmark, dtype, dim):
    """Benchmark batch column access (get_column API)."""
    nq, topk = 10, 1000
    res_data = build_search_result(
        nq=nq, topk=topk,
        vector_fields=[("vector", dtype, dim)]
    )
    
    def run_batch():
        result = ColumnarSearchResult(res_data)
        columns = get_column_batch(result, "vector")
        # Note: get_column for vectors returns flattened data (dim elements per hit)
        return len(columns)  # Should be nq queries
    
    num_queries = benchmark(run_batch)
    assert num_queries == nq

