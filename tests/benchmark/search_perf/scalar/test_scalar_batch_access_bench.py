#!/usr/bin/env python3
"""
Scalar Batch Access Benchmarks

Compares iteration-based access vs batch column access (get_column API).
"""

import pytest
from pymilvus.client.columnar_search_result import ColumnarSearchResult

from tests.benchmark.kernels import build_search_result, iterate_result, get_column_batch


# Test configurations
SCALAR_CONFIGS = [
    ("INT64", "int_field", None),
    ("VARCHAR", "varchar_field", "MEDIUM"),
    ("JSON", "json_field", "COMPLEX"),
]


@pytest.mark.parametrize("dtype, field_name, complexity", SCALAR_CONFIGS)
def test_scalar_access_iteration(benchmark, dtype, field_name, complexity):
    """Benchmark iteration-based scalar field access."""
    nq, topk = 10, 1000
    res_data = build_search_result(
        nq=nq, topk=topk,
        scalar_fields=[(field_name, dtype, complexity)]
    )
    
    def run_iteration():
        result = ColumnarSearchResult(res_data)
        return iterate_result(result, field_name)
    
    count = benchmark(run_iteration)
    assert count == nq * topk


@pytest.mark.parametrize("dtype, field_name, complexity", SCALAR_CONFIGS)
def test_scalar_access_batch(benchmark, dtype, field_name, complexity):
    """Benchmark batch column access (get_column API)."""
    nq, topk = 10, 1000
    res_data = build_search_result(
        nq=nq, topk=topk,
        scalar_fields=[(field_name, dtype, complexity)]
    )
    
    def run_batch():
        result = ColumnarSearchResult(res_data)
        columns = get_column_batch(result, field_name)
        return sum(len(c) for c in columns)
    
    count = benchmark(run_batch)
    assert count == nq * topk
