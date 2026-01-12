#!/usr/bin/env python3
"""
Vector Field Performance Benchmark

Measures actual proto → user object conversion time with real field access.
All tests are parametrized with four access modes for comprehensive coverage.

Access Modes:
1. full_iteration: Iterate all hits, access field each time
2. slice: res[q][0:100] slice access for each query
3. first_element: res[0][0][field] single element access
4. batch_column: get_column() API (ColumnarSearchResult only)
"""

import pytest
from pymilvus.client.columnar_search_result import ColumnarSearchResult
from pymilvus.client.search_result import SearchResult
from tests.benchmark.kernels import build_search_result, iterate_result, slice_access, get_column_batch


# =============================================================================
# Configuration
# =============================================================================

MODES = ["Legacy", "Columnar"]
ACCESS_MODES = ["full_iteration", "slice", "first_element", "batch_column"]

# Vector field types to test (complete coverage)
VECTOR_TYPES = [
    # Dense float vectors
    ("FLOAT_VECTOR", 128),
    ("FLOAT_VECTOR", 768),
    ("FLOAT16_VECTOR", 768),
    ("BFLOAT16_VECTOR", 768),
    # Binary and integer vectors
    ("BINARY_VECTOR", 1024),
    ("INT8_VECTOR", 128),
    # Sparse vectors (dim is ignored, use placeholder)
    ("SPARSE_FLOAT_VECTOR", 0),
]

# NQ scalability (fixed TopK=1 to prevent OOM for large NQ)
NQ_SCALES = [
    (1, 1, "nq_1"),
    (10, 1, "nq_10"),
    (100, 1, "nq_100"),
    (1000, 1, "nq_1000"),
    (10000, 1, "nq_10000"),
]

# TopK scalability (fixed NQ=1 to limit memory)
TOPK_SCALES = [
    (1, 1, "topk_1"),
    (1, 10, "topk_10"),
    (1, 100, "topk_100"),
    (1, 1000, "topk_1000"),
    (1, 10000, "topk_10000"),
]

# Dimension impact cases
DIMENSION_CASES = [
    (128, "small"),
    (768, "medium"),
    (1536, "large"),
    (3072, "xlarge"),
]

# Vector type comparison
TYPE_COMPARISON = [
    ("FLOAT_VECTOR", 1024),
    ("BINARY_VECTOR", 8192),
    ("FLOAT16_VECTOR", 1024),
]


# =============================================================================
# Access Mode Executor (Core Logic)
# =============================================================================

def execute_access(result, field_name: str, access_mode: str, nq: int, topk: int):
    """
    Execute the specified access mode on the result.
    
    All modes ensure actual field data is accessed (cold start).
    """
    if access_mode == "full_iteration":
        return iterate_result(result, field_name)
    
    elif access_mode == "slice":
        slice_size = min(100, topk)
        total = 0
        for q_idx in range(nq):
            sliced = slice_access(result, q_idx, 0, slice_size, field_name)
            total += len(sliced)
        return total
    
    elif access_mode == "first_element":
        values = []
        for _ in range(100):
            values.append(result[0][0][field_name])
        return len(values)
    
    elif access_mode == "batch_column":
        columns = get_column_batch(result, field_name)
        return len(columns)  # Returns nq for vectors
    
    return 0


def run_benchmark(res_data, field_name: str, mode: str, access_mode: str, nq: int, topk: int):
    """
    Run benchmark with specified protocol mode and access mode.
    """
    if access_mode == "batch_column" and mode == "Legacy":
        pytest.skip("batch_column only supported for ColumnarSearchResult")
    
    if mode == "Legacy":
        result = SearchResult(res_data)
    else:
        result = ColumnarSearchResult(res_data)
    
    return execute_access(result, field_name, access_mode, nq, topk)


# =============================================================================
# 1. Baseline: Vector Type Comparison
# =============================================================================

@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("access_mode", ACCESS_MODES)
@pytest.mark.parametrize("dtype, dim", VECTOR_TYPES)
def test_vector_type_baseline(benchmark, mode, access_mode, dtype, dim):
    """
    Baseline: Compare different vector types across all access modes.
    
    Fixed: NQ=10, TopK=1000
    """
    nq, topk = 10, 1000
    res_data = build_search_result(nq, topk, vector_fields=[("vector", dtype, dim)])
    
    def run():
        return run_benchmark(res_data, "vector", mode, access_mode, nq, topk)
    
    count = benchmark(run)
    if access_mode == "full_iteration":
        assert count == nq * topk
    elif access_mode == "slice":
        assert count == nq * min(100, topk)
    elif access_mode == "first_element":
        assert count == 100
    elif access_mode == "batch_column":
        assert count == nq


# =============================================================================
# 2. NQ Scalability (Fixed TopK=100)
# =============================================================================

@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("access_mode", ACCESS_MODES)
@pytest.mark.parametrize("nq, topk, label", NQ_SCALES)
def test_vector_nq_scalability(benchmark, mode, access_mode, nq, topk, label):
    """
    NQ Scalability: Measure how NQ growth affects performance.
    
    Fixed: TopK=100, FLOAT_VECTOR dim=128
    """
    res_data = build_search_result(nq, topk, vector_fields=[("vector", "FLOAT_VECTOR", 128)])
    
    def run():
        return run_benchmark(res_data, "vector", mode, access_mode, nq, topk)
    
    benchmark(run)


# =============================================================================
# 3. TopK Scalability (Fixed NQ=1)
# =============================================================================

@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("access_mode", ACCESS_MODES)
@pytest.mark.parametrize("nq, topk, label", TOPK_SCALES)
def test_vector_topk_scalability(benchmark, mode, access_mode, nq, topk, label):
    """
    TopK Scalability: Measure how TopK growth affects performance.
    
    Fixed: NQ=1, FLOAT_VECTOR dim=128
    """
    res_data = build_search_result(nq, topk, vector_fields=[("vector", "FLOAT_VECTOR", 128)])
    
    def run():
        return run_benchmark(res_data, "vector", mode, access_mode, nq, topk)
    
    benchmark(run)


# =============================================================================
# 4. Dimension Impact
# =============================================================================

@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("access_mode", ACCESS_MODES)
@pytest.mark.parametrize("dim, label", DIMENSION_CASES)
def test_vector_dimension_impact(benchmark, mode, access_mode, dim, label):
    """
    Dimension Impact: How vector dimension affects performance.
    
    Fixed: NQ=10, TopK=100, FLOAT_VECTOR
    """
    nq, topk = 10, 100
    res_data = build_search_result(nq, topk, vector_fields=[("vector", "FLOAT_VECTOR", dim)])
    
    def run():
        return run_benchmark(res_data, "vector", mode, access_mode, nq, topk)
    
    benchmark(run)


# =============================================================================
# 5. Vector Type Comparison (Same Data Size)
# =============================================================================

@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("access_mode", ACCESS_MODES)
@pytest.mark.parametrize("dtype, dim", TYPE_COMPARISON)
def test_vector_type_comparison(benchmark, mode, access_mode, dtype, dim):
    """
    Vector Type Comparison: Float vs Binary vs Float16.
    
    Compares parsing overhead for different vector representations.
    Fixed: NQ=10, TopK=100
    """
    nq, topk = 10, 100
    res_data = build_search_result(nq, topk, vector_fields=[("vector", dtype, dim)])
    
    def run():
        return run_benchmark(res_data, "vector", mode, access_mode, nq, topk)
    
    benchmark(run)
