#!/usr/bin/env python3
"""
Scalar Field Performance Benchmark

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

# Scalar field types to test (complete coverage)
SCALAR_TYPES = [
    # Numeric types
    ("BOOL", "bool_field", None),
    ("INT8", "int8_field", None),
    ("INT16", "int16_field", None),
    ("INT32", "int32_field", None),
    ("INT64", "int64_field", None),
    ("FLOAT", "float_field", None),
    ("DOUBLE", "double_field", None),
    # String types
    ("VARCHAR", "varchar_field", "MEDIUM"),
    # Complex types
    ("JSON", "json_field", "MEDIUM"),
    ("ARRAY", "array_field", None),
    # Special types
    ("GEOMETRY", "geometry_field", None),
    ("TIMESTAMPTZ", "timestamptz_field", None),
    # Composite types (internal use for STRUCT)
    ("_ARRAY_OF_STRUCT", "struct_array_field", None),
    ("_ARRAY_OF_VECTOR", "vector_array_field", None),
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

# Payload complexity cases
COMPLEXITY_CASES = [
    ("VARCHAR", "text", "SMALL"),
    ("VARCHAR", "text", "MEDIUM"),
    ("VARCHAR", "text", "LARGE"),
    ("JSON", "meta", "SMALL"),
    ("JSON", "meta", "MEDIUM"),
    ("JSON", "meta", "COMPLEX"),
    ("JSON", "meta", "UNEVEN"),
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
        # Iterate all hits and access field
        return iterate_result(result, field_name)
    
    elif access_mode == "slice":
        # Slice first 100 hits (or topk if smaller) from each query
        slice_size = min(100, topk)
        total = 0
        for q_idx in range(nq):
            sliced = slice_access(result, q_idx, 0, slice_size, field_name)
            total += len(sliced)
        return total
    
    elif access_mode == "first_element":
        # Access first element 100 times (simulates common usage)
        values = []
        for _ in range(100):
            values.append(result[0][0][field_name])
        return len(values)
    
    elif access_mode == "batch_column":
        # ColumnarSearchResult batch column access
        if hasattr(result, '__iter__') and hasattr(list(result)[0] if nq > 0 else None, 'get_column'):
            # Recreate result since we consumed it checking
            pass
        columns = get_column_batch(result, field_name)
        return sum(len(c) for c in columns)
    
    return 0


def run_benchmark(res_data, field_name: str, mode: str, access_mode: str, nq: int, topk: int):
    """
    Run benchmark with specified protocol mode and access mode.
    
    Returns the result count for assertion.
    """
    # Skip batch_column for Legacy mode (not supported)
    if access_mode == "batch_column" and mode == "Legacy":
        pytest.skip("batch_column only supported for ColumnarSearchResult")
    
    # Cold start: create result object each time
    if mode == "Legacy":
        result = SearchResult(res_data)
    else:
        result = ColumnarSearchResult(res_data)
    
    return execute_access(result, field_name, access_mode, nq, topk)


# =============================================================================
# 1. Baseline: Field Type Comparison
# =============================================================================

@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("access_mode", ACCESS_MODES)
@pytest.mark.parametrize("field_type, field_name, complexity", SCALAR_TYPES)
def test_scalar_type_baseline(benchmark, mode, access_mode, field_type, field_name, complexity):
    """
    Baseline: Compare different scalar types across all access modes.
    
    Fixed: NQ=10, TopK=1000
    """
    nq, topk = 10, 1000
    res_data = build_search_result(nq, topk, scalar_fields=[(field_name, field_type, complexity)])
    
    def run():
        return run_benchmark(res_data, field_name, mode, access_mode, nq, topk)
    
    count = benchmark(run)
    # Verify access happened
    if access_mode == "full_iteration":
        assert count == nq * topk
    elif access_mode == "slice":
        assert count == nq * min(100, topk)
    elif access_mode == "first_element":
        assert count == 100
    elif access_mode == "batch_column":
        assert count == nq * topk


# =============================================================================
# 2. NQ Scalability (Fixed TopK=100)
# =============================================================================

@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("access_mode", ACCESS_MODES)
@pytest.mark.parametrize("nq, topk, label", NQ_SCALES)
def test_scalar_nq_scalability(benchmark, mode, access_mode, nq, topk, label):
    """
    NQ Scalability: Measure how NQ growth affects performance.
    
    Fixed: TopK=100, INT64 field type
    """
    res_data = build_search_result(nq, topk, scalar_fields=[("id", "INT64", None)])
    
    def run():
        return run_benchmark(res_data, "id", mode, access_mode, nq, topk)
    
    benchmark(run)


# =============================================================================
# 3. TopK Scalability (Fixed NQ=1)
# =============================================================================

@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("access_mode", ACCESS_MODES)
@pytest.mark.parametrize("nq, topk, label", TOPK_SCALES)
def test_scalar_topk_scalability(benchmark, mode, access_mode, nq, topk, label):
    """
    TopK Scalability: Measure how TopK growth affects performance.
    
    Fixed: NQ=1, INT64 field type
    """
    res_data = build_search_result(nq, topk, scalar_fields=[("id", "INT64", None)])
    
    def run():
        return run_benchmark(res_data, "id", mode, access_mode, nq, topk)
    
    benchmark(run)


# =============================================================================
# 4. Payload Complexity: Data Size Impact
# =============================================================================

@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("access_mode", ACCESS_MODES)
@pytest.mark.parametrize("field_type, field_name, complexity", COMPLEXITY_CASES)
def test_scalar_payload_complexity(benchmark, mode, access_mode, field_type, field_name, complexity):
    """
    Payload Complexity: Impact of VARCHAR length and JSON structure.
    
    Fixed: NQ=10, TopK=100
    """
    nq, topk = 10, 100
    res_data = build_search_result(nq, topk, scalar_fields=[(field_name, field_type, complexity)])
    
    def run():
        return run_benchmark(res_data, field_name, mode, access_mode, nq, topk)
    
    benchmark(run)
