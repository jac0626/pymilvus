#!/usr/bin/env python3
"""
Access Pattern Benchmarks

Tests performance of different access patterns on search results:
1. Random point access: res[i][j]
2. Slice access: res[0][0:100]
3. Full iteration (baseline)
"""

import pytest
from pymilvus.client.columnar_search_result import ColumnarSearchResult
from pymilvus.client.search_result import SearchResult

from tests.benchmark.kernels import build_search_result, iterate_result, random_access, slice_access


# =============================================================================
# Test Configuration
# =============================================================================

# Standard test sizes
STANDARD_SIZES = [
    (10, 1000),      # 10k hits
    (100, 1000),     # 100k hits
]

# Extended scale (user requested: max 16384)
EXTENDED_SCALE = [
    (1, 16384),      # Single query, max topk
    (16384, 1),      # Max queries, single result
    (128, 128),      # Balanced (~16k hits)
]

# Slice sizes for partial result access
SLICE_SIZES = [10, 100, 1000]

# Number of random accesses to perform
RANDOM_ACCESS_COUNTS = [10, 100, 1000]


# =============================================================================
# Random Point Access Benchmarks
# =============================================================================

@pytest.mark.parametrize("nq, topk", STANDARD_SIZES)
@pytest.mark.parametrize("num_accesses", RANDOM_ACCESS_COUNTS)
def test_random_access_columnar(benchmark, nq, topk, num_accesses):
    """Benchmark random point access pattern on ColumnarSearchResult."""
    import random
    data = build_search_result(
        nq=nq, topk=topk,
        scalar_fields=[("age", "INT64", None)],
        vector_fields=[("vector", "FLOAT_VECTOR", 128)]
    )
    
    # Generate random indices
    random.seed(42)  # Reproducible
    indices = [(random.randint(0, nq-1), random.randint(0, topk-1)) 
               for _ in range(num_accesses)]
    
    def run_random_access():
        result = ColumnarSearchResult(data)
        return random_access(result, indices, "age")
    
    values = benchmark(run_random_access)
    assert len(values) == num_accesses


@pytest.mark.parametrize("nq, topk", STANDARD_SIZES)
@pytest.mark.parametrize("num_accesses", RANDOM_ACCESS_COUNTS)
def test_random_access_legacy(benchmark, nq, topk, num_accesses):
    """Benchmark random point access pattern on Legacy SearchResult."""
    import random
    data = build_search_result(
        nq=nq, topk=topk,
        scalar_fields=[("age", "INT64", None)],
        vector_fields=[("vector", "FLOAT_VECTOR", 128)]
    )
    
    random.seed(42)
    indices = [(random.randint(0, nq-1), random.randint(0, topk-1)) 
               for _ in range(num_accesses)]
    
    def run_random_access():
        result = SearchResult(data)
        return random_access(result, indices, "age")
    
    values = benchmark(run_random_access)
    assert len(values) == num_accesses


# =============================================================================
# Slice Access Benchmarks
# =============================================================================

@pytest.mark.parametrize("nq, topk", [(10, 10000), (100, 1000)])
@pytest.mark.parametrize("slice_size", SLICE_SIZES)
def test_slice_access_columnar(benchmark, nq, topk, slice_size):
    """Benchmark slice access pattern on ColumnarSearchResult."""
    data = build_search_result(
        nq=nq, topk=topk,
        scalar_fields=[("name", "VARCHAR", "SMALL")],
        vector_fields=[("vector", "FLOAT_VECTOR", 128)]
    )
    
    def run_slice_access():
        result = ColumnarSearchResult(data)
        # Access first `slice_size` hits from each query
        total = 0
        for q_idx in range(min(nq, 10)):  # Limit to 10 queries for fair comparison
            sliced = slice_access(result, q_idx, 0, slice_size, "name")
            total += len(sliced)
        return total
    
    count = benchmark(run_slice_access)
    assert count == min(nq, 10) * slice_size


@pytest.mark.parametrize("nq, topk", [(10, 10000), (100, 1000)])
@pytest.mark.parametrize("slice_size", SLICE_SIZES)
def test_slice_access_legacy(benchmark, nq, topk, slice_size):
    """Benchmark slice access pattern on Legacy SearchResult."""
    data = build_search_result(
        nq=nq, topk=topk,
        scalar_fields=[("name", "VARCHAR", "SMALL")],
        vector_fields=[("vector", "FLOAT_VECTOR", 128)]
    )
    
    def run_slice_access():
        result = SearchResult(data)
        total = 0
        for q_idx in range(min(nq, 10)):
            sliced = slice_access(result, q_idx, 0, slice_size, "name")
            total += len(sliced)
        return total
    
    count = benchmark(run_slice_access)
    assert count == min(nq, 10) * slice_size


# =============================================================================
# Extended Scale Tests (NQ/TopK = 16384)
# =============================================================================

@pytest.mark.parametrize("nq, topk", EXTENDED_SCALE)
def test_extended_scale_columnar(benchmark, nq, topk):
    """Test columnar at extended scale (16384 max)."""
    data = build_search_result(
        nq=nq, topk=topk,
        scalar_fields=[("id", "INT64", None)],
    )
    
    def run_iterate():
        result = ColumnarSearchResult(data)
        return iterate_result(result, "id")
    
    count = benchmark(run_iterate)
    assert count == nq * topk


@pytest.mark.parametrize("nq, topk", EXTENDED_SCALE)
def test_extended_scale_legacy(benchmark, nq, topk):
    """Test legacy at extended scale (16384 max)."""
    data = build_search_result(
        nq=nq, topk=topk,
        scalar_fields=[("id", "INT64", None)],
    )
    
    def run_iterate():
        result = SearchResult(data)
        return iterate_result(result, "id")
    
    count = benchmark(run_iterate)
    assert count == nq * topk


# =============================================================================
# First Element Access (Common Pattern)
# =============================================================================

@pytest.mark.parametrize("nq, topk", [(100, 1000), (1000, 100)])
def test_first_element_columnar(benchmark, nq, topk):
    """Benchmark accessing res[0][0] pattern (most common)."""
    data = build_search_result(
        nq=nq, topk=topk,
        scalar_fields=[("score", "FLOAT", None)],
        vector_fields=[("vector", "FLOAT_VECTOR", 128)]
    )
    
    def run_first_access():
        result = ColumnarSearchResult(data)
        # Access first element multiple times (simulating repeated access)
        values = []
        for _ in range(100):
            values.append(result[0][0]["score"])
        return values
    
    values = benchmark(run_first_access)
    assert len(values) == 100


@pytest.mark.parametrize("nq, topk", [(100, 1000), (1000, 100)])
def test_first_element_legacy(benchmark, nq, topk):
    """Benchmark accessing res[0][0] pattern on Legacy."""
    data = build_search_result(
        nq=nq, topk=topk,
        scalar_fields=[("score", "FLOAT", None)],
        vector_fields=[("vector", "FLOAT_VECTOR", 128)]
    )
    
    def run_first_access():
        result = SearchResult(data)
        values = []
        for _ in range(100):
            values.append(result[0][0]["score"])
        return values
    
    values = benchmark(run_first_access)
    assert len(values) == 100
