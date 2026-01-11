#!/usr/bin/env python3
"""
Vector Field Performance Benchmark (Scientific Matrix)

Implements the Variable Control Matrix:
1. Baseline: Protocol Comparison (Fixed NQ=10, TopK=100)
2. Batch Scalability: Throughput vs NQ (Fixed TopK=100, Dim=768)
3. Result Scalability: Deserialization vs TopK (Fixed NQ=10, Dim=768)
4. Vector Dimension: Copy Cost vs Dim
"""

import pytest
from pymilvus.client.columnar_search_result import ColumnarSearchResult
from pymilvus.client.search_result import SearchResult
from tests.benchmark.kernels import build_search_result, iterate_result

# One generic helper
MODES = ["Legacy", "Columnar"]

def build_vector_result(nq, topk, dim, dtype="FLOAT_VECTOR"):
    return build_search_result(nq=nq, topk=topk, vector_fields=[("vector", dtype, dim)])

# =============================================================================
# 1. Baseline Performance (Protocol Comparison)
# Goal: Compare overhead of Columnar vs Legacy for different vector types
# Fixed: NQ=10, TopK=100, Dim=768
# =============================================================================

BASELINE_VECTORS = [
    ("FLOAT_VECTOR", 768),
    ("FLOAT16_VECTOR", 768),
    ("BINARY_VECTOR", 768),
]

@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("dtype, dim", BASELINE_VECTORS)
def test_vector_baseline(benchmark, mode, dtype, dim):
    """Baseline: Compare Legacy vs Columnar for different vectors."""
    nq, topk = 10, 100
    res_data = build_vector_result(nq, topk, dim, dtype)
    
    def run():
        if mode == "Legacy":
            SearchResult(res_data)
        else:
            ColumnarSearchResult(res_data)
    benchmark(run)


# =============================================================================
# 2. Batch Scalability (NQ Growth)
# Goal: Measure throughput linearity for FLOAT_VECTOR
# Fixed: TopK=100, Dim=768
# =============================================================================

NQ_SCALES = [1, 10, 100, 1000, 10000]

@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("nq", NQ_SCALES)
def test_vector_batch_scalability(benchmark, mode, nq):
    """Batch Scalability: Throughput vs NQ."""
    topk, dim = 100, 768
    res_data = build_vector_result(nq, topk, dim, "FLOAT_VECTOR")
    
    def run():
        if mode == "Legacy":
            SearchResult(res_data)
        else:
            ColumnarSearchResult(res_data)
    benchmark(run)


# =============================================================================
# 3. Result Scalability (TopK Growth)
# Goal: Measure deserialization cost
# Fixed: NQ=10, Dim=768
# =============================================================================

TOPK_SCALES = [10, 100, 1000, 10000]

@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("topk", TOPK_SCALES)
def test_vector_result_scalability(benchmark, mode, topk):
    """Result Scalability: Cost vs TopK."""
    nq, dim = 10, 768
    res_data = build_vector_result(nq, topk, dim, "FLOAT_VECTOR")
    
    def run():
        if mode == "Legacy":
            SearchResult(res_data)
        else:
            ColumnarSearchResult(res_data)
    benchmark(run)


# =============================================================================
# 4. Payload Complexity (Dimension)
# Goal: Measure copy cost vs Dimension
# Fixed: NQ=10, TopK=100
# =============================================================================

DIMS = [128, 768, 1536]

@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("dim", DIMS)
def test_vector_dimension_cost(benchmark, mode, dim):
    """Dimension Cost: Throughput vs Dim."""
    nq, topk = 10, 100
    res_data = build_vector_result(nq, topk, dim, "FLOAT_VECTOR")
    
    def run():
        if mode == "Legacy":
            SearchResult(res_data)
        else:
            ColumnarSearchResult(res_data)
    benchmark(run)
