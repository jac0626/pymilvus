#!/usr/bin/env python3
"""
Scalar Field Performance Benchmark (Scientific Matrix)

Implements the Variable Control Matrix:
1. Baseline: Protocol Comparison (Fixed NQ=10, TopK=100)
2. Batch Scalability: Throughput vs NQ (Fixed TopK=100, INT64)
3. Result Scalability: Deserialization vs TopK (Fixed NQ=10, INT64)
4. Payload Complexity: Parsing vs Data Size
"""

import pytest
from pymilvus.client.columnar_search_result import ColumnarSearchResult
from pymilvus.client.search_result import SearchResult
from tests.benchmark.kernels import build_search_result, iterate_result

# =============================================================================
# 1. Baseline Performance (Protocol Comparison)
# Goal: Compare overhead of Columnar vs Legacy for different types
# Fixed: NQ=10, TopK=100
# =============================================================================

BASELINE_TYPES = [
    ("INT64", "int64_field", None),
    ("ARRAY", "array_field", None),
    ("VARCHAR", "varchar_field", "MEDIUM"),
    ("JSON", "json_field", "MEDIUM"),
]

@pytest.mark.parametrize("field_type, field_name, complexity", BASELINE_TYPES)
def test_scalar_baseline(benchmark, field_type, field_name, complexity):
    """Compare protocols under standard load (NQ=10, TopK=100)."""
    nq, topk = 10, 100
    res_data = build_search_result(nq, topk, scalar_fields=[(field_name, field_type, complexity)])
    
    def run_benchmark():
        # Test Columnar (Primary Target)
        # We can also test Legacy if needed, but Columnar is the optimization target
        # For verifying optimization, we should probably run both or just Columnar?
        # The plan implies comparing protocols. We'll run Columnar here.
        # To strictly compare, we might need separate tests or parametrize "mode".
        # Let's parametrize mode inside.
        ColumnarSearchResult(res_data)
        
    # We want to separate Columnar vs Legacy in reporting.
    # Let's use the explicit test structure from before but simplified.
    pass 

# Redefining structure to match pytest-benchmark style better: 
# One test function per scenario, parameterized by mode.

MODES = ["Legacy", "Columnar"]

@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("field_type, field_name, complexity", BASELINE_TYPES)
def test_baseline_protocol(benchmark, mode, field_type, field_name, complexity):
    """Baseline: Compare Legacy vs Columnar for different types."""
    nq, topk = 10, 100
    res_data = build_search_result(nq, topk, scalar_fields=[(field_name, field_type, complexity)])
    
    def run():
        if mode == "Legacy":
            SearchResult(res_data)
        else:
            ColumnarSearchResult(res_data)
    benchmark(run)


# =============================================================================
# 2. Batch Scalability (NQ Growth)
# Goal: Measure throughput linearity
# Fixed: TopK=100, Type=INT64
# =============================================================================

NQ_SCALES = [1, 10, 100, 1000, 10000]

@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("nq", NQ_SCALES)
def test_batch_scalability(benchmark, mode, nq):
    """Batch Scalability: Throughput vs NQ."""
    topk = 100
    res_data = build_search_result(nq, topk, scalar_fields=[("id", "INT64", None)])
    
    def run():
        if mode == "Legacy":
            SearchResult(res_data)
        else:
            ColumnarSearchResult(res_data)
    benchmark(run)


# =============================================================================
# 3. Result Scalability (TopK Growth)
# Goal: Measure deserialization cost
# Fixed: NQ=10, Type=INT64
# =============================================================================

TOPK_SCALES = [10, 100, 1000, 10000]

@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("topk", TOPK_SCALES)
def test_result_scalability(benchmark, mode, topk):
    """Result Scalability: Cost vs TopK."""
    nq = 10
    res_data = build_search_result(nq, topk, scalar_fields=[("id", "INT64", None)])
    
    def run():
        if mode == "Legacy":
            SearchResult(res_data)
        else:
            ColumnarSearchResult(res_data)
    benchmark(run)


# =============================================================================
# 4. Payload Complexity (Data Size)
# Goal: Measure parsing overhead
# Fixed: NQ=10, TopK=100
# =============================================================================

COMPLEXITY_CASES = [
    # VARCHAR
    ("VARCHAR", "varchar_field", "SMALL"),
    ("VARCHAR", "varchar_field", "MEDIUM"),
    ("VARCHAR", "varchar_field", "LARGE"),
    # JSON
    ("JSON", "json_field", "SMALL"),
    ("JSON", "json_field", "MEDIUM"),
    ("JSON", "json_field", "COMPLEX"),
]

@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("field_type, field_name, complexity", COMPLEXITY_CASES)
def test_payload_complexity(benchmark, mode, field_type, field_name, complexity):
    """Payload Complexity: Impact of data size/structure."""
    nq, topk = 10, 100
    res_data = build_search_result(nq, topk, scalar_fields=[(field_name, field_type, complexity)])
    
    def run():
        if mode == "Legacy":
            SearchResult(res_data)
        else:
            ColumnarSearchResult(res_data)
    benchmark(run)
