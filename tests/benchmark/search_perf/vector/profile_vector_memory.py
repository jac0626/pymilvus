#!/usr/bin/env python3
"""
Vector Field Memory Profiling using memray

Usage:
    memray run -o vector_mem.bin tests/benchmark/search_perf/vector/profile_vector_memory.py [legacy|columnar|all]
    memray summary vector_mem.bin
"""

import sys
from pymilvus.client.columnar_search_result import ColumnarSearchResult
from pymilvus.client.search_result import SearchResult

sys.path.insert(0, str(__file__).rsplit('/', 4)[0])
from tests.benchmark.kernels import build_search_result, iterate_result


# Test scenario: 10k vectors (Standard Large Batch)
# NQ=10, TopK=1000, Dim=768
NQ = 10
TOPK = 1000
DIM = 768


def build_vector_result():
    """Build result with FLOAT_VECTOR field."""
    return build_search_result(
        nq=NQ, topk=TOPK,
        vector_fields=[("vector", "FLOAT_VECTOR", DIM)],
    )


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    
    print(f"=== Vector Field Memory Profiling ({mode}) ===\n")
    print(f"Testing with {NQ * TOPK:,} vectors (Dim={DIM})...")
    
    # Build data once
    res_data = build_vector_result()
    
    if mode in ["all", "legacy"]:
        print("\n1. Legacy SearchResult...")
        sr = SearchResult(res_data)
        count = iterate_result(sr, "vector")
        print(f"   Processed {count} items")
    
    if mode in ["all", "columnar"]:
        print("\n2. Columnar SearchResult...")
        cr = ColumnarSearchResult(res_data)
        count = iterate_result(cr, "vector")
        print(f"   Processed {count} items")
    
    print("\n✅ Memory profiling complete!")
