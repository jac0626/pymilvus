#!/usr/bin/env python3
"""
Scalar Field Memory Profiling using memray

Usage:
    memray run -o scalar_mem.bin tests/benchmark/scalar_perf/profile_scalar_memory.py [legacy|columnar|all]
    memray summary scalar_mem.bin
"""

import sys
from pymilvus.client.columnar_search_result import ColumnarSearchResult
from pymilvus.client.search_result import SearchResult

sys.path.insert(0, str(__file__).rsplit('/', 3)[0])
from tests.benchmark.kernels import build_search_result, iterate_result


# Test scenario: 10k items
NQ, TOPK = 10, 1000


def build_json_result():
    """Build result with JSON field (worst case for memory)."""
    return build_search_result(
        nq=NQ, topk=TOPK,
        scalar_fields=[("json_field", "JSON", "COMPLEX")],
    )


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    
    print(f"=== Scalar Field Memory Profiling ({mode}) ===\n")
    print(f"Testing with {NQ * TOPK:,} JSON entries...")
    
    # Build data once
    res_data = build_json_result()
    
    if mode in ["all", "legacy"]:
        print("\n1. Legacy SearchResult...")
        sr = SearchResult(res_data)
        count = iterate_result(sr, "json_field")
        print(f"   Processed {count} items")
    
    if mode in ["all", "columnar"]:
        print("\n2. Columnar SearchResult...")
        cr = ColumnarSearchResult(res_data)
        count = iterate_result(cr, "json_field")
        print(f"   Processed {count} items")
    
    print("\n✅ Memory profiling complete!")
