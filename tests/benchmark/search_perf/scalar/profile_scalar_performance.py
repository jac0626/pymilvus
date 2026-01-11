#!/usr/bin/env python3
"""
Scalar Field CPU Profiling

Uses shared kernels for data generation and profiling.
"""

from pymilvus.client.columnar_search_result import ColumnarSearchResult
from pymilvus.client.search_result import SearchResult

import sys
sys.path.insert(0, str(__file__).rsplit('/', 3)[0])
from tests.benchmark.kernels import build_search_result, iterate_result, profile_cpu, get_output_dir


# Test scenario: 10k items (NQ=10, TopK=1000)
NQ, TOPK = 10, 1000


def build_json_result():
    """Build result with JSON field (worst case)."""
    return build_search_result(
        nq=NQ, topk=TOPK,
        scalar_fields=[("json_field", "JSON", "COMPLEX")],
    )


def build_int_result():
    """Build result with INT64 field (simplest case)."""
    return build_search_result(
        nq=NQ, topk=TOPK,
        scalar_fields=[("int64_field", "INT64", None)],
    )


def run_legacy_json():
    """Legacy + JSON workload."""
    res_data = build_json_result()
    sr = SearchResult(res_data)
    return iterate_result(sr, "json_field")


def run_columnar_json():
    """Columnar + JSON workload."""
    res_data = build_json_result()
    cr = ColumnarSearchResult(res_data)
    return iterate_result(cr, "json_field")


def run_legacy_int():
    """Legacy + INT64 workload."""
    res_data = build_int_result()
    sr = SearchResult(res_data)
    return iterate_result(sr, "int64_field")


if __name__ == "__main__":
    print("=== Scalar Field Profiling ===\n")
    
    print("1. Profiling Legacy SearchResult (JSON)...")
    count, _, elapsed = profile_cpu("scalar_legacy_json", run_legacy_json)
    print(f"   Processed {count} items in {elapsed*1000:.2f}ms\n")
    
    print("2. Profiling Columnar SearchResult (JSON)...")
    count, _, elapsed = profile_cpu("scalar_columnar_json", run_columnar_json)
    print(f"   Processed {count} items in {elapsed*1000:.2f}ms\n")
    
    print("3. Profiling Legacy SearchResult (INT64)...")
    count, _, elapsed = profile_cpu("scalar_legacy_int", run_legacy_int)
    print(f"   Processed {count} items in {elapsed*1000:.2f}ms\n")
    
    print(f"✅ CPU Profiling Complete!")
    print(f"Output dir: {get_output_dir()}")
