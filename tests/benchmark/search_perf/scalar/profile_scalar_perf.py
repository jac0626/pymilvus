#!/usr/bin/env python3
"""
Scalar Performance Profiling
===========================
Pure python profiling script for scalar parsing bottlenecks.
Focuses on Legacy SearchResult - specifically JSON parsing overhead.

Usage:
    python3 profile_scalar_perf.py
"""

import sys
import os

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../../"))
sys.path.insert(0, project_root)

from pymilvus.client.search_result import SearchResult
from tests.benchmark.kernels import build_search_result, iterate_result, ScalarComplexity, profiling

# =============================================================================
# Helper
# =============================================================================

def run_legacy_iteration(data, field_name):
    """Run full iteration on Legacy SearchResult."""
    res = SearchResult(data)
    iterate_result(res, field_name)

# =============================================================================
# Scenarios
# =============================================================================

def profile_large_json(use_memory: bool = False):
    """
    Scenario 2: Large JSON (Legacy)
    Target: JSON field with COMPLEX complexity
    Goal: Identify JSON parsing and object creation overhead (orjson/json).
    """
    print("\n[Scenario 2] Large JSON: Complex Structure - Legacy Mode")
    print("-" * 60)
    
    nq, topk = 10, 1000
    data = build_search_result(
        nq, topk, 
        scalar_fields=[("meta", "JSON", "COMPLEX")]
    )
    
    if use_memory:
        profiling.profile_memory(
            "scalar_json_legacy",
            lambda d: run_legacy_iteration(d, "meta"),
            data
        )
    else:
        profiling.profile_cpu(
            "scalar_json_legacy",
            lambda d: run_legacy_iteration(d, "meta"),
            data,
            top_n=20
        )

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Scalar Performance Profiling")
    parser.add_argument("--memory", action="store_true", help="Run memory profiling instead of CPU")
    args = parser.parse_args()

    print("============================================================")
    print(f"Scalar Performance Profiling Suite (Mode: {'Memory' if args.memory else 'CPU'})")
    print("============================================================")
    
    profile_large_json(args.memory)
    
    print("\nDone! Profile stats saved to .benchmarks/")
