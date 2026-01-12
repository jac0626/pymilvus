#!/usr/bin/env python3
"""
Vector Performance Profiling
===========================
Pure python profiling script for vector parsing bottlenecks.
Focuses on Legacy SearchResult to identify optimization opportunities.
Includes one comparison scenario to verify Columnar optimization.

Usage:
    python3 profile_vector_perf.py
"""

import sys
import os

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../../"))
sys.path.insert(0, project_root)

from pymilvus.client.search_result import SearchResult
from pymilvus.client.columnar_search_result import ColumnarSearchResult
from tests.benchmark.kernels import build_search_result, iterate_result, ScalarComplexity, profiling

# =============================================================================
# Common Setup
# =============================================================================

def run_legacy_iteration(data):
    """Run full iteration on Legacy SearchResult."""
    res = SearchResult(data)
    iterate_result(res, "vector")

def run_columnar_iteration(data):
    """Run full iteration on ColumnarSearchResult."""
    res = ColumnarSearchResult(data)
    iterate_result(res, "vector")

# =============================================================================
# Scenarios
# =============================================================================

def profile_baseline(use_memory: bool = False):
    """
    Scenario 1: Baseline (Legacy)
    Target: Standard FLOAT_VECTOR (768d)
    Goal: Identify general parsing bottlenecks.
    """
    print("\n[Scenario 1] Baseline: FLOAT_VECTOR (768d) - Legacy Mode")
    print("-" * 60)
    
    nq, topk = 10, 1000
    data = build_search_result(nq, topk, vector_fields=[("vector", "FLOAT_VECTOR", 768)])
    

    
    if use_memory:
        profiling.profile_memory(
            "vector_baseline_legacy",
            run_legacy_iteration,
            data
        )
    else:
        profiling.profile_cpu(
            "vector_baseline_legacy",
            run_legacy_iteration,
            data,
            top_n=20
        )

def profile_large_vector(use_memory: bool = False):
    """
    Scenario 3: Large Vector (Legacy)
    Target: Large FLOAT_VECTOR (3072d)
    Goal: Analyze memory copy and allocation overhead for large objects.
    """
    print("\n[Scenario 3] Large Vector: FLOAT_VECTOR (3072d) - Legacy Mode")
    print("-" * 60)
    
    nq, topk = 10, 1000
    data = build_search_result(nq, topk, vector_fields=[("vector", "FLOAT_VECTOR", 3072)])
    
    if use_memory:
        profiling.profile_memory(
            "vector_large_legacy",
            run_legacy_iteration,
            data
        )
    else:
        profiling.profile_cpu(
            "vector_large_legacy",
            run_legacy_iteration,
            data,
            top_n=20
        )

def profile_sparse_vector(use_memory: bool = False):
    """
    Scenario 4: Sparse Vector (Legacy)
    Target: SPARSE_FLOAT_VECTOR
    Goal: Analyze parsing logic for complex variable-length data.
    """
    print("\n[Scenario 4] Sparse Vector - Legacy Mode")
    print("-" * 60)
    
    nq, topk = 10, 1000
    data = build_search_result(nq, topk, vector_fields=[("vector", "SPARSE_FLOAT_VECTOR", 0)])
    
    if use_memory:
        profiling.profile_memory(
            "vector_sparse_legacy",
            run_legacy_iteration,
            data
        )
    else:
        profiling.profile_cpu(
            "vector_sparse_legacy",
            run_legacy_iteration,
            data,
            top_n=20
        )

def verify_optimization():
    """
    Scenario 5: Verification (Legacy vs Columnar)
    Target: Baseline FLOAT_VECTOR (768d)
    Goal: Confirm ColumnarSearchResult optimization effect.
    """
    print("\n[Scenario 5] Verification: Legacy vs Columnar")
    print("-" * 60)
    
    nq, topk = 10, 1000
    data = build_search_result(nq, topk, vector_fields=[("vector", "FLOAT_VECTOR", 768)])
    
    profiling.compare_implementations(
        "Vector Baseline Comparison",
        {
            "Legacy": run_legacy_iteration,
            "Columnar": run_columnar_iteration
        },
        data,
        warmup_runs=5,
        timed_runs=10
    )

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Vector Performance Profiling")
    parser.add_argument("--memory", action="store_true", help="Run memory profiling instead of CPU")
    args = parser.parse_args()

    print("============================================================")
    print(f"Vector Performance Profiling Suite (Mode: {'Memory' if args.memory else 'CPU'})")
    print("============================================================")
    
    # 1. Bottleneck Analysis (Legacy Only)
    profile_baseline(args.memory)
    profile_large_vector(args.memory)
    profile_sparse_vector(args.memory)
    
    # 2. Optimization Verification (Skip for memory mode, it's just timing)
    if not args.memory:
        verify_optimization()
    
    print("\nDone! Profile stats saved to .benchmarks/")
