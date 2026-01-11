#!/usr/bin/env python3
"""
Vector Performance CPU Profiling

Uses shared kernels for data generation and profiling.
"""

import argparse
import sys
from pymilvus.client.columnar_search_result import ColumnarSearchResult
from pymilvus.client.search_result import SearchResult

sys.path.insert(0, str(__file__).rsplit('/', 3)[0])
from tests.benchmark.kernels import build_search_result, iterate_result, profile_cpu, get_output_dir


def build_vector_result(nq: int, topk: int, dim: int):
    """Build result with FLOAT_VECTOR field."""
    return build_search_result(
        nq=nq,
        topk=topk,
        vector_fields=[("vector", "FLOAT_VECTOR", dim)],
    )


def run_legacy_workload(res_data):
    """Run the Legacy SearchResult iteration workload."""
    sr = SearchResult(res_data)
    return iterate_result(sr, "vector")


def run_columnar_workload(res_data):
    """Run the ColumnarSearchResult iteration workload."""
    cr = ColumnarSearchResult(res_data)
    return iterate_result(cr, "vector")


def main():
    parser = argparse.ArgumentParser(description="Profile Vector Search Result Performance")
    parser.add_argument("--nq", type=int, default=10, help="Number of queries")
    parser.add_argument("--topk", type=int, default=1000, help="Top K results per query")
    parser.add_argument("--dim", type=int, default=768, help="Vector dimension")
    parser.add_argument("--mode", choices=["all", "legacy", "columnar"], default="all")
    args = parser.parse_args()

    print(f"=== Vector Performance Profiling ===")
    print(f"NQ={args.nq}, TopK={args.topk}, Dim={args.dim}")
    print(f"Total hits: {args.nq * args.topk:,}")
    print()
    
    print("Generating Mock Data...")
    data = build_vector_result(args.nq, args.topk, args.dim)
    print("Done.\n")
    
    if args.mode in ["all", "legacy"]:
        count, _, elapsed = profile_cpu("vector_legacy", run_legacy_workload, data)
        print(f"Legacy: {count:,} items in {elapsed*1000:.2f}ms\n")
    
    if args.mode in ["all", "columnar"]:
        count, _, elapsed = profile_cpu("vector_columnar", run_columnar_workload, data)
        print(f"Columnar: {count:,} items in {elapsed*1000:.2f}ms\n")
    
    print(f"Output dir: {get_output_dir()}")


if __name__ == "__main__":
    main()
