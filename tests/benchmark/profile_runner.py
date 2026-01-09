#!/usr/bin/env python3
"""
Profile Runner - Standalone entry point for clean profiling.

This script runs benchmark kernels WITHOUT pytest overhead, making it suitable
for external profilers like py-spy (CPU) and memray (Memory).

Usage:
    # CPU Profiling with py-spy
    py-spy record -o profile.svg -- python tests/benchmark/profile_runner.py \\
        --scenario search_iteration --nq 1000 --topk 1000 --dtype float_vector

    # Memory Profiling with memray
    memray run -o mem.bin tests/benchmark/profile_runner.py \\
        --scenario search_iteration --nq 1000 --topk 1000

    # Direct Run (for timing)
    python tests/benchmark/profile_runner.py --scenario search_iteration --nq 100 --topk 1000 --loops 10
"""

import argparse
import time
import sys

# Ensure the package is importable
sys.path.insert(0, ".")

from pymilvus.client.types import DataType
from pymilvus.client.search_result import SearchResult
from pymilvus.client.columnar_search_result import ColumnarSearchResult

from tests.benchmark.kernels.data_gen import (
    create_search_result_data,
    get_vector_field,
    get_varchar_field,
    get_json_field,
    get_array_field,
    SCALAR_FIELDS_CORE,
)
from tests.benchmark.kernels.search_ops import (
    run_full_iteration_benchmark,
    run_random_access_benchmark,
    run_slice_access_benchmark,
    run_columnar_access_benchmark,
)
from tests.benchmark.kernels.insert_ops import (
    generate_insert_data,
    get_kitchen_sink_fields,
)


# =============================================================================
# Scenario Runners
# =============================================================================

def run_search_scenario(args):
    """Run a search scenario with specified parameters."""
    # Parse dtype
    dtype_map = {
        "float_vector": DataType.FLOAT_VECTOR,
        "binary_vector": DataType.BINARY_VECTOR,
        "float16_vector": DataType.FLOAT16_VECTOR,
        "int64": DataType.INT64,
        "varchar": DataType.VARCHAR,
        "json": DataType.JSON,
    }
    dtype = dtype_map.get(args.dtype, DataType.FLOAT_VECTOR)
    
    # Create field config
    if dtype == DataType.FLOAT_VECTOR:
        field = get_vector_field(dtype, args.dim)
    elif dtype == DataType.VARCHAR:
        field = get_varchar_field(args.length)
    elif dtype == DataType.JSON:
        field = get_json_field(args.complexity)
    else:
        field = {"name": f"field_{dtype.name.lower()}", "dtype": dtype}
    
    # Generate data
    print(f"Generating data: nq={args.nq}, topk={args.topk}, dtype={dtype.name}")
    data = create_search_result_data(args.nq, args.topk, [field])
    
    # Create result object
    if args.result_type == "legacy":
        result = SearchResult(data)
    else:
        result = ColumnarSearchResult(data)
    
    field_name = field["name"]
    print(f"Result type: {type(result).__name__}, field: {field_name}")
    
    # Select scenario
    scenario_map = {
        "search_iteration": lambda: run_full_iteration_benchmark(result, field_name),
        "search_random": lambda: run_random_access_benchmark(result, field_name, 1000),
        "search_slice": lambda: run_slice_access_benchmark(result, field_name, 100),
        "search_columnar": lambda: run_columnar_access_benchmark(result, field_name),
    }
    
    if args.scenario not in scenario_map:
        print(f"Unknown scenario: {args.scenario}")
        print(f"Available: {list(scenario_map.keys())}")
        return
    
    run_fn = scenario_map[args.scenario]
    
    # Run benchmark
    print(f"Running scenario: {args.scenario}, loops: {args.loops}")
    times = []
    for i in range(args.loops):
        start = time.perf_counter()
        count = run_fn()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  Loop {i+1}/{args.loops}: {elapsed*1000:.2f}ms (items: {count})")
    
    # Summary
    avg_time = sum(times) / len(times)
    print(f"\nSummary:")
    print(f"  Avg: {avg_time*1000:.2f}ms")
    print(f"  Min: {min(times)*1000:.2f}ms")
    print(f"  Max: {max(times)*1000:.2f}ms")


def run_insert_scenario(args):
    """Run an insert data generation scenario."""
    fields = get_kitchen_sink_fields(args.dim)
    
    print(f"Generating insert data: batch_size={args.batch_size}, fields={len(fields)}")
    
    times = []
    for i in range(args.loops):
        start = time.perf_counter()
        data = generate_insert_data(args.batch_size, fields)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  Loop {i+1}/{args.loops}: {elapsed*1000:.2f}ms (rows: {len(data)})")
    
    avg_time = sum(times) / len(times)
    print(f"\nSummary:")
    print(f"  Avg: {avg_time*1000:.2f}ms")
    print(f"  Min: {min(times)*1000:.2f}ms")
    print(f"  Max: {max(times)*1000:.2f}ms")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Profile Runner for Benchmark Kernels")
    
    # Common arguments
    parser.add_argument("--scenario", type=str, required=True,
                        help="Scenario to run: search_iteration, search_random, search_slice, search_columnar, insert")
    parser.add_argument("--loops", type=int, default=5,
                        help="Number of loops to run (for stable timing)")
    
    # Search arguments
    parser.add_argument("--nq", type=int, default=10, help="Number of queries")
    parser.add_argument("--topk", type=int, default=1000, help="TopK per query")
    parser.add_argument("--dtype", type=str, default="float_vector",
                        help="Data type: float_vector, binary_vector, int64, varchar, json")
    parser.add_argument("--dim", type=int, default=128, help="Vector dimension")
    parser.add_argument("--length", type=int, default=256, help="VARCHAR length")
    parser.add_argument("--complexity", type=str, default="simple", help="JSON complexity")
    parser.add_argument("--result-type", type=str, default="columnar",
                        choices=["legacy", "columnar"], help="Result type to use")
    
    # Insert arguments
    parser.add_argument("--batch-size", type=int, default=1000, help="Insert batch size")
    
    args = parser.parse_args()
    
    if args.scenario.startswith("search"):
        run_search_scenario(args)
    elif args.scenario == "insert":
        run_insert_scenario(args)
    else:
        print(f"Unknown scenario: {args.scenario}")
        sys.exit(1)


if __name__ == "__main__":
    main()
