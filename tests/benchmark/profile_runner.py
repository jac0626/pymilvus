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
    memray run -o mem.bin -- python tests/benchmark/profile_runner.py \\
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

from tests.benchmark.kernels.data_gen import (
    create_search_result_data,
    get_vector_field,
    get_varchar_field,
    get_json_field,
)
from tests.benchmark.kernels.search_ops import (
    benchmark_iteration_legacy,
    benchmark_iteration_columnar,
    benchmark_random_legacy,
    benchmark_random_columnar,
    benchmark_slice_legacy,
    benchmark_slice_columnar,
    benchmark_columnar_batch,
)
from tests.benchmark.kernels.insert_ops import (
    generate_insert_data,
    get_kitchen_sink_fields,
    benchmark_insert_prepare,
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
    
    # Generate data (once - simulates receiving protobuf from server)
    print(f"Generating data: nq={args.nq}, topk={args.topk}, dtype={dtype.name}")
    data = create_search_result_data(args.nq, args.topk, [field])
    
    field_name = field["name"]
    result_type_name = 'SearchResult' if args.result_type == 'legacy' else 'ColumnarSearchResult'
    print(f"Result type: {result_type_name}, field: {field_name}")
    
    # Map scenarios to cold-start benchmark functions
    if args.result_type == "legacy":
        scenario_map = {
            "search_iteration": lambda: benchmark_iteration_legacy(data, field_name),
            "search_random": lambda: benchmark_random_legacy(data, field_name, 1000),
            "search_slice": lambda: benchmark_slice_legacy(data, field_name, 100),
        }
    else:
        scenario_map = {
            "search_iteration": lambda: benchmark_iteration_columnar(data, field_name),
            "search_random": lambda: benchmark_random_columnar(data, field_name, 1000),
            "search_slice": lambda: benchmark_slice_columnar(data, field_name, 100),
            "search_columnar": lambda: benchmark_columnar_batch(data, field_name),
        }
    
    if args.scenario not in scenario_map:
        print(f"Unknown scenario: {args.scenario}")
        available = list(scenario_map.keys())
        print(f"Available for {result_type_name}: {available}")
        return
    
    run_fn = scenario_map[args.scenario]
    
    # Run benchmark (each call includes object creation = cold start)
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
    """Run an insert data preparation scenario."""
    fields = get_kitchen_sink_fields(args.dim)
    
    print(f"Generating insert data: batch_size={args.batch_size}, fields={len(fields)}")
    data = generate_insert_data(args.batch_size, fields)
    
    print(f"Running insert preparation benchmark (list -> protobuf)")
    times = []
    for i in range(args.loops):
        start = time.perf_counter()
        count = benchmark_insert_prepare(data, fields)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  Loop {i+1}/{args.loops}: {elapsed*1000:.2f}ms (rows: {count})")
    
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
                        help="Scenario: search_iteration, search_random, search_slice, search_columnar, insert")
    parser.add_argument("--loops", type=int, default=5,
                        help="Number of loops to run")
    
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
