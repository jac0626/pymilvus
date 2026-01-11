#!/usr/bin/env python3
# Copyright (c) PyMilvus Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""
Profiling utilities for benchmarking.

Provides CPU and memory profiling helpers with proper path handling.
"""

import cProfile
import pstats
import time
from pathlib import Path
from typing import Any, Callable, Tuple, Optional


def get_output_dir() -> Path:
    """
    Get the benchmark output directory (project-relative).
    
    Creates the directory if it doesn't exist.
    
    Returns:
        Path to .benchmarks/ directory relative to project root
    """
    # Navigate from this file to project root
    # tests/benchmark/kernels/profiling.py -> tests/benchmark/kernels -> tests/benchmark -> tests -> root
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent.parent
    output_dir = project_root / ".benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def profile_cpu(
    name: str,
    func: Callable,
    *args,
    save_stats: bool = True,
    print_summary: bool = True,
    top_n: int = 15,
    **kwargs
) -> Tuple[Any, pstats.Stats, float]:
    """
    Profile a function's CPU usage.
    
    Args:
        name: Name for the profile (used in output filename)
        func: Function to profile
        *args: Positional arguments to pass to func
        save_stats: Whether to save .stats file
        print_summary: Whether to print top functions
        top_n: Number of top functions to display
        **kwargs: Keyword arguments to pass to func
    
    Returns:
        Tuple of (function result, pstats.Stats, execution time in seconds)
    
    Example:
        >>> result, stats, elapsed = profile_cpu("search_legacy", run_search, data)
    """
    profiler = cProfile.Profile()
    
    profiler.enable()
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    profiler.disable()
    
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    
    if save_stats:
        output_path = get_output_dir() / f"profile_{name}.stats"
        stats.dump_stats(str(output_path))
        print(f"Stats saved to: {output_path}")
    
    if print_summary:
        print(f"\n--- Profile: {name} ---")
        print(f"Total time: {elapsed:.4f}s")
        stats.print_stats(top_n)
    
    return result, stats, elapsed


def run_with_timing(func: Callable, *args, **kwargs) -> Tuple[Any, float]:
    """
    Run a function and measure its execution time.
    
    Args:
        func: Function to run
        *args: Positional arguments
        **kwargs: Keyword arguments
    
    Returns:
        Tuple of (function result, execution time in seconds)
    """
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


def compare_implementations(
    name: str,
    implementations: dict,
    data: Any,
    warmup_runs: int = 1,
    timed_runs: int = 3,
) -> dict:
    """
    Compare multiple implementations with timing.
    
    Args:
        name: Name of the comparison (for output)
        implementations: Dict of {impl_name: callable}
        data: Input data to pass to each implementation
        warmup_runs: Number of warmup runs before timing
        timed_runs: Number of timed runs (results are averaged)
    
    Returns:
        Dict of {impl_name: avg_time_ms}
    
    Example:
        >>> results = compare_implementations(
        ...     "search",
        ...     {"legacy": SearchResult, "columnar": ColumnarSearchResult},
        ...     mock_data
        ... )
    """
    print(f"\n=== Comparison: {name} ===")
    results = {}
    
    for impl_name, impl_func in implementations.items():
        # Warmup
        for _ in range(warmup_runs):
            impl_func(data)
        
        # Timed runs
        times = []
        for _ in range(timed_runs):
            _, elapsed = run_with_timing(impl_func, data)
            times.append(elapsed)
        
        avg_time = sum(times) / len(times)
        results[impl_name] = avg_time * 1000  # Convert to ms
        print(f"  {impl_name}: {avg_time * 1000:.2f}ms (avg of {timed_runs} runs)")
    
    # Compute speedup
    if len(results) >= 2:
        values = list(results.values())
        baseline = max(values)
        fastest = min(values)
        print(f"  Speedup: {baseline / fastest:.2f}x")
    
    return results
