#!/usr/bin/env python3
"""
Benchmark Results Plotting Script

Generates visualizations from pytest-benchmark JSON output files.

Usage:
    # Generate from benchmark results
    python tests/benchmark/scripts/plot_benchmark_results.py results.json -o plots/

    # Compare multiple runs
    python tests/benchmark/scripts/plot_benchmark_results.py run1.json run2.json --compare
"""

import argparse
import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("Please install matplotlib and numpy: pip install matplotlib numpy")
    exit(1)


# =============================================================================
# Data Loading
# =============================================================================

def load_benchmark_data(filepath: str) -> Dict[str, Any]:
    """Load pytest-benchmark JSON output."""
    with open(filepath, "r") as f:
        return json.load(f)


def parse_test_name(name: str) -> Dict[str, str]:
    """
    Parse test name to extract parameters.
    
    Example: "test_float_vector_iteration_legacy[nq=10-topk=1000]"
    Returns: {"test": "test_float_vector_iteration_legacy", "nq": "10", "topk": "1000"}
    """
    match = re.match(r"([^\[]+)(?:\[(.+)\])?", name)
    if not match:
        return {"test": name}
    
    result = {"test": match.group(1)}
    if match.group(2):
        params = match.group(2).split("-")
        for param in params:
            if "=" in param:
                key, value = param.split("=", 1)
                result[key] = value
            else:
                # Handle positional parameters like "10" -> treat as numeric param
                result[f"param_{len(result)}"] = param
    return result


def extract_benchmarks(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract benchmark entries with parsed metadata."""
    benchmarks = data.get("benchmarks", [])
    results = []
    
    for bench in benchmarks:
        parsed = parse_test_name(bench["name"])
        entry = {
            "name": bench["name"],
            "test": parsed.get("test", bench["name"]),
            "mean": bench["stats"]["mean"] * 1000,  # Convert to ms
            "stddev": bench["stats"]["stddev"] * 1000,
            "min": bench["stats"]["min"] * 1000,
            "max": bench["stats"]["max"] * 1000,
            "rounds": bench["stats"]["rounds"],
            **{k: v for k, v in parsed.items() if k != "test"},
        }
        results.append(entry)
    
    return results


# =============================================================================
# Plot Generators
# =============================================================================

def plot_scaling_chart(
    benchmarks: List[Dict[str, Any]],
    x_param: str,
    group_param: str,
    output_dir: str,
    title: str = "Performance Scaling",
):
    """
    Generate a line chart showing scaling behavior.
    
    X-axis: x_param values (e.g., "nq")
    Lines: Different values of group_param (e.g., "topk")
    """
    # Group data
    groups = defaultdict(list)
    for bench in benchmarks:
        if x_param in bench and group_param in bench:
            key = bench.get(group_param, "default")
            x_val = int(bench.get(x_param, 0))
            groups[key].append((x_val, bench["mean"], bench["stddev"]))
    
    if not groups:
        print(f"No data for scaling chart: {x_param} x {group_param}")
        return
    
    plt.figure(figsize=(10, 6))
    
    for group_key, points in sorted(groups.items(), key=lambda x: str(x[0])):
        points.sort(key=lambda x: x[0])
        x_vals = [p[0] for p in points]
        y_vals = [p[1] for p in points]
        yerr = [p[2] for p in points]
        plt.errorbar(x_vals, y_vals, yerr=yerr, marker="o", label=f"{group_param}={group_key}", capsize=3)
    
    plt.xlabel(x_param)
    plt.ylabel("Time (ms)")
    plt.title(title)
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = f"scaling_{x_param}_by_{group_param}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=150)
    plt.close()
    print(f"Saved: {filename}")


def plot_comparison_bar_chart(
    benchmarks: List[Dict[str, Any]],
    group_by: str,
    output_dir: str,
    title: str = "Legacy vs Columnar Comparison",
):
    """
    Generate a grouped bar chart comparing Legacy vs Columnar.
    """
    # Separate legacy and columnar results
    legacy_data = {}
    columnar_data = {}
    
    for bench in benchmarks:
        test_name = bench["test"]
        key = bench.get(group_by, "default")
        
        if "legacy" in test_name.lower():
            legacy_data[key] = bench["mean"]
        elif "columnar" in test_name.lower():
            columnar_data[key] = bench["mean"]
    
    # Find common keys
    common_keys = sorted(set(legacy_data.keys()) & set(columnar_data.keys()))
    
    if not common_keys:
        print("No comparable Legacy vs Columnar data found")
        return
    
    x = np.arange(len(common_keys))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    legacy_vals = [legacy_data.get(k, 0) for k in common_keys]
    columnar_vals = [columnar_data.get(k, 0) for k in common_keys]
    
    bars1 = ax.bar(x - width/2, legacy_vals, width, label="Legacy (SearchResult)")
    bars2 = ax.bar(x + width/2, columnar_vals, width, label="Columnar (ColumnarSearchResult)")
    
    ax.set_xlabel(group_by)
    ax.set_ylabel("Time (ms)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(common_keys, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    
    # Add speedup annotations
    for i, (l, c) in enumerate(zip(legacy_vals, columnar_vals)):
        if c > 0:
            speedup = l / c
            ax.annotate(f"{speedup:.1f}x", xy=(x[i], max(l, c)), ha="center", fontsize=8)
    
    plt.tight_layout()
    
    filename = f"comparison_by_{group_by}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=150)
    plt.close()
    print(f"Saved: {filename}")


def plot_access_mode_comparison(
    benchmarks: List[Dict[str, Any]],
    output_dir: str,
):
    """
    Generate a bar chart comparing different access modes.
    """
    modes = ["random", "iteration", "slice", "columnar"]
    mode_data = {mode: [] for mode in modes}
    
    for bench in benchmarks:
        test_name = bench["test"].lower()
        for mode in modes:
            if mode in test_name:
                mode_data[mode].append(bench["mean"])
                break
    
    # Calculate averages
    mode_avgs = {mode: np.mean(vals) if vals else 0 for mode, vals in mode_data.items()}
    
    if not any(mode_avgs.values()):
        print("No access mode data found")
        return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    x = np.arange(len(modes))
    vals = [mode_avgs[m] for m in modes]
    colors = ["#ff7f0e", "#2ca02c", "#d62728", "#1f77b4"]
    
    bars = ax.bar(x, vals, color=colors)
    
    ax.set_xlabel("Access Mode")
    ax.set_ylabel("Average Time (ms)")
    ax.set_title("Access Mode Performance Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in modes])
    ax.grid(True, axis="y", alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, vals):
        if val > 0:
            ax.annotate(f"{val:.2f}ms", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       ha="center", va="bottom", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "access_mode_comparison.png"), dpi=150)
    plt.close()
    print("Saved: access_mode_comparison.png")


def plot_type_comparison(
    benchmarks: List[Dict[str, Any]],
    output_dir: str,
):
    """
    Generate a bar chart comparing different data types.
    """
    type_data = defaultdict(list)
    
    for bench in benchmarks:
        # Try to extract type from test name or parameters
        test_name = bench["test"].lower()
        dtype = bench.get("dtype", bench.get("vtype", None))
        
        if dtype:
            type_data[str(dtype)].append(bench["mean"])
        else:
            # Try to infer from name
            if "float_vector" in test_name:
                type_data["FLOAT_VECTOR"].append(bench["mean"])
            elif "binary_vector" in test_name:
                type_data["BINARY_VECTOR"].append(bench["mean"])
            elif "int64" in test_name:
                type_data["INT64"].append(bench["mean"])
            elif "varchar" in test_name:
                type_data["VARCHAR"].append(bench["mean"])
            elif "json" in test_name:
                type_data["JSON"].append(bench["mean"])
    
    if not type_data:
        print("No type comparison data found")
        return
    
    # Calculate averages
    type_avgs = {t: np.mean(vals) for t, vals in type_data.items()}
    types = sorted(type_avgs.keys())
    vals = [type_avgs[t] for t in types]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    x = np.arange(len(types))
    bars = ax.bar(x, vals)
    
    ax.set_xlabel("Data Type")
    ax.set_ylabel("Average Time (ms)")
    ax.set_title("Performance by Data Type")
    ax.set_xticks(x)
    ax.set_xticklabels(types, rotation=45, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "type_comparison.png"), dpi=150)
    plt.close()
    print("Saved: type_comparison.png")


# =============================================================================
# Main
# =============================================================================

def generate_all_plots(benchmarks: List[Dict[str, Any]], output_dir: str):
    """Generate all available plots from benchmark data."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Scaling charts
    plot_scaling_chart(benchmarks, "nq", "topk", output_dir, "Performance vs NQ (by TopK)")
    plot_scaling_chart(benchmarks, "topk", "nq", output_dir, "Performance vs TopK (by NQ)")
    
    # 2. Comparison charts
    plot_comparison_bar_chart(benchmarks, "nq", output_dir, "Legacy vs Columnar by NQ")
    plot_comparison_bar_chart(benchmarks, "dtype", output_dir, "Legacy vs Columnar by Type")
    
    # 3. Access mode comparison
    plot_access_mode_comparison(benchmarks, output_dir)
    
    # 4. Type comparison
    plot_type_comparison(benchmarks, output_dir)
    
    print(f"\nAll plots saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Plot Benchmark Results")
    parser.add_argument("input_files", nargs="+", help="Input JSON file(s) from pytest-benchmark")
    parser.add_argument("-o", "--output", default="./benchmark_plots", help="Output directory for plots")
    parser.add_argument("--compare", action="store_true", help="Compare multiple input files")
    
    args = parser.parse_args()
    
    all_benchmarks = []
    for filepath in args.input_files:
        data = load_benchmark_data(filepath)
        benchmarks = extract_benchmarks(data)
        print(f"Loaded {len(benchmarks)} benchmarks from {filepath}")
        all_benchmarks.extend(benchmarks)
    
    generate_all_plots(all_benchmarks, args.output)


if __name__ == "__main__":
    main()
