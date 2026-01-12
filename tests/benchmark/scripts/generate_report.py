
import os
import pstats
import subprocess
import sys
import json
import glob
from pathlib import Path
from collections import defaultdict

# =============================================================================
# Helper: Format Time
# =============================================================================
def format_time(seconds):
    if seconds < 0.001:
        return f"{seconds*1000*1000:.2f} us"
    elif seconds < 1.0:
        return f"{seconds*1000:.2f} ms"
    else:
        return f"{seconds:.2f} s"

# =============================================================================
# 1. Benchmark Speedup Analysis
# =============================================================================
def analyze_speedups(json_file_path, output_path):
    """Parse pytest-benchmark JSON and write speedup table to file."""
    if not os.path.exists(json_file_path):
        return

    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    comparisons = defaultdict(dict)
    for b in data['benchmarks']:
        name = b['name']
        if "Legacy" in name:
            key = name.replace("Legacy", "{}")
            comparisons[key]['Legacy'] = b['stats']['mean']
        elif "Columnar" in name:
            key = name.replace("Columnar", "{}")
            comparisons[key]['Columnar'] = b['stats']['mean']
            
    with open(output_path, 'w') as f:
        f.write(f"Benchmark Speedup Report: {os.path.basename(json_file_path)}\n")
        f.write("="*80 + "\n")
        f.write(f"{'Test Case':<60} | {'Legacy':<10} | {'Columnar':<10} | {'Speedup':<8}\n")
        f.write("-" * 100 + "\n")
        
        results = []
        for key, stats in comparisons.items():
            if 'Legacy' in stats and 'Columnar' in stats:
                leg = stats['Legacy']
                col = stats['Columnar']
                speedup = leg / col
                display_key = key.split("[")[1].replace("]", "").replace("{}", "")
                results.append((speedup, display_key, leg, col))
        
        results.sort(key=lambda x: x[1])
        
        for speedup, name, leg, col in results:
            f.write(f"{name:<60} | {format_time(leg):<10} | {format_time(col):<10} | {speedup:.2f}x\n")
            
    print(f"Generated Summary: {output_path}")

# =============================================================================
# 2. Convert CPU Stats
# =============================================================================
def convert_stats_to_txt(stats_file, output_file):
    with open(output_file, 'w') as f:
        f.write(f"CPU Profile Report: {os.path.basename(stats_file)}\n")
        f.write("="*80 + "\n\n")
        try:
            p = pstats.Stats(str(stats_file), stream=f)
            p.strip_dirs().sort_stats("cumulative").print_stats(50)
            print(f"Generated CPU Report: {output_file}")
        except Exception as e:
            f.write(f"Error parsing stats: {e}")

# =============================================================================
# 3. Convert Memray Stats
# =============================================================================
def convert_memray_to_txt(bin_file, output_file):
    # memray stats writes to stdout
    result = subprocess.run(["memray", "stats", str(bin_file)], capture_output=True, text=True)
    with open(output_file, 'w') as f:
        f.write(f"Memory Profile Report: {os.path.basename(bin_file)}\n")
        f.write("="*80 + "\n\n")
        f.write(result.stdout)
        if result.stderr:
            f.write("\n--- STDERR ---\n")
            f.write(result.stderr)
    print(f"Generated Memory Report: {output_file}")

# =============================================================================
# Main
# =============================================================================
def main():
    root_dir = Path(".benchmarks")
    report_dir = root_dir / "reports"
    
    # Define Sub-directories
    dir_cpu = report_dir / "cpu"
    dir_mem = report_dir / "memory"
    dir_sum = report_dir / "summary"
    
    for d in [dir_cpu, dir_mem, dir_sum]:
        d.mkdir(parents=True, exist_ok=True)
        
    print(f"Generating reports in {report_dir}...")
    
    # 1. Process CPU Profiles (*.stats)
    # Search recursively in data dir or root
    for stats_file in root_dir.rglob("*.stats"):
        convert_stats_to_txt(stats_file, dir_cpu / (stats_file.stem + ".txt"))

    # 2. Process Memory Profiles (*.bin)
    for bin_file in root_dir.rglob("*.bin"):
        convert_memray_to_txt(bin_file, dir_mem / (bin_file.stem + ".txt"))
        
    # 3. Process Benchmark JSONs (*.json)
    # They might be in root or subdirs, let's find them
    # NOTE: Benchmark JSONs are now output to .benchmarks root
    results_dir = root_dir
    if results_dir.exists():
        for json_file in results_dir.glob("*.json"):
            analyze_speedups(json_file, dir_sum / (json_file.stem + "_summary.txt"))

if __name__ == "__main__":
    main()
