
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import re
import os

# Set style for "beautiful charts"
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 12

ARTIFACT_DIR = "/Users/zilliz/.gemini/antigravity/brain/970059a5-6e08-4189-9c7d-289e402c1e9f"
JSON_FILE = "benchmark_results.json"

def load_data():
    with open(JSON_FILE, 'r') as f:
        data = json.load(f)
    
    records = []
    for bench in data['benchmarks']:
        name = bench['name']
        mean_time = bench['stats']['mean'] * 1000  # Convert to ms
        
        # Parse params from name: test_float_vector_matrix[128-10-100]
        # or test_other_vectors[BINARY_VECTOR-...]
        
        # Case 1: Matrix
        match = re.search(r"test_float_vector_matrix\[(\d+)-(\d+)-(\d+)\]", name)
        if match:
            dim, nq, topk = map(int, match.groups())
            records.append({
                "Type": "FLOAT_VECTOR",
                "Dim": dim,
                "NQ": nq,
                "TopK": topk,
                "Time (ms)": mean_time,
                "Load": f"{nq}x{topk}"
            })
            continue

        # Case 2: Other types
        match = re.search(r"test_other_vectors\[(\w+)-.*-(\d+)\]", name)
        if match:
            vtype, dim = match.groups()
            records.append({
                "Type": vtype,
                "Dim": int(dim),
                "NQ": 10,
                "TopK": 100,
                "Time (ms)": mean_time,
                "Load": "10x100"
            })
            continue
            
        # Case 3: Legacy
        if "legacy" in name:
             records.append({
                "Type": "Legacy (Float)",
                "Dim": 768,
                "NQ": 10,
                "TopK": 100,
                "Time (ms)": mean_time,
                "Load": "10x100"
            })

    return pd.DataFrame(records)

def plot_dimension_impact(df):
    """Plot impact of dimension on performance (fixed load)."""
    # Filter for Float Vector, NQ=10, TopK=100 (Middle ground)
    subset = df[(df["Type"] == "FLOAT_VECTOR") & (df["NQ"] == 10) & (df["TopK"] == 100)].copy()
    
    if subset.empty:
        return

    plt.figure(figsize=(8, 6))
    ax = sns.barplot(x="Dim", y="Time (ms)", data=subset, palette="viridis")
    
    plt.title("Impact of Vector Dimension on Performance\n(NQ=10, TopK=100)", fontsize=14, fontweight='bold')
    plt.xlabel("Vector Dimension", fontsize=12)
    plt.ylabel("Execution Time (ms)", fontsize=12)
    plt.ylim(0, max(subset["Time (ms)"]) * 1.2)
    
    for i in ax.containers:
        ax.bar_label(i, fmt='%.2f ms', padding=3)
        
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, "perf_dimension_impact.png"))
    print("Generated perf_dimension_impact.png")

def plot_load_heatmap(df):
    """Heatmap of execution time for NQ vs TopK (Dim=768)."""
    subset = df[(df["Type"] == "FLOAT_VECTOR") & (df["Dim"] == 768)].copy()
    
    if subset.empty:
        return

    pivot = subset.pivot(index="NQ", columns="TopK", values="Time (ms)")
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="magma_r", cbar_kws={'label': 'Time (ms)'}, annot_kws={"size": 12})
    
    plt.title("Performance Heatmap: NQ vs TopK\n(Dim=768, Float Vector)", fontsize=14, fontweight='bold')
    plt.xlabel("TopK", fontsize=12)
    plt.ylabel("Number of Queries (NQ)", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, "perf_load_heatmap.png"))
    print("Generated perf_load_heatmap.png")

def plot_type_comparison(df):
    """Compare different vector types at fixed load."""
    # NQ=10, TopK=100. Dim varies but roughly comparable in bytes? No, binary is bits.
    # Just show execution time for the single point we measured.
    
    subset = df[(df["NQ"] == 10) & (df["TopK"] == 100)].copy()
    # Filter for key types
    types = ["FLOAT_VECTOR", "BINARY_VECTOR", "FLOAT16_VECTOR", "Legacy (Float)"]
    subset = subset[subset["Type"].isin(types)]
    
    if subset.empty:
        return

    plt.figure(figsize=(10, 6))
    # Add dimension to label
    subset["Label"] = subset.apply(lambda r: f"{r['Type']}\n(Dim={r['Dim']})", axis=1)
    
    ax = sns.barplot(x="Label", y="Time (ms)", data=subset, palette="rocket")
    
    plt.title("Performance Comparison by Vector Type\n(NQ=10, TopK=100)", fontsize=14, fontweight='bold')
    plt.xlabel("", fontsize=12)
    plt.ylabel("Execution Time (ms)", fontsize=12)
    plt.xticks(rotation=15)
    
    # Add value labels
    for i in ax.containers:
        ax.bar_label(i, fmt='%.2f ms', padding=3)

    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, "perf_type_comparison.png"))
    print("Generated perf_type_comparison.png")

if __name__ == "__main__":
    df = load_data()
    plot_dimension_impact(df)
    plot_load_heatmap(df)
    plot_type_comparison(df)
