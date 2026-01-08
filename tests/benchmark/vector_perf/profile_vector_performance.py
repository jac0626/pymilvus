
import cProfile
import pstats
import time
import argparse
import sys
from pymilvus.client.columnar_search_result import ColumnarSearchResult
from pymilvus.client.search_result import SearchResult
from pymilvus.grpc_gen import schema_pb2

# =============================================================================
# Mock Data Builder
# =============================================================================

def build_float_vector_result(nq: int, topk: int, dim: int) -> schema_pb2.SearchResultData:
    """Create mock SearchResultData for profiling."""
    total = nq * topk
    res = schema_pb2.SearchResultData()
    res.num_queries = nq
    res.top_k = topk
    res.topks.extend([topk] * nq)
    res.ids.int_id.data.extend(list(range(total)))
    res.scores.extend([float(i) * 0.001 for i in range(total)])
    res.primary_field_name = "id"
    
    # Add scalar fields to make it realistic
    res.output_fields.append("id")

    # FLOAT_VECTOR field
    field = res.fields_data.add()
    field.field_name = "vector"
    field.type = schema_pb2.DataType.FloatVector
    field.vectors.dim = dim
    # Optimize: Use extend with pre-allocated list
    dummy_data = [0.123] * (total * dim)
    field.vectors.float_vector.data.extend(dummy_data)
    res.output_fields.append("vector")
    return res

# =============================================================================
# Workloads
# =============================================================================

def run_legacy_workload(res_data):
    """Run the Legacy SearchResult iteration workload."""
    sr = SearchResult(res_data)
    count = 0
    # Simulate full access
    for hits in sr:
        for hit in hits:
            _ = hit["vector"]
            count += 1
    return count

def run_columnar_workload(res_data):
    """Run the ColumnarSearchResult iteration workload."""
    cr = ColumnarSearchResult(res_data)
    count = 0
    # Simulate full access
    for hits in cr:
        for hit in hits:
            _ = hit["vector"]
            count += 1
    return count

# =============================================================================
# Profiling Logic
# =============================================================================

def profile_target(name, func, data):
    print(f"\n--- Profiling {name.upper()} ---")
    profiler = cProfile.Profile()
    profiler.enable()
    
    start = time.perf_counter()
    func(data)
    dt = time.perf_counter() - start
    
    profiler.disable()
    print(f"Total Execution Time: {dt:.4f}s")
    
    # Save statistics
    filename = f"profile_{name}.stats"
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumtime')
    stats.dump_stats(filename)
    print(f"Stats saved to {filename}")
    
    # Print summary
    stats.print_stats(15)

def main():
    parser = argparse.ArgumentParser(description="Profile Vector Search Result Performance")
    parser.add_argument("--nq", type=int, default=10, help="Number of queries")
    parser.add_argument("--topk", type=int, default=1000, help="Top K results per query")
    parser.add_argument("--dim", type=int, default=768, help="Vector dimension")
    parser.add_argument("--mode", choices=["all", "legacy", "columnar"], default="all", help="Profiling mode")
    args = parser.parse_args()

    print(f"Generating Mock Data (NQ={args.nq}, TopK={args.topk}, Dim={args.dim})...")
    data = build_float_vector_result(args.nq, args.topk, args.dim)
    
    if args.mode in ["all", "legacy"]:
        profile_target("legacy", run_legacy_workload, data)
    
    if args.mode in ["all", "columnar"]:
        profile_target("columnar", run_columnar_workload, data)

if __name__ == "__main__":
    main()
