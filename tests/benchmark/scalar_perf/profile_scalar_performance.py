#!/usr/bin/env python3
"""
Scalar Field CPU and Memory Profiling
"""

import cProfile
import pstats
import json
from pymilvus.client.columnar_search_result import ColumnarSearchResult
from pymilvus.client.search_result import SearchResult
from pymilvus.grpc_gen import schema_pb2

def build_scalar_result(nq: int, topk: int, scalar_type: str):
    """Build mock SearchResultData with scalar fields."""
    total = nq * topk
    res = schema_pb2.SearchResultData()
    res.num_queries = nq
    res.top_k = topk
    res.topks.extend([topk] * nq)
    res.ids.int_id.data.extend(list(range(total)))
    res.scores.extend([float(i) * 0.001 for i in range(total)])
    res.primary_field_name = "id"
    
    if scalar_type == "INT64":
        field = res.fields_data.add()
        field.field_name = "int64_field"
        field.type = schema_pb2.DataType.Int64
        field.scalars.long_data.data.extend([i * 100 for i in range(total)])
        res.output_fields.append("int64_field")
    elif scalar_type == "JSON":
        field = res.fields_data.add()
        field.field_name = "json_field"
        field.type = schema_pb2.DataType.JSON
        for i in range(total):
            json_obj = {
                "user": {"id": i, "name": f"user_{i}"},
                "metadata": {"tags": [f"tag_{j}" for j in range(5)]},
            }
            field.scalars.json_data.data.append(json.dumps(json_obj).encode())
        res.output_fields.append("json_field")
    
    return res

def iterate_result(results, field_name):
    """Access all scalar fields."""
    count = 0
    for hits in results:
        for hit in hits:
            _ = hit[field_name]
            count += 1
    return count

# Test scenario: 10k items
NQ, TOPK = 10, 1000

print("=== Scalar Field Profiling ===\n")

# Profile Legacy with JSON (worst case)
print("1. Profiling Legacy SearchResult (JSON)...")
res_data = build_scalar_result(NQ, TOPK, "JSON")

profiler = cProfile.Profile()
profiler.enable()
sr = SearchResult(res_data)
count = iterate_result(sr, "json_field")
profiler.disable()

stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.dump_stats('/Users/zilliz/pymilvus/.benchmarks/profile_scalar_legacy.stats')
print(f"  Processed {count} items")
print(f"  Saved to profile_scalar_legacy.stats\n")

# Profile Columnar with JSON
print("2. Profiling Columnar SearchResult (JSON)...")
res_data = build_scalar_result(NQ, TOPK, "JSON")

profiler = cProfile.Profile()
profiler.enable()
cr = ColumnarSearchResult(res_data)
count = iterate_result(cr, "json_field")
profiler.disable()

stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.dump_stats('/Users/zilliz/pymilvus/.benchmarks/profile_scalar_columnar.stats')
print(f"  Processed {count} items")
print(f"  Saved to profile_scalar_columnar.stats\n")

# Profile Legacy with INT64 (simplest case)
print("3. Profiling Legacy SearchResult (INT64)...")
res_data = build_scalar_result(NQ, TOPK, "INT64")

profiler = cProfile.Profile()
profiler.enable()
sr = SearchResult(res_data)
count = iterate_result(sr, "int64_field")
profiler.disable()

stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.dump_stats('/Users/zilliz/pymilvus/.benchmarks/profile_scalar_legacy_int.stats')
print(f"  Processed {count} items")
print(f"  Saved to profile_scalar_legacy_int.stats\n")

print("✅ CPU Profiling Complete!")
print("\nView results with:")
print("  python -m pstats /Users/zilliz/pymilvus/.benchmarks/profile_scalar_legacy.stats")
