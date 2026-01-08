#!/usr/bin/env python3
"""
Scalar Field Memory Profiling using memray
"""

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
    
    # JSON field (worst case for memory)
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

import sys

# Test scenario: 10k items
NQ, TOPK = 10, 1000

if __name__ == "__main__":
    mode = "all"
    if len(sys.argv) > 1:
        mode = sys.argv[1]

    print(f"=== Scalar Field Memory Profiling ({mode}) ===\n")
    print("Testing with 10,000 JSON entries...")

    # Build data once
    res_data = build_scalar_result(NQ, TOPK, "JSON")

    if mode in ["all", "legacy"]:
        print("\n1. Legacy SearchResult...")
        sr = SearchResult(res_data)
        count = iterate_result(sr, "json_field")
        print(f"   Processed {count} items")

    if mode in ["all", "columnar"]:
        print("\n2. Columnar SearchResult...")
        cr = ColumnarSearchResult(res_data)
        count = iterate_result(cr, "json_field")
        print(f"   Processed {count} items")

    print("\n✅ Memory profiling complete!")
