#!/usr/bin/env python3
"""
ColumnarSearchResult Memory Analysis

This script compares memory usage between SearchResult (Legacy) and
ColumnarSearchResult (Columnar) modes using memray.
"""

import sys
import contextlib
import numpy as np
import random
from pymilvus.client.columnar_search_result import ColumnarSearchResult
from pymilvus.client.search_result import SearchResult
from pymilvus.grpc_gen import schema_pb2, milvus_pb2

# --- Mock Data Generation ---

def generate_search_result_proto(nq=10, topk=1000, dim=768):
    """Generates a raw SearchResultData protobuf with FloatVectors."""
    total_hits = nq * topk
    
    # 1. IDs
    ids = schema_pb2.IDs(int_id=schema_pb2.LongArray(data=[i for i in range(total_hits)]))
    
    # 2. Scores
    scores = [random.random() for _ in range(total_hits)]
    
    # 3. Fields (FloatVector + Int64 + JSON)
    fields_data = []
    
    # Float Vector (Large payload)
    # 1536 float32s * 10000 hits = ~60MB data
    vectors = [random.random() for _ in range(total_hits * dim)]
    f_vec = schema_pb2.FieldData(
        type=schema_pb2.FloatVector,
        field_name="embedding",
        vectors=schema_pb2.VectorField(
            dim=dim,
            float_vector=schema_pb2.FloatArray(data=vectors)
        )
    )
    fields_data.append(f_vec)
    
    # Int64
    f_int = schema_pb2.FieldData(
        type=schema_pb2.Int64,
        field_name="id",
        scalars=schema_pb2.ScalarField(
            long_data=schema_pb2.LongArray(data=[i for i in range(total_hits)])
        )
    )
    fields_data.append(f_int)
    
    result = schema_pb2.SearchResultData(
        num_queries=nq,
        top_k=topk,
        ids=ids,
        scores=scores,
        fields_data=fields_data,
        topks=[topk] * nq
    )
    
    return result

# --- Profiling Workloads ---

def profile_legacy(proto):
    """Profile Legacy SearchResult initialization (Full Materialization)."""
    # Simply creating SearchResult triggers full parsing in legacy mode
    res = SearchResult(proto)
    return res

def profile_columnar_init(proto):
    """Profile ColumnarSearchResult initialization (Lazy)."""
    res = ColumnarSearchResult(proto)
    return res

def profile_columnar_access(proto):
    """Profile ColumnarSearchResult with partial access."""
    res = ColumnarSearchResult(proto)
    # Access first row to trigger minimal parsing
    _ = res[0] 
    return res

def profile_columnar_iter(proto):
    """Profile ColumnarSearchResult with full iteration."""
    res = ColumnarSearchResult(proto)
    # Iterate over all hits
    for hit in res:
        pass
    return res

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python profile_columnar_memory.py [legacy|columnar_init|columnar_access|columnar_iter]")
        sys.exit(1)
        
    mode = sys.argv[1]
    
    print(f"Generating Protobuf Data (NQ=10, TopK=1000, Dim=768)...")
    # This part should ideally NOT be profiled for heap, but memray tracks all.
    # We will look at peaks.
    proto = generate_search_result_proto(nq=10, topk=1000, dim=768)
    
    print(f"Running mode: {mode}")
    
    if mode == "legacy":
        profile_legacy(proto)
    elif mode == "columnar_init":
        profile_columnar_init(proto)
    elif mode == "columnar_access":
        profile_columnar_access(proto)
    elif mode == "columnar_iter":
        profile_columnar_iter(proto)
        
    print("Done.")
