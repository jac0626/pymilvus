"""
Comprehensive Performance Benchmark: Phase 1 vs Phase 2 Zero-Copy

This benchmark compares:
1. Original SearchResult (baseline)
2. Columnar Phase 1 only (zero_copy_vectors=False)
3. Columnar Phase 1+2 (zero_copy_vectors=True)

Tests include multiple data types:
- FLOAT_VECTOR (float32, 128-dim)
- BINARY_VECTOR (8-bit packed, 128-dim)  
- FLOAT16_VECTOR (half precision, 128-dim)
- INT8_VECTOR (int8, 128-dim)
- Scalar types: INT64, FLOAT, VARCHAR
"""

import time
import random
import os
import numpy as np
from typing import Tuple, Dict, List

from pymilvus.grpc_gen import schema_pb2
from pymilvus.client.search_result import SearchResult
from pymilvus.client.columnar_search_result import ColumnarSearchResult
from pymilvus.client.types import DataType


def create_test_data(nq: int, topk: int, dim: int = 128) -> schema_pb2.SearchResultData:
    """Create test data with multiple data types."""
    total = nq * topk
    
    # IDs and scores
    ids = schema_pb2.IDs(int_id=schema_pb2.LongArray(data=list(range(total))))
    scores = [random.random() for _ in range(total)]
    
    # Float vector (128-dim float32)
    float_vec_data = [random.random() for _ in range(total * dim)]
    float_vector_field = schema_pb2.FieldData(
        type=DataType.FLOAT_VECTOR,
        field_name="float_vector",
        vectors=schema_pb2.VectorField(
            dim=dim,
            float_vector=schema_pb2.FloatArray(data=float_vec_data)
        )
    )
    
    # Binary vector (128-dim = 16 bytes per vector)
    binary_vec_data = os.urandom(total * (dim // 8))
    binary_vector_field = schema_pb2.FieldData(
        type=DataType.BINARY_VECTOR,
        field_name="binary_vector",
        vectors=schema_pb2.VectorField(
            dim=dim,
            binary_vector=binary_vec_data
        )
    )
    
    # Float16 vector (128-dim = 256 bytes per vector)
    float16_vec_data = os.urandom(total * dim * 2)
    float16_vector_field = schema_pb2.FieldData(
        type=DataType.FLOAT16_VECTOR,
        field_name="float16_vector",
        vectors=schema_pb2.VectorField(
            dim=dim,
            float16_vector=float16_vec_data
        )
    )
    
    # INT8 vector (128-dim = 128 bytes per vector)
    int8_vec_data = os.urandom(total * dim)
    int8_vector_field = schema_pb2.FieldData(
        type=DataType.INT8_VECTOR,
        field_name="int8_vector",
        vectors=schema_pb2.VectorField(
            dim=dim,
            int8_vector=int8_vec_data
        )
    )
    
    # Scalar fields
    int64_field = schema_pb2.FieldData(
        type=DataType.INT64,
        field_name="int64_field",
        scalars=schema_pb2.ScalarField(
            long_data=schema_pb2.LongArray(data=list(range(total)))
        )
    )
    
    float_field = schema_pb2.FieldData(
        type=DataType.FLOAT,
        field_name="float_field",
        scalars=schema_pb2.ScalarField(
            float_data=schema_pb2.FloatArray(data=[random.random() for _ in range(total)])
        )
    )
    
    varchar_field = schema_pb2.FieldData(
        type=DataType.VARCHAR,
        field_name="varchar_field",
        scalars=schema_pb2.ScalarField(
            string_data=schema_pb2.StringArray(data=[f"str_{i}" for i in range(total)])
        )
    )
    
    return schema_pb2.SearchResultData(
        ids=ids,
        scores=scores,
        topks=[topk] * nq,
        num_queries=nq,
        fields_data=[
            float_vector_field,
            binary_vector_field,
            float16_vector_field,
            int8_vector_field,
            int64_field,
            float_field,
            varchar_field,
        ],
        output_fields=[
            "float_vector", "binary_vector", "float16_vector", "int8_vector",
            "int64_field", "float_field", "varchar_field"
        ]
    )


def benchmark_init_time(data: schema_pb2.SearchResultData, iterations: int = 100) -> Dict[str, float]:
    """Benchmark initialization time."""
    results = {}
    
    # Original SearchResult
    start = time.perf_counter()
    for _ in range(iterations):
        _ = SearchResult(data)
    results["Original"] = (time.perf_counter() - start) / iterations * 1000  # ms
    
    # Columnar Phase 1 only
    start = time.perf_counter()
    for _ in range(iterations):
        _ = ColumnarSearchResult(data, zero_copy_vectors=False)
    results["Columnar P1"] = (time.perf_counter() - start) / iterations * 1000
    
    # Columnar Phase 1+2
    start = time.perf_counter()
    for _ in range(iterations):
        _ = ColumnarSearchResult(data, zero_copy_vectors=True)
    results["Columnar P1+P2"] = (time.perf_counter() - start) / iterations * 1000
    
    return results


def benchmark_field_access(data: schema_pb2.SearchResultData, field_name: str, iterations: int = 10) -> Dict[str, float]:
    """Benchmark single field access across all results."""
    results = {}
    
    # Original
    original = SearchResult(data)
    start = time.perf_counter()
    for _ in range(iterations):
        for hits in original:
            for hit in hits:
                _ = hit.entity.get(field_name)
    results["Original"] = (time.perf_counter() - start) / iterations * 1000
    
    # Columnar Phase 1 only
    columnar_p1 = ColumnarSearchResult(data, zero_copy_vectors=False)
    start = time.perf_counter()
    for _ in range(iterations):
        for hits in columnar_p1:
            for hit in hits:
                _ = hit[field_name]
    results["Columnar P1"] = (time.perf_counter() - start) / iterations * 1000
    
    # Columnar Phase 1+2
    columnar_p2 = ColumnarSearchResult(data, zero_copy_vectors=True)
    start = time.perf_counter()
    for _ in range(iterations):
        for hits in columnar_p2:
            for hit in hits:
                _ = hit[field_name]
    results["Columnar P1+P2"] = (time.perf_counter() - start) / iterations * 1000
    
    return results


def benchmark_all_fields_access(data: schema_pb2.SearchResultData, iterations: int = 10) -> Dict[str, float]:
    """Benchmark accessing all fields."""
    results = {}
    fields = ["float_vector", "binary_vector", "float16_vector", "int8_vector",
              "int64_field", "float_field", "varchar_field"]
    
    # Original
    original = SearchResult(data)
    start = time.perf_counter()
    for _ in range(iterations):
        for hits in original:
            for hit in hits:
                for f in fields:
                    _ = hit.entity.get(f)
    results["Original"] = (time.perf_counter() - start) / iterations * 1000
    
    # Columnar Phase 1 only
    columnar_p1 = ColumnarSearchResult(data, zero_copy_vectors=False)
    start = time.perf_counter()
    for _ in range(iterations):
        for hits in columnar_p1:
            for hit in hits:
                for f in fields:
                    _ = hit[f]
    results["Columnar P1"] = (time.perf_counter() - start) / iterations * 1000
    
    # Columnar Phase 1+2
    columnar_p2 = ColumnarSearchResult(data, zero_copy_vectors=True)
    start = time.perf_counter()
    for _ in range(iterations):
        for hits in columnar_p2:
            for hit in hits:
                for f in fields:
                    _ = hit[f]
    results["Columnar P1+P2"] = (time.perf_counter() - start) / iterations * 1000
    
    return results


def run_experiments():
    """Run 4 groups of experiments."""
    print("=" * 70)
    print("Performance Benchmark: Phase 1 vs Phase 2 Zero-Copy")
    print("=" * 70)
    
    # Test configurations
    configs = [
        {"name": "Small", "nq": 10, "topk": 10, "dim": 128},
        {"name": "Medium", "nq": 100, "topk": 100, "dim": 128},
        {"name": "Large", "nq": 100, "topk": 1000, "dim": 128},
        {"name": "High-Dim", "nq": 50, "topk": 100, "dim": 512},
    ]
    
    for config in configs:
        print(f"\n{'='*70}")
        print(f"Experiment: {config['name']} (nq={config['nq']}, topk={config['topk']}, dim={config['dim']})")
        print(f"Total results: {config['nq'] * config['topk']:,}")
        print("=" * 70)
        
        data = create_test_data(config['nq'], config['topk'], config['dim'])
        
        # 1. Initialization Time
        print("\n[1] Initialization Time (ms):")
        init_results = benchmark_init_time(data, iterations=50)
        for name, time_ms in init_results.items():
            speedup = init_results["Original"] / time_ms if time_ms > 0 else float('inf')
            print(f"  {name:15s}: {time_ms:8.3f} ms  ({speedup:.1f}x)")
        
        # 2. FLOAT_VECTOR Access
        print("\n[2] FLOAT_VECTOR Access (ms):")
        float_results = benchmark_field_access(data, "float_vector", iterations=5)
        for name, time_ms in float_results.items():
            speedup = float_results["Original"] / time_ms if time_ms > 0 else float('inf')
            print(f"  {name:15s}: {time_ms:8.3f} ms  ({speedup:.1f}x)")
        
        # 3. FLOAT16_VECTOR Access (Phase 2 optimized)
        print("\n[3] FLOAT16_VECTOR Access (ms) - Phase 2 Optimized:")
        f16_results = benchmark_field_access(data, "float16_vector", iterations=5)
        for name, time_ms in f16_results.items():
            speedup = f16_results["Original"] / time_ms if time_ms > 0 else float('inf')
            print(f"  {name:15s}: {time_ms:8.3f} ms  ({speedup:.1f}x)")
        
        # 4. INT8_VECTOR Access (Phase 2 optimized)
        print("\n[4] INT8_VECTOR Access (ms) - Phase 2 Optimized:")
        int8_results = benchmark_field_access(data, "int8_vector", iterations=5)
        for name, time_ms in int8_results.items():
            speedup = int8_results["Original"] / time_ms if time_ms > 0 else float('inf')
            print(f"  {name:15s}: {time_ms:8.3f} ms  ({speedup:.1f}x)")
        
        # 5. All Fields Access
        print("\n[5] All Fields Access (ms):")
        all_results = benchmark_all_fields_access(data, iterations=3)
        for name, time_ms in all_results.items():
            speedup = all_results["Original"] / time_ms if time_ms > 0 else float('inf')
            print(f"  {name:15s}: {time_ms:8.3f} ms  ({speedup:.1f}x)")
    
    print("\n" + "=" * 70)
    print("Benchmark Complete!")
    print("=" * 70)


if __name__ == "__main__":
    run_experiments()
