#!/usr/bin/env python3
"""
Protobuf Deserialization Overhead Benchmark

Measures the pure protobuf layer overhead separately from pymilvus conversion logic.
This helps evaluate whether switching protocols (e.g., to Arrow) is worthwhile.

Key insight: This measures what we CANNOT optimize in pymilvus itself.
"""

import pytest
from pymilvus.grpc_gen import schema_pb2
from tests.benchmark.kernels import build_search_result


# =============================================================================
# Configuration
# =============================================================================

# Simulate different result sizes
RESULT_SIZES = [
    (1, 100, "tiny"),
    (10, 100, "small"),
    (10, 1000, "medium"),
    (100, 1000, "large"),
]

# Different payload types to measure serialization cost
PAYLOAD_TYPES = [
    ("scalar_only", [("int_field", "INT64", None)]),
    ("varchar", [("text", "VARCHAR", "MEDIUM")]),
    ("json", [("meta", "JSON", "COMPLEX")]),
    ("vector_float", None),  # Special handling for vector
]


# =============================================================================
# Pure Protobuf Serialization/Deserialization
# =============================================================================

@pytest.mark.parametrize("nq, topk, label", RESULT_SIZES)
def test_proto_serialize(benchmark, nq, topk, label):
    """
    Benchmark: Protobuf.SerializeToString()
    
    Measures the cost of serializing protobuf to bytes (simulates server-side).
    """
    res_data = build_search_result(nq, topk, scalar_fields=[("id", "INT64", None)])
    
    def run_serialize():
        return res_data.SerializeToString()
    
    bytes_data = benchmark(run_serialize)
    assert len(bytes_data) > 0


@pytest.mark.parametrize("nq, topk, label", RESULT_SIZES)
def test_proto_deserialize(benchmark, nq, topk, label):
    """
    Benchmark: Protobuf.ParseFromString()
    
    Measures the cost of deserializing bytes to protobuf object.
    This is the fundamental overhead from gRPC that pymilvus cannot optimize.
    """
    res_data = build_search_result(nq, topk, scalar_fields=[("id", "INT64", None)])
    bytes_data = res_data.SerializeToString()
    
    def run_deserialize():
        pb = schema_pb2.SearchResultData()
        pb.ParseFromString(bytes_data)
        return pb
    
    result = benchmark(run_deserialize)
    assert result.num_queries == nq


@pytest.mark.parametrize("nq, topk, label", RESULT_SIZES)
def test_proto_roundtrip(benchmark, nq, topk, label):
    """
    Benchmark: Full Serialize + Deserialize roundtrip
    
    Simulates the complete network layer overhead.
    """
    res_data = build_search_result(nq, topk, scalar_fields=[("id", "INT64", None)])
    
    def run_roundtrip():
        bytes_data = res_data.SerializeToString()
        pb = schema_pb2.SearchResultData()
        pb.ParseFromString(bytes_data)
        return pb
    
    result = benchmark(run_roundtrip)
    assert result.num_queries == nq


# =============================================================================
# Payload Type Impact on Proto Overhead
# =============================================================================

@pytest.mark.parametrize("dtype, label", [
    ("INT64", "int64"),
    ("VARCHAR", "varchar"),
    ("JSON", "json"),
])
def test_proto_deserialize_by_type(benchmark, dtype, label):
    """
    Benchmark: Proto deserialization cost by data type
    
    Measures how different field types affect protobuf parsing overhead.
    """
    nq, topk = 10, 1000
    
    if dtype == "JSON":
        res_data = build_search_result(nq, topk, scalar_fields=[("field", dtype, "COMPLEX")])
    elif dtype == "VARCHAR":
        res_data = build_search_result(nq, topk, scalar_fields=[("field", dtype, "MEDIUM")])
    else:
        res_data = build_search_result(nq, topk, scalar_fields=[("field", dtype, None)])
    
    bytes_data = res_data.SerializeToString()
    
    def run_deserialize():
        pb = schema_pb2.SearchResultData()
        pb.ParseFromString(bytes_data)
        return pb
    
    benchmark(run_deserialize)


@pytest.mark.parametrize("dtype, dim", [
    ("FLOAT_VECTOR", 128),
    ("FLOAT_VECTOR", 768),
    ("FLOAT16_VECTOR", 768),
    ("BINARY_VECTOR", 1024),
])
def test_proto_deserialize_vector(benchmark, dtype, dim):
    """
    Benchmark: Proto deserialization cost for vector fields
    
    Vectors are large binary blobs - measures their impact on proto parsing.
    """
    nq, topk = 10, 1000
    res_data = build_search_result(nq, topk, vector_fields=[("vector", dtype, dim)])
    bytes_data = res_data.SerializeToString()
    
    def run_deserialize():
        pb = schema_pb2.SearchResultData()
        pb.ParseFromString(bytes_data)
        return pb
    
    benchmark(run_deserialize)


# =============================================================================
# Comparison: Proto Layer vs PyMilvus Conversion Layer
# =============================================================================

@pytest.mark.parametrize("nq, topk, label", RESULT_SIZES)
def test_breakdown_proto_vs_pymilvus(benchmark, nq, topk, label):
    """
    Benchmark breakdown: Proto deserialization ONLY (for comparison)
    
    Run this alongside scalar/vector benchmarks to see:
    - Proto overhead: This test
    - PyMilvus overhead: test_scalar_bench.py / test_vector_bench.py
    - Total = Proto + PyMilvus
    """
    from pymilvus.client.columnar_search_result import ColumnarSearchResult
    
    # Build with multiple field types (realistic scenario)
    res_data = build_search_result(
        nq, topk,
        scalar_fields=[("id", "INT64", None), ("name", "VARCHAR", "MEDIUM")],
        vector_fields=[("embedding", "FLOAT_VECTOR", 768)]
    )
    bytes_data = res_data.SerializeToString()
    
    def run_proto_only():
        pb = schema_pb2.SearchResultData()
        pb.ParseFromString(bytes_data)
        return pb
    
    # This measures ONLY proto layer
    benchmark(run_proto_only)


@pytest.mark.parametrize("nq, topk, label", RESULT_SIZES)
def test_breakdown_full_pipeline(benchmark, nq, topk, label):
    """
    Benchmark breakdown: Proto + PyMilvus ColumnarSearchResult
    
    Measures the complete client-side parsing pipeline.
    Compare with test_breakdown_proto_vs_pymilvus to see PyMilvus overhead.
    """
    from pymilvus.client.columnar_search_result import ColumnarSearchResult
    
    res_data = build_search_result(
        nq, topk,
        scalar_fields=[("id", "INT64", None), ("name", "VARCHAR", "MEDIUM")],
        vector_fields=[("embedding", "FLOAT_VECTOR", 768)]
    )
    bytes_data = res_data.SerializeToString()
    
    def run_full_pipeline():
        pb = schema_pb2.SearchResultData()
        pb.ParseFromString(bytes_data)
        return ColumnarSearchResult(pb)
    
    benchmark(run_full_pipeline)
