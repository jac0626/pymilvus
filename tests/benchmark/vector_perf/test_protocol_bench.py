
import pytest
from pymilvus.client.columnar_search_result import ColumnarSearchResult
from pymilvus.client.search_result import SearchResult
from pymilvus.grpc_gen import schema_pb2
import struct

# =============================================================================
# Benchmarking Configuration
# =============================================================================

# Scenario: Returns 1MB of vector data
# NQ=10, TopK=1000
NQ = 10
TOPK = 1000

# Baseline: 768 dim Float Vector (3072 bytes per vector)
DIM_FLOAT = 768

# Proxy: 1536 dim Float16 Vector (3072 bytes per vector)
# We use Float16 as a proxy because it uses 'bytes' protocol in Protobuf
DIM_BYTES_PROXY = 1536

def build_baseline_data():
    """Build standardized FLOAT_VECTOR data (Repeated Float Protocol)."""
    total = NQ * TOPK
    res = schema_pb2.SearchResultData()
    res.num_queries = NQ
    res.top_k = TOPK
    res.topks.extend([TOPK] * NQ)
    res.ids.int_id.data.extend(list(range(total)))
    res.scores.extend([0.1] * total)
    res.primary_field_name = "id"
    
    field = res.fields_data.add()
    field.field_name = "vector"
    field.type = schema_pb2.DataType.FloatVector
    field.vectors.dim = DIM_FLOAT
    
    # Fill with repeated float data
    dummy_data = [0.123] * (total * DIM_FLOAT)
    field.vectors.float_vector.data.extend(dummy_data)
    res.output_fields.append("vector")
    return res

def build_bytes_proxy_data():
    """Build proxy FLOAT16_VECTOR data (Bytes Protocol). Same payload size."""
    total = NQ * TOPK
    res = schema_pb2.SearchResultData()
    res.num_queries = NQ
    res.top_k = TOPK
    res.topks.extend([TOPK] * NQ)
    res.ids.int_id.data.extend(list(range(total)))
    res.scores.extend([0.1] * total)
    res.primary_field_name = "id"
    
    field = res.fields_data.add()
    field.field_name = "vector"
    field.type = schema_pb2.DataType.Float16Vector
    field.vectors.dim = DIM_BYTES_PROXY 
    
    # Fill with bytes data (same total size in bytes as Float32)
    # Total bytes = total * DIM_BYTES_PROXY * 2 = total * 3072
    payload_size = total * DIM_BYTES_PROXY * 2
    field.vectors.float16_vector = b'x' * payload_size
    res.output_fields.append("vector")
    return res

def iterate_result(results):
    count = 0
    for hits in results:
        for hit in hits:
            v = hit["vector"]
            count += 1
    return count

# =============================================================================
# Benchmarks
# =============================================================================

def test_protocol_baseline_legacy(benchmark):
    """Baseline: Float32 (Repeated) on Legacy."""
    data = build_baseline_data()
    benchmark(lambda: iterate_result(SearchResult(data)))

def test_protocol_bytes_legacy(benchmark):
    """Proposal: Float32 (Bytes Proxy) on Legacy."""
    # Simulates what would happen if we switched Float32 to use bytes protocol
    data = build_bytes_proxy_data()
    benchmark(lambda: iterate_result(SearchResult(data)))

def test_protocol_baseline_columnar(benchmark):
    """Baseline: Float32 (Repeated) on Columnar."""
    data = build_baseline_data()
    benchmark(lambda: iterate_result(ColumnarSearchResult(data)))

def test_protocol_bytes_columnar(benchmark):
    """Proposal: Float32 (Bytes Proxy) on Columnar."""
    data = build_bytes_proxy_data()
    benchmark(lambda: iterate_result(ColumnarSearchResult(data)))
