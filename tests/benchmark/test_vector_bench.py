
import struct
import pytest
from pymilvus.client.columnar_search_result import ColumnarSearchResult
from pymilvus.client.search_result import SearchResult
from pymilvus.grpc_gen import schema_pb2

# =============================================================================
# Benchmarking Configuration
# =============================================================================

# Scientific Matrix for FLOAT_VECTOR
NQ_VALUES = [1, 10, 100]
TOPK_VALUES = [10, 100, 1000]
DIM_VALUES = [128, 768, 1536]

# Representative cases for other types to avoid explosion of tests
OTHER_TYPES_CONFIG = [
    # (type_name, builder_func, dim)
    ("BINARY_VECTOR", "build_binary_vector_result", 1024),
    ("FLOAT16_VECTOR", "build_float16_vector_result", 128),
    ("BFLOAT16_VECTOR", "build_bfloat16_vector_result", 128),
]

# =============================================================================
# Mock Data Builders
# =============================================================================

def build_float_vector_result(nq: int, topk: int, dim: int) -> schema_pb2.SearchResultData:
    """Create mock SearchResultData with FLOAT_VECTOR field."""
    total = nq * topk
    res = schema_pb2.SearchResultData()
    res.num_queries = nq
    res.top_k = topk
    res.topks.extend([topk] * nq)
    res.ids.int_id.data.extend(list(range(total)))
    res.scores.extend([float(i) * 0.001 for i in range(total)])
    res.primary_field_name = "id"

    # FLOAT_VECTOR field
    field = res.fields_data.add()
    field.field_name = "vector"
    field.type = schema_pb2.DataType.FloatVector
    field.vectors.dim = dim
    for i in range(total * dim):
        field.vectors.float_vector.data.append(float(i % 1000) * 0.001)
    res.output_fields.append("vector")
    return res

def build_binary_vector_result(nq: int, topk: int, dim: int) -> schema_pb2.SearchResultData:
    """Create mock SearchResultData with BINARY_VECTOR field (dim in bits)."""
    total = nq * topk
    bytes_per_vec = dim // 8
    res = schema_pb2.SearchResultData()
    res.num_queries = nq
    res.top_k = topk
    res.topks.extend([topk] * nq)
    res.ids.int_id.data.extend(list(range(total)))
    res.scores.extend([float(i) * 0.001 for i in range(total)])
    res.primary_field_name = "id"

    # BINARY_VECTOR field
    field = res.fields_data.add()
    field.field_name = "vector"
    field.type = schema_pb2.DataType.BinaryVector
    field.vectors.dim = dim
    field.vectors.binary_vector = bytes([i % 256 for i in range(total * bytes_per_vec)])
    res.output_fields.append("vector")
    return res

def build_float16_vector_result(nq: int, topk: int, dim: int) -> schema_pb2.SearchResultData:
    total = nq * topk
    bytes_per_vec = dim * 2
    res = schema_pb2.SearchResultData()
    res.num_queries = nq
    res.top_k = topk
    res.topks.extend([topk] * nq)
    res.ids.int_id.data.extend(list(range(total)))
    res.scores.extend([float(i) * 0.001 for i in range(total)])
    res.primary_field_name = "id"

    field = res.fields_data.add()
    field.field_name = "vector"
    field.type = schema_pb2.DataType.Float16Vector
    field.vectors.dim = dim
    field.vectors.float16_vector = bytes([i % 256 for i in range(total * bytes_per_vec)])
    res.output_fields.append("vector")
    return res

def build_bfloat16_vector_result(nq: int, topk: int, dim: int) -> schema_pb2.SearchResultData:
    total = nq * topk
    bytes_per_vec = dim * 2
    res = schema_pb2.SearchResultData()
    res.num_queries = nq
    res.top_k = topk
    res.topks.extend([topk] * nq)
    res.ids.int_id.data.extend(list(range(total)))
    res.scores.extend([float(i) * 0.001 for i in range(total)])
    res.primary_field_name = "id"

    field = res.fields_data.add()
    field.field_name = "vector"
    field.type = schema_pb2.DataType.BFloat16Vector
    field.vectors.dim = dim
    field.vectors.bfloat16_vector = bytes([i % 256 for i in range(total * bytes_per_vec)])
    res.output_fields.append("vector")
    return res

def build_int8_vector_result(nq: int, topk: int, dim: int) -> schema_pb2.SearchResultData:
    total = nq * topk
    res = schema_pb2.SearchResultData()
    res.num_queries = nq
    res.top_k = topk
    res.topks.extend([topk] * nq)
    res.ids.int_id.data.extend(list(range(total)))
    res.scores.extend([float(i) * 0.001 for i in range(total)])
    res.primary_field_name = "id"

    # INT8_VECTOR field
    field = res.fields_data.add()
    field.field_name = "vector"
    field.type = schema_pb2.DataType.Int8Vector
    field.vectors.dim = dim
    field.vectors.int8_vector = bytes([i % 256 for i in range(total * dim)])
    res.output_fields.append("vector")
    return res

# =============================================================================
# Benchmark Tests
# =============================================================================

def iterate_result(results):
    """Helper to simulate full iteration + field access."""
    count = 0
    for hits in results:
        for hit in hits:
            _ = hit["vector"]
            count += 1
    return count

@pytest.mark.parametrize("nq", NQ_VALUES)
@pytest.mark.parametrize("topk", TOPK_VALUES)
@pytest.mark.parametrize("dim", DIM_VALUES)
def test_float_vector_matrix(benchmark, nq, topk, dim):
    """
    Comprehensive Matrix Test for FLOAT_VECTOR
    Covers: Initialization + Iteration
    """
    # 1. Build Data
    res_data = build_float_vector_result(nq, topk, dim)
    
    # 2. Benchmark ColumnarSearchResult
    def run_columnar():
        cr = ColumnarSearchResult(res_data)
        iterate_result(cr)
        
    benchmark(run_columnar)


# Legacy comparison for reference (Spot check only)
def test_float_legacy_baseline(benchmark):
    """Keep one baseline check for legacy SearchResult to ensure we track regression/improvement."""
    nq, topk, dim = 10, 100, 768
    res_data = build_float_vector_result(nq, topk, dim)
    def run_legacy():
        sr = SearchResult(res_data)
        iterate_result(sr)
    benchmark(run_legacy)


@pytest.mark.parametrize("type_name, builder_func_name, dim", OTHER_TYPES_CONFIG)
def test_other_vectors(benchmark, type_name, builder_func_name, dim):
    """
    Representative benchmarks for BINARY, FLOAT16, etc.
    Using fixed NQ=10, TopK=100 for efficiency.
    """
    nq, topk = 10, 100
    builder = globals()[builder_func_name]
    res_data = builder(nq, topk, dim)
    
    def run_columnar():
        cr = ColumnarSearchResult(res_data)
        iterate_result(cr)
        
    benchmark(run_columnar)
