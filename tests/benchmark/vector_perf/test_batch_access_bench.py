
import pytest
import time
import random
import string
import orjson
from pymilvus.grpc_gen import schema_pb2
from pymilvus.client.types import DataType
from pymilvus.client.columnar_search_result import ColumnarSearchResult

# Helper to generate random string
def gen_str(length=100):
    return "".join(random.choices(string.ascii_letters, k=length))

def create_mock_data(nq, topk, dim=128, vector_type=DataType.FLOAT_VECTOR):
    res = schema_pb2.SearchResultData()
    res.topks.extend([topk] * nq)
    res.scores.extend([0.1] * (nq * topk))
    total_hits = nq * topk
    
    res.ids.int_id.data.extend(range(total_hits))
    
    # INT64 Field (Baseline required by schema usually, but focusing on vector)
    f_int = res.fields_data.add()
    f_int.field_name = "id_field"
    f_int.type = DataType.INT64
    f_int.scalars.long_data.data.extend(range(total_hits))
    
    # VECTOR Field
    f_vec = res.fields_data.add()
    f_vec.field_name = "vector"
    f_vec.type = vector_type
    f_vec.vectors.dim = dim
    if vector_type == DataType.FLOAT_VECTOR:
        f_vec.vectors.float_vector.data.extend([0.1] * (total_hits * dim))
    elif vector_type == DataType.FLOAT16_VECTOR:
        f_vec.vectors.float16_vector = bytes([0] * (total_hits * dim * 2))
    elif vector_type == DataType.BFLOAT16_VECTOR:
        f_vec.vectors.bfloat16_vector = bytes([0] * (total_hits * dim * 2))
    elif vector_type == DataType.BINARY_VECTOR:
        f_vec.vectors.binary_vector = bytes([0] * (total_hits * dim // 8))
    
    res.output_fields.extend(["id_field", "vector"])
    return res

@pytest.fixture(scope="function")
def raw_res_data(request):
    nq = request.param.get("nq", 1)
    topk = request.param.get("topk", 100)
    dim = request.param.get("dim", 128)
    vector_type = request.param.get("vtype", DataType.FLOAT_VECTOR)
    
    return create_mock_data(nq, topk, dim, vector_type)

# ==============================================================================
# Vector Benchmark (FLOAT, FLOAT16, BINARY)
# ==============================================================================

@pytest.mark.parametrize("raw_res_data", [
    {"nq": 10, "topk": 1000, "vtype": DataType.FLOAT_VECTOR},
    {"nq": 10, "topk": 1000, "vtype": DataType.FLOAT16_VECTOR},
    {"nq": 10, "topk": 1000, "vtype": DataType.BINARY_VECTOR, "dim": 1024},
], indirect=True)
def test_vector_access_legacy(benchmark, raw_res_data):
    def _access():
        # INITIALIZATION INSIDE
        benchmark_data = ColumnarSearchResult(raw_res_data)
        cnt = 0
        for hits in benchmark_data:
            for hit in hits:
                _ = hit["vector"]
                cnt += 1
        return cnt
    benchmark(_access)

@pytest.mark.parametrize("raw_res_data", [
    {"nq": 10, "topk": 1000, "vtype": DataType.FLOAT_VECTOR},
    {"nq": 10, "topk": 1000, "vtype": DataType.FLOAT16_VECTOR},
    {"nq": 10, "topk": 1000, "vtype": DataType.BINARY_VECTOR, "dim": 1024},
], indirect=True)
def test_vector_access_batch(benchmark, raw_res_data):
    def _access():
        # INITIALIZATION INSIDE
        benchmark_data = ColumnarSearchResult(raw_res_data)
        cnt = 0
        for hits in benchmark_data:
            col = hits.get_column("vector")
            cnt += len(col)
        return cnt
    benchmark(_access)
