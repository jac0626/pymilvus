
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

def create_mock_data(nq, topk, str_len=0, has_json=False):
    res = schema_pb2.SearchResultData()
    res.topks.extend([topk] * nq)
    res.scores.extend([0.1] * (nq * topk))
    total_hits = nq * topk
    
    res.ids.int_id.data.extend(range(total_hits))
    
    # 1. INT64 Field (Fixed size)
    f_int = res.fields_data.add()
    f_int.field_name = "age"
    f_int.type = DataType.INT64
    f_int.scalars.long_data.data.extend(range(total_hits))
    
    # Placeholder for VECTOR field (required by SearchResult but not accessed in scalar tests)
    f_vec = res.fields_data.add()
    f_vec.field_name = "vector"
    f_vec.type = DataType.FLOAT_VECTOR
    f_vec.vectors.dim = 1
    f_vec.vectors.float_vector.data.extend([0.1] * total_hits)
    
    # 2. VARCHAR Field (Variable Length)
    if str_len > 0:
        f_str = res.fields_data.add()
        f_str.field_name = "desc"
        f_str.type = DataType.VARCHAR
        s = gen_str(str_len)
        f_str.scalars.string_data.data.extend([s] * total_hits)
        
    # 3. JSON Field (Complex)
    if has_json:
        f_json = res.fields_data.add()
        f_json.field_name = "meta"
        f_json.type = DataType.JSON
        dummy_json = orjson.dumps({
            "user": {"id": 123, "name": "benchmark"},
            "tags": ["perf", "test", "milvus", "python"],
            "score": 99.9,
            "extra": "x" * 50
        })
        f_json.scalars.json_data.data.extend([dummy_json] * total_hits)
    
    fields = ["age", "vector"]
    if str_len > 0: fields.append("desc")
    if has_json: fields.append("meta")
    res.output_fields.extend(fields)
    
    return res

@pytest.fixture(scope="function")
def raw_res_data(request):
    nq = request.param.get("nq", 1)
    topk = request.param.get("topk", 100)
    str_len = request.param.get("len", 0)
    has_json = request.param.get("json", False)
    
    return create_mock_data(nq, topk, str_len, has_json)

# ==============================================================================
# Scalar Benchmark (INT64, VARCHAR, JSON)
# ==============================================================================

@pytest.mark.parametrize("raw_res_data", [
    {"nq": 10, "topk": 1000, "len": 0, "json": False},    # INT64 Baseline
    {"nq": 10, "topk": 1000, "len": 64, "json": False},   # VARCHAR(64)
    {"nq": 10, "topk": 1000, "len": 2048, "json": False}, # VARCHAR(2048)
    {"nq": 10, "topk": 1000, "len": 0, "json": True},     # JSON
], indirect=True)
def test_scalar_access_legacy(benchmark, raw_res_data):
    # Detect field
    fields = [fd.field_name for fd in raw_res_data.fields_data]
    if "meta" in fields: field = "meta"
    elif "desc" in fields: field = "desc"
    else: field = "age"

    def _access():
        # INITIALIZATION INSIDE
        benchmark_data = ColumnarSearchResult(raw_res_data)
        cnt = 0
        for hits in benchmark_data:
            for hit in hits:
                _ = hit[field]
                cnt += 1
        return cnt
    benchmark(_access)

@pytest.mark.parametrize("raw_res_data", [
    {"nq": 10, "topk": 1000, "len": 0, "json": False},    # INT64 Baseline
    {"nq": 10, "topk": 1000, "len": 64, "json": False},   # VARCHAR(64)
    {"nq": 10, "topk": 1000, "len": 2048, "json": False}, # VARCHAR(2048)
    {"nq": 10, "topk": 1000, "len": 0, "json": True},     # JSON
], indirect=True)
def test_scalar_access_batch(benchmark, raw_res_data):
    fields = [fd.field_name for fd in raw_res_data.fields_data]
    if "meta" in fields: field = "meta"
    elif "desc" in fields: field = "desc"
    else: field = "age"

    def _access():
        # INITIALIZATION INSIDE
        benchmark_data = ColumnarSearchResult(raw_res_data)
        cnt = 0
        for hits in benchmark_data:
            col = hits.get_column(field)
            cnt += len(col)
        return cnt
    benchmark(_access)
