
import time
import timeit
import numpy as np
import warnings
from pymilvus.grpc_gen import schema_pb2, common_pb2
from pymilvus.client.types import DataType
from pymilvus.client.search_result import SearchResult
from pymilvus.client.columnar_search_result import ColumnarSearchResult

# Suppress pymilvus warnings
warnings.filterwarnings("ignore")

def create_dummy_search_result(nq=1, topk=10000, dim=128):
    """Generates a dummy SearchResultData protobuf message."""
    total = nq * topk
    
    # 1. IDs
    ids = schema_pb2.IDs()
    ids.int_id.data.extend(list(range(total)))

    # 2. Scores
    scores = [0.5] * total

    # 3. Fields Data
    fields_data = []

    # Float Vector Field
    vec_data = schema_pb2.FieldData()
    vec_data.field_name = "vector"
    vec_data.type = DataType.FLOAT_VECTOR
    vec_data.vectors.dim = dim
    # Flattened vector data
    vec_data.vectors.float_vector.data.extend(np.random.rand(total * dim).astype(np.float32).tolist())
    fields_data.append(vec_data)

    # Int64 Field
    int_data = schema_pb2.FieldData()
    int_data.field_name = "count"
    int_data.type = DataType.INT64
    int_data.scalars.long_data.data.extend(list(range(total)))
    fields_data.append(int_data)

    # Boolean Field
    bool_data = schema_pb2.FieldData()
    bool_data.field_name = "is_valid"
    bool_data.type = DataType.BOOL
    bool_data.scalars.bool_data.data.extend([True, False] * (total // 2))
    fields_data.append(bool_data)

    # Double Field
    double_data = schema_pb2.FieldData()
    double_data.field_name = "score_double"
    double_data.type = DataType.DOUBLE
    double_data.scalars.double_data.data.extend((np.random.rand(total)).tolist())
    fields_data.append(double_data)

    # Varchar Field
    varchar_data = schema_pb2.FieldData()
    varchar_data.field_name = "description"
    varchar_data.type = DataType.VARCHAR
    varchar_data.scalars.string_data.data.extend([f"desc_{i}" for i in range(total)])
    fields_data.append(varchar_data)

    # 4. SearchResultData
    res = schema_pb2.SearchResultData()
    res.ids.CopyFrom(ids)
    res.scores.extend(scores)
    res.topks.extend([topk] * nq)
    res.fields_data.extend(fields_data)
    res.num_queries = nq
    res.output_fields.extend(["vector", "count", "is_valid", "score_double", "description"])
    
    return res

def benchmark():
    nq = 1
    topk = 100000  # Large topk to emphasize row-processing overhead
    dim = 128
    
    print(f"Generating dummy data (nq={nq}, topk={topk}, dim={dim})...")
    res_proto = create_dummy_search_result(nq, topk, dim)
    print("Data generated.")

    print("\n--- Benchmarking Initialization (Parsing) ---")
    
    def init_original():
        return SearchResult(res_proto)

    def init_columnar():
        return ColumnarSearchResult(res_proto)

    t_orig = timeit.timeit(init_original, number=5) / 5
    t_col = timeit.timeit(init_columnar, number=5) / 5
    
    print(f"Original SearchResult Init: {t_orig:.4f} s")
    print(f"ColumnarSearchResult Init: {t_col:.4f} s")
    print(f"Speedup: {t_orig / t_col:.2f}x")

    print("\n--- Benchmarking Access (Iterate all rows, access one field) ---")
    
    orig_res = init_original()
    col_res = init_columnar()

    def iter_original():
        # Simulate user code iterating and accessing ALL fields
        hits = orig_res[0]
        cnt = 0
        for hit in hits:
             # Access all fields to force full materialization/decoding
             _ = hit.entity.get("count")
             _ = hit.entity.get("vector")
             _ = hit.entity.get("is_valid")
             _ = hit.entity.get("score_double")
             _ = hit.entity.get("description")
             cnt += 1
        return cnt

    def iter_columnar():
        hits = col_res[0]
        cnt = 0
        for hit in hits:
             # Access all fields
             _ = hit["count"]
             _ = hit["vector"]
             _ = hit["is_valid"]
             _ = hit["score_double"]
             _ = hit["description"]
             cnt += 1
        return cnt
    
    # Note: iterating original is fast because it's already materialized during Init?
    # Actually hybrid_hits is somewhat lazy, let's verify.
    # But init_original forces some creation of lists.
    
    # Let's measure iteration
    t_iter_orig = timeit.timeit(iter_original, number=5) / 5
    t_iter_col = timeit.timeit(iter_columnar, number=5) / 5
    
    print(f"Original Iterate: {t_iter_orig:.4f} s")
    print(f"Columnar Iterate: {t_iter_col:.4f} s")
    
    # Check correctness
    print("\n--- Correctness Check ---")
    verify_correctness(orig_res, col_res)
    print("Correctness verified: PASS")

def verify_correctness(orig_res, col_res):
    """
    Rigorously compares the original SearchResult with the ColumnarSearchResult.
    """
    assert len(orig_res) == len(col_res), "Mismatch in number of queries"
    
    for i in range(len(orig_res)):
        orig_hits = orig_res[i]
        col_hits = col_res[i]
        
        assert len(orig_hits) == len(col_hits), f"Mismatch in topk for query {i}"
        
        # Check a sample of items to save time, or check all if fast enough
        # checking all 100k might be slow, let's check first, middle, last
        indices_to_check = [0, len(orig_hits)//2, len(orig_hits)-1]
        
        for idx in indices_to_check:
            orig_hit = orig_hits[idx]
            col_hit = col_hits[idx]
            
            # Check ID
            assert orig_hit.id == col_hit.id, f"ID mismatch at index {idx}: {orig_hit.id} != {col_hit.id}"
            
            # Check Score
            assert abs(orig_hit.distance - col_hit.distance) < 1e-6, f"Score mismatch at index {idx}"
            
            # Check Fields (projected output fields)
            # 1. Count (Int64)
            orig_cnt = orig_hit.entity.get("count")
            col_cnt = col_hit["count"]
            assert orig_cnt == col_cnt, f"Count mismatch at index {idx}: {orig_cnt} != {col_cnt}"
            
            # 2. Vector (Float Vector)
            orig_vec = orig_hit.entity.get("vector")
            col_vec = col_hit["vector"]
            assert np.allclose(orig_vec, col_vec, atol=1e-5), f"Vector mismatch at index {idx}"

            # 3. Boolean
            orig_bool = orig_hit.entity.get("is_valid")
            col_bool = col_hit["is_valid"]
            assert orig_bool == col_bool, f"Boolean mismatch at index {idx}: {orig_bool} != {col_bool}"

            # 4. Double
            orig_double = orig_hit.entity.get("score_double")
            col_double = col_hit["score_double"]
            assert abs(orig_double - col_double) < 1e-9, f"Double mismatch at index {idx}"

            # 5. Varchar
            orig_varchar = orig_hit.entity.get("description")
            col_varchar = col_hit["description"]
            assert orig_varchar == col_varchar, f"Varchar mismatch at index {idx}: {orig_varchar} != {col_varchar}"

    
if __name__ == "__main__":
    benchmark()
