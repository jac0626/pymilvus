
import pytest
import numpy as np
import orjson
from pymilvus.client.columnar_search_result import ColumnarSearchResult
from pymilvus.grpc_gen import schema_pb2, common_pb2
from pymilvus.client.types import DataType

def create_mock_result(nq=1, topk=2, dim=4):
    """
    Create a Mock SearchResultData with varied fields.
    """
    res = schema_pb2.SearchResultData()
    res.topks.extend([topk] * nq)
    res.scores.extend([0.1] * (nq * topk))
    
    # IDs
    total_hits = nq * topk
    res.ids.int_id.data.extend(range(total_hits))
    
    # Int64 Field
    f_int = res.fields_data.add()
    f_int.field_name = "age"
    f_int.type = DataType.INT64
    f_int.scalars.long_data.data.extend(range(100, 100 + total_hits))
    
    # Float Vector Field
    f_vec = res.fields_data.add()
    f_vec.field_name = "emb"
    f_vec.type = DataType.FLOAT_VECTOR
    f_vec.vectors.dim = dim
    # Flattened vector data: [0.1, 0.1, ..., 0.2, 0.2, ...]
    for i in range(total_hits):
        f_vec.vectors.float_vector.data.extend([float(i)] * dim)
        
    # JSON Field
    f_json = res.fields_data.add()
    f_json.field_name = "meta"
    f_json.type = DataType.JSON
    for i in range(total_hits):
        f_json.scalars.json_data.data.append(orjson.dumps({"idx": i}))
        
    # Dynamic Field ($meta)
    f_dyn = res.fields_data.add()
    f_dyn.field_name = "$meta"
    f_dyn.type = DataType.JSON
    for i in range(total_hits):
        f_dyn.scalars.json_data.data.append(orjson.dumps({"color": "red" if i % 2 == 0 else "blue"}))
        
    # Output fields (include dynamic 'color')
    res.output_fields.extend(["age", "emb", "meta", "color"])
    
    return res

def test_get_column_int64():
    nq, topk = 2, 5
    res_proto = create_mock_result(nq, topk)
    res = ColumnarSearchResult(res_proto)
    
    # Check first query results
    hits0 = res[0]
    col_age = hits0.get_column("age")
    
    assert len(col_age) == topk
    assert col_age == [100, 101, 102, 103, 104]
    
    # Check second query results (offset)
    hits1 = res[1]
    col_age1 = hits1.get_column("age")
    assert col_age1 == [105, 106, 107, 108, 109]

def test_get_column_float_vector():
    nq, topk, dim = 1, 3, 4
    res_proto = create_mock_result(nq, topk, dim)
    res = ColumnarSearchResult(res_proto)
    
    hits = res[0]
    col_vec = hits.get_column("emb")
    
    # Should be flattened: 3 vectors * 4 dim = 12 floats
    assert len(col_vec) == topk * dim
    assert col_vec[:4] == [0.0, 0.0, 0.0, 0.0]  # First vec
    assert col_vec[4:8] == [1.0, 1.0, 1.0, 1.0] # Second vec

def test_get_column_json_raw():
    nq, topk = 1, 3
    res_proto = create_mock_result(nq, topk)
    res = ColumnarSearchResult(res_proto)
    
    hits = res[0]
    col_json = hits.get_column("meta")
    
    assert len(col_json) == topk
    assert isinstance(col_json[0], bytes)
    assert orjson.loads(col_json[0]) == {"idx": 0}

def test_get_column_dynamic():
    nq, topk = 1, 4
    res_proto = create_mock_result(nq, topk)
    res = ColumnarSearchResult(res_proto)
    
    hits = res[0]
    col_color = hits.get_column("color")
    
    assert len(col_color) == topk
    assert col_color == ["red", "blue", "red", "blue"]

def test_missing_field_no_dynamic():
    # create default mock (has $meta) -> manually remove it
    res_proto = create_mock_result()
    # Find index of $meta and remove it
    meta_idx = -1
    for i, f in enumerate(res_proto.fields_data):
        if f.field_name == "$meta":
            meta_idx = i
            break
    if meta_idx != -1:
        del res_proto.fields_data[meta_idx]

    res = ColumnarSearchResult(res_proto)
    hits = res[0]
    
    with pytest.raises(KeyError):
        hits.get_column("non_existent")

def test_missing_field_with_dynamic():
    # Default mock has $meta
    res_proto = create_mock_result(nq=1, topk=2)
    res = ColumnarSearchResult(res_proto)
    hits = res[0]
    
    # Should return list of None
    col = hits.get_column("non_existent")
    assert col == [None, None]

if __name__ == "__main__":
    test_get_column_int64()
    test_get_column_float_vector()
    test_get_column_json_raw()
    test_get_column_dynamic()
    test_missing_field_no_dynamic()
    test_missing_field_with_dynamic()
    print("All tests passed!")
