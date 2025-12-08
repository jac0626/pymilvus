"""
Test script to verify the output_mode integration.
This tests that the switch between SearchResult and ColumnarSearchResult works.
"""

import sys
import numpy as np
from pymilvus.grpc_gen import schema_pb2
from pymilvus.client.types import DataType
from pymilvus.client.search_result import SearchResult
from pymilvus.client.columnar_search_result import ColumnarSearchResult

def create_dummy_search_result(nq=1, topk=100, dim=128):
    """Generates a dummy SearchResultData protobuf message."""
    total = nq * topk
    
    # IDs
    ids = schema_pb2.IDs()
    ids.int_id.data.extend(list(range(total)))

    # Scores
    scores = [0.5] * total

    # Fields Data
    fields_data = []

    # Float Vector Field
    vec_data = schema_pb2.FieldData()
    vec_data.field_name = "vector"
    vec_data.type = DataType.FLOAT_VECTOR
    vec_data.vectors.dim = dim
    vec_data.vectors.float_vector.data.extend(np.random.rand(total * dim).astype(np.float32).tolist())
    fields_data.append(vec_data)

    # Int64 Field
    int_data = schema_pb2.FieldData()
    int_data.field_name = "count"
    int_data.type = DataType.INT64
    int_data.scalars.long_data.data.extend(list(range(total)))
    fields_data.append(int_data)

    # SearchResultData
    res = schema_pb2.SearchResultData()
    res.ids.CopyFrom(ids)
    res.scores.extend(scores)
    res.topks.extend([topk] * nq)
    res.fields_data.extend(fields_data)
    res.num_queries = nq
    res.output_fields.extend(["vector", "count"])
    
    return res

def test_both_modes():
    print("Testing output_mode integration...")
    
    res_proto = create_dummy_search_result(nq=1, topk=100, dim=8)
    
    # Test default mode (SearchResult)
    print("\n1. Testing default mode (SearchResult)...")
    default_result = SearchResult(res_proto)
    print(f"   Type: {type(default_result).__name__}")
    print(f"   Length: {len(default_result)}")
    print(f"   First hit type: {type(default_result[0][0]).__name__}")
    print(f"   First hit ID: {default_result[0][0].id}")
    
    # Test columnar mode (ColumnarSearchResult)
    print("\n2. Testing columnar mode (ColumnarSearchResult)...")
    columnar_result = ColumnarSearchResult(res_proto)
    print(f"   Type: {type(columnar_result).__name__}")
    print(f"   Length: {len(columnar_result)}")
    print(f"   First hit type: {type(columnar_result[0][0]).__name__}")
    print(f"   First hit ID: {columnar_result[0][0].id}")
    
    # Verify data consistency
    print("\n3. Verifying data consistency...")
    for i in range(min(5, len(default_result[0]))):
        default_hit = default_result[0][i]
        columnar_hit = columnar_result[0][i]
        
        assert default_hit.id == columnar_hit.id, f"ID mismatch at index {i}"
        assert abs(default_hit.distance - columnar_hit.distance) < 1e-6, f"Distance mismatch at index {i}"
        
        # Check field values
        default_count = default_hit.entity.get("count")
        columnar_count = columnar_hit["count"]
        assert default_count == columnar_count, f"Count mismatch at index {i}"
        
    print("   All checks passed!")
    
    print("\n✅ Integration test PASSED!")
    return True

if __name__ == "__main__":
    success = test_both_modes()
    sys.exit(0 if success else 1)
