"""
Test compatibility between ColumnarSearchResult and SearchResult.
Verifies that return types match exactly.
"""
import numpy as np
from pymilvus.grpc_gen import schema_pb2
from pymilvus.client.search_result import SearchResult
from pymilvus.client.columnar_search_result import ColumnarSearchResult


def create_mock_search_result(nq=2, topk=5, dim=4):
    """Create mock SearchResultData with various field types."""
    total = nq * topk
    
    res = schema_pb2.SearchResultData()
    res.num_queries = nq
    res.top_k = topk
    res.topks.extend([topk] * nq)
    res.ids.int_id.data.extend(list(range(total)))
    res.scores.extend([float(i) * 0.1 for i in range(total)])
    res.primary_field_name = "id"
    
    # FLOAT_VECTOR
    vec_field = res.fields_data.add()
    vec_field.field_name = "float_vector"
    vec_field.type = schema_pb2.DataType.FloatVector
    vec_field.vectors.dim = dim
    # Add float data (RepeatedScalarFieldContainer)
    for i in range(total * dim):
        vec_field.vectors.float_vector.data.append(float(i) * 0.01)
    
    # BINARY_VECTOR
    bin_field = res.fields_data.add()
    bin_field.field_name = "binary_vector"
    bin_field.type = schema_pb2.DataType.BinaryVector
    bin_field.vectors.dim = 32  # 32 bits = 4 bytes per vector
    bin_field.vectors.binary_vector = bytes([i % 256 for i in range(total * 4)])
    
    # INT64 scalar
    int_field = res.fields_data.add()
    int_field.field_name = "int_field"
    int_field.type = schema_pb2.DataType.Int64
    int_field.scalars.long_data.data.extend(list(range(total)))
    
    # VARCHAR scalar
    str_field = res.fields_data.add()
    str_field.field_name = "str_field"
    str_field.type = schema_pb2.DataType.VarChar
    str_field.scalars.string_data.data.extend([f"str_{i}" for i in range(total)])
    
    # JSON
    json_field = res.fields_data.add()
    json_field.field_name = "json_field"
    json_field.type = schema_pb2.DataType.JSON
    for i in range(total):
        json_field.scalars.json_data.data.append(f'{{"value": {i}}}'.encode())
    
    # ARRAY (INT64)
    array_field = res.fields_data.add()
    array_field.field_name = "array_field"
    array_field.type = schema_pb2.DataType.Array
    array_field.scalars.array_data.element_type = schema_pb2.DataType.Int64
    for i in range(total):
        arr = array_field.scalars.array_data.data.add()
        arr.long_data.data.extend([i, i+1, i+2])
    
    res.output_fields.extend([
        "float_vector", "binary_vector", "int_field", "str_field", "json_field", "array_field"
    ])
    
    return res


def test_type_compatibility():
    """Verify that ColumnarSearchResult returns the same types as SearchResult."""
    print("=" * 60)
    print("Testing Type Compatibility")
    print("=" * 60)
    
    res_data = create_mock_search_result(nq=2, topk=5, dim=4)
    
    sr = SearchResult(res_data)
    cr = ColumnarSearchResult(res_data)
    
    # Get first hit from each
    sr_hit = sr[0][0]
    cr_hit = cr[0][0]
    
    fields_to_check = ["float_vector", "binary_vector", "int_field", "str_field", "json_field", "array_field"]
    
    print("\nField Type Comparison:")
    print("-" * 60)
    all_match = True
    
    for field in fields_to_check:
        sr_val = sr_hit[field]
        cr_val = cr_hit[field]
        
        sr_type = type(sr_val).__name__
        cr_type = type(cr_val).__name__
        
        type_match = type(sr_val) == type(cr_val)
        value_match = sr_val == cr_val
        
        status = "✅" if type_match and value_match else "❌"
        if not (type_match and value_match):
            all_match = False
        
        print(f"{status} {field}:")
        print(f"   SearchResult:   {sr_type} = {sr_val}")
        print(f"   Columnar:       {cr_type} = {cr_val}")
    
    # Check special properties
    print("\nProperty Comparison:")
    print("-" * 60)
    
    props = [
        ("id", sr_hit.id, cr_hit.id),
        ("distance", sr_hit.distance, cr_hit.distance),
    ]
    
    for name, sr_val, cr_val in props:
        match = sr_val == cr_val
        status = "✅" if match else "❌"
        if not match:
            all_match = False
        print(f"{status} {name}: SearchResult={sr_val}, Columnar={cr_val}")
    
    print("\n" + "=" * 60)
    if all_match:
        print("✅ All types and values match!")
    else:
        print("❌ Some mismatches found!")
    print("=" * 60)
    
    return all_match


def test_iteration_compatibility():
    """Verify that iteration works the same way."""
    print("\n" + "=" * 60)
    print("Testing Iteration Compatibility")
    print("=" * 60)
    
    res_data = create_mock_search_result(nq=2, topk=3, dim=4)
    
    sr = SearchResult(res_data)
    cr = ColumnarSearchResult(res_data)
    
    print(f"\nSearchResult structure: len={len(sr)}, sub-lens={[len(h) for h in sr]}")
    print(f"Columnar structure:     len={len(cr)}, sub-lens={[len(h) for h in cr]}")
    
    # Compare all hits
    all_match = True
    for q_idx, (sr_hits, cr_hits) in enumerate(zip(sr, cr)):
        for h_idx, (sr_hit, cr_hit) in enumerate(zip(sr_hits, cr_hits)):
            if sr_hit.id != cr_hit.id or sr_hit.distance != cr_hit.distance:
                print(f"❌ Mismatch at [{q_idx}][{h_idx}]")
                all_match = False
    
    if all_match:
        print("✅ All iterations match!")
    
    return all_match


def test_dict_compatibility():
    """Verify dict-like interface compatibility."""
    print("\n" + "=" * 60)
    print("Testing Dict-like Interface Compatibility")
    print("=" * 60)
    
    res_data = create_mock_search_result(nq=1, topk=2, dim=4)
    
    cr = ColumnarSearchResult(res_data)
    hit = cr[0][0]
    
    print(f"keys(): {hit.keys()}")
    print(f"'int_field' in hit: {'int_field' in hit}")
    print(f"hit.get('int_field'): {hit.get('int_field')}")
    print(f"hit.get('nonexistent', 'default'): {hit.get('nonexistent', 'default')}")
    print(f"hit.to_dict(): {hit.to_dict()}")
    
    # Test read-only
    try:
        hit["int_field"] = 999
        print("❌ Should have raised TypeError!")
        return False
    except TypeError as e:
        print(f"✅ Read-only enforcement: {e}")
    
    return True


def test_performance():
    """Compare initialization performance."""
    import time
    
    print("\n" + "=" * 60)
    print("Testing Performance")
    print("=" * 60)
    
    # Create larger mock data
    nq, topk, dim = 100, 1000, 128
    total = nq * topk
    
    res = schema_pb2.SearchResultData()
    res.num_queries = nq
    res.top_k = topk
    res.topks.extend([topk] * nq)
    res.ids.int_id.data.extend(list(range(total)))
    res.scores.extend([float(i) * 0.001 for i in range(total)])
    res.primary_field_name = "id"
    
    # Add float vector
    vec_field = res.fields_data.add()
    vec_field.field_name = "vector"
    vec_field.type = schema_pb2.DataType.FloatVector
    vec_field.vectors.dim = dim
    for i in range(total * dim):
        vec_field.vectors.float_vector.data.append(float(i % 1000) * 0.001)
    res.output_fields.append("vector")
    
    print(f"Data size: nq={nq}, topk={topk}, dim={dim}, total={total:,} hits")
    
    # Benchmark SearchResult
    start = time.perf_counter()
    sr = SearchResult(res)
    sr_init_time = (time.perf_counter() - start) * 1000
    
    # Benchmark ColumnarSearchResult
    start = time.perf_counter()
    cr = ColumnarSearchResult(res)
    cr_init_time = (time.perf_counter() - start) * 1000
    
    print(f"\nInitialization time:")
    print(f"  SearchResult:   {sr_init_time:.2f} ms")
    print(f"  Columnar:       {cr_init_time:.2f} ms")
    print(f"  Speedup:        {sr_init_time / cr_init_time:.0f}x")
    
    return True


if __name__ == "__main__":
    test_type_compatibility()
    test_iteration_compatibility()
    test_dict_compatibility()
    test_performance()
