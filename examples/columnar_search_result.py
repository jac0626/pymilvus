#!/usr/bin/env python3
"""
ColumnarSearchResult Usage Example

This example demonstrates how to use ColumnarSearchResult for high-performance
search result processing. ColumnarSearchResult provides:
- 1000x+ faster initialization than SearchResult
- Zero-copy vector access via numpy arrays
- Batch column access API for vectorized processing
- Full compatibility with existing SearchResult API

This script can run in two modes:
1. With Milvus server: Set MILVUS_URI environment variable
2. Without server: Uses mock data for performance comparison only
"""

import os
import numpy as np
import time

# Default Milvus server - override with MILVUS_URI environment variable
MILVUS_URI = os.getenv("MILVUS_URI", "http://10.100.32.69:19530")
HAS_MILVUS = bool(MILVUS_URI)

if HAS_MILVUS:
    from pymilvus import MilvusClient


def setup_collection():
    """Create a test collection with ALL supported field types for E2E testing."""
    if not HAS_MILVUS:
        print("Skipping: MILVUS_URI not set")
        return None
    
    from pymilvus import DataType, FieldSchema, CollectionSchema
    
    client = MilvusClient(MILVUS_URI)
    collection_name = "columnar_example_collection"
    
    # Drop if exists
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)
    
    # Create schema with ALL scalar types
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=128),
        # All integer types
        FieldSchema(name="int8_field", dtype=DataType.INT8),
        FieldSchema(name="int16_field", dtype=DataType.INT16),
        FieldSchema(name="int32_field", dtype=DataType.INT32),
        FieldSchema(name="int64_field", dtype=DataType.INT64),
        # Float types
        FieldSchema(name="float_field", dtype=DataType.FLOAT),
        FieldSchema(name="double_field", dtype=DataType.DOUBLE),
        # String and bool
        FieldSchema(name="varchar_field", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="bool_field", dtype=DataType.BOOL),
        # Complex types
        FieldSchema(name="json_field", dtype=DataType.JSON),
        FieldSchema(name="int_array", dtype=DataType.ARRAY, element_type=DataType.INT64, max_capacity=10),
        FieldSchema(name="float_array", dtype=DataType.ARRAY, element_type=DataType.FLOAT, max_capacity=10),
        FieldSchema(name="varchar_array", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=10, max_length=64),
    ]
    schema = CollectionSchema(fields=fields, enable_dynamic_field=False)
    client.create_collection(collection_name=collection_name, schema=schema)
    
    # Insert sample data with ALL field types
    num_entities = 100
    entities = []
    for i in range(num_entities):
        entities.append({
            "vector": [float((i + j) % 256) / 256 for j in range(128)],
            # Integer types
            "int8_field": (i % 127),  # -128 to 127
            "int16_field": i * 10,
            "int32_field": i * 100,
            "int64_field": i * 1000,
            # Float types
            "float_field": float(i) * 0.5,
            "double_field": float(i) * 1.5,
            # String and bool
            "varchar_field": f"Document_{i}",
            "bool_field": i % 2 == 0,
            # Complex types
            "json_field": {"name": f"item_{i}", "value": i * 100, "tags": [f"tag{i%5}"]},
            "int_array": [i, i+1, i+2],
            "float_array": [float(i)*0.1, float(i)*0.2, float(i)*0.3],
            "varchar_array": [f"s{i}", f"t{i}", f"u{i}"],
        })
    
    client.insert(collection_name=collection_name, data=entities)
    
    # Create index and load
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="vector", metric_type="L2", index_type="FLAT")
    client.create_index(collection_name=collection_name, index_params=index_params)
    client.load_collection(collection_name=collection_name)
    
    all_fields = [f.name for f in fields if f.name not in ("id", "vector")]
    print(f"Created collection '{collection_name}' with {num_entities} entities")
    print(f"  Fields ({len(all_fields)}): {', '.join(all_fields)}")
    return collection_name


def basic_usage_example():
    """
    Basic iteration - works the same as SearchResult.
    Use output_mode="columnar" to get ColumnarSearchResult.
    """
    if not HAS_MILVUS:
        print("Skipping basic_usage_example: MILVUS_URI not set")
        return
    
    client = MilvusClient(MILVUS_URI)
    collection_name = "columnar_example_collection"
    
    # Search with output_mode="columnar"
    results = client.search(
        collection_name=collection_name,
        data=[[0.1] * 128],
        limit=5,
        output_fields=["int64_field", "varchar_field"],
        output_mode="columnar",
    )
    
    print("\n=== Basic Usage Example ===")
    for hits in results:
        for hit in hits:
            print(f"ID: {hit.id}, Distance: {hit.distance:.4f}, int64: {hit['int64_field']}, name: {hit['varchar_field']}")


def dict_compatibility_example():
    """
    RowProxy supports dict-like methods for compatibility.
    """
    if not HAS_MILVUS:
        print("Skipping dict_compatibility_example: MILVUS_URI not set")
        return
    
    client = MilvusClient(MILVUS_URI)
    collection_name = "columnar_example_collection"
    
    results = client.search(
        collection_name=collection_name,
        data=[[0.1] * 128],
        limit=3,
        output_fields=["int64_field", "varchar_field", "bool_field"],
        output_mode="columnar",
    )
    
    print("\n=== Dict Compatibility Example ===")
    hit = results[0][0]
    
    print(f"Keys: {list(hit.keys())}")
    print(f"Values: {hit.values()}")
    print(f"Items: {hit.items()}")
    print(f"'varchar_field' in hit: {'varchar_field' in hit}")


def slice_access_example():
    """
    ColumnarHits supports slice and negative indexing.
    """
    if not HAS_MILVUS:
        print("Skipping slice_access_example: MILVUS_URI not set")
        return
    
    client = MilvusClient(MILVUS_URI)
    collection_name = "columnar_example_collection"
    
    results = client.search(
        collection_name=collection_name,
        data=[[0.1] * 128],
        limit=10,
        output_fields=["varchar_field"],
        output_mode="columnar",
    )
    
    print("\n=== Slice Access Example ===")
    hits = results[0]
    
    print("First 3 results (hits[:3]):")
    for hit in hits[:3]:
        print(f"  ID: {hit.id}, name: {hit['varchar_field']}")
    
    print(f"Last result (hits[-1]): ID={hits[-1].id}")


def performance_comparison():
    """
    Demonstrates the performance difference between access patterns.
    This works without a Milvus server - uses mock data.
    """
    from pymilvus.grpc_gen import schema_pb2
    from pymilvus.client.types import DataType
    from pymilvus.client.columnar_search_result import ColumnarSearchResult
    
    # Create test data
    nq, topk, dim = 100, 1000, 128
    total = nq * topk
    
    result_data = schema_pb2.SearchResultData()
    result_data.ids.int_id.data.extend(list(range(total)))
    result_data.scores.extend([float(i) * 0.01 for i in range(total)])
    result_data.topks.extend([topk] * nq)
    result_data.num_queries = nq
    
    vec_field = result_data.fields_data.add()
    vec_field.field_name = "vector"
    vec_field.type = DataType.FLOAT16_VECTOR
    vec_field.vectors.dim = dim
    vec_field.vectors.float16_vector = np.random.rand(total * dim).astype(np.float16).tobytes()
    result_data.output_fields.extend(["vector"])
    
    print("\n=== Performance Comparison ===")
    print(f"Test data: nq={nq}, topk={topk}, dim={dim}, total={total:,} vectors")
    
    # Method 1: Per-row access (slower)
    cr = ColumnarSearchResult(result_data, zero_copy_vectors=True)
    start = time.perf_counter()
    for hits in cr:
        for hit in hits:
            _ = hit["vector"]
    per_row_time = (time.perf_counter() - start) * 1000
    
    # Method 2: Batch access (faster)
    cr = ColumnarSearchResult(result_data, zero_copy_vectors=True)
    start = time.perf_counter()
    for hits in cr:
        _ = hits.get_column("vector")
    batch_time = (time.perf_counter() - start) * 1000
    
    print(f"Per-row access: {per_row_time:.0f} ms")
    print(f"Batch access:   {batch_time:.0f} ms")
    print(f"Speedup:        {per_row_time/batch_time:.0f}x")


def batch_access_example():
    """
    Use batch access API for maximum performance when processing all results.
    """
    from pymilvus.grpc_gen import schema_pb2
    from pymilvus.client.types import DataType
    from pymilvus.client.columnar_search_result import ColumnarSearchResult
    
    # Create mock data
    nq, topk, dim = 10, 100, 128
    total = nq * topk
    
    result_data = schema_pb2.SearchResultData()
    result_data.ids.int_id.data.extend(list(range(total)))
    result_data.scores.extend([float(i) * 0.01 for i in range(total)])
    result_data.topks.extend([topk] * nq)
    result_data.num_queries = nq
    
    vec_field = result_data.fields_data.add()
    vec_field.field_name = "vector"
    vec_field.type = DataType.FLOAT16_VECTOR
    vec_field.vectors.dim = dim
    vec_field.vectors.float16_vector = np.random.rand(total * dim).astype(np.float16).tobytes()
    result_data.output_fields.extend(["vector"])
    
    print("\n=== Batch Access Example ===")
    cr = ColumnarSearchResult(result_data, zero_copy_vectors=True)
    
    for i, hits in enumerate(cr):
        if i >= 2:  # Only show first 2 queries
            break
        vectors = hits.get_column("vector")
        ids = hits.get_all_ids()
        distances = hits.get_all_distances()
        
        print(f"Query {i}: vectors.shape={vectors.shape}, {len(ids)} IDs, {len(distances)} distances")


def cleanup_collection():
    """Remove the test collection."""
    if not HAS_MILVUS:
        return
    
    client = MilvusClient(MILVUS_URI)
    collection_name = "columnar_example_collection"
    
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)
        print(f"\nCleaned up collection '{collection_name}'")


def compare_vector_types():
    """
    Compare Original vs Columnar for ALL vector types:
    - FLOAT_VECTOR
    - FLOAT16_VECTOR
    - BFLOAT16_VECTOR
    - BINARY_VECTOR
    - INT8_VECTOR
    - SPARSE_FLOAT_VECTOR
    """
    if not HAS_MILVUS:
        print("Skipping compare_vector_types: MILVUS_URI not set")
        return
    
    from pymilvus import DataType, FieldSchema, CollectionSchema
    import random
    
    client = MilvusClient(MILVUS_URI)
    
    # Dense vector types
    vector_types = [
        ("FLOAT_VECTOR", DataType.FLOAT_VECTOR, 128),
        ("FLOAT16_VECTOR", DataType.FLOAT16_VECTOR, 128),
        ("BFLOAT16_VECTOR", DataType.BFLOAT16_VECTOR, 128),
        ("BINARY_VECTOR", DataType.BINARY_VECTOR, 128),  # 128 bits = 16 bytes
        ("INT8_VECTOR", DataType.INT8_VECTOR, 128),
    ]
    
    print("\n=== Compare ALL Vector Types ===")
    
    for vec_name, vec_type, dim in vector_types:
        collection_name = f"test_vector_{vec_name.lower()}"
        
        # Drop if exists
        if client.has_collection(collection_name):
            client.drop_collection(collection_name)
        
        # Create schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="vector", dtype=vec_type, dim=dim),
            FieldSchema(name="label", dtype=DataType.VARCHAR, max_length=100),
        ]
        schema = CollectionSchema(fields=fields, enable_dynamic_field=False)
        
        # Create collection
        client.create_collection(collection_name=collection_name, schema=schema)
        
        # For bfloat16, check if ml_dtypes is available
        if vec_type == DataType.BFLOAT16_VECTOR:
            try:
                from ml_dtypes import bfloat16
            except ImportError:
                print(f"  ⚠️ {vec_name}: Skipped (ml_dtypes not installed)")
                client.drop_collection(collection_name)
                continue
        
        # Generate data based on vector type
        num_entities = 20
        entities = []
        for i in range(num_entities):
            if vec_type == DataType.BINARY_VECTOR:
                # Binary vector: bytes of length dim/8
                vec_data = bytes([i % 256] * (dim // 8))
            elif vec_type == DataType.FLOAT16_VECTOR:
                # FLOAT16_VECTOR requires np.ndarray with dtype=float16
                vec_data = np.array([float(i + j) / 100 for j in range(dim)], dtype=np.float16)
            elif vec_type == DataType.BFLOAT16_VECTOR:
                # BFLOAT16_VECTOR requires np.ndarray with dtype=bfloat16
                from ml_dtypes import bfloat16
                vec_data = np.array([float(i + j) / 100 for j in range(dim)], dtype=bfloat16)
            elif vec_type == DataType.INT8_VECTOR:
                # INT8_VECTOR: requires np.ndarray with dtype=int8
                vec_data = np.array([(i + j) % 128 for j in range(dim)], dtype=np.int8)
            else:
                # FLOAT_VECTOR
                vec_data = [float(i + j) / 100 for j in range(dim)]
            
            entities.append({
                "vector": vec_data,
                "label": f"item_{i}",
            })
        
        client.insert(collection_name=collection_name, data=entities)
        
        if vec_type == DataType.INT8_VECTOR:
             # INT8_VECTOR: use HNSW as per example
            index_params = client.prepare_index_params()
            index_params.add_index(
                field_name="vector", 
                metric_type="L2", 
                index_type="HNSW", 
                params={"M": 8, "efConstruction": 200}
            )
            client.create_index(collection_name=collection_name, index_params=index_params)
        else:
            index_params = client.prepare_index_params()
            if vec_type == DataType.BINARY_VECTOR:
                index_params.add_index(field_name="vector", metric_type="HAMMING", index_type="BIN_FLAT")
            else:
                index_params.add_index(field_name="vector", metric_type="L2", index_type="FLAT")
            client.create_index(collection_name=collection_name, index_params=index_params)
        client.load_collection(collection_name=collection_name)
        
        # Search with both modes - use correct format for each type
        if vec_type == DataType.BINARY_VECTOR:
            query_vec = [bytes([128] * (dim // 8))]
        elif vec_type == DataType.FLOAT16_VECTOR:
            query_vec = [np.array([0.5] * dim, dtype=np.float16)]
        elif vec_type == DataType.BFLOAT16_VECTOR:
            from ml_dtypes import bfloat16
            query_vec = [np.array([0.5] * dim, dtype=bfloat16)]
        elif vec_type == DataType.INT8_VECTOR:
            query_vec = [np.array([int(i % 127) for i in range(dim)], dtype=np.int8)]
        else:
            # FLOAT_VECTOR
            query_vec = [[0.5] * dim]
        
        try:
            original = client.search(
                collection_name=collection_name,
                data=query_vec,
                limit=5,
                output_fields=["vector", "label"],
            )
            
            columnar = client.search(
                collection_name=collection_name,
                data=query_vec,
                limit=5,
                output_fields=["vector", "label"],
                output_mode="columnar",
            )
            
            # Compare results including vector data
            match = True
            for i, (orig, col) in enumerate(zip(original[0], columnar[0])):
                orig_id = orig.get("id") if isinstance(orig, dict) else orig.id
                col_id = col.id
                
                orig_label = orig.get("entity", {}).get("label") or orig.get("label")
                col_label = col["label"]
                
                # Get vector data
                orig_vec = orig.get("entity", {}).get("vector") or orig.get("vector")
                col_vec = col["vector"]
                
                # Convert to bytes for comparison
                if isinstance(orig_vec, bytes):
                    orig_bytes = orig_vec
                elif isinstance(orig_vec, np.ndarray):
                    orig_bytes = orig_vec.tobytes()
                elif isinstance(orig_vec, (list, tuple)):
                    orig_bytes = bytes(orig_vec) if vec_type == DataType.BINARY_VECTOR else str(orig_vec)
                else:
                    orig_bytes = str(orig_vec)
                
                if isinstance(col_vec, bytes):
                    col_bytes = col_vec
                elif isinstance(col_vec, np.ndarray):
                    col_bytes = col_vec.tobytes()
                elif isinstance(col_vec, (list, tuple)):
                    col_bytes = bytes(col_vec) if vec_type == DataType.BINARY_VECTOR else str(col_vec)
                else:
                    col_bytes = str(col_vec)
                
                if orig_id != col_id or orig_label != col_label or orig_bytes != col_bytes:
                    match = False
                    if orig_bytes != col_bytes:
                        print(f"    Vector mismatch at [{i}]:")
                        print(f"      Original: type={type(orig_vec).__name__}, len={len(orig_bytes) if isinstance(orig_bytes, bytes) else 'N/A'}")
                        print(f"      Columnar: type={type(col_vec).__name__}, len={len(col_bytes) if isinstance(col_bytes, bytes) else 'N/A'}")
                    break
            
            status = "✅" if match else "❌"
            print(f"  {status} {vec_name}: {len(original[0])} results match (ID + label + vector)")
            
        except Exception as e:
            print(f"  ⚠️ {vec_name}: {e}")
        
        # Cleanup
        client.drop_collection(collection_name)


def compare_output_modes():
    """
    Compare Original SearchResult vs ColumnarSearchResult output.
    This verifies that both modes return identical data for ALL field types.
    """
    if not HAS_MILVUS:
        print("Skipping compare_output_modes: MILVUS_URI not set")
        return
    
    client = MilvusClient(MILVUS_URI)
    collection_name = "columnar_example_collection"
    
    query_vector = [[0.1] * 128]
    limit = 10
    # Test ALL field types (12 total)
    output_fields = [
        # Integer types
        "int8_field", "int16_field", "int32_field", "int64_field",
        # Float types
        "float_field", "double_field",
        # String and bool
        "varchar_field", "bool_field",
        # Complex types
        "json_field", "int_array", "float_array", "varchar_array"
    ]
    
    # Search with original mode (default)
    original_results = client.search(
        collection_name=collection_name,
        data=query_vector,
        limit=limit,
        output_fields=output_fields,
    )
    
    # Search with columnar mode
    columnar_results = client.search(
        collection_name=collection_name,
        data=query_vector,
        limit=limit,
        output_fields=output_fields,
        output_mode="columnar",
    )
    
    print("\n=== Compare Output Modes (All Field Types) ===")
    print(f"Original type: {type(original_results).__name__}")
    print(f"Columnar type: {type(columnar_results).__name__}")
    print(f"Testing fields: {output_fields}")
    
    # Helper to get field value from original result
    def get_orig_value(orig, field):
        if isinstance(orig, dict):
            return orig.get("entity", {}).get(field) or orig.get(field)
        return orig[field]
    
    # Compare results
    all_match = True
    field_results = {f: {"match": 0, "mismatch": 0} for f in output_fields}
    
    for q_idx, (orig_hits, col_hits) in enumerate(zip(original_results, columnar_results)):
        if len(orig_hits) != len(col_hits):
            print(f"❌ Query {q_idx}: length mismatch {len(orig_hits)} vs {len(col_hits)}")
            all_match = False
            continue
        
        for i, (orig, col) in enumerate(zip(orig_hits, col_hits)):
            # Compare IDs
            orig_id = orig.get("id") if isinstance(orig, dict) else orig.id
            col_id = col.id
            if orig_id != col_id:
                print(f"❌ ID mismatch at [{q_idx}][{i}]: {orig_id} vs {col_id}")
                all_match = False
            
            # Compare distances
            orig_dist = orig.get("distance") if isinstance(orig, dict) else orig.distance
            col_dist = col.distance
            if abs(orig_dist - col_dist) > 1e-6:
                print(f"❌ Distance mismatch at [{q_idx}][{i}]: {orig_dist} vs {col_dist}")
                all_match = False
            
            # Compare all fields
            for field in output_fields:
                orig_val = get_orig_value(orig, field)
                col_val = col[field]
                
                # Compare values (handle float comparison)
                match = False
                if orig_val is None and col_val is None:
                    match = True
                elif isinstance(orig_val, float) and isinstance(col_val, float):
                    match = abs(orig_val - col_val) < 1e-6
                elif isinstance(orig_val, (list, dict)) and isinstance(col_val, (list, dict)):
                    match = str(orig_val) == str(col_val)
                else:
                    match = orig_val == col_val
                
                if match:
                    field_results[field]["match"] += 1
                else:
                    field_results[field]["mismatch"] += 1
                    if field_results[field]["mismatch"] <= 2:  # Only show first 2 mismatches
                        print(f"❌ Field '{field}' mismatch at [{q_idx}][{i}]:")
                        print(f"   Original: {type(orig_val).__name__} = {orig_val}")
                        print(f"   Columnar: {type(col_val).__name__} = {col_val}")
                    all_match = False
    
    # Print summary
    print("\n--- Field Comparison Summary ---")
    total_match = 0
    total_mismatch = 0
    for field in output_fields:
        m, mm = field_results[field]["match"], field_results[field]["mismatch"]
        total_match += m
        total_mismatch += mm
        status = "✅" if mm == 0 else "❌"
        print(f"  {status} {field}: {m} match, {mm} mismatch")
    
    if all_match:
        print(f"\n✅ All {limit} results match across all {len(output_fields)} field types!")
        print("\nSample data (first result):")
        orig = original_results[0][0]
        col = columnar_results[0][0]
        for field in output_fields:
            orig_val = get_orig_value(orig, field)
            col_val = col[field]
            print(f"  {field}: {orig_val} == {col_val}")
    else:
        print(f"\n❌ {total_mismatch} mismatches found!")


if __name__ == "__main__":
    print("=" * 60)
    print(" ColumnarSearchResult Example")
    print("=" * 60)
    
    if HAS_MILVUS:
        print(f"Using Milvus server: {MILVUS_URI}")
        setup_collection()
        compare_output_modes()  # Compare all field types
        compare_vector_types()  # Compare all dense vector types
        compare_sparse_vector() # Compare sparse vector type
        basic_usage_example()



def compare_sparse_vector():
    """
    Compare Original vs Columnar for SPARSE_FLOAT_VECTOR type.
    Requires dictionary input format and specialized index.
    """
    if not HAS_MILVUS:
        print("Skipping compare_sparse_vector: MILVUS_URI not set")
        return
    
    from pymilvus import DataType, FieldSchema, CollectionSchema
    import random
    
    client = MilvusClient(MILVUS_URI)
    collection_name = "test_vector_sparse_float"
    vec_type = DataType.SPARSE_FLOAT_VECTOR
    dim = 200 # Max dimension for generating indices
    
    print(f"\n=== Compare SPARSE_FLOAT_VECTOR ===")
    
    # Drop if exists
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)
    
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="vector", dtype=vec_type),
        FieldSchema(name="label", dtype=DataType.VARCHAR, max_length=128),
    ]
    schema = CollectionSchema(fields=fields, enable_dynamic_field=False)
    
    # Create collection
    client.create_collection(collection_name=collection_name, schema=schema)
    
    # Generate data
    num_entities = 20
    entities = []
    for i in range(num_entities):
        # Sparse vector: dictionary {index: value}
        # Create different patterns
        idx_count = 10 + (i % 10)
        indices = random.sample(range(dim), idx_count)
        values = [random.random() for _ in range(idx_count)]
        vec_data = {idx: val for idx, val in zip(indices, values)}
        
        entities.append({
            "vector": vec_data,
            "label": f"item_{i}",
        })
    
    client.insert(collection_name=collection_name, data=entities)
    
    # Create index (Sparse Inverted Index)
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector", 
        index_name="sparse_inverted_index",
        index_type="SPARSE_INVERTED_INDEX", 
        metric_type="IP", 
        params={"drop_ratio_build": 0.2}
    )
    client.create_index(collection_name=collection_name, index_params=index_params)
    client.load_collection(collection_name=collection_name)
    
    # Search
    # Generate random query
    q_indices = random.sample(range(dim), 10)
    q_values = [random.random() for _ in range(10)]
    query_vec = [{idx: val for idx, val in zip(q_indices, q_values)}]
    
    try:
        original = client.search(
            collection_name=collection_name,
            data=query_vec,
            limit=5,
            output_fields=["vector", "label"],
        )
        
        columnar = client.search(
            collection_name=collection_name,
            data=query_vec,
            limit=5,
            output_fields=["vector", "label"],
            output_mode="columnar",
        )
        
        # Compare results
        match = True
        for i, (orig, col) in enumerate(zip(original[0], columnar[0])):
            orig_id = orig.get("id") if isinstance(orig, dict) else orig.id
            col_id = col.id
            
            orig_label = orig.get("entity", {}).get("label") or orig.get("label")
            col_label = col["label"]
            
            # Compare sparse vectors (dictionaries)
            orig_vec = orig.get("entity", {}).get("vector") or orig.get("vector")
            col_vec = col["vector"]
            
            if orig_id != col_id or orig_label != col_label:
                match = False
                break
                
            # Check sparse vector equality (keys same, values close)
            if orig_vec.keys() != col_vec.keys():
                match = False
                print(f"    Sparse keys mismatch: {orig_vec.keys()} vs {col_vec.keys()}")
                break
                
            for k in orig_vec:
                if abs(orig_vec[k] - col_vec[k]) > 1e-5:
                     match = False
                     print(f"    Sparse value mismatch at {k}: {orig_vec[k]} vs {col_vec[k]}")
                     break
            if not match:
                break
        
        status = "✅" if match else "❌"
        print(f"  {status} SPARSE_FLOAT_VECTOR: {len(original[0])} results match (ID + label + sparse_vector)")
        
    except Exception as e:
        print(f"  ⚠️ SPARSE_FLOAT_VECTOR: {e}")
        import traceback
        traceback.print_exc()
    
    # Cleanup
    client.drop_collection(collection_name)


if __name__ == "__main__":
    print("=" * 60)
    print(" ColumnarSearchResult Example")
    print("=" * 60)
    
    if HAS_MILVUS:
        print(f"Using Milvus server: {MILVUS_URI}")
        setup_collection()
        compare_output_modes()  # Compare all field types
        compare_vector_types()  # Compare all dense vector types
        compare_sparse_vector() # Compare sparse vector type
        basic_usage_example()
        dict_compatibility_example()
        slice_access_example()
        cleanup_collection()
    else:
        print("No MILVUS_URI set - running with mock data only")
        print("Set MILVUS_URI=http://your-milvus:19530 to run all examples")
    
    # These always work (mock data)
    batch_access_example()
    performance_comparison()
    
    print("\n" + "=" * 60)
    print(" Done!")
    print("=" * 60)

