"""
Comprehensive unit tests for ColumnarSearchResult.

Tests all supported data types and ensures compatibility with SearchResult.
"""
import pytest
from typing import List, Any
from pymilvus.grpc_gen import schema_pb2
from pymilvus.client.search_result import SearchResult
from pymilvus.client.columnar_search_result import (
    ColumnarSearchResult,
    ColumnarHits,
    RowProxy,
)


# ==============================================================================
# Test Fixtures and Mock Data Builders
# ==============================================================================

def build_base_result(nq: int = 2, topk: int = 5) -> schema_pb2.SearchResultData:
    """Create a base SearchResultData with IDs and scores."""
    total = nq * topk
    res = schema_pb2.SearchResultData()
    res.num_queries = nq
    res.top_k = topk
    res.topks.extend([topk] * nq)
    res.ids.int_id.data.extend(list(range(total)))
    res.scores.extend([float(i) * 0.1 for i in range(total)])
    res.primary_field_name = "id"
    return res


def add_bool_field(res: schema_pb2.SearchResultData, name: str = "bool_field"):
    """Add BOOL field to mock data."""
    total = sum(res.topks)
    field = res.fields_data.add()
    field.field_name = name
    field.type = schema_pb2.DataType.Bool
    field.scalars.bool_data.data.extend([i % 2 == 0 for i in range(total)])
    res.output_fields.append(name)


def add_int8_field(res: schema_pb2.SearchResultData, name: str = "int8_field"):
    """Add INT8 field to mock data."""
    total = sum(res.topks)
    field = res.fields_data.add()
    field.field_name = name
    field.type = schema_pb2.DataType.Int8
    field.scalars.int_data.data.extend([i % 128 for i in range(total)])
    res.output_fields.append(name)


def add_int16_field(res: schema_pb2.SearchResultData, name: str = "int16_field"):
    """Add INT16 field to mock data."""
    total = sum(res.topks)
    field = res.fields_data.add()
    field.field_name = name
    field.type = schema_pb2.DataType.Int16
    field.scalars.int_data.data.extend([i * 100 for i in range(total)])
    res.output_fields.append(name)


def add_int32_field(res: schema_pb2.SearchResultData, name: str = "int32_field"):
    """Add INT32 field to mock data."""
    total = sum(res.topks)
    field = res.fields_data.add()
    field.field_name = name
    field.type = schema_pb2.DataType.Int32
    field.scalars.int_data.data.extend([i * 1000 for i in range(total)])
    res.output_fields.append(name)


def add_int64_field(res: schema_pb2.SearchResultData, name: str = "int64_field"):
    """Add INT64 field to mock data."""
    total = sum(res.topks)
    field = res.fields_data.add()
    field.field_name = name
    field.type = schema_pb2.DataType.Int64
    field.scalars.long_data.data.extend([i * 10000 for i in range(total)])
    res.output_fields.append(name)


def add_float_field(res: schema_pb2.SearchResultData, name: str = "float_field"):
    """Add FLOAT field to mock data."""
    total = sum(res.topks)
    field = res.fields_data.add()
    field.field_name = name
    field.type = schema_pb2.DataType.Float
    field.scalars.float_data.data.extend([float(i) * 0.1 for i in range(total)])
    res.output_fields.append(name)


def add_double_field(res: schema_pb2.SearchResultData, name: str = "double_field"):
    """Add DOUBLE field to mock data."""
    total = sum(res.topks)
    field = res.fields_data.add()
    field.field_name = name
    field.type = schema_pb2.DataType.Double
    field.scalars.double_data.data.extend([float(i) * 0.001 for i in range(total)])
    res.output_fields.append(name)


def add_varchar_field(res: schema_pb2.SearchResultData, name: str = "varchar_field"):
    """Add VARCHAR field to mock data."""
    total = sum(res.topks)
    field = res.fields_data.add()
    field.field_name = name
    field.type = schema_pb2.DataType.VarChar
    field.scalars.string_data.data.extend([f"str_{i}" for i in range(total)])
    res.output_fields.append(name)


def add_json_field(res: schema_pb2.SearchResultData, name: str = "json_field"):
    """Add JSON field to mock data."""
    total = sum(res.topks)
    field = res.fields_data.add()
    field.field_name = name
    field.type = schema_pb2.DataType.JSON
    for i in range(total):
        field.scalars.json_data.data.append(f'{{"value": {i}, "name": "item_{i}"}}'.encode())
    res.output_fields.append(name)


def add_array_int64_field(res: schema_pb2.SearchResultData, name: str = "array_int64"):
    """Add ARRAY<INT64> field to mock data."""
    total = sum(res.topks)
    field = res.fields_data.add()
    field.field_name = name
    field.type = schema_pb2.DataType.Array
    field.scalars.array_data.element_type = schema_pb2.DataType.Int64
    for i in range(total):
        arr = field.scalars.array_data.data.add()
        arr.long_data.data.extend([i, i + 1, i + 2])
    res.output_fields.append(name)


def add_array_varchar_field(res: schema_pb2.SearchResultData, name: str = "array_varchar"):
    """Add ARRAY<VARCHAR> field to mock data."""
    total = sum(res.topks)
    field = res.fields_data.add()
    field.field_name = name
    field.type = schema_pb2.DataType.Array
    field.scalars.array_data.element_type = schema_pb2.DataType.VarChar
    for i in range(total):
        arr = field.scalars.array_data.data.add()
        arr.string_data.data.extend([f"a{i}", f"b{i}"])
    res.output_fields.append(name)


def add_float_vector_field(res: schema_pb2.SearchResultData, dim: int = 4, name: str = "float_vector"):
    """Add FLOAT_VECTOR field to mock data."""
    total = sum(res.topks)
    field = res.fields_data.add()
    field.field_name = name
    field.type = schema_pb2.DataType.FloatVector
    field.vectors.dim = dim
    for i in range(total * dim):
        field.vectors.float_vector.data.append(float(i) * 0.01)
    res.output_fields.append(name)


def add_binary_vector_field(res: schema_pb2.SearchResultData, dim: int = 32, name: str = "binary_vector"):
    """Add BINARY_VECTOR field to mock data (dim is in bits)."""
    total = sum(res.topks)
    bytes_per_vec = dim // 8
    field = res.fields_data.add()
    field.field_name = name
    field.type = schema_pb2.DataType.BinaryVector
    field.vectors.dim = dim
    field.vectors.binary_vector = bytes([i % 256 for i in range(total * bytes_per_vec)])
    res.output_fields.append(name)


def add_float16_vector_field(res: schema_pb2.SearchResultData, dim: int = 4, name: str = "float16_vector"):
    """Add FLOAT16_VECTOR field to mock data."""
    total = sum(res.topks)
    bytes_per_vec = dim * 2
    field = res.fields_data.add()
    field.field_name = name
    field.type = schema_pb2.DataType.Float16Vector
    field.vectors.dim = dim
    field.vectors.float16_vector = bytes([i % 256 for i in range(total * bytes_per_vec)])
    res.output_fields.append(name)


def add_bfloat16_vector_field(res: schema_pb2.SearchResultData, dim: int = 4, name: str = "bfloat16_vector"):
    """Add BFLOAT16_VECTOR field to mock data."""
    total = sum(res.topks)
    bytes_per_vec = dim * 2
    field = res.fields_data.add()
    field.field_name = name
    field.type = schema_pb2.DataType.BFloat16Vector
    field.vectors.dim = dim
    field.vectors.bfloat16_vector = bytes([i % 256 for i in range(total * bytes_per_vec)])
    res.output_fields.append(name)


def add_int8_vector_field(res: schema_pb2.SearchResultData, dim: int = 4, name: str = "int8_vector"):
    """Add INT8_VECTOR field to mock data."""
    total = sum(res.topks)
    field = res.fields_data.add()
    field.field_name = name
    field.type = schema_pb2.DataType.Int8Vector
    field.vectors.dim = dim
    field.vectors.int8_vector = bytes([i % 256 for i in range(total * dim)])
    res.output_fields.append(name)


def add_sparse_vector_field(res: schema_pb2.SearchResultData, name: str = "sparse_vector"):
    """Add SPARSE_FLOAT_VECTOR field to mock data."""
    import struct
    total = sum(res.topks)
    field = res.fields_data.add()
    field.field_name = name
    field.type = schema_pb2.DataType.SparseFloatVector
    # Create sparse vectors in the format expected by Milvus
    for i in range(total):
        # Simple sparse vector: {0: 0.1, 1: 0.2}
        indices = [0, 1]
        values = [0.1, 0.2]
        content = struct.pack(f"<{len(indices)}I{len(values)}f", *indices, *values)
        field.vectors.sparse_float_vector.contents.append(content)
    res.output_fields.append(name)


def add_dynamic_field(res: schema_pb2.SearchResultData):
    """Add $meta field for dynamic fields."""
    total = sum(res.topks)
    field = res.fields_data.add()
    field.field_name = "$meta"
    field.type = schema_pb2.DataType.JSON
    field.is_dynamic = True
    for i in range(total):
        field.scalars.json_data.data.append(f'{{"dyn_field": {i * 100}}}'.encode())


# ==============================================================================
# Test Classes
# ==============================================================================

class TestColumnarSearchResultBasic:
    """Basic functionality tests."""

    def test_empty_result(self):
        """Test empty search result."""
        res = schema_pb2.SearchResultData()
        res.num_queries = 0
        res.top_k = 0
        cr = ColumnarSearchResult(res)
        assert len(cr) == 0

    def test_single_query_single_result(self):
        """Test single query with single result."""
        res = build_base_result(nq=1, topk=1)
        cr = ColumnarSearchResult(res)
        assert len(cr) == 1
        assert len(cr[0]) == 1
        assert cr[0][0].id == 0

    def test_multiple_queries(self):
        """Test multiple queries."""
        res = build_base_result(nq=3, topk=5)
        cr = ColumnarSearchResult(res)
        assert len(cr) == 3
        for hits in cr:
            assert len(hits) == 5

    def test_uneven_topk(self):
        """Test when different queries have different topk."""
        res = schema_pb2.SearchResultData()
        res.num_queries = 3
        res.top_k = 5
        res.topks.extend([5, 3, 2])  # Different topk per query
        res.ids.int_id.data.extend(list(range(10)))
        res.scores.extend([float(i) for i in range(10)])
        res.primary_field_name = "id"
        
        cr = ColumnarSearchResult(res)
        assert len(cr) == 3
        assert len(cr[0]) == 5
        assert len(cr[1]) == 3
        assert len(cr[2]) == 2


class TestColumnarSearchResultScalarTypes:
    """Tests for all scalar data types."""

    def test_bool_field(self):
        """Test BOOL field type."""
        res = build_base_result(nq=1, topk=3)
        add_bool_field(res)
        
        sr = SearchResult(res)
        cr = ColumnarSearchResult(res)
        
        for i in range(3):
            assert sr[0][i]["bool_field"] == cr[0][i]["bool_field"]

    def test_int8_field(self):
        """Test INT8 field type."""
        res = build_base_result(nq=1, topk=3)
        add_int8_field(res)
        
        sr = SearchResult(res)
        cr = ColumnarSearchResult(res)
        
        for i in range(3):
            assert sr[0][i]["int8_field"] == cr[0][i]["int8_field"]

    def test_int16_field(self):
        """Test INT16 field type."""
        res = build_base_result(nq=1, topk=3)
        add_int16_field(res)
        
        sr = SearchResult(res)
        cr = ColumnarSearchResult(res)
        
        for i in range(3):
            assert sr[0][i]["int16_field"] == cr[0][i]["int16_field"]

    def test_int32_field(self):
        """Test INT32 field type."""
        res = build_base_result(nq=1, topk=3)
        add_int32_field(res)
        
        sr = SearchResult(res)
        cr = ColumnarSearchResult(res)
        
        for i in range(3):
            assert sr[0][i]["int32_field"] == cr[0][i]["int32_field"]

    def test_int64_field(self):
        """Test INT64 field type."""
        res = build_base_result(nq=1, topk=3)
        add_int64_field(res)
        
        sr = SearchResult(res)
        cr = ColumnarSearchResult(res)
        
        for i in range(3):
            assert sr[0][i]["int64_field"] == cr[0][i]["int64_field"]

    def test_float_field(self):
        """Test FLOAT field type."""
        res = build_base_result(nq=1, topk=3)
        add_float_field(res)
        
        sr = SearchResult(res)
        cr = ColumnarSearchResult(res)
        
        for i in range(3):
            assert sr[0][i]["float_field"] == cr[0][i]["float_field"]

    def test_double_field(self):
        """Test DOUBLE field type."""
        res = build_base_result(nq=1, topk=3)
        add_double_field(res)
        
        sr = SearchResult(res)
        cr = ColumnarSearchResult(res)
        
        for i in range(3):
            assert sr[0][i]["double_field"] == cr[0][i]["double_field"]

    def test_varchar_field(self):
        """Test VARCHAR field type."""
        res = build_base_result(nq=1, topk=3)
        add_varchar_field(res)
        
        sr = SearchResult(res)
        cr = ColumnarSearchResult(res)
        
        for i in range(3):
            assert sr[0][i]["varchar_field"] == cr[0][i]["varchar_field"]

    def test_json_field(self):
        """Test JSON field type."""
        res = build_base_result(nq=1, topk=3)
        add_json_field(res)
        
        sr = SearchResult(res)
        cr = ColumnarSearchResult(res)
        
        for i in range(3):
            sr_val = sr[0][i]["json_field"]
            cr_val = cr[0][i]["json_field"]
            assert type(sr_val) == type(cr_val) == dict
            assert sr_val == cr_val


class TestColumnarSearchResultArrayTypes:
    """Tests for ARRAY data types."""

    def test_array_int64_field(self):
        """Test ARRAY<INT64> field type."""
        res = build_base_result(nq=1, topk=3)
        add_array_int64_field(res)
        
        sr = SearchResult(res)
        cr = ColumnarSearchResult(res)
        
        for i in range(3):
            sr_val = list(sr[0][i]["array_int64"])  # Convert to list for comparison
            cr_val = cr[0][i]["array_int64"]
            assert sr_val == cr_val

    def test_array_varchar_field(self):
        """Test ARRAY<VARCHAR> field type."""
        res = build_base_result(nq=1, topk=3)
        add_array_varchar_field(res)
        
        sr = SearchResult(res)
        cr = ColumnarSearchResult(res)
        
        for i in range(3):
            sr_val = list(sr[0][i]["array_varchar"])
            cr_val = cr[0][i]["array_varchar"]
            assert sr_val == cr_val


class TestColumnarSearchResultVectorTypes:
    """Tests for all vector data types."""

    def test_float_vector_field(self):
        """Test FLOAT_VECTOR field type - should return list."""
        res = build_base_result(nq=1, topk=3)
        add_float_vector_field(res, dim=4)
        
        sr = SearchResult(res)
        cr = ColumnarSearchResult(res)
        
        for i in range(3):
            sr_val = sr[0][i]["float_vector"]
            cr_val = cr[0][i]["float_vector"]
            # Both should be list type
            assert isinstance(cr_val, list)
            assert list(sr_val) == cr_val

    def test_binary_vector_field(self):
        """Test BINARY_VECTOR field type - should return bytes."""
        res = build_base_result(nq=1, topk=3)
        add_binary_vector_field(res, dim=32)
        
        sr = SearchResult(res)
        cr = ColumnarSearchResult(res)
        
        for i in range(3):
            sr_val = sr[0][i]["binary_vector"]
            cr_val = cr[0][i]["binary_vector"]
            assert type(sr_val) == type(cr_val) == bytes
            assert sr_val == cr_val

    def test_float16_vector_field(self):
        """Test FLOAT16_VECTOR field type - should return bytes."""
        res = build_base_result(nq=1, topk=3)
        add_float16_vector_field(res, dim=4)
        
        sr = SearchResult(res)
        cr = ColumnarSearchResult(res)
        
        for i in range(3):
            sr_val = sr[0][i]["float16_vector"]
            cr_val = cr[0][i]["float16_vector"]
            assert type(sr_val) == type(cr_val) == bytes
            assert sr_val == cr_val

    def test_bfloat16_vector_field(self):
        """Test BFLOAT16_VECTOR field type - should return bytes."""
        res = build_base_result(nq=1, topk=3)
        add_bfloat16_vector_field(res, dim=4)
        
        sr = SearchResult(res)
        cr = ColumnarSearchResult(res)
        
        for i in range(3):
            sr_val = sr[0][i]["bfloat16_vector"]
            cr_val = cr[0][i]["bfloat16_vector"]
            assert type(sr_val) == type(cr_val) == bytes
            assert sr_val == cr_val

    def test_int8_vector_field(self):
        """Test INT8_VECTOR field type - should return bytes."""
        res = build_base_result(nq=1, topk=3)
        add_int8_vector_field(res, dim=4)
        
        sr = SearchResult(res)
        cr = ColumnarSearchResult(res)
        
        for i in range(3):
            sr_val = sr[0][i]["int8_vector"]
            cr_val = cr[0][i]["int8_vector"]
            assert type(sr_val) == type(cr_val) == bytes
            assert sr_val == cr_val


class TestColumnarHits:
    """Tests for ColumnarHits class."""

    def test_length(self):
        """Test ColumnarHits length."""
        res = build_base_result(nq=2, topk=10)
        cr = ColumnarSearchResult(res)
        
        assert len(cr) == 2
        assert len(cr[0]) == 10
        assert len(cr[1]) == 10

    def test_iteration(self):
        """Test iteration over ColumnarHits."""
        res = build_base_result(nq=1, topk=5)
        cr = ColumnarSearchResult(res)
        
        hits = cr[0]
        count = 0
        for hit in hits:
            assert isinstance(hit, RowProxy)
            count += 1
        assert count == 5

    def test_ids_and_distances(self):
        """Test that ColumnarHits exposes ids and distances properties."""
        res = build_base_result(nq=2, topk=5)
        cr = ColumnarSearchResult(res)
        
        hits = cr[0]
        assert len(hits.ids) == 5
        assert len(hits.distances) == 5
        assert hits.ids == list(range(5))

    def test_slice_access(self):
        """Test slice access on ColumnarHits."""
        res = build_base_result(nq=1, topk=10)
        cr = ColumnarSearchResult(res)
        
        hits = cr[0]
        sliced = hits[2:5]
        assert len(sliced) == 3
        assert all(isinstance(h, RowProxy) for h in sliced)
        assert sliced[0].id == 2
        assert sliced[2].id == 4

    def test_negative_index(self):
        """Test negative indexing on ColumnarHits."""
        res = build_base_result(nq=1, topk=5)
        cr = ColumnarSearchResult(res)
        
        hits = cr[0]
        assert hits[-1].id == 4
        assert hits[-2].id == 3


class TestRowProxy:
    """Tests for RowProxy class."""

    def test_basic_access(self):
        """Test basic field access."""
        res = build_base_result(nq=1, topk=3)
        add_int64_field(res)
        cr = ColumnarSearchResult(res)
        
        hit = cr[0][0]
        assert hit.id == 0
        assert hit.distance == 0.0
        assert hit["int64_field"] == 0

    def test_dict_like_interface(self):
        """Test dict-like interface."""
        res = build_base_result(nq=1, topk=3)
        add_int64_field(res)
        add_varchar_field(res)
        cr = ColumnarSearchResult(res)
        
        hit = cr[0][0]
        
        # keys()
        keys = hit.keys()
        assert "int64_field" in keys
        assert "varchar_field" in keys
        
        # __contains__
        assert "int64_field" in hit
        assert "nonexistent" not in hit
        
        # get()
        assert hit.get("int64_field") == 0
        assert hit.get("nonexistent", "default") == "default"
        
        # items()
        items = dict(hit.items())
        assert "int64_field" in items
        assert "varchar_field" in items

    def test_entity_access(self):
        """Test entity property for nested access."""
        res = build_base_result(nq=1, topk=3)
        add_int64_field(res)
        cr = ColumnarSearchResult(res)
        
        hit = cr[0][0]
        # hit.entity should return self for compatibility
        assert hit.entity["int64_field"] == 0
        assert hit["entity"]["int64_field"] == 0

    def test_to_dict(self):
        """Test to_dict() method."""
        res = build_base_result(nq=1, topk=3)
        add_int64_field(res)
        add_varchar_field(res)
        cr = ColumnarSearchResult(res)
        
        hit = cr[0][0]
        d = hit.to_dict()
        
        assert d["id"] == 0
        assert d["distance"] == 0.0
        assert "entity" in d
        assert d["entity"]["int64_field"] == 0
        assert d["entity"]["varchar_field"] == "str_0"

    def test_read_only(self):
        """Test that RowProxy is read-only."""
        res = build_base_result(nq=1, topk=3)
        add_int64_field(res)
        cr = ColumnarSearchResult(res)
        
        hit = cr[0][0]
        with pytest.raises(TypeError):
            hit["int64_field"] = 999

    def test_properties(self):
        """Test id, distance, pk, score properties."""
        res = build_base_result(nq=1, topk=3)
        cr = ColumnarSearchResult(res)
        
        hit = cr[0][1]
        assert hit.id == 1
        assert hit.pk == 1
        assert hit.distance == pytest.approx(0.1, rel=1e-5)
        assert hit.score == pytest.approx(0.1, rel=1e-5)


class TestDynamicFields:
    """Tests for dynamic field support."""

    def test_dynamic_field_access(self):
        """Test accessing dynamic fields from $meta."""
        res = build_base_result(nq=1, topk=3)
        add_dynamic_field(res)
        res.output_fields.append("dyn_field")
        
        cr = ColumnarSearchResult(res)
        hit = cr[0][0]
        
        # Should be able to access dynamic field
        assert hit["dyn_field"] == 0

    def test_dynamic_field_in_keys(self):
        """Test that dynamic fields appear in keys()."""
        res = build_base_result(nq=1, topk=3)
        add_int64_field(res)
        add_dynamic_field(res)
        res.output_fields.append("dyn_field")
        
        cr = ColumnarSearchResult(res)
        hit = cr[0][0]
        
        keys = hit.keys()
        assert "int64_field" in keys
        assert "dyn_field" in keys


class TestCompatibilityWithSearchResult:
    """Tests ensuring full compatibility with SearchResult."""

    def test_iteration_pattern(self):
        """Test common iteration pattern."""
        res = build_base_result(nq=2, topk=5)
        add_int64_field(res)
        add_float_vector_field(res, dim=4)
        
        sr = SearchResult(res)
        cr = ColumnarSearchResult(res)
        
        sr_results = []
        for hits in sr:
            for hit in hits:
                sr_results.append((hit.id, hit.distance, hit["int64_field"]))
        
        cr_results = []
        for hits in cr:
            for hit in hits:
                cr_results.append((hit.id, hit.distance, hit["int64_field"]))
        
        assert sr_results == cr_results

    def test_all_scalar_types_together(self):
        """Test all scalar types in a single result."""
        res = build_base_result(nq=1, topk=3)
        add_bool_field(res)
        add_int8_field(res)
        add_int16_field(res)
        add_int32_field(res)
        add_int64_field(res)
        add_float_field(res)
        add_double_field(res)
        add_varchar_field(res)
        add_json_field(res)
        
        sr = SearchResult(res)
        cr = ColumnarSearchResult(res)
        
        for i in range(3):
            for field in res.output_fields:
                sr_val = sr[0][i][field]
                cr_val = cr[0][i][field]
                # Values should be equal (may need type conversion for some)
                if hasattr(sr_val, '__iter__') and not isinstance(sr_val, (str, dict)):
                    assert list(sr_val) == list(cr_val), f"Mismatch for {field}"
                else:
                    assert sr_val == cr_val, f"Mismatch for {field}"

    def test_all_vector_types_together(self):
        """Test all vector types in a single result."""
        res = build_base_result(nq=1, topk=3)
        add_float_vector_field(res, dim=4)
        add_binary_vector_field(res, dim=32)
        add_float16_vector_field(res, dim=4)
        add_bfloat16_vector_field(res, dim=4)
        add_int8_vector_field(res, dim=4)
        
        sr = SearchResult(res)
        cr = ColumnarSearchResult(res)
        
        for i in range(3):
            # FLOAT_VECTOR returns list
            sr_float = list(sr[0][i]["float_vector"])
            cr_float = cr[0][i]["float_vector"]
            assert sr_float == cr_float
            
            # Other vectors return bytes
            for field in ["binary_vector", "float16_vector", "bfloat16_vector", "int8_vector"]:
                sr_val = sr[0][i][field]
                cr_val = cr[0][i][field]
                assert sr_val == cr_val


class TestPerformance:
    """Performance-related tests (not measuring time, just ensuring efficiency)."""

    def test_initialization_creates_minimal_objects(self):
        """Test that initialization doesn't create many objects."""
        res = build_base_result(nq=100, topk=100)
        add_float_vector_field(res, dim=128)
        
        # This should complete quickly as it doesn't create 10,000 Hit objects
        cr = ColumnarSearchResult(res)
        
        assert len(cr) == 100
        assert len(cr[0]) == 100

    def test_materialize_is_noop(self):
        """Test that materialize() is a no-op."""
        res = build_base_result(nq=1, topk=5)
        cr = ColumnarSearchResult(res)
        
        # Should not raise any error
        cr.materialize()
        
        # Data should still be accessible
        assert cr[0][0].id == 0


class TestSpecialCases:
    """Tests for special/edge cases."""

    def test_string_primary_key(self):
        """Test with string primary key."""
        res = schema_pb2.SearchResultData()
        res.num_queries = 1
        res.top_k = 3
        res.topks.extend([3])
        res.ids.str_id.data.extend(["a", "b", "c"])
        res.scores.extend([0.1, 0.2, 0.3])
        res.primary_field_name = "pk"
        
        cr = ColumnarSearchResult(res)
        assert cr[0][0].id == "a"
        assert cr[0][1].id == "b"
        assert cr[0][2].id == "c"

    def test_round_decimal(self):
        """Test score rounding."""
        res = build_base_result(nq=1, topk=3)
        res.scores[:] = [0.123456, 0.234567, 0.345678]
        
        cr = ColumnarSearchResult(res, round_decimal=2)
        assert cr[0][0].distance == 0.12
        assert cr[0][1].distance == 0.23
        assert cr[0][2].distance == 0.35

    def test_recalls(self):
        """Test recalls attribute."""
        res = build_base_result(nq=2, topk=3)
        res.recalls.extend([0.95, 0.98])
        
        cr = ColumnarSearchResult(res)
        assert cr.recalls is not None
        recalls_list = list(cr.recalls)
        assert recalls_list[0] == pytest.approx(0.95, rel=1e-5)
        assert recalls_list[1] == pytest.approx(0.98, rel=1e-5)

    def test_extra_info(self):
        """Test extra info from status."""
        res = build_base_result(nq=1, topk=3)
        
        # Without status, extra should be empty
        cr = ColumnarSearchResult(res)
        assert cr.extra == {}
