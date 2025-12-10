"""
Unit tests for ColumnarSearchResult - testing data parsing correctness for all types.
This mirrors the tests in test_search_result.py but validates the columnar implementation.
"""

import logging
import os
import random

import numpy as np
import pytest
import orjson

from pymilvus.client.columnar_search_result import ColumnarSearchResult, ColumnarHits, RowProxy
from pymilvus.client.search_result import SearchResult
from pymilvus.client.types import DataType
from pymilvus.grpc_gen import schema_pb2

LOGGER = logging.getLogger(__name__)


class TestRowProxy:
    """Tests for the RowProxy class."""

    def test_row_proxy_id_and_distance(self):
        """Test that RowProxy correctly returns id and distance."""
        result = _create_simple_search_result(nq=1, topk=5)
        columnar = ColumnarSearchResult(result)
        
        hit = columnar[0][0]
        assert isinstance(hit, RowProxy)
        assert hit.id == 0
        assert hit.distance == 0.0
    
    def test_row_proxy_getitem(self):
        """Test RowProxy __getitem__ for field access."""
        result = _create_simple_search_result(nq=1, topk=5)
        columnar = ColumnarSearchResult(result)
        
        hit = columnar[0][2]
        assert hit["count"] == 2
    
    def test_row_proxy_get_with_default(self):
        """Test RowProxy.get() with default value."""
        result = _create_simple_search_result(nq=1, topk=5)
        columnar = ColumnarSearchResult(result)
        
        hit = columnar[0][0]
        assert hit.get("count") == 0
        assert hit.get("nonexistent", "default_value") == "default_value"
    
    def test_row_proxy_entity_property(self):
        """Test that hit.entity returns self for compatibility."""
        result = _create_simple_search_result(nq=1, topk=5)
        columnar = ColumnarSearchResult(result)
        
        hit = columnar[0][0]
        assert hit.entity is hit
        # This allows hit.entity.get("field") to work
        assert hit.entity.get("count") == 0
    
    def test_row_proxy_to_dict(self):
        """Test RowProxy.to_dict() materialization."""
        result = _create_simple_search_result(nq=1, topk=5)
        columnar = ColumnarSearchResult(result)
        
        hit = columnar[0][0]
        d = hit.to_dict()
        
        assert isinstance(d, dict)
        assert d["id"] == 0
        assert d["distance"] == 0.0
        assert "entity" in d
        assert d["entity"]["count"] == 0


class TestColumnarHits:
    """Tests for the ColumnarHits class."""
    
    def test_columnar_hits_length(self):
        """Test ColumnarHits length."""
        result = _create_simple_search_result(nq=2, topk=10)
        columnar = ColumnarSearchResult(result)
        
        assert len(columnar) == 2
        assert len(columnar[0]) == 10
        assert len(columnar[1]) == 10
    
    def test_columnar_hits_iteration(self):
        """Test iteration over ColumnarHits."""
        result = _create_simple_search_result(nq=1, topk=5)
        columnar = ColumnarSearchResult(result)
        
        hits = columnar[0]
        count = 0
        for hit in hits:
            assert isinstance(hit, RowProxy)
            assert hit.id == count
            count += 1
        assert count == 5
    
    def test_columnar_hits_ids_and_distances(self):
        """Test that ColumnarHits exposes ids and distances properties."""
        result = _create_simple_search_result(nq=1, topk=5)
        columnar = ColumnarSearchResult(result)
        
        hits = columnar[0]
        assert len(hits.ids) == 5
        assert len(hits.distances) == 5
        assert list(hits.ids) == [0, 1, 2, 3, 4]
    
    def test_columnar_hits_slice_access(self):
        """Test slice access on ColumnarHits."""
        result = _create_simple_search_result(nq=1, topk=10)
        columnar = ColumnarSearchResult(result)
        
        hits = columnar[0]
        sliced = hits[:3]
        
        assert len(sliced) == 3
        assert all(isinstance(h, RowProxy) for h in sliced)
        assert sliced[0].id == 0
        assert sliced[1].id == 1
        assert sliced[2].id == 2
    
    def test_columnar_hits_negative_index(self):
        """Test negative indexing on ColumnarHits."""
        result = _create_simple_search_result(nq=1, topk=5)
        columnar = ColumnarSearchResult(result)
        
        hits = columnar[0]
        assert hits[-1].id == 4
        assert hits[-2].id == 3


class TestRowProxyDictCompatibility:
    """Tests for RowProxy dict-like compatibility methods."""
    
    def test_row_proxy_keys(self):
        """Test RowProxy.keys() returns field names."""
        result = _create_simple_search_result(nq=1, topk=5)
        columnar = ColumnarSearchResult(result)
        
        hit = columnar[0][0]
        keys = hit.keys()
        
        assert "count" in keys
    
    def test_row_proxy_values(self):
        """Test RowProxy.values() returns field values."""
        result = _create_simple_search_result(nq=1, topk=5)
        columnar = ColumnarSearchResult(result)
        
        hit = columnar[0][0]
        values = hit.values()
        
        assert isinstance(values, list)
        assert 0 in values  # count field value
    
    def test_row_proxy_items(self):
        """Test RowProxy.items() returns (field, value) pairs."""
        result = _create_simple_search_result(nq=1, topk=5)
        columnar = ColumnarSearchResult(result)
        
        hit = columnar[0][0]
        items = hit.items()
        
        assert isinstance(items, list)
        assert all(isinstance(item, tuple) for item in items)
        assert ("count", 0) in items
    
    def test_row_proxy_contains(self):
        """Test 'field in hit' syntax."""
        result = _create_simple_search_result(nq=1, topk=5)
        columnar = ColumnarSearchResult(result)
        
        hit = columnar[0][0]
        assert "count" in hit
        assert "nonexistent" not in hit
    
    def test_row_proxy_iter(self):
        """Test iterating over RowProxy yields field names."""
        result = _create_simple_search_result(nq=1, topk=5)
        columnar = ColumnarSearchResult(result)
        
        hit = columnar[0][0]
        field_names = list(hit)
        
        assert "count" in field_names


class TestColumnarSearchResultDataTypes:
    """Tests for all data type parsing correctness."""
    
    @pytest.mark.parametrize("pk", [
        schema_pb2.IDs(int_id=schema_pb2.LongArray(data=list(range(6)))),
        schema_pb2.IDs(str_id=schema_pb2.StringArray(data=[str(i*10) for i in range(6)]))
    ])
    def test_no_fields_data(self, pk):
        """Test ColumnarSearchResult with no fields data."""
        result = schema_pb2.SearchResultData(
            num_queries=2,
            top_k=3,
            scores=[1.*i for i in range(6)],
            ids=pk,
            topks=[3, 3],
        )
        columnar = ColumnarSearchResult(result)
        
        assert len(columnar) == 2
        assert len(columnar[0]) == 3
        assert len(columnar[1]) == 3
    
    def test_bool_type(self):
        """Test BOOL data type parsing."""
        original, columnar = _create_paired_results_with_field(
            DataType.BOOL, "bool_field",
            lambda fd: fd.scalars.bool_data.data.extend([True, False, True, False, True, False])
        )
        _assert_field_values_match(original, columnar, "bool_field")
    
    def test_int8_type(self):
        """Test INT8 data type parsing."""
        original, columnar = _create_paired_results_with_field(
            DataType.INT8, "int8_field",
            lambda fd: fd.scalars.int_data.data.extend([1, 2, 3, 4, 5, 6])
        )
        _assert_field_values_match(original, columnar, "int8_field")
    
    def test_int16_type(self):
        """Test INT16 data type parsing."""
        original, columnar = _create_paired_results_with_field(
            DataType.INT16, "int16_field",
            lambda fd: fd.scalars.int_data.data.extend([100, 200, 300, 400, 500, 600])
        )
        _assert_field_values_match(original, columnar, "int16_field")
    
    def test_int32_type(self):
        """Test INT32 data type parsing."""
        original, columnar = _create_paired_results_with_field(
            DataType.INT32, "int32_field",
            lambda fd: fd.scalars.int_data.data.extend([1000, 2000, 3000, 4000, 5000, 6000])
        )
        _assert_field_values_match(original, columnar, "int32_field")
    
    def test_int64_type(self):
        """Test INT64 data type parsing."""
        original, columnar = _create_paired_results_with_field(
            DataType.INT64, "int64_field",
            lambda fd: fd.scalars.long_data.data.extend([10000, 20000, 30000, 40000, 50000, 60000])
        )
        _assert_field_values_match(original, columnar, "int64_field")
    
    def test_float_type(self):
        """Test FLOAT data type parsing."""
        original, columnar = _create_paired_results_with_field(
            DataType.FLOAT, "float_field",
            lambda fd: fd.scalars.float_data.data.extend([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
        )
        _assert_field_values_match(original, columnar, "float_field", rtol=1e-5)
    
    def test_double_type(self):
        """Test DOUBLE data type parsing."""
        original, columnar = _create_paired_results_with_field(
            DataType.DOUBLE, "double_field",
            lambda fd: fd.scalars.double_data.data.extend([1.111, 2.222, 3.333, 4.444, 5.555, 6.666])
        )
        _assert_field_values_match(original, columnar, "double_field", rtol=1e-9)
    
    def test_varchar_type(self):
        """Test VARCHAR data type parsing."""
        original, columnar = _create_paired_results_with_field(
            DataType.VARCHAR, "varchar_field",
            lambda fd: fd.scalars.string_data.data.extend(["a", "bb", "ccc", "dddd", "eeeee", "ffffff"])
        )
        _assert_field_values_match(original, columnar, "varchar_field")
    
    def test_json_type(self):
        """Test JSON data type parsing."""
        json_data = [orjson.dumps({"key": i, "value": f"val_{i}"}) for i in range(6)]
        
        original, columnar = _create_paired_results_with_field(
            DataType.JSON, "json_field",
            lambda fd: fd.scalars.json_data.data.extend(json_data)
        )
        
        # Compare JSON values
        for q in range(len(original)):
            for i in range(len(original[q])):
                orig_val = original[q][i].entity.get("json_field")
                col_val = columnar[q][i]["json_field"]
                assert orig_val == col_val, f"json_field mismatch at q={q}, i={i}"
    
    def test_float_vector_type(self):
        """Test FLOAT_VECTOR data type parsing."""
        dim = 4
        vectors = [random.random() for _ in range(6 * dim)]
        
        def setup_field(fd):
            fd.vectors.dim = dim
            fd.vectors.float_vector.data.extend(vectors)
        
        original, columnar = _create_paired_results_with_vector_field(
            DataType.FLOAT_VECTOR, "float_vector_field", setup_field
        )
        
        # Compare vectors
        for i in range(6):
            orig_vec = original[i // 3][i % 3].entity.get("float_vector_field")
            col_vec = columnar[i // 3][i % 3]["float_vector_field"]
            assert np.allclose(orig_vec, col_vec, rtol=1e-5), f"Vector mismatch at index {i}"
    
    def test_binary_vector_type(self):
        """Test BINARY_VECTOR data type parsing."""
        dim = 8  # 8 bits = 1 byte per vector
        binary_data = os.urandom(6)  # 6 vectors * 1 byte each
        
        def setup_field(fd):
            fd.vectors.dim = dim
            fd.vectors.binary_vector = binary_data
        
        original, columnar = _create_paired_results_with_vector_field(
            DataType.BINARY_VECTOR, "binary_vector_field", setup_field
        )
        
        # Compare binary vectors
        for i in range(6):
            orig_vec = original[i // 3][i % 3].entity.get("binary_vector_field")
            col_vec = columnar[i // 3][i % 3]["binary_vector_field"]
            # Both should be bytes of length 1
            assert len(col_vec) == dim // 8
    
    def test_float16_vector_type(self):
        """Test FLOAT16_VECTOR data type parsing with Phase 2 zero-copy."""
        dim = 4
        bytes_per_vector = dim * 2  # 2 bytes per float16
        float16_data = os.urandom(6 * bytes_per_vector)
        
        def setup_field(fd):
            fd.vectors.dim = dim
            fd.vectors.float16_vector = float16_data
        
        original, columnar = _create_paired_results_with_vector_field(
            DataType.FLOAT16_VECTOR, "float16_vector_field", setup_field
        )
        
        # Phase 2: columnar returns numpy arrays with element count, not byte count
        for i in range(6):
            col_vec = columnar[i // 3][i % 3]["float16_vector_field"]
            # With zero_copy_vectors=True, should return numpy array with dim elements
            assert isinstance(col_vec, np.ndarray), f"Expected numpy array, got {type(col_vec)}"
            assert len(col_vec) == dim, f"Expected {dim} elements, got {len(col_vec)}"
    
    def test_bfloat16_vector_type(self):
        """Test BFLOAT16_VECTOR data type parsing with Phase 2 zero-copy."""
        dim = 4
        bytes_per_vector = dim * 2
        bfloat16_data = os.urandom(6 * bytes_per_vector)
        
        def setup_field(fd):
            fd.vectors.dim = dim
            fd.vectors.bfloat16_vector = bfloat16_data
        
        original, columnar = _create_paired_results_with_vector_field(
            DataType.BFLOAT16_VECTOR, "bfloat16_vector_field", setup_field
        )
        
        # Phase 2: columnar returns numpy arrays with element count, not byte count
        for i in range(6):
            col_vec = columnar[i // 3][i % 3]["bfloat16_vector_field"]
            assert isinstance(col_vec, np.ndarray), f"Expected numpy array, got {type(col_vec)}"
            assert len(col_vec) == dim, f"Expected {dim} elements, got {len(col_vec)}"
    
    def test_int8_vector_type(self):
        """Test INT8_VECTOR data type parsing."""
        dim = 8
        int8_data = os.urandom(6 * dim)
        
        def setup_field(fd):
            fd.vectors.dim = dim
            fd.vectors.int8_vector = int8_data
        
        original, columnar = _create_paired_results_with_vector_field(
            DataType.INT8_VECTOR, "int8_vector_field", setup_field
        )
        
        for i in range(6):
            col_vec = columnar[i // 3][i % 3]["int8_vector_field"]
            assert len(col_vec) == dim
    
    def test_int64_array_type(self):
        """Test ARRAY of INT64 data type parsing."""
        fields_data = [
            schema_pb2.FieldData(
                type=DataType.ARRAY, 
                field_name="int64_array_field",
                scalars=schema_pb2.ScalarField(
                    array_data=schema_pb2.ArrayArray(
                        data=[schema_pb2.ScalarField(long_data=schema_pb2.LongArray(data=list(range(10)))) for _ in range(6)],
                        element_type=DataType.INT64,
                    ),
                )
            ),
        ]
        
        ids = schema_pb2.IDs(int_id=schema_pb2.LongArray(data=list(range(6))))
        
        result = schema_pb2.SearchResultData(
            fields_data=fields_data,
            num_queries=2,
            top_k=3,
            scores=[1.*i for i in range(6)],
            ids=ids,
            topks=[3, 3],
            output_fields=["int64_array_field"]
        )
        
        original = SearchResult(result)
        columnar = ColumnarSearchResult(result)
        
        for q in range(2):
            for i in range(3):
                orig_val = original[q][i].entity.get("int64_array_field")
                col_val = columnar[q][i]["int64_array_field"]
                assert orig_val == col_val, f"int64_array_field mismatch at q={q}, i={i}"
    
    def test_varchar_array_type(self):
        """Test ARRAY of VARCHAR data type parsing."""
        fields_data = [
            schema_pb2.FieldData(
                type=DataType.ARRAY, 
                field_name="varchar_array_field",
                scalars=schema_pb2.ScalarField(
                    array_data=schema_pb2.ArrayArray(
                        data=[schema_pb2.ScalarField(string_data=schema_pb2.StringArray(data=[f"item_{j}" for j in range(5)])) for _ in range(6)],
                        element_type=DataType.VARCHAR,
                    ),
                )
            ),
        ]
        
        ids = schema_pb2.IDs(int_id=schema_pb2.LongArray(data=list(range(6))))
        
        result = schema_pb2.SearchResultData(
            fields_data=fields_data,
            num_queries=2,
            top_k=3,
            scores=[1.*i for i in range(6)],
            ids=ids,
            topks=[3, 3],
            output_fields=["varchar_array_field"]
        )
        
        original = SearchResult(result)
        columnar = ColumnarSearchResult(result)
        
        for q in range(2):
            for i in range(3):
                orig_val = original[q][i].entity.get("varchar_array_field")
                col_val = columnar[q][i]["varchar_array_field"]
                assert orig_val == col_val, f"varchar_array_field mismatch at q={q}, i={i}"
    
    def test_geometry_type(self):
        """Test GEOMETRY data type parsing (WKT strings stored as bytes)."""
        wkt_data = [
            b"POINT(1.0 2.0)",
            b"POINT(3.0 4.0)",
            b"LINESTRING(0 0, 1 1)",
            b"POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))",
            b"POINT(5.0 6.0)",
            b"POINT(7.0 8.0)",
        ]
        
        fields_data = [
            schema_pb2.FieldData(
                type=DataType.GEOMETRY, 
                field_name="geometry_field",
                scalars=schema_pb2.ScalarField(
                    geometry_wkt_data=schema_pb2.GeometryWktArray(data=wkt_data)
                )
            ),
        ]
        
        ids = schema_pb2.IDs(int_id=schema_pb2.LongArray(data=list(range(6))))
        
        result = schema_pb2.SearchResultData(
            fields_data=fields_data,
            num_queries=2,
            top_k=3,
            scores=[1.*i for i in range(6)],
            ids=ids,
            topks=[3, 3],
            output_fields=["geometry_field"]
        )
        
        original = SearchResult(result)
        columnar = ColumnarSearchResult(result)
        
        for q in range(2):
            for i in range(3):
                orig_val = original[q][i].entity.get("geometry_field")
                col_val = columnar[q][i]["geometry_field"]
                assert orig_val == col_val, f"geometry_field mismatch at q={q}, i={i}"
    
    def test_timestamptz_type(self):
        """Test TIMESTAMPTZ data type parsing (stored as strings)."""
        timestamp_data = [
            "2024-01-01T00:00:00Z",
            "2024-01-02T00:00:00Z",
            "2024-01-03T00:00:00Z",
            "2024-01-04T00:00:00Z",
            "2024-01-05T00:00:00Z",
            "2024-01-06T00:00:00Z",
        ]
        
        fields_data = [
            schema_pb2.FieldData(
                type=DataType.TIMESTAMPTZ, 
                field_name="timestamp_field",
                scalars=schema_pb2.ScalarField(
                    string_data=schema_pb2.StringArray(data=timestamp_data)
                )
            ),
        ]
        
        ids = schema_pb2.IDs(int_id=schema_pb2.LongArray(data=list(range(6))))
        
        result = schema_pb2.SearchResultData(
            fields_data=fields_data,
            num_queries=2,
            top_k=3,
            scores=[1.*i for i in range(6)],
            ids=ids,
            topks=[3, 3],
            output_fields=["timestamp_field"]
        )
        
        original = SearchResult(result)
        columnar = ColumnarSearchResult(result)
        
        for q in range(2):
            for i in range(3):
                orig_val = original[q][i].entity.get("timestamp_field")
                col_val = columnar[q][i]["timestamp_field"]
                assert orig_val == col_val, f"timestamp_field mismatch at q={q}, i={i}"


class TestColumnarVsOriginalConsistency:
    """Tests to ensure ColumnarSearchResult produces the same data as SearchResult."""
    
    @pytest.mark.parametrize("pk", [
        schema_pb2.IDs(int_id=schema_pb2.LongArray(data=list(range(6)))),
        schema_pb2.IDs(str_id=schema_pb2.StringArray(data=[str(i*10) for i in range(6)]))
    ])
    def test_full_consistency_with_multiple_types(self, pk):
        """Test that ColumnarSearchResult matches SearchResult for all supported types."""
        fields_data = [
            schema_pb2.FieldData(type=DataType.BOOL, field_name="bool_field",
                                 scalars=schema_pb2.ScalarField(bool_data=schema_pb2.BoolArray(data=[True for _ in range(6)]))),
            schema_pb2.FieldData(type=DataType.INT64, field_name="int64_field",
                                 scalars=schema_pb2.ScalarField(long_data=schema_pb2.LongArray(data=list(range(6))))),
            schema_pb2.FieldData(type=DataType.FLOAT, field_name="float_field",
                                 scalars=schema_pb2.ScalarField(float_data=schema_pb2.FloatArray(data=[i*1.5 for i in range(6)]))),
            schema_pb2.FieldData(type=DataType.DOUBLE, field_name="double_field",
                                 scalars=schema_pb2.ScalarField(double_data=schema_pb2.DoubleArray(data=[i*2.5 for i in range(6)]))),
            schema_pb2.FieldData(type=DataType.VARCHAR, field_name="varchar_field",
                                 scalars=schema_pb2.ScalarField(string_data=schema_pb2.StringArray(data=[f"str_{i}" for i in range(6)]))),
            schema_pb2.FieldData(type=DataType.FLOAT_VECTOR, field_name="vector_field",
                                 vectors=schema_pb2.VectorField(
                                     dim=4,
                                     float_vector=schema_pb2.FloatArray(data=[random.random() for _ in range(24)]),
                                 )),
        ]
        
        result = schema_pb2.SearchResultData(
            fields_data=fields_data,
            num_queries=2,
            top_k=3,
            scores=[1.*i for i in range(6)],
            ids=pk,
            topks=[3, 3],
            output_fields=["bool_field", "int64_field", "float_field", "double_field", "varchar_field", "vector_field"]
        )
        
        original = SearchResult(result)
        columnar = ColumnarSearchResult(result)
        
        # Compare structure
        assert len(original) == len(columnar)
        
        for q in range(len(original)):
            assert len(original[q]) == len(columnar[q])
            
            for i in range(len(original[q])):
                orig_hit = original[q][i]
                col_hit = columnar[q][i]
                
                # Compare IDs
                assert orig_hit.id == col_hit.id, f"ID mismatch at q={q}, i={i}"
                
                # Compare distances
                assert abs(orig_hit.distance - col_hit.distance) < 1e-6, f"Distance mismatch at q={q}, i={i}"
                
                # Compare scalar fields
                for field in ["bool_field", "int64_field", "varchar_field"]:
                    orig_val = orig_hit.entity.get(field)
                    col_val = col_hit[field]
                    assert orig_val == col_val, f"{field} mismatch at q={q}, i={i}: {orig_val} != {col_val}"
                
                # Compare float fields with tolerance
                for field in ["float_field", "double_field"]:
                    orig_val = orig_hit.entity.get(field)
                    col_val = col_hit[field]
                    assert abs(orig_val - col_val) < 1e-6, f"{field} mismatch at q={q}, i={i}"
                
                # Compare vector fields
                orig_vec = orig_hit.entity.get("vector_field")
                col_vec = col_hit["vector_field"]
                assert np.allclose(orig_vec, col_vec, rtol=1e-5), f"vector_field mismatch at q={q}, i={i}"


# Helper functions

def _create_simple_search_result(nq=1, topk=10):
    """Create a simple SearchResultData for testing."""
    total = nq * topk
    
    ids = schema_pb2.IDs()
    ids.int_id.data.extend(list(range(total)))
    
    count_field = schema_pb2.FieldData()
    count_field.field_name = "count"
    count_field.type = DataType.INT64
    count_field.scalars.long_data.data.extend(list(range(total)))
    
    result = schema_pb2.SearchResultData()
    result.ids.CopyFrom(ids)
    result.scores.extend([float(i) for i in range(total)])
    result.topks.extend([topk] * nq)
    result.fields_data.append(count_field)
    result.num_queries = nq
    result.output_fields.extend(["count"])
    
    return result


def _create_paired_results_with_field(data_type, field_name, setup_fn):
    """Create both SearchResult and ColumnarSearchResult with a specific field."""
    field_data = schema_pb2.FieldData()
    field_data.field_name = field_name
    field_data.type = data_type
    setup_fn(field_data)
    
    ids = schema_pb2.IDs()
    ids.int_id.data.extend(list(range(6)))
    
    result = schema_pb2.SearchResultData(
        fields_data=[field_data],
        num_queries=2,
        top_k=3,
        scores=[1.*i for i in range(6)],
        ids=ids,
        topks=[3, 3],
        output_fields=[field_name]
    )
    
    return SearchResult(result), ColumnarSearchResult(result)


def _create_paired_results_with_vector_field(data_type, field_name, setup_fn):
    """Create both SearchResult and ColumnarSearchResult with a vector field."""
    field_data = schema_pb2.FieldData()
    field_data.field_name = field_name
    field_data.type = data_type
    setup_fn(field_data)
    
    ids = schema_pb2.IDs()
    ids.int_id.data.extend(list(range(6)))
    
    result = schema_pb2.SearchResultData(
        fields_data=[field_data],
        num_queries=2,
        top_k=3,
        scores=[1.*i for i in range(6)],
        ids=ids,
        topks=[3, 3],
        output_fields=[field_name]
    )
    
    return SearchResult(result), ColumnarSearchResult(result)


def _assert_field_values_match(original, columnar, field_name, rtol=None):
    """Assert that field values match between SearchResult and ColumnarSearchResult."""
    for q in range(len(original)):
        for i in range(len(original[q])):
            orig_val = original[q][i].entity.get(field_name)
            col_val = columnar[q][i][field_name]
            
            if rtol is not None:
                assert abs(orig_val - col_val) < rtol, f"{field_name} mismatch at q={q}, i={i}: {orig_val} != {col_val}"
            else:
                assert orig_val == col_val, f"{field_name} mismatch at q={q}, i={i}: {orig_val} != {col_val}"
