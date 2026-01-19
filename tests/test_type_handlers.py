"""Unit tests for TypeHandler module."""
import pytest
from pymilvus.grpc_gen import schema_pb2
from pymilvus.client.types import DataType
from pymilvus.client.type_handlers import (
    get_handler,
    BoolHandler,
    Int64Handler,
    FloatVectorHandler,
    JsonHandler,
    ArrayHandler,
)


def create_field_data(dtype: DataType) -> schema_pb2.FieldData:
    return schema_pb2.FieldData(type=dtype, field_name="test")


class TestRegistry:
    def test_all_types_registered(self):
        types = [
            DataType.BOOL, DataType.INT8, DataType.INT16, DataType.INT32, DataType.INT64,
            DataType.FLOAT, DataType.DOUBLE, DataType.VARCHAR,
            DataType.FLOAT_VECTOR, DataType.BINARY_VECTOR, DataType.FLOAT16_VECTOR,
            DataType.BFLOAT16_VECTOR, DataType.INT8_VECTOR, DataType.SPARSE_FLOAT_VECTOR,
            DataType.JSON, DataType.ARRAY,
        ]
        for dt in types:
            h = get_handler(dt)
            assert h is not None, f"No handler for {dt}"


class TestScalarHandlers:
    def test_bool_pack_and_read(self):
        h = BoolHandler()
        fd = create_field_data(DataType.BOOL)
        h.pack_batch([True, False, True], fd, {})
        
        payload = h.extract_payload(fd)
        accessor = h.create_accessor(payload, 0)
        assert accessor(0) is True
        assert accessor(1) is False
        assert accessor(2) is True
    
    def test_int64_pack_and_read(self):
        h = Int64Handler()
        fd = create_field_data(DataType.INT64)
        h.pack_batch([100, 200, 300], fd, {})
        
        payload = h.extract_payload(fd)
        assert h.get_slice(payload, 0, 3) == [100, 200, 300]


class TestVectorHandlers:
    def test_float_vector_pack_and_read(self):
        h = FloatVectorHandler()
        fd = create_field_data(DataType.FLOAT_VECTOR)
        vectors = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        h.pack_batch(vectors, fd, {"params": {"dim": 3}})
        
        payload = h.extract_payload(fd)
        dim = h.get_dim(fd)
        accessor = h.create_accessor(payload, 0, dim)
        
        assert dim == 3
        assert accessor(0) == [1.0, 2.0, 3.0]
        assert accessor(1) == [4.0, 5.0, 6.0]
    
    def test_float_vector_accessor_with_offset(self):
        h = FloatVectorHandler()
        fd = create_field_data(DataType.FLOAT_VECTOR)
        vectors = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        h.pack_batch(vectors, fd, {})
        
        payload = h.extract_payload(fd)
        dim = h.get_dim(fd)
        accessor = h.create_accessor(payload, start=1, dim=dim)
        
        assert accessor(0) == [3.0, 4.0]  # starts at index 1
        assert accessor(1) == [5.0, 6.0]


class TestComplexHandlers:
    def test_json_pack_and_read(self):
        h = JsonHandler()
        fd = create_field_data(DataType.JSON)
        h.pack_batch([{"a": 1}, {"b": 2}], fd, {})
        
        payload = h.extract_payload(fd)
        accessor = h.create_accessor(payload, 0)
        
        assert accessor(0) == {"a": 1}
        assert accessor(1) == {"b": 2}
    
    def test_array_pack_and_read(self):
        h = ArrayHandler()
        fd = create_field_data(DataType.ARRAY)
        h.pack_batch([[1, 2, 3], [4, 5, 6]], fd, {"element_type": DataType.INT64})
        
        payload = h.extract_payload(fd)
        accessor = h.create_accessor(payload, 0, element_type=DataType.INT64)
        
        assert accessor(0) == [1, 2, 3]
        assert accessor(1) == [4, 5, 6]


class TestValidData:
    def test_scalar_with_valid_data(self):
        h = Int64Handler()
        fd = create_field_data(DataType.INT64)
        h.pack_batch([100, 200, 300], fd, {}, valid_data=[True, False, True])
        
        payload = h.extract_payload(fd)
        assert list(payload) == [100, 300]  # 200 filtered out
