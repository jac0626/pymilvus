"""
Unit tests for TypeHandler module.

Tests all handler types for both read and write operations.
"""
import pytest
from typing import Dict, Any

from pymilvus.grpc_gen import schema_pb2
from pymilvus.client.types import DataType
from pymilvus.client.type_handlers import (
    get_handler,
    TypeHandlerRegistry,
    BoolHandler,
    Int32Handler,
    Int64Handler,
    FloatHandler,
    DoubleHandler,
    VarCharHandler,
    FloatVectorHandler,
    BinaryVectorHandler,
    Float16VectorHandler,
    Int8VectorHandler,
    JsonHandler,
    ArrayHandler,
)


# ==============================================================================
# Test Fixtures
# ==============================================================================

def create_field_data(dtype: DataType) -> schema_pb2.FieldData:
    """Create empty FieldData with type set."""
    return schema_pb2.FieldData(type=dtype, field_name="test_field")


def create_field_info(name: str = "test_field", **kwargs) -> Dict[str, Any]:
    """Create field info dict."""
    return {"name": name, **kwargs}


# ==============================================================================
# Registry Tests
# ==============================================================================

class TestTypeHandlerRegistry:
    """Tests for handler registration and lookup."""
    
    def test_get_handler_for_all_scalar_types(self):
        """Test that all scalar types have registered handlers."""
        scalar_types = [
            DataType.BOOL, DataType.INT8, DataType.INT16, DataType.INT32,
            DataType.INT64, DataType.FLOAT, DataType.DOUBLE, DataType.VARCHAR,
        ]
        for dtype in scalar_types:
            handler = get_handler(dtype)
            assert handler is not None, f"No handler for {dtype}"
    
    def test_get_handler_for_all_vector_types(self):
        """Test that all vector types have registered handlers."""
        vector_types = [
            DataType.FLOAT_VECTOR, DataType.BINARY_VECTOR,
            DataType.FLOAT16_VECTOR, DataType.BFLOAT16_VECTOR,
            DataType.INT8_VECTOR, DataType.SPARSE_FLOAT_VECTOR,
        ]
        for dtype in vector_types:
            handler = get_handler(dtype)
            assert handler is not None, f"No handler for {dtype}"
    
    def test_get_handler_for_complex_types(self):
        """Test that complex types have registered handlers."""
        complex_types = [DataType.JSON, DataType.ARRAY]
        for dtype in complex_types:
            handler = get_handler(dtype)
            assert handler is not None, f"No handler for {dtype}"
    
    def test_get_handler_raises_for_unknown_type(self):
        """Test that unknown type raises ValueError."""
        with pytest.raises(ValueError):
            get_handler(DataType.NONE)


# ==============================================================================
# Scalar Handler Tests
# ==============================================================================

class TestBoolHandler:
    """Tests for BoolHandler."""
    
    def test_pack_and_extract(self):
        """Test writing and reading bool values."""
        handler = BoolHandler()
        fd = create_field_data(DataType.BOOL)
        fi = create_field_info()
        
        # Pack values
        handler.pack_value(True, fd, fi)
        handler.pack_value(False, fd, fi)
        handler.pack_value(True, fd, fi)
        
        # Extract and verify
        data = handler.extract_data(fd)
        assert handler.get_value(data, 0) is True
        assert handler.get_value(data, 1) is False
        assert handler.get_value(data, 2) is True
    
    def test_pack_values_bulk(self):
        """Test bulk packing."""
        handler = BoolHandler()
        fd = create_field_data(DataType.BOOL)
        fi = create_field_info()
        
        handler.pack_values([True, False, True, False], fd, fi)
        
        data = handler.extract_data(fd)
        assert list(data) == [True, False, True, False]


class TestIntHandlers:
    """Tests for INT8, INT16, INT32 handlers."""
    
    @pytest.mark.parametrize("dtype", [DataType.INT8, DataType.INT16, DataType.INT32])
    def test_pack_and_extract(self, dtype):
        """Test writing and reading int values."""
        handler = get_handler(dtype)
        fd = create_field_data(dtype)
        fi = create_field_info()
        
        handler.pack_values([10, 20, 30], fd, fi)
        
        data = handler.extract_data(fd)
        assert handler.get_value(data, 0) == 10
        assert handler.get_value(data, 1) == 20
        assert handler.get_value(data, 2) == 30


class TestInt64Handler:
    """Tests for Int64Handler."""
    
    def test_pack_and_extract(self):
        """Test writing and reading int64 values."""
        handler = Int64Handler()
        fd = create_field_data(DataType.INT64)
        fi = create_field_info()
        
        handler.pack_values([1000000000000, 2000000000000], fd, fi)
        
        data = handler.extract_data(fd)
        assert handler.get_value(data, 0) == 1000000000000
        assert handler.get_value(data, 1) == 2000000000000


class TestFloatHandler:
    """Tests for FloatHandler."""
    
    def test_pack_and_extract(self):
        """Test writing and reading float values."""
        handler = FloatHandler()
        fd = create_field_data(DataType.FLOAT)
        fi = create_field_info()
        
        handler.pack_values([1.5, 2.5, 3.5], fd, fi)
        
        data = handler.extract_data(fd)
        assert handler.get_value(data, 0) == pytest.approx(1.5)
        assert handler.get_value(data, 1) == pytest.approx(2.5)


class TestVarCharHandler:
    """Tests for VarCharHandler."""
    
    def test_pack_and_extract(self):
        """Test writing and reading string values."""
        handler = VarCharHandler()
        fd = create_field_data(DataType.VARCHAR)
        fi = create_field_info()
        
        handler.pack_values(["hello", "world", "test"], fd, fi)
        
        data = handler.extract_data(fd)
        assert handler.get_value(data, 0) == "hello"
        assert handler.get_value(data, 1) == "world"
        assert handler.get_value(data, 2) == "test"


# ==============================================================================
# Vector Handler Tests
# ==============================================================================

class TestFloatVectorHandler:
    """Tests for FloatVectorHandler."""
    
    def test_pack_and_extract(self):
        """Test writing and reading float vectors."""
        handler = FloatVectorHandler()
        fd = create_field_data(DataType.FLOAT_VECTOR)
        fi = create_field_info(params={"dim": 4})
        
        # Pack two 4-dim vectors
        handler.pack_values([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], fd, fi)
        
        data = handler.extract_data(fd)
        dim = handler.get_dim(fd)
        
        assert dim == 4
        assert handler.get_value(data, 0, dim) == [1.0, 2.0, 3.0, 4.0]
        assert handler.get_value(data, 1, dim) == [5.0, 6.0, 7.0, 8.0]
    
    def test_accessor(self):
        """Test accessor creation."""
        handler = FloatVectorHandler()
        fd = create_field_data(DataType.FLOAT_VECTOR)
        fi = create_field_info(params={"dim": 3})
        
        handler.pack_values([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], fd, fi)
        
        data = handler.extract_data(fd)
        dim = handler.get_dim(fd)
        accessor = handler.create_accessor(data, start=0, dim=dim)
        
        assert accessor(0) == [1.0, 2.0, 3.0]
        assert accessor(1) == [4.0, 5.0, 6.0]
    
    def test_accessor_with_offset(self):
        """Test accessor with non-zero start offset."""
        handler = FloatVectorHandler()
        fd = create_field_data(DataType.FLOAT_VECTOR)
        fi = create_field_info(params={"dim": 2})
        
        handler.pack_values([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], fd, fi)
        
        data = handler.extract_data(fd)
        dim = handler.get_dim(fd)
        accessor = handler.create_accessor(data, start=1, dim=dim)
        
        # accessor(0) should return second vector
        assert accessor(0) == [3.0, 4.0]
        assert accessor(1) == [5.0, 6.0]


class TestBinaryVectorHandler:
    """Tests for BinaryVectorHandler."""
    
    def test_pack_and_extract(self):
        """Test writing and reading binary vectors."""
        handler = BinaryVectorHandler()
        fd = create_field_data(DataType.BINARY_VECTOR)
        fi = create_field_info(params={"dim": 32})  # 32 bits = 4 bytes
        
        # Pack two 32-bit vectors (4 bytes each)
        handler.pack_values([bytes([1, 2, 3, 4]), bytes([5, 6, 7, 8])], fd, fi)
        
        data = handler.extract_data(fd)
        dim = handler.get_dim(fd)
        
        assert dim == 32
        assert handler.get_value(data, 0, dim) == bytes([1, 2, 3, 4])
        assert handler.get_value(data, 1, dim) == bytes([5, 6, 7, 8])


class TestFloat16VectorHandler:
    """Tests for Float16VectorHandler."""
    
    def test_pack_and_extract_bytes(self):
        """Test writing and reading float16 vectors as bytes."""
        handler = Float16VectorHandler()
        fd = create_field_data(DataType.FLOAT16_VECTOR)
        fi = create_field_info(params={"dim": 4})
        
        # 4-dim float16 = 8 bytes per vector
        vec1 = bytes([0, 1, 2, 3, 4, 5, 6, 7])
        vec2 = bytes([8, 9, 10, 11, 12, 13, 14, 15])
        
        handler.pack_values([vec1, vec2], fd, fi)
        
        data = handler.extract_data(fd)
        dim = handler.get_dim(fd)
        
        assert dim == 4
        assert handler.get_value(data, 0, dim) == vec1
        assert handler.get_value(data, 1, dim) == vec2


# ==============================================================================
# Complex Handler Tests
# ==============================================================================

class TestJsonHandler:
    """Tests for JsonHandler."""
    
    def test_pack_and_extract(self):
        """Test writing and reading JSON values."""
        handler = JsonHandler()
        fd = create_field_data(DataType.JSON)
        fi = create_field_info()
        
        handler.pack_values(
            [{"name": "alice", "age": 30}, {"name": "bob", "age": 25}],
            fd, fi
        )
        
        data = handler.extract_data(fd)
        
        assert handler.get_value(data, 0) == {"name": "alice", "age": 30}
        assert handler.get_value(data, 1) == {"name": "bob", "age": 25}
    
    def test_accessor(self):
        """Test JSON accessor deserialization."""
        handler = JsonHandler()
        fd = create_field_data(DataType.JSON)
        fi = create_field_info()
        
        handler.pack_values([{"key": "value1"}, {"key": "value2"}], fd, fi)
        
        data = handler.extract_data(fd)
        accessor = handler.create_accessor(data, start=0)
        
        assert accessor(0) == {"key": "value1"}
        assert accessor(1) == {"key": "value2"}


class TestArrayHandler:
    """Tests for ArrayHandler."""
    
    def test_pack_and_extract_int64_array(self):
        """Test writing and reading int64 arrays."""
        handler = ArrayHandler()
        fd = create_field_data(DataType.ARRAY)
        fi = create_field_info(element_type=DataType.INT64)
        
        handler.pack_values([[1, 2, 3], [4, 5, 6]], fd, fi)
        
        data = handler.extract_data(fd)
        
        # Check extraction works
        result = handler.get_value(data, 0)
        assert list(result) == [1, 2, 3]


# ==============================================================================
# Accessor Performance Tests (Conceptual)
# ==============================================================================

class TestAccessorEfficiency:
    """Tests to verify accessor pattern works correctly."""
    
    def test_scalar_accessor_repeated_access(self):
        """Test that scalar accessor works for repeated access."""
        handler = Int64Handler()
        fd = create_field_data(DataType.INT64)
        fi = create_field_info()
        
        values = list(range(1000))
        handler.pack_values(values, fd, fi)
        
        data = handler.extract_data(fd)
        accessor = handler.create_accessor(data, start=0)
        
        # Access multiple times
        for i in range(1000):
            assert accessor(i) == i
    
    def test_vector_accessor_with_offset(self):
        """Test vector accessor correctly handles offset."""
        handler = FloatVectorHandler()
        fd = create_field_data(DataType.FLOAT_VECTOR)
        fi = create_field_info(params={"dim": 3})
        
        vectors = [[float(i), float(i+1), float(i+2)] for i in range(100)]
        handler.pack_values(vectors, fd, fi)
        
        data = handler.extract_data(fd)
        dim = handler.get_dim(fd)
        
        # Create accessor starting at offset 50
        accessor = handler.create_accessor(data, start=50, dim=dim)
        
        # accessor(0) should return vectors[50]
        assert accessor(0) == [50.0, 51.0, 52.0]
        assert accessor(10) == [60.0, 61.0, 62.0]
