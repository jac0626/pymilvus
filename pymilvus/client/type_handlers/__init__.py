"""
TypeHandler Module - Unified type-specific operations for PyMilvus.

Provides handlers for reading from and writing to protobuf FieldData,
eliminating the need for large if-elif chains.

Usage:
    from pymilvus.client.type_handlers import get_handler
    
    handler = get_handler(DataType.FLOAT_VECTOR)
    
    # Write
    handler.pack_single(value, field_data, field_info)
    handler.pack_batch(values, field_data, field_info)
    
    # Read
    payload = handler.extract_payload(field_data)
    accessor = handler.create_accessor(payload, start, dim)
    value = accessor(index)
"""

from pymilvus.client.types import DataType

from .base import (
    BytesVectorHandler,
    DataTypeHandler,
    TypeHandlerRegistry,
    VectorHandler,
    get_handler,
)
from .complex import ArrayHandler, JsonHandler
from .scalar import (
    BoolHandler,
    DoubleHandler,
    FloatHandler,
    GeometryHandler,
    Int64Handler,
    IntHandler,
    TimestampTzHandler,
    VarCharHandler,
)
from .vector import (
    BFloat16VectorHandler,
    BinaryVectorHandler,
    Float16VectorHandler,
    FloatVectorHandler,
    Int8VectorHandler,
    SparseFloatVectorHandler,
)

# Register all handlers
_handlers = [
    # Scalars
    BoolHandler(),
    IntHandler(),
    Int64Handler(),
    FloatHandler(),
    DoubleHandler(),
    VarCharHandler(),
    TimestampTzHandler(),
    GeometryHandler(),
    # Vectors
    FloatVectorHandler(),
    BinaryVectorHandler(),
    Float16VectorHandler(),
    BFloat16VectorHandler(),
    Int8VectorHandler(),
    SparseFloatVectorHandler(),
    # Complex
    JsonHandler(),
    ArrayHandler(),
]

for handler in _handlers:
    TypeHandlerRegistry.register(handler)

__all__ = [
    "DataTypeHandler",
    "VectorHandler",
    "BytesVectorHandler",
    "TypeHandlerRegistry",
    "get_handler",
    # Scalar
    "BoolHandler",
    "IntHandler",
    "Int64Handler",
    "FloatHandler",
    "DoubleHandler",
    "VarCharHandler",
    "TimestampTzHandler",
    "GeometryHandler",
    # Vector
    "FloatVectorHandler",
    "BinaryVectorHandler",
    "Float16VectorHandler",
    "BFloat16VectorHandler",
    "Int8VectorHandler",
    "SparseFloatVectorHandler",
    # Complex
    "JsonHandler",
    "ArrayHandler",
]
