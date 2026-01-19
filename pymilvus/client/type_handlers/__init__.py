"""
TypeHandler Module - Unified type-specific read/write operations.

This package provides handlers for all Milvus data types, eliminating
the need for large if-elif chains throughout the codebase.

Usage:
    from pymilvus.client.type_handlers import get_handler, DataType
    
    handler = get_handler(DataType.FLOAT_VECTOR)
    
    # Reading
    data = handler.extract_data(field_data)
    value = handler.get_value(data, index=0, dim=128)
    
    # Writing
    handler.pack_value(value, field_data, field_info)
"""

from pymilvus.client.types import DataType

from .base import TypeHandler, TypeHandlerRegistry, get_handler

# Import and register all handlers
from .scalar import (
    BoolHandler,
    Int8Handler,
    Int16Handler,
    Int32Handler,
    Int64Handler,
    FloatHandler,
    DoubleHandler,
    VarCharHandler,
    GeometryHandler,
    TimestampTzHandler,
)
from .vector import (
    FloatVectorHandler,
    BinaryVectorHandler,
    Float16VectorHandler,
    BFloat16VectorHandler,
    Int8VectorHandler,
    SparseFloatVectorHandler,
)
from .complex import (
    JsonHandler,
    ArrayHandler,
)

# Register all handlers
_all_handlers = [
    # Scalars
    BoolHandler(),
    Int8Handler(),
    Int16Handler(),
    Int32Handler(),
    Int64Handler(),
    FloatHandler(),
    DoubleHandler(),
    VarCharHandler(),
    GeometryHandler(),
    TimestampTzHandler(),
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

for handler in _all_handlers:
    TypeHandlerRegistry.register(handler)


__all__ = [
    "TypeHandler",
    "TypeHandlerRegistry",
    "get_handler",
    "DataType",
    # Scalar handlers
    "BoolHandler",
    "Int8Handler",
    "Int16Handler",
    "Int32Handler",
    "Int64Handler",
    "FloatHandler",
    "DoubleHandler",
    "VarCharHandler",
    "GeometryHandler",
    "TimestampTzHandler",
    # Vector handlers
    "FloatVectorHandler",
    "BinaryVectorHandler",
    "Float16VectorHandler",
    "BFloat16VectorHandler",
    "Int8VectorHandler",
    "SparseFloatVectorHandler",
    # Complex handlers
    "JsonHandler",
    "ArrayHandler",
]
