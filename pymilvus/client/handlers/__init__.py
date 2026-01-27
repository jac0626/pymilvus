"""
TypeHandler pattern for PyMilvus type processing.

This module provides a unified, extensible approach to handling different data types
in Milvus. Instead of scattered if-else chains, each type has a dedicated handler
that implements packing (write) and extraction (read) operations.

Usage:
    from pymilvus.client.handlers import get_handler, TypeHandlerRegistry

    # Get handler for a specific type
    handler = get_handler(DataType.FLOAT_VECTOR)

    # Pack a value into field data
    handler.pack_value(value, field_data, field_info, context)

    # Extract a value from field data
    result = handler.extract_value(field_data, index, context)
"""

from .base import TypeHandler
from .context import ExtractContext, PackContext
from .registry import TypeHandlerRegistry, get_handler

__all__ = [
    "TypeHandler",
    "TypeHandlerRegistry",
    "PackContext",
    "ExtractContext",
    "get_handler",
]
