"""
TypeHandler Base Module - Abstract interface for type-specific operations.

This module provides a unified interface for reading from and writing to
protobuf FieldData structures. Each data type (INT32, FLOAT_VECTOR, JSON, etc.)
has a corresponding handler that knows how to:
1. Read: Extract Python values from protobuf FieldData
2. Write: Convert Python values to protobuf FieldData
3. Validate: Check if input values are valid for this type

Design Benefits:
- Eliminates massive if-elif chains in entity_helper.py and search_result.py
- Single Responsibility: Each type's logic is encapsulated in one class
- Open/Closed: Add new types by adding new handlers, not modifying existing code
- Testable: Each handler can be unit tested in isolation
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from pymilvus.client.types import DataType
from pymilvus.grpc_gen import schema_pb2

__all__ = [
    "TypeHandler",
    "TypeHandlerRegistry",
    "get_handler",
]


class TypeHandler(ABC):
    """Abstract base class for type-specific read/write operations.
    
    Each subclass handles one or more related DataTypes.
    """
    
    # Class attribute: which DataType(s) this handler supports
    supported_types: Tuple[DataType, ...] = ()
    
    # =========================================================================
    # Reading (for search results)
    # =========================================================================
    
    @abstractmethod
    def extract_data(self, field_data: schema_pb2.FieldData) -> Any:
        """Extract raw data container from protobuf FieldData.
        
        Returns the underlying data structure (list, bytes, etc.) that can
        be efficiently sliced or indexed.
        
        Args:
            field_data: The protobuf FieldData message
            
        Returns:
            The raw data container (e.g., repeated field, bytes)
        """
        pass
    
    @abstractmethod
    def get_value(self, data: Any, index: int, dim: int = 0) -> Any:
        """Get a single value at the given index.
        
        Args:
            data: The raw data container from extract_data()
            index: Absolute index into the data
            dim: Dimension for vector types (ignored for scalars)
            
        Returns:
            The value at the given index
        """
        pass
    
    def get_slice(self, data: Any, start: int, end: int, dim: int = 0) -> Any:
        """Get a slice of values. Default implementation uses get_value().
        
        Override for efficient bulk access (e.g., bytes slicing).
        
        Args:
            data: The raw data container
            start: Start index (inclusive)
            end: End index (exclusive)
            dim: Dimension for vector types
            
        Returns:
            Slice of data (list, bytes, etc.)
        """
        return [self.get_value(data, i, dim) for i in range(start, end)]
    
    def create_accessor(
        self, data: Any, start: int, dim: int = 0
    ) -> Callable[[int], Any]:
        """Create a fast accessor function for repeated access.
        
        Returns a callable that takes a relative index and returns the value.
        Default implementation creates a simple closure.
        
        Args:
            data: The raw data container
            start: Starting offset
            dim: Dimension for vector types
            
        Returns:
            Accessor function: (relative_index) -> value
        """
        def accessor(i: int) -> Any:
            return self.get_value(data, i + start, dim)
        return accessor
    
    # =========================================================================
    # Writing (for insert/upsert)
    # =========================================================================
    
    @abstractmethod
    def pack_value(
        self,
        value: Any,
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        """Pack a single Python value into protobuf FieldData.
        
        This is used for row-by-row insert operations.
        
        Args:
            value: The Python value to pack (can be None for nullable)
            field_data: The target protobuf FieldData message
            field_info: Schema info for the field
        """
        pass
    
    @abstractmethod
    def pack_values(
        self,
        values: List[Any],
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        """Pack multiple Python values into protobuf FieldData.
        
        This is used for columnar insert operations and is typically
        more efficient than calling pack_value() in a loop.
        
        Args:
            values: List of Python values to pack
            field_data: The target protobuf FieldData message
            field_info: Schema info for the field
        """
        pass
    
    # =========================================================================
    # Validation (optional, can be overridden)
    # =========================================================================
    
    def validate(self, value: Any, field_info: Dict[str, Any]) -> bool:
        """Validate that a value is appropriate for this type.
        
        Default implementation returns True. Override for type-specific validation.
        
        Args:
            value: The value to validate
            field_info: Schema info for the field
            
        Returns:
            True if valid, False otherwise
        """
        return True
    
    # =========================================================================
    # Metadata
    # =========================================================================
    
    def get_dim(self, field_data: schema_pb2.FieldData) -> int:
        """Get dimension for vector types. Returns 0 for scalars."""
        return 0
    
    def get_bytes_per_element(self, dim: int) -> int:
        """Get bytes per element for this type. Used for byte-based vectors."""
        return 0


class TypeHandlerRegistry:
    """Registry for TypeHandler instances, keyed by DataType."""
    
    _handlers: Dict[DataType, TypeHandler] = {}
    
    @classmethod
    def register(cls, handler: TypeHandler) -> None:
        """Register a handler for its supported types."""
        for dtype in handler.supported_types:
            cls._handlers[dtype] = handler
    
    @classmethod
    def get(cls, dtype: DataType) -> Optional[TypeHandler]:
        """Get the handler for a DataType, or None if not found."""
        return cls._handlers.get(dtype)
    
    @classmethod
    def get_or_raise(cls, dtype: DataType) -> TypeHandler:
        """Get the handler for a DataType, raising if not found."""
        handler = cls._handlers.get(dtype)
        if handler is None:
            raise ValueError(f"No handler registered for DataType: {dtype}")
        return handler
    
    @classmethod
    def all_handlers(cls) -> Dict[DataType, TypeHandler]:
        """Get all registered handlers."""
        return cls._handlers.copy()


def get_handler(dtype: DataType) -> TypeHandler:
    """Convenience function to get a handler for a DataType."""
    return TypeHandlerRegistry.get_or_raise(dtype)
