"""
TypeHandler Base Module - Abstract interfaces for type-specific operations.

Provides a unified interface for reading from and writing to protobuf FieldData.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple

from pymilvus.client.types import DataType
from pymilvus.grpc_gen import schema_pb2

__all__ = [
    "DataTypeHandler",
    "VectorHandler", 
    "BytesVectorHandler",
    "TypeHandlerRegistry",
    "get_handler",
]


class DataTypeHandler(ABC):
    """Base class for all type handlers."""
    
    supported_types: Tuple[DataType, ...] = ()
    
    # === Write Operations ===
    
    @abstractmethod
    def pack_single(
        self, value: Any, field_data: schema_pb2.FieldData, field_info: Dict
    ) -> None:
        """Pack a single value into FieldData (row insert)."""
        pass
    
    @abstractmethod
    def pack_batch(
        self,
        values: List[Any],
        field_data: schema_pb2.FieldData,
        field_info: Dict,
        valid_data: Optional[List[bool]] = None,
    ) -> None:
        """Pack multiple values into FieldData (columnar insert)."""
        pass
    
    # === Read Operations ===
    
    @abstractmethod
    def extract_payload(self, field_data: schema_pb2.FieldData) -> Any:
        """Extract raw data container from FieldData (zero-copy reference)."""
        pass
    
    @abstractmethod
    def create_accessor(
        self, payload: Any, start: int, valid_data: Optional[Any] = None
    ) -> Callable[[int], Any]:
        """Create a fast accessor closure with pre-bound parameters."""
        pass
    
    def get_slice(self, payload: Any, start: int, end: int) -> List[Any]:
        """Get a slice of values. Default uses accessor."""
        accessor = self.create_accessor(payload, 0)
        return [accessor(i) for i in range(start, end)]


class VectorHandler(DataTypeHandler):
    """Base class for vector type handlers - adds dimension methods."""
    
    @abstractmethod
    def get_dim(self, field_data: schema_pb2.FieldData) -> int:
        """Get vector dimension from FieldData."""
        pass
    
    def get_bytes_per_element(self, dim: int) -> int:
        """Get bytes per vector element. Override for bytes vectors."""
        return 0
    
    @abstractmethod
    def create_accessor(
        self, payload: Any, start: int, dim: int = 0, valid_data: Optional[Any] = None
    ) -> Callable[[int], Any]:
        """Create accessor with dimension parameter."""
        pass
    
    def get_slice(
        self, payload: Any, start: int, end: int, dim: int = 0
    ) -> List[Any]:
        """Get slice of vectors."""
        accessor = self.create_accessor(payload, 0, dim)
        return [accessor(i) for i in range(start, end)]


class BytesVectorHandler(VectorHandler):
    """Base for bytes-based vectors (BINARY, FLOAT16, BFLOAT16, INT8).
    
    Maintains internal cache to avoid O(n²) byte concatenation.
    """
    
    def __init__(self):
        self._cache: Dict[int, List[bytes]] = {}
    
    def flush(self, field_data: schema_pb2.FieldData) -> None:
        """Flush cached bytes to FieldData. Call after all pack_single calls."""
        field_id = id(field_data)
        bytes_list = self._cache.pop(field_id, None)
        if bytes_list:
            self._set_bytes(field_data, b"".join(bytes_list))
    
    @abstractmethod
    def _set_bytes(self, field_data: schema_pb2.FieldData, data: bytes) -> None:
        """Set bytes on the appropriate field_data attribute."""
        pass


class TypeHandlerRegistry:
    """Registry for TypeHandler lookup by DataType."""
    
    _handlers: Dict[DataType, DataTypeHandler] = {}
    
    @classmethod
    def register(cls, handler: DataTypeHandler) -> None:
        for dtype in handler.supported_types:
            cls._handlers[dtype] = handler
    
    @classmethod
    def get(cls, dtype: DataType) -> Optional[DataTypeHandler]:
        return cls._handlers.get(dtype)
    
    @classmethod
    def get_or_raise(cls, dtype: DataType) -> DataTypeHandler:
        handler = cls._handlers.get(dtype)
        if handler is None:
            raise ValueError(f"No handler for DataType: {dtype}")
        return handler


def get_handler(dtype: DataType) -> DataTypeHandler:
    """Get handler for a DataType."""
    return TypeHandlerRegistry.get_or_raise(dtype)
