"""
Complex Type Handlers - JSON and ARRAY types.

These handlers manage complex/nested types that require special
serialization and deserialization logic.
"""

from typing import Any, Callable, Dict, List

import orjson

from pymilvus.client.types import DataType
from pymilvus.exceptions import DataNotMatchException, ExceptionsMessage
from pymilvus.grpc_gen import schema_pb2

from .base import TypeHandler


class JsonHandler(TypeHandler):
    """Handler for JSON type.
    
    JSON data is stored as bytes in protobuf and deserialized on read.
    """
    
    supported_types = (DataType.JSON,)
    
    def extract_data(self, field_data: schema_pb2.FieldData) -> Any:
        return field_data.scalars.json_data.data
    
    def get_value(self, data: Any, index: int, dim: int = 0) -> Any:
        val = data[index]
        return orjson.loads(val) if val else None
    
    def get_slice(self, data: Any, start: int, end: int, dim: int = 0) -> List[Any]:
        """Return raw bytes for efficiency - caller can deserialize if needed."""
        return data[start:end]
    
    def create_accessor(
        self, data: Any, start: int, dim: int = 0
    ) -> Callable[[int], Any]:
        """Accessor that deserializes JSON on access."""
        def accessor(i: int) -> Any:
            val = data[i + start]
            return orjson.loads(val) if val else None
        return accessor
    
    def pack_value(
        self,
        value: Any,
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        from pymilvus.client.entity_helper import convert_to_json
        
        try:
            if value is None:
                field_data.scalars.json_data.data.extend([])
            else:
                field_data.scalars.json_data.data.append(convert_to_json(value))
        except (TypeError, ValueError) as e:
            raise DataNotMatchException(
                message=ExceptionsMessage.FieldDataInconsistent
                % (field_info["name"], "json", type(value))
                + f" Detail: {e!s}"
            ) from e
    
    def pack_values(
        self,
        values: List[Any],
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        from pymilvus.client.entity_helper import convert_to_json
        
        json_bytes = [convert_to_json(v) for v in values]
        field_data.scalars.json_data.data.extend(json_bytes)


class ArrayHandler(TypeHandler):
    """Handler for ARRAY type.
    
    Arrays are stored as repeated ScalarField messages with element_type.
    """
    
    supported_types = (DataType.ARRAY,)
    
    def extract_data(self, field_data: schema_pb2.FieldData) -> Any:
        return field_data.scalars.array_data.data
    
    def get_element_type(self, field_data: schema_pb2.FieldData) -> DataType:
        return DataType(field_data.scalars.array_data.element_type)
    
    def get_value(self, data: Any, index: int, dim: int = 0) -> List[Any]:
        # Need element_type to extract properly - this is a simplified version
        array_data = data[index]
        return self._extract_array_data(array_data)
    
    def get_value_with_element_type(
        self, data: Any, index: int, element_type: DataType
    ) -> List[Any]:
        """Get value with explicit element type for proper extraction."""
        array_data = data[index]
        return self._extract_typed_array_data(array_data, element_type)
    
    def _extract_array_data(self, array_data: Any) -> List[Any]:
        """Extract array data by trying each type."""
        if array_data.long_data.data:
            return list(array_data.long_data.data)
        if array_data.int_data.data:
            return list(array_data.int_data.data)
        if array_data.float_data.data:
            return list(array_data.float_data.data)
        if array_data.double_data.data:
            return list(array_data.double_data.data)
        if array_data.string_data.data:
            return list(array_data.string_data.data)
        if array_data.bool_data.data:
            return list(array_data.bool_data.data)
        return []
    
    def _extract_typed_array_data(
        self, array_data: Any, element_type: DataType
    ) -> List[Any]:
        """Extract array data with known element type."""
        if element_type == DataType.INT64:
            return list(array_data.long_data.data)
        if element_type in (DataType.INT8, DataType.INT16, DataType.INT32):
            return list(array_data.int_data.data)
        if element_type == DataType.FLOAT:
            return list(array_data.float_data.data)
        if element_type == DataType.DOUBLE:
            return list(array_data.double_data.data)
        if element_type in (DataType.VARCHAR, DataType.STRING):
            return list(array_data.string_data.data)
        if element_type == DataType.BOOL:
            return list(array_data.bool_data.data)
        return []
    
    def create_accessor(
        self, data: Any, start: int, dim: int = 0
    ) -> Callable[[int], List[Any]]:
        """Accessor for array type."""
        def accessor(i: int) -> List[Any]:
            return self._extract_array_data(data[i + start])
        return accessor
    
    def create_typed_accessor(
        self, data: Any, start: int, element_type: DataType
    ) -> Callable[[int], List[Any]]:
        """Create accessor with known element type for better performance."""
        def accessor(i: int) -> List[Any]:
            return self._extract_typed_array_data(data[i + start], element_type)
        return accessor
    
    def pack_value(
        self,
        value: Any,
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        from pymilvus.client.entity_helper import convert_to_array
        
        try:
            if value is None:
                field_data.scalars.array_data.data.extend([])
            else:
                field_data.scalars.array_data.data.append(
                    convert_to_array(value, field_info)
                )
        except (TypeError, ValueError) as e:
            raise DataNotMatchException(
                message=ExceptionsMessage.FieldDataInconsistent
                % (field_info["name"], "array", type(value))
                + f" Detail: {e!s}"
            ) from e
    
    def pack_values(
        self,
        values: List[Any],
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        from pymilvus.client.entity_helper import convert_to_array
        
        array_data = [convert_to_array(v, field_info) for v in values]
        field_data.scalars.array_data.data.extend(array_data)
