"""
Complex Type Handlers - JSON and ARRAY types.
"""

from typing import Any, Callable, Dict, List, Optional

import orjson

from pymilvus.client.types import DataType
from pymilvus.grpc_gen import schema_pb2

from .base import DataTypeHandler, TypeHandlerRegistry


class JsonHandler(DataTypeHandler):
    supported_types = (DataType.JSON,)
    
    def extract_payload(self, field_data: schema_pb2.FieldData) -> Any:
        return field_data.scalars.json_data.data
    
    def create_accessor(
        self, payload: Any, start: int, valid_data: Optional[Any] = None
    ) -> Callable[[int], Any]:
        if valid_data is not None:
            def accessor(i: int) -> Any:
                idx = i + start
                if not valid_data[idx]:
                    return None
                val = payload[idx]
                return orjson.loads(val) if val else None
            return accessor
        def accessor(i: int) -> Any:
            val = payload[i + start]
            return orjson.loads(val) if val else None
        return accessor
    
    def pack_single(self, value: Any, field_data: schema_pb2.FieldData, field_info: Dict) -> None:
        from pymilvus.client.entity_helper import convert_to_json
        
        if value is None:
            field_data.scalars.json_data.data.extend([])
        else:
            field_data.scalars.json_data.data.append(convert_to_json(value))
    
    def pack_batch(self, values: List, field_data: schema_pb2.FieldData, field_info: Dict, valid_data: Optional[List[bool]] = None) -> None:
        from pymilvus.client.entity_helper import convert_to_json
        
        if valid_data:
            values = [v for i, v in enumerate(values) if valid_data[i]]
        json_bytes = [convert_to_json(v) for v in values]
        field_data.scalars.json_data.data.extend(json_bytes)


class ArrayHandler(DataTypeHandler):
    """Handler for ARRAY type - delegates to element handlers."""
    supported_types = (DataType.ARRAY,)
    
    def extract_payload(self, field_data: schema_pb2.FieldData) -> Any:
        return field_data.scalars.array_data.data
    
    def get_element_type(self, field_data: schema_pb2.FieldData) -> DataType:
        return DataType(field_data.scalars.array_data.element_type)
    
    def _extract_array_data(self, array_data: Any, element_type: Optional[DataType] = None) -> List[Any]:
        """Extract list from ScalarField based on element type."""
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
        # Try all types if element_type unknown
        for attr in ["long_data", "int_data", "float_data", "double_data", "string_data", "bool_data"]:
            data = getattr(array_data, attr, None)
            if data and len(data.data) > 0:
                return list(data.data)
        return []
    
    def create_accessor(
        self, payload: Any, start: int, valid_data: Optional[Any] = None, element_type: Optional[DataType] = None
    ) -> Callable[[int], List[Any]]:
        if valid_data is not None:
            def accessor(i: int) -> Optional[List[Any]]:
                idx = i + start
                if not valid_data[idx]:
                    return None
                return self._extract_array_data(payload[idx], element_type)
            return accessor
        def accessor(i: int) -> List[Any]:
            return self._extract_array_data(payload[i + start], element_type)
        return accessor
    
    def pack_single(self, value: Any, field_data: schema_pb2.FieldData, field_info: Dict) -> None:
        from pymilvus.client.entity_helper import convert_to_array
        
        if value is None:
            field_data.scalars.array_data.data.extend([])
        else:
            field_data.scalars.array_data.data.append(convert_to_array(value, field_info))
    
    def pack_batch(self, values: List, field_data: schema_pb2.FieldData, field_info: Dict, valid_data: Optional[List[bool]] = None) -> None:
        from pymilvus.client.entity_helper import convert_to_array
        
        if valid_data:
            values = [v for i, v in enumerate(values) if valid_data[i]]
        array_data = [convert_to_array(v, field_info) for v in values]
        field_data.scalars.array_data.data.extend(array_data)
