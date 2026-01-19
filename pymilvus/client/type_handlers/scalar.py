"""
Scalar Type Handlers - BOOL, INT8-64, FLOAT, DOUBLE, VARCHAR, etc.
"""

from typing import Any, Callable, Dict, List, Optional

from pymilvus.client.types import DataType
from pymilvus.exceptions import DataNotMatchException, ExceptionsMessage
from pymilvus.grpc_gen import schema_pb2
from pymilvus.settings import Config

from .base import DataTypeHandler


class ScalarHandler(DataTypeHandler):
    """Base for scalar types with common logic."""
    
    def create_accessor(
        self, payload: Any, start: int, valid_data: Optional[Any] = None
    ) -> Callable[[int], Any]:
        if valid_data is not None:
            def accessor(i: int) -> Any:
                idx = i + start
                return payload[idx] if valid_data[idx] else None
            return accessor
        def accessor(i: int) -> Any:
            return payload[i + start]
        return accessor
    
    def get_slice(self, payload: Any, start: int, end: int) -> List[Any]:
        return list(payload[start:end])


class BoolHandler(ScalarHandler):
    supported_types = (DataType.BOOL,)
    
    def extract_payload(self, field_data: schema_pb2.FieldData) -> Any:
        return field_data.scalars.bool_data.data
    
    def pack_single(self, value: Any, field_data: schema_pb2.FieldData, field_info: Dict) -> None:
        if value is None:
            field_data.scalars.bool_data.data.extend([])
        else:
            field_data.scalars.bool_data.data.append(value)
    
    def pack_batch(self, values: List, field_data: schema_pb2.FieldData, field_info: Dict, valid_data: Optional[List[bool]] = None) -> None:
        if valid_data:
            values = [v for i, v in enumerate(values) if valid_data[i]]
        field_data.scalars.bool_data.data.extend(values)


class IntHandler(ScalarHandler):
    """Handler for INT8, INT16, INT32 (share int_data)."""
    supported_types = (DataType.INT8, DataType.INT16, DataType.INT32)
    
    def extract_payload(self, field_data: schema_pb2.FieldData) -> Any:
        return field_data.scalars.int_data.data
    
    def pack_single(self, value: Any, field_data: schema_pb2.FieldData, field_info: Dict) -> None:
        if value is None:
            field_data.scalars.int_data.data.extend([])
        else:
            field_data.scalars.int_data.data.append(value)
    
    def pack_batch(self, values: List, field_data: schema_pb2.FieldData, field_info: Dict, valid_data: Optional[List[bool]] = None) -> None:
        if valid_data:
            values = [v for i, v in enumerate(values) if valid_data[i]]
        field_data.scalars.int_data.data.extend(values)


class Int64Handler(ScalarHandler):
    supported_types = (DataType.INT64,)
    
    def extract_payload(self, field_data: schema_pb2.FieldData) -> Any:
        return field_data.scalars.long_data.data
    
    def pack_single(self, value: Any, field_data: schema_pb2.FieldData, field_info: Dict) -> None:
        if value is None:
            field_data.scalars.long_data.data.extend([])
        else:
            field_data.scalars.long_data.data.append(value)
    
    def pack_batch(self, values: List, field_data: schema_pb2.FieldData, field_info: Dict, valid_data: Optional[List[bool]] = None) -> None:
        if valid_data:
            values = [v for i, v in enumerate(values) if valid_data[i]]
        field_data.scalars.long_data.data.extend(values)


class FloatHandler(ScalarHandler):
    supported_types = (DataType.FLOAT,)
    
    def extract_payload(self, field_data: schema_pb2.FieldData) -> Any:
        return field_data.scalars.float_data.data
    
    def pack_single(self, value: Any, field_data: schema_pb2.FieldData, field_info: Dict) -> None:
        if value is None:
            field_data.scalars.float_data.data.extend([])
        else:
            field_data.scalars.float_data.data.append(value)
    
    def pack_batch(self, values: List, field_data: schema_pb2.FieldData, field_info: Dict, valid_data: Optional[List[bool]] = None) -> None:
        if valid_data:
            values = [v for i, v in enumerate(values) if valid_data[i]]
        field_data.scalars.float_data.data.extend(values)


class DoubleHandler(ScalarHandler):
    supported_types = (DataType.DOUBLE,)
    
    def extract_payload(self, field_data: schema_pb2.FieldData) -> Any:
        return field_data.scalars.double_data.data
    
    def pack_single(self, value: Any, field_data: schema_pb2.FieldData, field_info: Dict) -> None:
        if value is None:
            field_data.scalars.double_data.data.extend([])
        else:
            field_data.scalars.double_data.data.append(value)
    
    def pack_batch(self, values: List, field_data: schema_pb2.FieldData, field_info: Dict, valid_data: Optional[List[bool]] = None) -> None:
        if valid_data:
            values = [v for i, v in enumerate(values) if valid_data[i]]
        field_data.scalars.double_data.data.extend(values)


class VarCharHandler(ScalarHandler):
    supported_types = (DataType.VARCHAR, DataType.STRING)
    
    def extract_payload(self, field_data: schema_pb2.FieldData) -> Any:
        return field_data.scalars.string_data.data
    
    def pack_single(self, value: Any, field_data: schema_pb2.FieldData, field_info: Dict) -> None:
        if value is None:
            field_data.scalars.string_data.data.extend([])
        else:
            v = value
            if Config.EncodeProtocol.lower() != "utf-8".lower():
                v = value.encode(Config.EncodeProtocol)
            field_data.scalars.string_data.data.append(v)
    
    def pack_batch(self, values: List, field_data: schema_pb2.FieldData, field_info: Dict, valid_data: Optional[List[bool]] = None) -> None:
        if valid_data:
            values = [v for i, v in enumerate(values) if valid_data[i]]
        if Config.EncodeProtocol.lower() != "utf-8".lower():
            values = [v.encode(Config.EncodeProtocol) for v in values]
        field_data.scalars.string_data.data.extend(values)


class TimestampTzHandler(ScalarHandler):
    supported_types = (DataType.TIMESTAMPTZ,)
    
    def extract_payload(self, field_data: schema_pb2.FieldData) -> Any:
        return field_data.scalars.string_data.data
    
    def pack_single(self, value: Any, field_data: schema_pb2.FieldData, field_info: Dict) -> None:
        if value is None:
            field_data.scalars.string_data.data.extend([])
        else:
            field_data.scalars.string_data.data.append(value)
    
    def pack_batch(self, values: List, field_data: schema_pb2.FieldData, field_info: Dict, valid_data: Optional[List[bool]] = None) -> None:
        if valid_data:
            values = [v for i, v in enumerate(values) if valid_data[i]]
        field_data.scalars.string_data.data.extend(values)


class GeometryHandler(ScalarHandler):
    supported_types = (DataType.GEOMETRY,)
    
    def extract_payload(self, field_data: schema_pb2.FieldData) -> Any:
        return field_data.scalars.geometry_wkt_data.data
    
    def pack_single(self, value: Any, field_data: schema_pb2.FieldData, field_info: Dict) -> None:
        if value is None:
            field_data.scalars.geometry_wkt_data.data.extend([])
        else:
            field_data.scalars.geometry_wkt_data.data.append(value)
    
    def pack_batch(self, values: List, field_data: schema_pb2.FieldData, field_info: Dict, valid_data: Optional[List[bool]] = None) -> None:
        if valid_data:
            values = [v for i, v in enumerate(values) if valid_data[i]]
        field_data.scalars.geometry_wkt_data.data.extend(values)
