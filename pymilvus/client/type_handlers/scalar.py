"""
Scalar Type Handlers - BOOL, INT8-64, FLOAT, DOUBLE, VARCHAR, etc.

These handlers manage simple scalar types that map directly to protobuf
scalar fields.
"""

from typing import Any, Callable, Dict, List

from pymilvus.client.types import DataType
from pymilvus.exceptions import DataNotMatchException, ExceptionsMessage
from pymilvus.grpc_gen import schema_pb2
from pymilvus.settings import Config

from .base import TypeHandler


class ScalarHandler(TypeHandler):
    """Base class for scalar type handlers with common logic."""
    
    def get_dim(self, field_data: schema_pb2.FieldData) -> int:
        return 0  # Scalars have no dimension
    
    def create_accessor(
        self, data: Any, start: int, dim: int = 0
    ) -> Callable[[int], Any]:
        """Optimized accessor for scalars - simple list indexing."""
        def accessor(i: int) -> Any:
            return data[i + start]
        return accessor
    
    def get_slice(self, data: Any, start: int, end: int, dim: int = 0) -> Any:
        """Efficient slice for scalar lists."""
        return data[start:end]


class BoolHandler(ScalarHandler):
    """Handler for BOOL type."""
    
    supported_types = (DataType.BOOL,)
    
    def extract_data(self, field_data: schema_pb2.FieldData) -> Any:
        return field_data.scalars.bool_data.data
    
    def get_value(self, data: Any, index: int, dim: int = 0) -> bool:
        return data[index]
    
    def pack_value(
        self,
        value: Any,
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        try:
            if value is None:
                field_data.scalars.bool_data.data.extend([])
            else:
                field_data.scalars.bool_data.data.append(value)
        except (TypeError, ValueError) as e:
            raise DataNotMatchException(
                message=ExceptionsMessage.FieldDataInconsistent
                % (field_info["name"], "bool", type(value))
                + f" Detail: {e!s}"
            ) from e
    
    def pack_values(
        self,
        values: List[Any],
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        field_data.scalars.bool_data.data.extend(values)


class IntHandler(ScalarHandler):
    """Base handler for integer types sharing int_data."""
    
    def extract_data(self, field_data: schema_pb2.FieldData) -> Any:
        return field_data.scalars.int_data.data
    
    def get_value(self, data: Any, index: int, dim: int = 0) -> int:
        return data[index]
    
    def pack_value(
        self,
        value: Any,
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        try:
            if value is None:
                field_data.scalars.int_data.data.extend([])
            else:
                field_data.scalars.int_data.data.append(value)
        except (TypeError, ValueError) as e:
            raise DataNotMatchException(
                message=ExceptionsMessage.FieldDataInconsistent
                % (field_info["name"], "int", type(value))
                + f" Detail: {e!s}"
            ) from e
    
    def pack_values(
        self,
        values: List[Any],
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        field_data.scalars.int_data.data.extend(values)


class Int8Handler(IntHandler):
    supported_types = (DataType.INT8,)


class Int16Handler(IntHandler):
    supported_types = (DataType.INT16,)


class Int32Handler(IntHandler):
    supported_types = (DataType.INT32,)


class Int64Handler(ScalarHandler):
    """Handler for INT64 type (uses long_data)."""
    
    supported_types = (DataType.INT64,)
    
    def extract_data(self, field_data: schema_pb2.FieldData) -> Any:
        return field_data.scalars.long_data.data
    
    def get_value(self, data: Any, index: int, dim: int = 0) -> int:
        return data[index]
    
    def pack_value(
        self,
        value: Any,
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        try:
            if value is None:
                field_data.scalars.long_data.data.extend([])
            else:
                field_data.scalars.long_data.data.append(value)
        except (TypeError, ValueError) as e:
            raise DataNotMatchException(
                message=ExceptionsMessage.FieldDataInconsistent
                % (field_info["name"], "int64", type(value))
                + f" Detail: {e!s}"
            ) from e
    
    def pack_values(
        self,
        values: List[Any],
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        field_data.scalars.long_data.data.extend(values)


class FloatHandler(ScalarHandler):
    """Handler for FLOAT type."""
    
    supported_types = (DataType.FLOAT,)
    
    def extract_data(self, field_data: schema_pb2.FieldData) -> Any:
        return field_data.scalars.float_data.data
    
    def get_value(self, data: Any, index: int, dim: int = 0) -> float:
        return data[index]
    
    def pack_value(
        self,
        value: Any,
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        try:
            if value is None:
                field_data.scalars.float_data.data.extend([])
            else:
                field_data.scalars.float_data.data.append(value)
        except (TypeError, ValueError) as e:
            raise DataNotMatchException(
                message=ExceptionsMessage.FieldDataInconsistent
                % (field_info["name"], "float", type(value))
                + f" Detail: {e!s}"
            ) from e
    
    def pack_values(
        self,
        values: List[Any],
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        field_data.scalars.float_data.data.extend(values)


class DoubleHandler(ScalarHandler):
    """Handler for DOUBLE type."""
    
    supported_types = (DataType.DOUBLE,)
    
    def extract_data(self, field_data: schema_pb2.FieldData) -> Any:
        return field_data.scalars.double_data.data
    
    def get_value(self, data: Any, index: int, dim: int = 0) -> float:
        return data[index]
    
    def pack_value(
        self,
        value: Any,
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        try:
            if value is None:
                field_data.scalars.double_data.data.extend([])
            else:
                field_data.scalars.double_data.data.append(value)
        except (TypeError, ValueError) as e:
            raise DataNotMatchException(
                message=ExceptionsMessage.FieldDataInconsistent
                % (field_info["name"], "double", type(value))
                + f" Detail: {e!s}"
            ) from e
    
    def pack_values(
        self,
        values: List[Any],
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        field_data.scalars.double_data.data.extend(values)


class VarCharHandler(ScalarHandler):
    """Handler for VARCHAR/STRING type."""
    
    supported_types = (DataType.VARCHAR, DataType.STRING)
    
    def extract_data(self, field_data: schema_pb2.FieldData) -> Any:
        return field_data.scalars.string_data.data
    
    def get_value(self, data: Any, index: int, dim: int = 0) -> str:
        return data[index]
    
    def pack_value(
        self,
        value: Any,
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        try:
            if value is None:
                field_data.scalars.string_data.data.extend([])
            else:
                # Apply encoding if needed
                v = value
                if Config.EncodeProtocol.lower() != "utf-8".lower():
                    v = value.encode(Config.EncodeProtocol)
                field_data.scalars.string_data.data.append(v)
        except (TypeError, ValueError) as e:
            raise DataNotMatchException(
                message=ExceptionsMessage.FieldDataInconsistent
                % (field_info["name"], "varchar", type(value))
                + f" Detail: {e!s}"
            ) from e
    
    def pack_values(
        self,
        values: List[Any],
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        if Config.EncodeProtocol.lower() != "utf-8".lower():
            values = [v.encode(Config.EncodeProtocol) for v in values]
        field_data.scalars.string_data.data.extend(values)
    
    def validate(self, value: Any, field_info: Dict[str, Any]) -> bool:
        if not isinstance(value, str):
            return False
        max_len = field_info.get("params", {}).get(Config.MaxVarCharLengthKey, Config.MaxVarCharLength)
        return len(value) <= int(max_len)


class GeometryHandler(ScalarHandler):
    """Handler for GEOMETRY type (GeoJSON as WKT string)."""
    
    supported_types = (DataType.GEOMETRY,)
    
    def extract_data(self, field_data: schema_pb2.FieldData) -> Any:
        return field_data.scalars.geometry_wkt_data.data
    
    def get_value(self, data: Any, index: int, dim: int = 0) -> str:
        return data[index]
    
    def pack_value(
        self,
        value: Any,
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        try:
            if value is None:
                field_data.scalars.geometry_wkt_data.data.extend([])
            else:
                field_data.scalars.geometry_wkt_data.data.append(value)
        except (TypeError, ValueError) as e:
            raise DataNotMatchException(
                message=ExceptionsMessage.FieldDataInconsistent
                % (field_info["name"], "geometry", type(value))
            ) from e
    
    def pack_values(
        self,
        values: List[Any],
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        field_data.scalars.geometry_wkt_data.data.extend(values)


class TimestampTzHandler(ScalarHandler):
    """Handler for TIMESTAMPTZ type (stored as string)."""
    
    supported_types = (DataType.TIMESTAMPTZ,)
    
    def extract_data(self, field_data: schema_pb2.FieldData) -> Any:
        return field_data.scalars.string_data.data
    
    def get_value(self, data: Any, index: int, dim: int = 0) -> str:
        return data[index]
    
    def pack_value(
        self,
        value: Any,
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        try:
            if value is None:
                field_data.scalars.string_data.data.extend([])
            else:
                field_data.scalars.string_data.data.append(value)
        except (TypeError, ValueError) as e:
            raise DataNotMatchException(
                message=ExceptionsMessage.FieldDataInconsistent
                % (field_info["name"], "timestamptz", type(value))
            ) from e
    
    def pack_values(
        self,
        values: List[Any],
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        field_data.scalars.string_data.data.extend(values)
