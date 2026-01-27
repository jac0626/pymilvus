"""
Scalar type handlers for primitive types.

Handles: BOOL, INT8, INT16, INT32, INT64, FLOAT, DOUBLE, VARCHAR, GEOMETRY, TIMESTAMPTZ
"""

from typing import Any, Dict, List, Tuple

from pymilvus.client.types import DataType
from pymilvus.exceptions import DataNotMatchException, ExceptionsMessage, ParamError
from pymilvus.grpc_gen import schema_pb2
from pymilvus.settings import Config

from .base import TypeHandler
from .context import ExtractContext, PackContext


class ScalarHandler(TypeHandler):
    """Base class for scalar type handlers."""

    # Override in subclasses
    type_name: str = "scalar"

    def _handle_pack_error(
        self, field_name: str, value: Any, error: Exception
    ) -> None:
        """Raise a consistent error for pack failures."""
        raise DataNotMatchException(
            message=ExceptionsMessage.FieldDataInconsistent
            % (field_name, self.type_name, type(value))
            + f" Detail: {error!s}"
        ) from error


class BoolHandler(ScalarHandler):
    """Handler for BOOL type."""

    supported_types = (DataType.BOOL,)
    type_name = "bool"

    def pack_value(
        self,
        value: Any,
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
        context: PackContext,
    ) -> None:
        try:
            if value is None:
                field_data.scalars.bool_data.data.extend([])
            else:
                field_data.scalars.bool_data.data.append(value)
        except (TypeError, ValueError) as e:
            self._handle_pack_error(self.get_field_name(field_info), value, e)

    def pack_values(
        self,
        values: List[Any],
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        field_data.scalars.bool_data.data.extend(values)

    def get_raw_data(self, field_data: schema_pb2.FieldData) -> Any:
        return field_data.scalars.bool_data.data

    def extract_value(
        self,
        field_data: schema_pb2.FieldData,
        index: int,
        context: ExtractContext,
    ) -> Any:
        if self.is_nullable_null(field_data, index):
            return None
        data = field_data.scalars.bool_data.data
        if len(data) > index:
            return data[index]
        return None

    def extract_range(
        self,
        field_data: schema_pb2.FieldData,
        start: int,
        end: int,
    ) -> Tuple[List[Any], schema_pb2.FieldData]:
        from pymilvus.client.search_result import apply_valid_data

        data = list(field_data.scalars.bool_data.data[start:end])
        data = apply_valid_data(data, field_data.valid_data, start, end)
        field_meta = schema_pb2.FieldData(
            type=field_data.type,
            field_name=field_data.field_name,
            field_id=field_data.field_id,
            is_dynamic=field_data.is_dynamic,
        )
        return data, field_meta


class IntHandler(ScalarHandler):
    """Handler for INT8, INT16, INT32 types."""

    supported_types = (DataType.INT8, DataType.INT16, DataType.INT32)
    type_name = "int"

    def pack_value(
        self,
        value: Any,
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
        context: PackContext,
    ) -> None:
        try:
            if value is None:
                field_data.scalars.int_data.data.extend([])
            else:
                field_data.scalars.int_data.data.append(value)
        except (TypeError, ValueError) as e:
            self._handle_pack_error(self.get_field_name(field_info), value, e)

    def pack_values(
        self,
        values: List[Any],
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        field_data.scalars.int_data.data.extend(values)

    def get_raw_data(self, field_data: schema_pb2.FieldData) -> Any:
        return field_data.scalars.int_data.data

    def extract_value(
        self,
        field_data: schema_pb2.FieldData,
        index: int,
        context: ExtractContext,
    ) -> Any:
        if self.is_nullable_null(field_data, index):
            return None
        data = field_data.scalars.int_data.data
        if len(data) > index:
            return data[index]
        return None

    def extract_range(
        self,
        field_data: schema_pb2.FieldData,
        start: int,
        end: int,
    ) -> Tuple[List[Any], schema_pb2.FieldData]:
        from pymilvus.client.search_result import apply_valid_data

        data = list(field_data.scalars.int_data.data[start:end])
        data = apply_valid_data(data, field_data.valid_data, start, end)
        field_meta = schema_pb2.FieldData(
            type=field_data.type,
            field_name=field_data.field_name,
            field_id=field_data.field_id,
            is_dynamic=field_data.is_dynamic,
        )
        return data, field_meta


class Int64Handler(ScalarHandler):
    """Handler for INT64 type."""

    supported_types = (DataType.INT64,)
    type_name = "int64"

    def pack_value(
        self,
        value: Any,
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
        context: PackContext,
    ) -> None:
        try:
            if value is None:
                field_data.scalars.long_data.data.extend([])
            else:
                field_data.scalars.long_data.data.append(value)
        except (TypeError, ValueError) as e:
            self._handle_pack_error(self.get_field_name(field_info), value, e)

    def pack_values(
        self,
        values: List[Any],
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        field_data.scalars.long_data.data.extend(values)

    def get_raw_data(self, field_data: schema_pb2.FieldData) -> Any:
        return field_data.scalars.long_data.data

    def extract_value(
        self,
        field_data: schema_pb2.FieldData,
        index: int,
        context: ExtractContext,
    ) -> Any:
        if self.is_nullable_null(field_data, index):
            return None
        data = field_data.scalars.long_data.data
        if len(data) > index:
            return data[index]
        return None

    def extract_range(
        self,
        field_data: schema_pb2.FieldData,
        start: int,
        end: int,
    ) -> Tuple[List[Any], schema_pb2.FieldData]:
        from pymilvus.client.search_result import apply_valid_data

        data = list(field_data.scalars.long_data.data[start:end])
        data = apply_valid_data(data, field_data.valid_data, start, end)
        field_meta = schema_pb2.FieldData(
            type=field_data.type,
            field_name=field_data.field_name,
            field_id=field_data.field_id,
            is_dynamic=field_data.is_dynamic,
        )
        return data, field_meta


class FloatHandler(ScalarHandler):
    """Handler for FLOAT type."""

    supported_types = (DataType.FLOAT,)
    type_name = "float"

    def pack_value(
        self,
        value: Any,
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
        context: PackContext,
    ) -> None:
        try:
            if value is None:
                field_data.scalars.float_data.data.extend([])
            else:
                field_data.scalars.float_data.data.append(value)
        except (TypeError, ValueError) as e:
            self._handle_pack_error(self.get_field_name(field_info), value, e)

    def pack_values(
        self,
        values: List[Any],
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        field_data.scalars.float_data.data.extend(values)

    def get_raw_data(self, field_data: schema_pb2.FieldData) -> Any:
        return field_data.scalars.float_data.data

    def extract_value(
        self,
        field_data: schema_pb2.FieldData,
        index: int,
        context: ExtractContext,
    ) -> Any:
        import numpy as np

        if self.is_nullable_null(field_data, index):
            return None
        data = field_data.scalars.float_data.data
        if len(data) > index:
            return np.single(data[index])
        return None

    def extract_range(
        self,
        field_data: schema_pb2.FieldData,
        start: int,
        end: int,
    ) -> Tuple[List[Any], schema_pb2.FieldData]:
        from pymilvus.client.search_result import apply_valid_data

        data = list(field_data.scalars.float_data.data[start:end])
        data = apply_valid_data(data, field_data.valid_data, start, end)
        field_meta = schema_pb2.FieldData(
            type=field_data.type,
            field_name=field_data.field_name,
            field_id=field_data.field_id,
            is_dynamic=field_data.is_dynamic,
        )
        return data, field_meta


class DoubleHandler(ScalarHandler):
    """Handler for DOUBLE type."""

    supported_types = (DataType.DOUBLE,)
    type_name = "double"

    def pack_value(
        self,
        value: Any,
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
        context: PackContext,
    ) -> None:
        try:
            if value is None:
                field_data.scalars.double_data.data.extend([])
            else:
                field_data.scalars.double_data.data.append(value)
        except (TypeError, ValueError) as e:
            self._handle_pack_error(self.get_field_name(field_info), value, e)

    def pack_values(
        self,
        values: List[Any],
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        field_data.scalars.double_data.data.extend(values)

    def get_raw_data(self, field_data: schema_pb2.FieldData) -> Any:
        return field_data.scalars.double_data.data

    def extract_value(
        self,
        field_data: schema_pb2.FieldData,
        index: int,
        context: ExtractContext,
    ) -> Any:
        if self.is_nullable_null(field_data, index):
            return None
        data = field_data.scalars.double_data.data
        if len(data) > index:
            return data[index]
        return None

    def extract_range(
        self,
        field_data: schema_pb2.FieldData,
        start: int,
        end: int,
    ) -> Tuple[List[Any], schema_pb2.FieldData]:
        from pymilvus.client.search_result import apply_valid_data

        data = list(field_data.scalars.double_data.data[start:end])
        data = apply_valid_data(data, field_data.valid_data, start, end)
        field_meta = schema_pb2.FieldData(
            type=field_data.type,
            field_name=field_data.field_name,
            field_id=field_data.field_id,
            is_dynamic=field_data.is_dynamic,
        )
        return data, field_meta


def _get_max_len_of_var_char(field_info: Dict) -> int:
    """Get the maximum length of a VARCHAR field."""
    k = Config.MaxVarCharLengthKey
    v = Config.MaxVarCharLength
    return field_info.get("params", {}).get(k, v)


def _convert_to_str(value: Any, field_info: Dict, check: bool = True) -> str:
    """Convert a value to string with validation."""
    if check:
        if not isinstance(value, str):
            raise DataNotMatchException(
                message=ExceptionsMessage.FieldDataInconsistent
                % (field_info.get("name", ""), "varchar", type(value))
            )

    if Config.EncodeProtocol.lower() != "utf-8".lower():
        value = value.encode(Config.EncodeProtocol)
        max_len = int(_get_max_len_of_var_char(field_info))
        if len(value) > max_len:
            raise ParamError(
                message=f"invalid input of field ({field_info['name']}), "
                f"length of string exceeds max length. length: {len(value)}, max length: {max_len}"
            )
    return value


class VarCharHandler(ScalarHandler):
    """Handler for VARCHAR type."""

    supported_types = (DataType.VARCHAR,)
    type_name = "varchar"

    def pack_value(
        self,
        value: Any,
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
        context: PackContext,
    ) -> None:
        try:
            if value is None:
                field_data.scalars.string_data.data.extend([])
            else:
                converted = _convert_to_str(value, field_info, check=True)
                field_data.scalars.string_data.data.append(converted)
        except (TypeError, ValueError) as e:
            self._handle_pack_error(self.get_field_name(field_info), value, e)

    def pack_values(
        self,
        values: List[Any],
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        converted = [_convert_to_str(v, field_info, check=True) for v in values]
        field_data.scalars.string_data.data.extend(converted)

    def get_raw_data(self, field_data: schema_pb2.FieldData) -> Any:
        return field_data.scalars.string_data.data

    def extract_value(
        self,
        field_data: schema_pb2.FieldData,
        index: int,
        context: ExtractContext,
    ) -> Any:
        if self.is_nullable_null(field_data, index):
            return None
        data = field_data.scalars.string_data.data
        if len(data) > index:
            return data[index]
        return None

    def extract_range(
        self,
        field_data: schema_pb2.FieldData,
        start: int,
        end: int,
    ) -> Tuple[List[Any], schema_pb2.FieldData]:
        from pymilvus.client.search_result import apply_valid_data

        data = list(field_data.scalars.string_data.data[start:end])
        data = apply_valid_data(data, field_data.valid_data, start, end)
        field_meta = schema_pb2.FieldData(
            type=field_data.type,
            field_name=field_data.field_name,
            field_id=field_data.field_id,
            is_dynamic=field_data.is_dynamic,
        )
        return data, field_meta


class GeometryHandler(ScalarHandler):
    """Handler for GEOMETRY type."""

    supported_types = (DataType.GEOMETRY,)
    type_name = "geometry"

    def pack_value(
        self,
        value: Any,
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
        context: PackContext,
    ) -> None:
        try:
            if value is None:
                field_data.scalars.geometry_wkt_data.data.extend([])
            else:
                converted = _convert_to_str(value, field_info, check=True)
                field_data.scalars.geometry_wkt_data.data.append(converted)
        except (TypeError, ValueError) as e:
            self._handle_pack_error(self.get_field_name(field_info), value, e)

    def pack_values(
        self,
        values: List[Any],
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        converted = [_convert_to_str(v, field_info, check=True) for v in values]
        field_data.scalars.geometry_wkt_data.data.extend(converted)

    def get_raw_data(self, field_data: schema_pb2.FieldData) -> Any:
        return field_data.scalars.geometry_wkt_data.data

    def extract_value(
        self,
        field_data: schema_pb2.FieldData,
        index: int,
        context: ExtractContext,
    ) -> Any:
        if self.is_nullable_null(field_data, index):
            return None
        data = field_data.scalars.geometry_wkt_data.data
        if len(data) > index:
            return data[index]
        return None

    def extract_range(
        self,
        field_data: schema_pb2.FieldData,
        start: int,
        end: int,
    ) -> Tuple[List[Any], schema_pb2.FieldData]:
        from pymilvus.client.search_result import apply_valid_data

        data = list(field_data.scalars.geometry_wkt_data.data[start:end])
        data = apply_valid_data(data, field_data.valid_data, start, end)
        field_meta = schema_pb2.FieldData(
            type=field_data.type,
            field_name=field_data.field_name,
            field_id=field_data.field_id,
            is_dynamic=field_data.is_dynamic,
        )
        return data, field_meta


class TimestampTzHandler(ScalarHandler):
    """Handler for TIMESTAMPTZ type."""

    supported_types = (DataType.TIMESTAMPTZ,)
    type_name = "string"  # TIMESTAMPTZ is passed as string

    def pack_value(
        self,
        value: Any,
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
        context: PackContext,
    ) -> None:
        try:
            if value is None:
                field_data.scalars.string_data.data.extend([])
            else:
                field_data.scalars.string_data.data.append(value)
        except (TypeError, ValueError) as e:
            self._handle_pack_error(self.get_field_name(field_info), value, e)

    def pack_values(
        self,
        values: List[Any],
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        field_data.scalars.string_data.data.extend(values)

    def get_raw_data(self, field_data: schema_pb2.FieldData) -> Any:
        return field_data.scalars.string_data.data

    def extract_value(
        self,
        field_data: schema_pb2.FieldData,
        index: int,
        context: ExtractContext,
    ) -> Any:
        if self.is_nullable_null(field_data, index):
            return None
        data = field_data.scalars.string_data.data
        if len(data) > index:
            return data[index]
        return None

    def extract_range(
        self,
        field_data: schema_pb2.FieldData,
        start: int,
        end: int,
    ) -> Tuple[List[Any], schema_pb2.FieldData]:
        from pymilvus.client.search_result import apply_valid_data

        data = list(field_data.scalars.string_data.data[start:end])
        data = apply_valid_data(data, field_data.valid_data, start, end)
        field_meta = schema_pb2.FieldData(
            type=field_data.type,
            field_name=field_data.field_name,
            field_id=field_data.field_id,
            is_dynamic=field_data.is_dynamic,
        )
        return data, field_meta
