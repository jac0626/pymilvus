"""
Complex type handlers for JSON, ARRAY, and struct types.

Handles: JSON, ARRAY, _ARRAY_OF_STRUCT, _ARRAY_OF_VECTOR
"""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import orjson

from pymilvus.client.types import DataType
from pymilvus.exceptions import DataNotMatchException, ExceptionsMessage, ParamError
from pymilvus.grpc_gen import schema_pb2

from .base import TypeHandler
from .context import ExtractContext, PackContext

logger = logging.getLogger(__name__)


class JsonHandler(TypeHandler):
    """Handler for JSON type."""

    supported_types = (DataType.JSON,)

    def pack_value(
        self,
        value: Any,
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
        context: PackContext,
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
                % (field_info.get("name", ""), "json", type(value))
                + f" Detail: {e!s}"
            ) from e

    def pack_values(
        self,
        values: List[Any],
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        from pymilvus.client.entity_helper import entity_to_json_arr

        field_data.scalars.json_data.data.extend(entity_to_json_arr(values, field_info))

    def get_raw_data(self, field_data: schema_pb2.FieldData) -> Any:
        return field_data.scalars.json_data.data

    def extract_value(
        self,
        field_data: schema_pb2.FieldData,
        index: int,
        context: ExtractContext,
    ) -> Any:
        if self.is_nullable_null(field_data, index):
            return None

        data = field_data.scalars.json_data.data
        if len(data) <= index:
            return None

        try:
            json_dict = orjson.loads(data[index])
        except Exception as e:
            logger.error(
                f"JsonHandler::extract_value::Failed to load JSON data: {e}, original data: {data[index]}"
            )
            raise

        return json_dict

    def extract_into_row(
        self,
        field_data: schema_pb2.FieldData,
        row_dict: Dict[str, Any],
        index: int,
        context: ExtractContext,
    ) -> bool:
        if self.is_nullable_null(field_data, index):
            row_dict[field_data.field_name] = None
            return False

        data = field_data.scalars.json_data.data
        if len(data) <= index:
            return False

        try:
            json_dict = orjson.loads(data[index])
        except Exception as e:
            logger.error(
                f"JsonHandler::extract_into_row::Failed to load JSON data: {e}, original data: {data[index]}"
            )
            raise

        if not field_data.is_dynamic:
            row_dict[field_data.field_name] = json_dict
            return False

        # Handle dynamic fields
        if not context.dynamic_output_fields:
            row_dict.update(json_dict)
        else:
            row_dict.update({k: v for k, v in json_dict.items() if k in context.dynamic_output_fields})
        return False

    def extract_range(
        self,
        field_data: schema_pb2.FieldData,
        start: int,
        end: int,
    ) -> Tuple[List[Any], schema_pb2.FieldData]:
        from pymilvus.client.search_result import apply_valid_data

        res = apply_valid_data(
            list(field_data.scalars.json_data.data[start:end]),
            field_data.valid_data,
            start,
            end,
        )
        json_dict_list = []
        for item in res:
            if item is not None:
                try:
                    json_dict_list.append(orjson.loads(item))
                except Exception as e:
                    logger.error(
                        f"JsonHandler::extract_range::Failed to load JSON item: {e}, original item: {item}"
                    )
                    raise
            else:
                json_dict_list.append(item)

        field_meta = schema_pb2.FieldData(
            type=field_data.type,
            field_name=field_data.field_name,
            field_id=field_data.field_id,
            is_dynamic=field_data.is_dynamic,
        )
        return json_dict_list, field_meta


class ArrayHandler(TypeHandler):
    """Handler for ARRAY type."""

    supported_types = (DataType.ARRAY,)

    def pack_value(
        self,
        value: Any,
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
        context: PackContext,
    ) -> None:
        from pymilvus.client.entity_helper import convert_to_array

        try:
            if value is None:
                field_data.scalars.array_data.data.extend([])
            else:
                field_data.scalars.array_data.data.append(convert_to_array(value, field_info))
        except (TypeError, ValueError) as e:
            raise DataNotMatchException(
                message=ExceptionsMessage.FieldDataInconsistent
                % (field_info.get("name", ""), "array", type(value))
                + f" Detail: {e!s}"
            ) from e

    def pack_values(
        self,
        values: List[Any],
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        from pymilvus.client.entity_helper import entity_to_array_arr

        field_data.scalars.array_data.data.extend(entity_to_array_arr(values, field_info))

    def get_raw_data(self, field_data: schema_pb2.FieldData) -> Any:
        return field_data.scalars.array_data.data

    def extract_value(
        self,
        field_data: schema_pb2.FieldData,
        index: int,
        context: ExtractContext,
    ) -> Any:
        if self.is_nullable_null(field_data, index):
            return None

        return self._extract_array_at_index(field_data, index)

    def _extract_array_at_index(self, field_data: schema_pb2.FieldData, index: int) -> Any:
        """Extract array data at given index."""
        data = field_data.scalars.array_data.data
        if len(data) <= index:
            return None

        array = data[index]
        element_type = field_data.scalars.array_data.element_type

        if element_type == DataType.INT64:
            return list(array.long_data.data)
        if element_type == DataType.BOOL:
            return list(array.bool_data.data)
        if element_type in (DataType.INT8, DataType.INT16, DataType.INT32):
            return list(array.int_data.data)
        if element_type == DataType.FLOAT:
            return list(array.float_data.data)
        if element_type == DataType.DOUBLE:
            return list(array.double_data.data)
        if element_type in (DataType.STRING, DataType.VARCHAR):
            return list(array.string_data.data)
        return None

    def extract_range(
        self,
        field_data: schema_pb2.FieldData,
        start: int,
        end: int,
    ) -> Tuple[List[Any], schema_pb2.FieldData]:
        from pymilvus.client.search_result import apply_valid_data, extract_array_row_data

        res = apply_valid_data(
            list(field_data.scalars.array_data.data[start:end]),
            field_data.valid_data,
            start,
            end,
        )
        field_meta = schema_pb2.FieldData(
            type=field_data.type,
            field_name=field_data.field_name,
            field_id=field_data.field_id,
            is_dynamic=field_data.is_dynamic,
        )
        return extract_array_row_data(res, field_data.scalars.array_data.element_type), field_meta


class ArrayOfStructHandler(TypeHandler):
    """Handler for _ARRAY_OF_STRUCT type."""

    supported_types = (DataType._ARRAY_OF_STRUCT,)

    def pack_value(
        self,
        value: Any,
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
        context: PackContext,
    ) -> None:
        # This is an internal type, packing is handled elsewhere
        raise NotImplementedError("_ARRAY_OF_STRUCT packing is not directly supported")

    def pack_values(
        self,
        values: List[Any],
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        raise NotImplementedError("_ARRAY_OF_STRUCT packing is not directly supported")

    def get_raw_data(self, field_data: schema_pb2.FieldData) -> Any:
        return field_data.struct_arrays

    def extract_value(
        self,
        field_data: schema_pb2.FieldData,
        index: int,
        context: ExtractContext,
    ) -> Any:
        from pymilvus.client.entity_helper import extract_struct_array_from_column_data

        if not hasattr(field_data, "struct_arrays") or not field_data.struct_arrays:
            return None
        return extract_struct_array_from_column_data(field_data.struct_arrays, index)

    def extract_into_row(
        self,
        field_data: schema_pb2.FieldData,
        row_dict: Dict[str, Any],
        index: int,
        context: ExtractContext,
    ) -> bool:
        # Return True to indicate lazy extraction
        return True

    def extract_range(
        self,
        field_data: schema_pb2.FieldData,
        start: int,
        end: int,
    ) -> Tuple[List[Any], schema_pb2.FieldData]:
        from pymilvus.client.entity_helper import extract_struct_array_from_column_data

        struct_array_data = []
        if hasattr(field_data, "struct_arrays") and field_data.struct_arrays:
            for row_idx in range(start, end):
                struct_array_data.append(
                    extract_struct_array_from_column_data(field_data.struct_arrays, row_idx)
                )

        field_meta = schema_pb2.FieldData(
            type=field_data.type,
            field_name=field_data.field_name,
            field_id=field_data.field_id,
            is_dynamic=field_data.is_dynamic,
        )
        return struct_array_data, field_meta


class ArrayOfVectorHandler(TypeHandler):
    """Handler for _ARRAY_OF_VECTOR type."""

    supported_types = (DataType._ARRAY_OF_VECTOR,)

    def pack_value(
        self,
        value: Any,
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
        context: PackContext,
    ) -> None:
        # This is an internal type, packing is handled elsewhere
        raise NotImplementedError("_ARRAY_OF_VECTOR packing is not directly supported")

    def pack_values(
        self,
        values: List[Any],
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        raise NotImplementedError("_ARRAY_OF_VECTOR packing is not directly supported")

    def get_raw_data(self, field_data: schema_pb2.FieldData) -> Any:
        return field_data.vectors.vector_array

    def extract_value(
        self,
        field_data: schema_pb2.FieldData,
        index: int,
        context: ExtractContext,
    ) -> Any:
        """Extract array of vectors at given index."""
        if not hasattr(field_data, "vectors") or not hasattr(field_data.vectors, "vector_array"):
            return []

        vector_array = field_data.vectors.vector_array
        if index >= len(vector_array.data):
            return []

        vector_data = vector_array.data[index]
        element_type = vector_array.element_type

        return self._extract_vectors(vector_data, element_type)

    def _extract_vectors(self, vector_data: Any, element_type: DataType) -> List[Any]:
        """Extract vectors from vector_data based on element_type."""
        dim = vector_data.dim

        if element_type == DataType.FLOAT_VECTOR:
            float_data = vector_data.float_vector.data
            num_vectors = len(float_data) // dim
            return [
                list(float_data[i * dim : (i + 1) * dim])
                for i in range(num_vectors)
            ]

        if element_type == DataType.FLOAT16_VECTOR:
            byte_data = vector_data.float16_vector
            bytes_per_vec = dim * 2
            num_vectors = len(byte_data) // bytes_per_vec
            return [
                list(np.frombuffer(byte_data[i * bytes_per_vec : (i + 1) * bytes_per_vec], dtype=np.float16))
                for i in range(num_vectors)
            ]

        if element_type == DataType.BFLOAT16_VECTOR:
            byte_data = vector_data.bfloat16_vector
            bytes_per_vec = dim * 2
            num_vectors = len(byte_data) // bytes_per_vec
            dtype = "bfloat16" if hasattr(np, "bfloat16") else np.uint16
            return [
                list(np.frombuffer(byte_data[i * bytes_per_vec : (i + 1) * bytes_per_vec], dtype=dtype))
                for i in range(num_vectors)
            ]

        if element_type == DataType.INT8_VECTOR:
            byte_data = vector_data.int8_vector
            num_vectors = len(byte_data) // dim
            return [
                list(np.frombuffer(byte_data[i * dim : (i + 1) * dim], dtype=np.int8))
                for i in range(num_vectors)
            ]

        if element_type == DataType.BINARY_VECTOR:
            byte_data = vector_data.binary_vector
            bytes_per_vec = dim // 8
            num_vectors = len(byte_data) // bytes_per_vec
            return [
                [byte_data[i * bytes_per_vec : (i + 1) * bytes_per_vec]]
                for i in range(num_vectors)
            ]

        raise ParamError(message=f"Unsupported element type: {element_type} for vector array extraction")

    def extract_into_row(
        self,
        field_data: schema_pb2.FieldData,
        row_dict: Dict[str, Any],
        index: int,
        context: ExtractContext,
    ) -> bool:
        # Return True to indicate lazy extraction
        return True

    def extract_range(
        self,
        field_data: schema_pb2.FieldData,
        start: int,
        end: int,
    ) -> Tuple[List[Any], schema_pb2.FieldData]:
        context = ExtractContext()
        data = [self.extract_value(field_data, i, context) for i in range(start, end)]
        field_meta = schema_pb2.FieldData(
            type=field_data.type,
            field_name=field_data.field_name,
            field_id=field_data.field_id,
            is_dynamic=field_data.is_dynamic,
        )
        return data, field_meta
