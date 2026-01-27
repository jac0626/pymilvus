"""
Vector type handlers for dense and sparse vectors.

Handles: FLOAT_VECTOR, BINARY_VECTOR, FLOAT16_VECTOR, BFLOAT16_VECTOR, INT8_VECTOR, SPARSE_FLOAT_VECTOR
"""

from typing import Any, Dict, List, Tuple

import numpy as np

from pymilvus.client.types import DataType
from pymilvus.exceptions import DataNotMatchException, ExceptionsMessage, ParamError
from pymilvus.grpc_gen import schema_pb2

from .base import TypeHandler
from .context import ExtractContext, PackContext


class VectorHandler(TypeHandler):
    """Base class for vector type handlers."""

    type_name: str = "vector"

    def _handle_pack_error(
        self, field_name: str, value: Any, error: Exception
    ) -> None:
        """Raise a consistent error for pack failures."""
        raise DataNotMatchException(
            message=ExceptionsMessage.FieldDataInconsistent
            % (field_name, self.type_name, type(value))
            + f" Detail: {error!s}"
        ) from error

    def _get_dim_from_field_info(self, field_info: Dict[str, Any]) -> int:
        """Get dimension from field_info params."""
        return field_info.get("params", {}).get("dim", 0)


class FloatVectorHandler(VectorHandler):
    """Handler for FLOAT_VECTOR type."""

    supported_types = (DataType.FLOAT_VECTOR,)
    type_name = "float_vector"

    def pack_value(
        self,
        value: Any,
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
        context: PackContext,
    ) -> None:
        try:
            if value is None:
                if field_data.vectors.dim == 0:
                    field_data.vectors.dim = self._get_dim_from_field_info(field_info)
            else:
                f_value = value
                if isinstance(value, np.ndarray):
                    if value.dtype not in ("float32", "float64"):
                        raise ParamError(
                            message="invalid input for float32 vector. Expected an np.ndarray with dtype=float32"
                        )
                    f_value = value.tolist()

                field_data.vectors.dim = len(f_value)
                field_data.vectors.float_vector.data.extend(f_value)
        except (TypeError, ValueError) as e:
            self._handle_pack_error(self.get_field_name(field_info), value, e)

    def pack_values(
        self,
        values: List[Any],
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        if len(values) > 0:
            field_data.vectors.dim = len(values[0])
        else:
            field_data.vectors.dim = self._get_dim_from_field_info(field_info)
        all_floats = [f for vector in values for f in vector]
        field_data.vectors.float_vector.data.extend(all_floats)

    def get_raw_data(self, field_data: schema_pb2.FieldData) -> Any:
        return field_data.vectors.float_vector.data

    def extract_value(
        self,
        field_data: schema_pb2.FieldData,
        index: int,
        context: ExtractContext,
    ) -> Any:
        if self.is_nullable_null(field_data, index):
            return None

        dim = field_data.vectors.dim
        phys_idx = context.get_physical_index(field_data, index)

        if len(field_data.vectors.float_vector.data) >= (phys_idx + 1) * dim:
            start_pos, end_pos = phys_idx * dim, (phys_idx + 1) * dim
            arr = np.array(
                field_data.vectors.float_vector.data[start_pos:end_pos], dtype=np.float32
            )
            return list(arr)
        return None

    def extract_range(
        self,
        field_data: schema_pb2.FieldData,
        start: int,
        end: int,
    ) -> Tuple[List[Any], schema_pb2.FieldData]:
        dim = field_data.vectors.dim
        vectors = field_data.vectors
        field_meta = schema_pb2.FieldData(
            type=field_data.type,
            field_name=field_data.field_name,
            field_id=field_data.field_id,
            is_dynamic=field_data.is_dynamic,
        )
        field_meta.vectors.dim = dim

        # Optimization: if range equals full data, return directly
        if start == 0 and (end - start) * dim >= len(vectors.float_vector.data):
            return vectors.float_vector.data, field_meta
        return vectors.float_vector.data[start * dim : end * dim], field_meta

    def extract_into_row(
        self,
        field_data: schema_pb2.FieldData,
        row_dict: Dict[str, Any],
        index: int,
        context: ExtractContext,
    ) -> bool:
        # Return True to indicate lazy extraction for vectors
        return True


class BytesVectorHandler(VectorHandler):
    """Base class for byte-based vector handlers (Binary, Float16, BFloat16, Int8)."""

    # Override in subclasses
    vector_attr: str = ""
    bytes_per_element: int = 1
    expected_dtype: str = ""

    def _get_bytes_from_value(self, value: Any) -> bytes:
        """Convert value to bytes. Override in subclasses."""
        return bytes(value)

    def _compute_dim(self, byte_length: int) -> int:
        """Compute dimension from byte length. Override in subclasses."""
        return byte_length

    def pack_value(
        self,
        value: Any,
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
        context: PackContext,
    ) -> None:
        try:
            if value is None:
                if field_data.vectors.dim == 0:
                    field_data.vectors.dim = self._get_dim_from_field_info(field_info)
            else:
                v_bytes = self._get_bytes_from_value(value)
                field_data.vectors.dim = self._compute_dim(len(v_bytes))
                context.append_bytes(id(field_data), v_bytes)
        except (TypeError, ValueError) as e:
            self._handle_pack_error(self.get_field_name(field_info), value, e)

    def flush(self, field_data: schema_pb2.FieldData, context: PackContext) -> None:
        """Flush accumulated bytes to field_data."""
        context.flush_vector_bytes(field_data, self.vector_attr)

    def get_raw_data(self, field_data: schema_pb2.FieldData) -> Any:
        return getattr(field_data.vectors, self.vector_attr)


class BinaryVectorHandler(BytesVectorHandler):
    """Handler for BINARY_VECTOR type."""

    supported_types = (DataType.BINARY_VECTOR,)
    type_name = "binary_vector"
    vector_attr = "binary_vector"

    def _get_bytes_from_value(self, value: Any) -> bytes:
        # Validate that value is iterable with length (not just an int)
        if not hasattr(value, '__len__') or isinstance(value, (int, float)):
            raise TypeError(f"Binary vector value must be bytes-like or iterable, got {type(value)}")
        return bytes(value)

    def _compute_dim(self, byte_length: int) -> int:
        return byte_length * 8  # Each byte = 8 bits

    def pack_values(
        self,
        values: List[Any],
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        if len(values) > 0:
            field_data.vectors.dim = len(values[0]) * 8
        else:
            field_data.vectors.dim = self._get_dim_from_field_info(field_info)
        field_data.vectors.binary_vector = b"".join(values)

    def extract_value(
        self,
        field_data: schema_pb2.FieldData,
        index: int,
        context: ExtractContext,
    ) -> Any:
        if self.is_nullable_null(field_data, index):
            return None

        dim = field_data.vectors.dim
        blen = dim // 8
        phys_idx = context.get_physical_index(field_data, index)

        if len(field_data.vectors.binary_vector) >= (phys_idx + 1) * blen:
            start_pos, end_pos = phys_idx * blen, (phys_idx + 1) * blen
            return [field_data.vectors.binary_vector[start_pos:end_pos]]
        return None

    def extract_range(
        self,
        field_data: schema_pb2.FieldData,
        start: int,
        end: int,
    ) -> Tuple[List[Any], schema_pb2.FieldData]:
        dim = field_data.vectors.dim
        vectors = field_data.vectors
        field_meta = schema_pb2.FieldData(
            type=field_data.type,
            field_name=field_data.field_name,
            field_id=field_data.field_id,
            is_dynamic=field_data.is_dynamic,
        )
        field_meta.vectors.dim = dim
        blen = dim // 8
        return vectors.binary_vector[start * blen : end * blen], field_meta

    def extract_into_row(
        self,
        field_data: schema_pb2.FieldData,
        row_dict: Dict[str, Any],
        index: int,
        context: ExtractContext,
    ) -> bool:
        return True


class Float16VectorHandler(BytesVectorHandler):
    """Handler for FLOAT16_VECTOR type."""

    supported_types = (DataType.FLOAT16_VECTOR,)
    type_name = "float16_vector"
    vector_attr = "float16_vector"

    def _get_bytes_from_value(self, value: Any) -> bytes:
        if isinstance(value, bytes):
            return value
        if isinstance(value, np.ndarray):
            if value.dtype != "float16":
                raise ParamError(
                    message="invalid input for float16 vector. Expected an np.ndarray with dtype=float16"
                )
            return value.view(np.uint8).tobytes()
        raise ParamError(
            message="invalid input type for float16 vector. Expected an np.ndarray with dtype=float16"
        )

    def _compute_dim(self, byte_length: int) -> int:
        return byte_length // 2  # 2 bytes per float16

    def pack_values(
        self,
        values: List[Any],
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        if len(values) > 0:
            field_data.vectors.dim = len(values[0]) // 2
        else:
            field_data.vectors.dim = self._get_dim_from_field_info(field_info)
        field_data.vectors.float16_vector = b"".join(values)

    def extract_value(
        self,
        field_data: schema_pb2.FieldData,
        index: int,
        context: ExtractContext,
    ) -> Any:
        if self.is_nullable_null(field_data, index):
            return None

        dim = field_data.vectors.dim
        byte_per_row = dim * 2
        phys_idx = context.get_physical_index(field_data, index)

        if len(field_data.vectors.float16_vector) >= (phys_idx + 1) * byte_per_row:
            start_pos, end_pos = phys_idx * byte_per_row, (phys_idx + 1) * byte_per_row
            return [field_data.vectors.float16_vector[start_pos:end_pos]]
        return None

    def extract_range(
        self,
        field_data: schema_pb2.FieldData,
        start: int,
        end: int,
    ) -> Tuple[List[Any], schema_pb2.FieldData]:
        dim = field_data.vectors.dim
        vectors = field_data.vectors
        field_meta = schema_pb2.FieldData(
            type=field_data.type,
            field_name=field_data.field_name,
            field_id=field_data.field_id,
            is_dynamic=field_data.is_dynamic,
        )
        field_meta.vectors.dim = dim
        byte_per_row = dim * 2
        return vectors.float16_vector[start * byte_per_row : end * byte_per_row], field_meta

    def extract_into_row(
        self,
        field_data: schema_pb2.FieldData,
        row_dict: Dict[str, Any],
        index: int,
        context: ExtractContext,
    ) -> bool:
        return True


class BFloat16VectorHandler(BytesVectorHandler):
    """Handler for BFLOAT16_VECTOR type."""

    supported_types = (DataType.BFLOAT16_VECTOR,)
    type_name = "bfloat16_vector"
    vector_attr = "bfloat16_vector"

    def _get_bytes_from_value(self, value: Any) -> bytes:
        if isinstance(value, bytes):
            return value
        if isinstance(value, np.ndarray):
            if value.dtype != "bfloat16":
                raise ParamError(
                    message="invalid input for bfloat16 vector. Expected an np.ndarray with dtype=bfloat16"
                )
            return value.view(np.uint8).tobytes()
        raise ParamError(
            message="invalid input type for bfloat16 vector. Expected an np.ndarray with dtype=bfloat16"
        )

    def _compute_dim(self, byte_length: int) -> int:
        return byte_length // 2  # 2 bytes per bfloat16

    def pack_values(
        self,
        values: List[Any],
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        if len(values) > 0:
            field_data.vectors.dim = len(values[0]) // 2
        else:
            field_data.vectors.dim = self._get_dim_from_field_info(field_info)
        field_data.vectors.bfloat16_vector = b"".join(values)

    def extract_value(
        self,
        field_data: schema_pb2.FieldData,
        index: int,
        context: ExtractContext,
    ) -> Any:
        if self.is_nullable_null(field_data, index):
            return None

        dim = field_data.vectors.dim
        byte_per_row = dim * 2
        phys_idx = context.get_physical_index(field_data, index)

        if len(field_data.vectors.bfloat16_vector) >= (phys_idx + 1) * byte_per_row:
            start_pos, end_pos = phys_idx * byte_per_row, (phys_idx + 1) * byte_per_row
            return [field_data.vectors.bfloat16_vector[start_pos:end_pos]]
        return None

    def extract_range(
        self,
        field_data: schema_pb2.FieldData,
        start: int,
        end: int,
    ) -> Tuple[List[Any], schema_pb2.FieldData]:
        dim = field_data.vectors.dim
        vectors = field_data.vectors
        field_meta = schema_pb2.FieldData(
            type=field_data.type,
            field_name=field_data.field_name,
            field_id=field_data.field_id,
            is_dynamic=field_data.is_dynamic,
        )
        field_meta.vectors.dim = dim
        byte_per_row = dim * 2
        return vectors.bfloat16_vector[start * byte_per_row : end * byte_per_row], field_meta

    def extract_into_row(
        self,
        field_data: schema_pb2.FieldData,
        row_dict: Dict[str, Any],
        index: int,
        context: ExtractContext,
    ) -> bool:
        return True


class Int8VectorHandler(BytesVectorHandler):
    """Handler for INT8_VECTOR type."""

    supported_types = (DataType.INT8_VECTOR,)
    type_name = "int8_vector"
    vector_attr = "int8_vector"

    def _get_bytes_from_value(self, value: Any) -> bytes:
        if isinstance(value, np.ndarray):
            if value.dtype != "int8":
                raise ParamError(
                    message="invalid input for int8 vector. Expected an np.ndarray with dtype=int8"
                )
            return value.view(np.int8).tobytes()
        raise ParamError(
            message="invalid input for int8 vector. Expected an np.ndarray with dtype=int8"
        )

    def _compute_dim(self, byte_length: int) -> int:
        return byte_length  # 1 byte per int8

    def pack_values(
        self,
        values: List[Any],
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        if len(values) > 0:
            field_data.vectors.dim = len(values[0])
        else:
            field_data.vectors.dim = self._get_dim_from_field_info(field_info)
        field_data.vectors.int8_vector = b"".join(values)

    def extract_value(
        self,
        field_data: schema_pb2.FieldData,
        index: int,
        context: ExtractContext,
    ) -> Any:
        if self.is_nullable_null(field_data, index):
            return None

        dim = field_data.vectors.dim
        phys_idx = context.get_physical_index(field_data, index)

        if len(field_data.vectors.int8_vector) >= (phys_idx + 1) * dim:
            start_pos, end_pos = phys_idx * dim, (phys_idx + 1) * dim
            return [field_data.vectors.int8_vector[start_pos:end_pos]]
        return None

    def extract_range(
        self,
        field_data: schema_pb2.FieldData,
        start: int,
        end: int,
    ) -> Tuple[List[Any], schema_pb2.FieldData]:
        dim = field_data.vectors.dim
        vectors = field_data.vectors
        field_meta = schema_pb2.FieldData(
            type=field_data.type,
            field_name=field_data.field_name,
            field_id=field_data.field_id,
            is_dynamic=field_data.is_dynamic,
        )
        field_meta.vectors.dim = dim
        return vectors.int8_vector[start * dim : end * dim], field_meta

    def extract_into_row(
        self,
        field_data: schema_pb2.FieldData,
        row_dict: Dict[str, Any],
        index: int,
        context: ExtractContext,
    ) -> bool:
        return True


class SparseFloatVectorHandler(VectorHandler):
    """Handler for SPARSE_FLOAT_VECTOR type."""

    supported_types = (DataType.SPARSE_FLOAT_VECTOR,)
    type_name = "sparse_float_vector"

    def pack_value(
        self,
        value: Any,
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
        context: PackContext,
    ) -> None:
        from pymilvus.client.entity_helper import (
            entity_is_sparse_matrix,
            sparse_rows_to_proto,
        )
        from pymilvus.client.utils import SciPyHelper

        try:
            if value is None:
                if field_data.vectors.dim == 0:
                    field_data.vectors.dim = 0
            else:
                if not SciPyHelper.is_scipy_sparse(value):
                    value = [value]
                elif value.shape[0] != 1:
                    raise ParamError(message="invalid input for sparse float vector: expect 1 row")
                if not entity_is_sparse_matrix(value):
                    raise ParamError(message="invalid input for sparse float vector")
                field_data.vectors.sparse_float_vector.contents.append(
                    sparse_rows_to_proto(value).contents[0]
                )
        except (TypeError, ValueError) as e:
            self._handle_pack_error(self.get_field_name(field_info), value, e)

    def pack_values(
        self,
        values: List[Any],
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        from pymilvus.client.entity_helper import sparse_rows_to_proto

        if len(values) > 0:
            field_data.vectors.sparse_float_vector.CopyFrom(sparse_rows_to_proto(values))

    def get_raw_data(self, field_data: schema_pb2.FieldData) -> Any:
        return field_data.vectors.sparse_float_vector

    def extract_value(
        self,
        field_data: schema_pb2.FieldData,
        index: int,
        context: ExtractContext,
    ) -> Any:
        from pymilvus.client.utils import sparse_parse_single_row

        if self.is_nullable_null(field_data, index):
            return None

        phys_idx = context.get_physical_index(field_data, index)
        return sparse_parse_single_row(
            field_data.vectors.sparse_float_vector.contents[phys_idx]
        )

    def extract_range(
        self,
        field_data: schema_pb2.FieldData,
        start: int,
        end: int,
    ) -> Tuple[List[Any], schema_pb2.FieldData]:
        from pymilvus.client.entity_helper import sparse_proto_to_rows

        field_meta = schema_pb2.FieldData(
            type=field_data.type,
            field_name=field_data.field_name,
            field_id=field_data.field_id,
            is_dynamic=field_data.is_dynamic,
        )
        return (
            sparse_proto_to_rows(field_data.vectors.sparse_float_vector, start, end),
            field_meta,
        )

    def extract_into_row(
        self,
        field_data: schema_pb2.FieldData,
        row_dict: Dict[str, Any],
        index: int,
        context: ExtractContext,
    ) -> bool:
        return True
