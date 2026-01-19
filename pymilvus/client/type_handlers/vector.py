"""
Vector Type Handlers - FLOAT_VECTOR, BINARY, FLOAT16, BFLOAT16, INT8, SPARSE.

These handlers manage vector types that require dimension-aware access
and may use bytes or repeated float storage.
"""

from typing import Any, Callable, Dict, List

import numpy as np

from pymilvus.client.types import DataType
from pymilvus.exceptions import DataNotMatchException, ExceptionsMessage, ParamError
from pymilvus.grpc_gen import schema_pb2

from .base import TypeHandler


class VectorHandler(TypeHandler):
    """Base class for vector type handlers."""
    
    def get_dim(self, field_data: schema_pb2.FieldData) -> int:
        return field_data.vectors.dim


class FloatVectorHandler(VectorHandler):
    """Handler for FLOAT_VECTOR type.
    
    Data is stored as flattened floats: [v0_d0, v0_d1, ..., v1_d0, v1_d1, ...]
    """
    
    supported_types = (DataType.FLOAT_VECTOR,)
    
    def extract_data(self, field_data: schema_pb2.FieldData) -> Any:
        return field_data.vectors.float_vector.data
    
    def get_value(self, data: Any, index: int, dim: int = 0) -> List[float]:
        start = index * dim
        return data[start : start + dim]
    
    def get_slice(self, data: Any, start: int, end: int, dim: int = 0) -> Any:
        """Return flattened slice for efficiency."""
        return data[start * dim : end * dim]
    
    def create_accessor(
        self, data: Any, start: int, dim: int = 0
    ) -> Callable[[int], List[float]]:
        """Optimized accessor for float vectors."""
        def accessor(i: int) -> List[float]:
            start_idx = (i + start) * dim
            return data[start_idx : start_idx + dim]
        return accessor
    
    def pack_value(
        self,
        value: Any,
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        try:
            if value is None:
                if field_data.vectors.dim == 0:
                    field_data.vectors.dim = field_info.get("params", {}).get("dim", 0)
            else:
                f_value = value
                if isinstance(value, np.ndarray):
                    if value.dtype not in ("float32", "float64"):
                        raise ParamError(
                            message="invalid input for float32 vector. Expected np.ndarray with dtype=float32"
                        )
                    f_value = value.tolist()
                field_data.vectors.dim = len(f_value)
                field_data.vectors.float_vector.data.extend(f_value)
        except (TypeError, ValueError) as e:
            raise DataNotMatchException(
                message=ExceptionsMessage.FieldDataInconsistent
                % (field_info["name"], "float_vector", type(value))
                + f" Detail: {e!s}"
            ) from e
    
    def pack_values(
        self,
        values: List[Any],
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        if len(values) > 0:
            first = values[0]
            if isinstance(first, np.ndarray):
                field_data.vectors.dim = len(first)
            else:
                field_data.vectors.dim = len(first)
            # Flatten all vectors
            all_floats = [f for v in values for f in (v.tolist() if isinstance(v, np.ndarray) else v)]
            field_data.vectors.float_vector.data.extend(all_floats)
        else:
            field_data.vectors.dim = field_info.get("params", {}).get("dim", 0)


class BytesVectorHandler(VectorHandler):
    """Base class for byte-based vector types (BINARY, FLOAT16, BFLOAT16, INT8)."""
    
    # Subclasses must set this to the protobuf attribute name
    proto_attr: str = ""
    bytes_per_dim: float = 1.0  # bytes per dimension element
    
    def extract_data(self, field_data: schema_pb2.FieldData) -> bytes:
        return getattr(field_data.vectors, self.proto_attr)
    
    def get_bytes_per_vector(self, dim: int) -> int:
        """Calculate bytes per vector based on dimension."""
        return int(dim * self.bytes_per_dim)
    
    def get_value(self, data: Any, index: int, dim: int = 0) -> bytes:
        bpv = self.get_bytes_per_vector(dim)
        start = index * bpv
        return data[start : start + bpv]
    
    def get_slice(self, data: Any, start: int, end: int, dim: int = 0) -> bytes:
        """Return byte slice."""
        bpv = self.get_bytes_per_vector(dim)
        return data[start * bpv : end * bpv]
    
    def create_accessor(
        self, data: Any, start: int, dim: int = 0
    ) -> Callable[[int], bytes]:
        """Optimized accessor for byte vectors."""
        bpv = self.get_bytes_per_vector(dim)
        def accessor(i: int) -> bytes:
            start_idx = (i + start) * bpv
            return data[start_idx : start_idx + bpv]
        return accessor


class BinaryVectorHandler(BytesVectorHandler):
    """Handler for BINARY_VECTOR type.
    
    Dimension is in bits, bytes = dim / 8.
    """
    
    supported_types = (DataType.BINARY_VECTOR,)
    proto_attr = "binary_vector"
    bytes_per_dim = 1 / 8  # dim is in bits
    
    def get_bytes_per_vector(self, dim: int) -> int:
        return dim // 8
    
    def pack_value(
        self,
        value: Any,
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        try:
            if value is None:
                if field_data.vectors.dim == 0:
                    field_data.vectors.dim = field_info.get("params", {}).get("dim", 0)
            else:
                field_data.vectors.dim = len(value) * 8
                field_data.vectors.binary_vector += bytes(value)
        except (TypeError, ValueError) as e:
            raise DataNotMatchException(
                message=ExceptionsMessage.FieldDataInconsistent
                % (field_info["name"], "binary_vector", type(value))
                + f" Detail: {e!s}"
            ) from e
    
    def pack_values(
        self,
        values: List[Any],
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        if len(values) > 0:
            field_data.vectors.dim = len(values[0]) * 8
            field_data.vectors.binary_vector = b"".join(bytes(v) for v in values)
        else:
            field_data.vectors.dim = field_info.get("params", {}).get("dim", 0)


class Float16VectorHandler(BytesVectorHandler):
    """Handler for FLOAT16_VECTOR type.
    
    Each dimension is 2 bytes (16 bits).
    """
    
    supported_types = (DataType.FLOAT16_VECTOR,)
    proto_attr = "float16_vector"
    bytes_per_dim = 2
    
    def pack_value(
        self,
        value: Any,
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        try:
            if value is None:
                if field_data.vectors.dim == 0:
                    field_data.vectors.dim = field_info.get("params", {}).get("dim", 0)
            else:
                if isinstance(value, bytes):
                    v_bytes = value
                elif isinstance(value, np.ndarray):
                    if value.dtype != "float16":
                        raise ParamError(
                            message="invalid input for float16 vector. Expected np.ndarray with dtype=float16"
                        )
                    v_bytes = value.view(np.uint8).tobytes()
                else:
                    raise ParamError(
                        message="invalid input type for float16 vector. Expected bytes or np.ndarray"
                    )
                field_data.vectors.dim = len(v_bytes) // 2
                field_data.vectors.float16_vector += v_bytes
        except (TypeError, ValueError) as e:
            raise DataNotMatchException(
                message=ExceptionsMessage.FieldDataInconsistent
                % (field_info["name"], "float16_vector", type(value))
                + f" Detail: {e!s}"
            ) from e
    
    def pack_values(
        self,
        values: List[Any],
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        if len(values) > 0:
            first = values[0]
            if isinstance(first, np.ndarray):
                field_data.vectors.dim = len(first)
                field_data.vectors.float16_vector = b"".join(
                    v.view(np.uint8).tobytes() for v in values
                )
            else:
                field_data.vectors.dim = len(first) // 2
                field_data.vectors.float16_vector = b"".join(values)
        else:
            field_data.vectors.dim = field_info.get("params", {}).get("dim", 0)


class BFloat16VectorHandler(BytesVectorHandler):
    """Handler for BFLOAT16_VECTOR type.
    
    Each dimension is 2 bytes (16 bits).
    """
    
    supported_types = (DataType.BFLOAT16_VECTOR,)
    proto_attr = "bfloat16_vector"
    bytes_per_dim = 2
    
    def pack_value(
        self,
        value: Any,
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        try:
            if value is None:
                if field_data.vectors.dim == 0:
                    field_data.vectors.dim = field_info.get("params", {}).get("dim", 0)
            else:
                if isinstance(value, bytes):
                    v_bytes = value
                elif isinstance(value, np.ndarray):
                    if value.dtype != "bfloat16":
                        raise ParamError(
                            message="invalid input for bfloat16 vector. Expected np.ndarray with dtype=bfloat16"
                        )
                    v_bytes = value.view(np.uint8).tobytes()
                else:
                    raise ParamError(
                        message="invalid input type for bfloat16 vector. Expected bytes or np.ndarray"
                    )
                field_data.vectors.dim = len(v_bytes) // 2
                field_data.vectors.bfloat16_vector += v_bytes
        except (TypeError, ValueError) as e:
            raise DataNotMatchException(
                message=ExceptionsMessage.FieldDataInconsistent
                % (field_info["name"], "bfloat16_vector", type(value))
                + f" Detail: {e!s}"
            ) from e
    
    def pack_values(
        self,
        values: List[Any],
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        if len(values) > 0:
            first = values[0]
            if isinstance(first, np.ndarray):
                field_data.vectors.dim = len(first)
                field_data.vectors.bfloat16_vector = b"".join(
                    v.view(np.uint8).tobytes() for v in values
                )
            else:
                field_data.vectors.dim = len(first) // 2
                field_data.vectors.bfloat16_vector = b"".join(values)
        else:
            field_data.vectors.dim = field_info.get("params", {}).get("dim", 0)


class Int8VectorHandler(BytesVectorHandler):
    """Handler for INT8_VECTOR type.
    
    Each dimension is 1 byte.
    """
    
    supported_types = (DataType.INT8_VECTOR,)
    proto_attr = "int8_vector"
    bytes_per_dim = 1
    
    def pack_value(
        self,
        value: Any,
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        try:
            if value is None:
                if field_data.vectors.dim == 0:
                    field_data.vectors.dim = field_info.get("params", {}).get("dim", 0)
            else:
                if isinstance(value, np.ndarray):
                    if value.dtype != "int8":
                        raise ParamError(
                            message="invalid input for int8 vector. Expected np.ndarray with dtype=int8"
                        )
                    i_bytes = value.view(np.int8).tobytes()
                else:
                    raise ParamError(
                        message="invalid input for int8 vector. Expected np.ndarray with dtype=int8"
                    )
                field_data.vectors.dim = len(i_bytes)
                field_data.vectors.int8_vector += i_bytes
        except (TypeError, ValueError) as e:
            raise DataNotMatchException(
                message=ExceptionsMessage.FieldDataInconsistent
                % (field_info["name"], "int8_vector", type(value))
                + f" Detail: {e!s}"
            ) from e
    
    def pack_values(
        self,
        values: List[Any],
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        if len(values) > 0:
            first = values[0]
            if isinstance(first, np.ndarray):
                field_data.vectors.dim = len(first)
                field_data.vectors.int8_vector = b"".join(
                    v.view(np.int8).tobytes() for v in values
                )
            else:
                field_data.vectors.dim = len(first)
                field_data.vectors.int8_vector = b"".join(values)
        else:
            field_data.vectors.dim = field_info.get("params", {}).get("dim", 0)


class SparseFloatVectorHandler(VectorHandler):
    """Handler for SPARSE_FLOAT_VECTOR type.
    
    Sparse vectors have variable size and require special handling.
    """
    
    supported_types = (DataType.SPARSE_FLOAT_VECTOR,)
    
    def extract_data(self, field_data: schema_pb2.FieldData) -> Any:
        return field_data.vectors.sparse_float_vector
    
    def get_value(self, data: Any, index: int, dim: int = 0) -> Dict[int, float]:
        # Import here to avoid circular dependency
        from pymilvus.client.entity_helper import sparse_proto_to_rows
        return sparse_proto_to_rows(data, index, index + 1)[0]
    
    def get_slice(self, data: Any, start: int, end: int, dim: int = 0) -> List[Dict[int, float]]:
        from pymilvus.client.entity_helper import sparse_proto_to_rows
        return sparse_proto_to_rows(data, start, end)
    
    def create_accessor(
        self, data: Any, start: int, dim: int = 0
    ) -> Callable[[int], Dict[int, float]]:
        from pymilvus.client.entity_helper import sparse_proto_to_rows
        def accessor(i: int) -> Dict[int, float]:
            return sparse_proto_to_rows(data, i + start, i + start + 1)[0]
        return accessor
    
    def pack_value(
        self,
        value: Any,
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        from pymilvus.client.entity_helper import entity_is_sparse_matrix, sparse_rows_to_proto
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
            raise DataNotMatchException(
                message=ExceptionsMessage.FieldDataInconsistent
                % (field_info["name"], "sparse_float_vector", type(value))
                + f" Detail: {e!s}"
            ) from e
    
    def pack_values(
        self,
        values: List[Any],
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        from pymilvus.client.entity_helper import sparse_rows_to_proto
        if len(values) > 0:
            field_data.vectors.sparse_float_vector.CopyFrom(sparse_rows_to_proto(values))
