"""
Vector Type Handlers - FLOAT_VECTOR, BINARY, FLOAT16, BFLOAT16, INT8, SPARSE.
"""

from typing import Any, Callable, Dict, List, Optional

import numpy as np

from pymilvus.client.types import DataType
from pymilvus.exceptions import ParamError
from pymilvus.grpc_gen import schema_pb2

from .base import BytesVectorHandler, VectorHandler


class FloatVectorHandler(VectorHandler):
    supported_types = (DataType.FLOAT_VECTOR,)
    
    def get_dim(self, field_data: schema_pb2.FieldData) -> int:
        return field_data.vectors.dim
    
    def extract_payload(self, field_data: schema_pb2.FieldData) -> Any:
        return field_data.vectors.float_vector.data
    
    def create_accessor(
        self, payload: Any, start: int, dim: int = 0, valid_data: Optional[Any] = None
    ) -> Callable[[int], List[float]]:
        if valid_data is not None:
            def accessor(i: int) -> Optional[List[float]]:
                idx = i + start
                if not valid_data[idx]:
                    return None
                s = idx * dim
                return list(payload[s : s + dim])
            return accessor
        def accessor(i: int) -> List[float]:
            s = (i + start) * dim
            return list(payload[s : s + dim])
        return accessor
    
    def pack_single(self, value: Any, field_data: schema_pb2.FieldData, field_info: Dict) -> None:
        if value is None:
            if field_data.vectors.dim == 0:
                field_data.vectors.dim = field_info.get("params", {}).get("dim", 0)
        else:
            f_value = value.tolist() if isinstance(value, np.ndarray) else value
            field_data.vectors.dim = len(f_value)
            field_data.vectors.float_vector.data.extend(f_value)
    
    def pack_batch(self, values: List, field_data: schema_pb2.FieldData, field_info: Dict, valid_data: Optional[List[bool]] = None) -> None:
        if valid_data:
            values = [v for i, v in enumerate(values) if valid_data[i]]
        if values:
            field_data.vectors.dim = len(values[0]) if not isinstance(values[0], np.ndarray) else len(values[0])
            all_floats = [f for v in values for f in (v.tolist() if isinstance(v, np.ndarray) else v)]
            field_data.vectors.float_vector.data.extend(all_floats)
        else:
            field_data.vectors.dim = field_info.get("params", {}).get("dim", 0)


class BinaryVectorHandler(BytesVectorHandler):
    supported_types = (DataType.BINARY_VECTOR,)
    proto_attr = "binary_vector"
    
    def get_dim(self, field_data: schema_pb2.FieldData) -> int:
        return field_data.vectors.dim
    
    def get_bytes_per_element(self, dim: int) -> int:
        return dim // 8
    
    def extract_payload(self, field_data: schema_pb2.FieldData) -> bytes:
        return field_data.vectors.binary_vector
    
    def _set_bytes(self, field_data: schema_pb2.FieldData, data: bytes) -> None:
        field_data.vectors.binary_vector = data
    
    def create_accessor(
        self, payload: Any, start: int, dim: int = 0, valid_data: Optional[Any] = None
    ) -> Callable[[int], bytes]:
        bpv = dim // 8
        if valid_data is not None:
            def accessor(i: int) -> Optional[bytes]:
                idx = i + start
                if not valid_data[idx]:
                    return None
                s = idx * bpv
                return payload[s : s + bpv]
            return accessor
        def accessor(i: int) -> bytes:
            s = (i + start) * bpv
            return payload[s : s + bpv]
        return accessor
    
    def pack_single(self, value: Any, field_data: schema_pb2.FieldData, field_info: Dict) -> None:
        if value is None:
            if field_data.vectors.dim == 0:
                field_data.vectors.dim = field_info.get("params", {}).get("dim", 0)
        else:
            field_data.vectors.dim = len(value) * 8
            field_id = id(field_data)
            if field_id not in self._cache:
                self._cache[field_id] = []
            self._cache[field_id].append(bytes(value))
    
    def pack_batch(self, values: List, field_data: schema_pb2.FieldData, field_info: Dict, valid_data: Optional[List[bool]] = None) -> None:
        if valid_data:
            values = [v for i, v in enumerate(values) if valid_data[i]]
        if values:
            field_data.vectors.dim = len(values[0]) * 8
            field_data.vectors.binary_vector = b"".join(bytes(v) for v in values)
        else:
            field_data.vectors.dim = field_info.get("params", {}).get("dim", 0)


class Float16VectorHandler(BytesVectorHandler):
    supported_types = (DataType.FLOAT16_VECTOR,)
    proto_attr = "float16_vector"
    
    def get_dim(self, field_data: schema_pb2.FieldData) -> int:
        return field_data.vectors.dim
    
    def get_bytes_per_element(self, dim: int) -> int:
        return dim * 2
    
    def extract_payload(self, field_data: schema_pb2.FieldData) -> bytes:
        return field_data.vectors.float16_vector
    
    def _set_bytes(self, field_data: schema_pb2.FieldData, data: bytes) -> None:
        field_data.vectors.float16_vector = data
    
    def create_accessor(
        self, payload: Any, start: int, dim: int = 0, valid_data: Optional[Any] = None
    ) -> Callable[[int], bytes]:
        bpv = dim * 2
        if valid_data is not None:
            def accessor(i: int) -> Optional[bytes]:
                idx = i + start
                if not valid_data[idx]:
                    return None
                s = idx * bpv
                return payload[s : s + bpv]
            return accessor
        def accessor(i: int) -> bytes:
            s = (i + start) * bpv
            return payload[s : s + bpv]
        return accessor
    
    def pack_single(self, value: Any, field_data: schema_pb2.FieldData, field_info: Dict) -> None:
        if value is None:
            if field_data.vectors.dim == 0:
                field_data.vectors.dim = field_info.get("params", {}).get("dim", 0)
        else:
            if isinstance(value, bytes):
                v_bytes = value
            elif isinstance(value, np.ndarray):
                v_bytes = value.view(np.uint8).tobytes()
            else:
                raise ParamError(message="invalid input type for float16 vector")
            field_data.vectors.dim = len(v_bytes) // 2
            field_id = id(field_data)
            if field_id not in self._cache:
                self._cache[field_id] = []
            self._cache[field_id].append(v_bytes)
    
    def pack_batch(self, values: List, field_data: schema_pb2.FieldData, field_info: Dict, valid_data: Optional[List[bool]] = None) -> None:
        if valid_data:
            values = [v for i, v in enumerate(values) if valid_data[i]]
        if values:
            first = values[0]
            if isinstance(first, np.ndarray):
                field_data.vectors.dim = len(first)
                field_data.vectors.float16_vector = b"".join(v.view(np.uint8).tobytes() for v in values)
            else:
                field_data.vectors.dim = len(first) // 2
                field_data.vectors.float16_vector = b"".join(values)
        else:
            field_data.vectors.dim = field_info.get("params", {}).get("dim", 0)


class BFloat16VectorHandler(BytesVectorHandler):
    supported_types = (DataType.BFLOAT16_VECTOR,)
    proto_attr = "bfloat16_vector"
    
    def get_dim(self, field_data: schema_pb2.FieldData) -> int:
        return field_data.vectors.dim
    
    def get_bytes_per_element(self, dim: int) -> int:
        return dim * 2
    
    def extract_payload(self, field_data: schema_pb2.FieldData) -> bytes:
        return field_data.vectors.bfloat16_vector
    
    def _set_bytes(self, field_data: schema_pb2.FieldData, data: bytes) -> None:
        field_data.vectors.bfloat16_vector = data
    
    def create_accessor(
        self, payload: Any, start: int, dim: int = 0, valid_data: Optional[Any] = None
    ) -> Callable[[int], bytes]:
        bpv = dim * 2
        if valid_data is not None:
            def accessor(i: int) -> Optional[bytes]:
                idx = i + start
                if not valid_data[idx]:
                    return None
                s = idx * bpv
                return payload[s : s + bpv]
            return accessor
        def accessor(i: int) -> bytes:
            s = (i + start) * bpv
            return payload[s : s + bpv]
        return accessor
    
    def pack_single(self, value: Any, field_data: schema_pb2.FieldData, field_info: Dict) -> None:
        if value is None:
            if field_data.vectors.dim == 0:
                field_data.vectors.dim = field_info.get("params", {}).get("dim", 0)
        else:
            if isinstance(value, bytes):
                v_bytes = value
            elif isinstance(value, np.ndarray):
                v_bytes = value.view(np.uint8).tobytes()
            else:
                raise ParamError(message="invalid input type for bfloat16 vector")
            field_data.vectors.dim = len(v_bytes) // 2
            field_id = id(field_data)
            if field_id not in self._cache:
                self._cache[field_id] = []
            self._cache[field_id].append(v_bytes)
    
    def pack_batch(self, values: List, field_data: schema_pb2.FieldData, field_info: Dict, valid_data: Optional[List[bool]] = None) -> None:
        if valid_data:
            values = [v for i, v in enumerate(values) if valid_data[i]]
        if values:
            first = values[0]
            if isinstance(first, np.ndarray):
                field_data.vectors.dim = len(first)
                field_data.vectors.bfloat16_vector = b"".join(v.view(np.uint8).tobytes() for v in values)
            else:
                field_data.vectors.dim = len(first) // 2
                field_data.vectors.bfloat16_vector = b"".join(values)
        else:
            field_data.vectors.dim = field_info.get("params", {}).get("dim", 0)


class Int8VectorHandler(BytesVectorHandler):
    supported_types = (DataType.INT8_VECTOR,)
    proto_attr = "int8_vector"
    
    def get_dim(self, field_data: schema_pb2.FieldData) -> int:
        return field_data.vectors.dim
    
    def get_bytes_per_element(self, dim: int) -> int:
        return dim
    
    def extract_payload(self, field_data: schema_pb2.FieldData) -> bytes:
        return field_data.vectors.int8_vector
    
    def _set_bytes(self, field_data: schema_pb2.FieldData, data: bytes) -> None:
        field_data.vectors.int8_vector = data
    
    def create_accessor(
        self, payload: Any, start: int, dim: int = 0, valid_data: Optional[Any] = None
    ) -> Callable[[int], bytes]:
        bpv = dim
        if valid_data is not None:
            def accessor(i: int) -> Optional[bytes]:
                idx = i + start
                if not valid_data[idx]:
                    return None
                s = idx * bpv
                return payload[s : s + bpv]
            return accessor
        def accessor(i: int) -> bytes:
            s = (i + start) * bpv
            return payload[s : s + bpv]
        return accessor
    
    def pack_single(self, value: Any, field_data: schema_pb2.FieldData, field_info: Dict) -> None:
        if value is None:
            if field_data.vectors.dim == 0:
                field_data.vectors.dim = field_info.get("params", {}).get("dim", 0)
        else:
            if isinstance(value, np.ndarray):
                i_bytes = value.view(np.int8).tobytes()
            else:
                raise ParamError(message="invalid input for int8 vector. Expected np.ndarray with dtype=int8")
            field_data.vectors.dim = len(i_bytes)
            field_id = id(field_data)
            if field_id not in self._cache:
                self._cache[field_id] = []
            self._cache[field_id].append(i_bytes)
    
    def pack_batch(self, values: List, field_data: schema_pb2.FieldData, field_info: Dict, valid_data: Optional[List[bool]] = None) -> None:
        if valid_data:
            values = [v for i, v in enumerate(values) if valid_data[i]]
        if values:
            first = values[0]
            if isinstance(first, np.ndarray):
                field_data.vectors.dim = len(first)
                field_data.vectors.int8_vector = b"".join(v.view(np.int8).tobytes() for v in values)
            else:
                field_data.vectors.dim = len(first)
                field_data.vectors.int8_vector = b"".join(values)
        else:
            field_data.vectors.dim = field_info.get("params", {}).get("dim", 0)


class SparseFloatVectorHandler(VectorHandler):
    supported_types = (DataType.SPARSE_FLOAT_VECTOR,)
    
    def get_dim(self, field_data: schema_pb2.FieldData) -> int:
        return 0  # Sparse vectors don't have fixed dim
    
    def extract_payload(self, field_data: schema_pb2.FieldData) -> Any:
        return field_data.vectors.sparse_float_vector
    
    def create_accessor(
        self, payload: Any, start: int, dim: int = 0, valid_data: Optional[Any] = None
    ) -> Callable[[int], Dict[int, float]]:
        from pymilvus.client.entity_helper import sparse_proto_to_rows
        
        if valid_data is not None:
            def accessor(i: int) -> Optional[Dict[int, float]]:
                idx = i + start
                if not valid_data[idx]:
                    return None
                return sparse_proto_to_rows(payload, idx, idx + 1)[0]
            return accessor
        def accessor(i: int) -> Dict[int, float]:
            idx = i + start
            return sparse_proto_to_rows(payload, idx, idx + 1)[0]
        return accessor
    
    def pack_single(self, value: Any, field_data: schema_pb2.FieldData, field_info: Dict) -> None:
        from pymilvus.client.entity_helper import entity_is_sparse_matrix, sparse_rows_to_proto
        from pymilvus.client.utils import SciPyHelper
        
        if value is None:
            return
        if not SciPyHelper.is_scipy_sparse(value):
            value = [value]
        elif value.shape[0] != 1:
            raise ParamError(message="invalid input for sparse float vector: expect 1 row")
        if not entity_is_sparse_matrix(value):
            raise ParamError(message="invalid input for sparse float vector")
        field_data.vectors.sparse_float_vector.contents.append(
            sparse_rows_to_proto(value).contents[0]
        )
    
    def pack_batch(self, values: List, field_data: schema_pb2.FieldData, field_info: Dict, valid_data: Optional[List[bool]] = None) -> None:
        from pymilvus.client.entity_helper import sparse_rows_to_proto
        
        if valid_data:
            values = [v for i, v in enumerate(values) if valid_data[i]]
        if values:
            field_data.vectors.sparse_float_vector.CopyFrom(sparse_rows_to_proto(values))
