from typing import Any, List, Dict, Union, Optional
import numpy as np
from pymilvus.grpc_gen import schema_pb2
from pymilvus.client.types import DataType
from pymilvus.client import entity_helper
from pymilvus.exceptions import MilvusException
from pymilvus.client.search_result import BaseSearchResult

class RowProxy:
    """
    A lightweight proxy object that represents a single row in the search result.
    It does not store data itself, but retrieves it from the ColumnarHits object on demand.
    """
    def __init__(self, hits: "ColumnarHits", idx: int):
        self._hits = hits
        self._idx = idx

    def __getitem__(self, key: str) -> Any:
        return self._hits.get_value(key, self._idx)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self.__getitem__(key)
        except (KeyError, ValueError):
            return default

    def to_dict(self) -> Dict[str, Any]:
        """Materialize this row into a dictionary (expensive operation)."""
        return {
            "id": self._hits.ids[self._idx],
            "distance": self._hits.distances[self._idx],
            "entity": {
                field_name: self._hits.get_value(field_name, self._idx)
                for field_name in self._hits.fields
            }
        }
    
    @property
    def id(self):
        return self._hits.ids[self._idx]
    
    @property
    def distance(self):
        return self._hits.distances[self._idx]

    @property
    def entity(self):
        # For compatibility with existing Hit object structure
        # In the optimization, users should access fields directly if possible, 
        # but this properties maintains backward compat logic: hit.entity.get('field')
        return self

    def __str__(self):
        return str(self.to_dict())

    def __repr__(self):
        return self.__str__()


class ColumnarHits:
    """
    Holds search results in a columnar format (directly wrapping Protobuf structures).
    Avoids row-wise materialization during initialization.
    """
    def __init__(
        self,
        start: int,
        end: int,
        all_pks: List[Union[str, int]],
        all_scores: List[float],
        fields_data: List[schema_pb2.FieldData],
        output_fields: List[str],
        pk_name: str,
        zero_copy_vectors: bool = True,
        numpy_vector_cache: Optional[Dict[str, np.ndarray]] = None,  # Shared cache from parent
    ):
        self.start = start
        self.end = end
        self.ids = all_pks[start:end]
        self.distances = all_scores[start:end]
        self.pk_name = pk_name
        self.output_fields = output_fields
        
        # Map field name to FieldData protobuf object for quick access
        self._fields_data_map = {fd.field_name: fd for fd in fields_data}
        self._fields = [fd.field_name for fd in fields_data]
        self._zero_copy_vectors = zero_copy_vectors
        
        # Cache for simple scalar containers to avoid repeated lookup
        self._data_containers = {}
        
        # Phase 2: Use shared numpy cache from parent (avoids repeated Protobuf access)
        self._vector_numpy_cache = numpy_vector_cache if numpy_vector_cache else {}

    @property
    def fields(self):
        return self._fields

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            # Not implemented for prototype, but would return a sub-slice of ColumnarHits
            raise NotImplementedError("Slicing not supported in prototype")
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")
        return RowProxy(self, idx)

    def __iter__(self):
        for i in range(len(self)):
            yield RowProxy(self, i)

    def _get_container(self, field_name: str) -> Any:
        """Helper to get the raw data container from protobuf."""
        if field_name not in self._data_containers:
            field_data = self._fields_data_map.get(field_name)
            if not field_data:
                raise KeyError(f"Field {field_name} not found in results")
            
            # Simplified container retrieval logic (subset of supported types for prototype)
            if field_data.type == DataType.FLOAT_VECTOR:
                self._data_containers[field_name] = field_data.vectors.float_vector.data
            elif field_data.type == DataType.BINARY_VECTOR:
                self._data_containers[field_name] = field_data.vectors.binary_vector
            elif field_data.type == DataType.FLOAT16_VECTOR:
                self._data_containers[field_name] = field_data.vectors.float16_vector
            elif field_data.type == DataType.BFLOAT16_VECTOR:
                self._data_containers[field_name] = field_data.vectors.bfloat16_vector
            elif field_data.type == DataType.INT8_VECTOR:
                self._data_containers[field_name] = field_data.vectors.int8_vector
            elif field_data.type == DataType.BOOL:
                self._data_containers[field_name] = field_data.scalars.bool_data.data
            elif field_data.type in (DataType.INT8, DataType.INT16, DataType.INT32):
                self._data_containers[field_name] = field_data.scalars.int_data.data
            elif field_data.type == DataType.INT64:
                self._data_containers[field_name] = field_data.scalars.long_data.data
            elif field_data.type == DataType.FLOAT:
                self._data_containers[field_name] = field_data.scalars.float_data.data
            elif field_data.type == DataType.DOUBLE:
                self._data_containers[field_name] = field_data.scalars.double_data.data
            elif field_data.type in (DataType.VARCHAR, DataType.STRING):
                self._data_containers[field_name] = field_data.scalars.string_data.data
            elif field_data.type == DataType.JSON:
                 self._data_containers[field_name] = field_data.scalars.json_data.data
            else:
                 # Fallback for other types
                 pass
        
        return self._data_containers.get(field_name) or self._fields_data_map[field_name]

    def get_value(self, field_name: str, idx: int) -> Any:
        """Retrieve a single value for a field at a specific relative index."""
        if field_name == self.pk_name:
            return self.ids[idx]  # ID is stored separately

        abs_idx = idx + self.start
        field_data = self._fields_data_map.get(field_name)
        
        if field_data is None:
            raise KeyError(f"Field {field_name} not found")

        dtype = field_data.type
        dim = field_data.vectors.dim if field_data.HasField("vectors") else 0
        
        # Phase 2: Use numpy cache for bytes-based vectors (true zero-copy)
        if self._zero_copy_vectors and hasattr(self, '_vector_numpy_cache') and field_name in self._vector_numpy_cache:
            np_array = self._vector_numpy_cache[field_name]
            if dtype == DataType.BINARY_VECTOR:
                bytes_per_vector = dim // 8
                v_start = abs_idx * bytes_per_vector
                return np_array[v_start : v_start + bytes_per_vector]  # numpy view, zero-copy
            elif dtype in (DataType.FLOAT16_VECTOR, DataType.BFLOAT16_VECTOR):
                v_start = abs_idx * dim
                return np_array[v_start : v_start + dim]  # numpy view, zero-copy
            elif dtype == DataType.INT8_VECTOR:
                v_start = abs_idx * dim
                return np_array[v_start : v_start + dim]  # numpy view, zero-copy
        
        # Fallback: get container for non-cached access
        container = self._get_container(field_name)
        
        # Handle vector types with dimension-based slicing (fallback mode)
        if dtype == DataType.FLOAT_VECTOR:
            v_start = abs_idx * dim
            return container[v_start : v_start + dim]
        
        elif dtype == DataType.BINARY_VECTOR:
            bytes_per_vector = dim // 8
            v_start = abs_idx * bytes_per_vector
            return container[v_start : v_start + bytes_per_vector]
        
        elif dtype in (DataType.FLOAT16_VECTOR, DataType.BFLOAT16_VECTOR):
            bytes_per_vector = dim * 2
            v_start = abs_idx * bytes_per_vector
            return container[v_start : v_start + bytes_per_vector]
        
        elif dtype == DataType.INT8_VECTOR:
            v_start = abs_idx * dim
            return container[v_start : v_start + dim]
        
        # Handle JSON
        elif dtype == DataType.JSON:
            import orjson
            val = container[abs_idx]
            return orjson.loads(val) if val else None
        
        # Handle ARRAY types
        elif dtype == DataType.ARRAY:
            array_data = field_data.scalars.array_data.data[abs_idx]
            element_type = field_data.scalars.array_data.element_type
            if element_type in (DataType.INT8, DataType.INT16, DataType.INT32):
                return list(array_data.int_data.data)
            elif element_type == DataType.INT64:
                return list(array_data.long_data.data)
            elif element_type == DataType.FLOAT:
                return list(array_data.float_data.data)
            elif element_type == DataType.DOUBLE:
                return list(array_data.double_data.data)
            elif element_type in (DataType.VARCHAR, DataType.STRING):
                return list(array_data.string_data.data)
            elif element_type == DataType.BOOL:
                return list(array_data.bool_data.data)
            return None
        
        # Simple scalar types - direct indexing
        elif hasattr(container, '__getitem__'):
            return container[abs_idx]

        raise NotImplementedError(f"Field type {dtype} access not fully implemented")


class ColumnarSearchResult(BaseSearchResult):
    """
    A drop-in replacement (mostly) for SearchResult that uses ColumnarHits.
    Optimized for zero-copy access patterns with lazy evaluation.
    """
    def __init__(
        self,
        res: schema_pb2.SearchResultData,
        round_decimal: Optional[int] = None,
        status: Optional[Any] = None,
        session_ts: Optional[int] = 0,
        zero_copy_vectors: bool = True,  # Phase 2: enable np.frombuffer for bytes-based vectors
    ):
        self._res = res
        self._round_decimal = round_decimal
        self._zero_copy_vectors = zero_copy_vectors
        _pk_name = res.primary_field_name or "id"
        
        # ids parsing
        if res.ids.HasField("int_id"):
            all_pks = res.ids.int_id.data
        elif res.ids.HasField("str_id"):
            all_pks = res.ids.str_id.data
        else:
            all_pks = []

        all_scores = res.scores
        
        # Phase 2: Create numpy cache ONCE at this level (avoids repeated Protobuf access)
        numpy_cache = None
        if zero_copy_vectors:
            numpy_cache = self._create_numpy_vector_cache(res.fields_data)

        data = []
        nq_thres = 0
        for topk in res.topks:
            start, end = nq_thres, nq_thres + topk
            data.append(
                ColumnarHits(
                    start,
                    end,
                    all_pks,
                    all_scores,
                    res.fields_data,
                    res.output_fields,
                    _pk_name,
                    zero_copy_vectors=zero_copy_vectors,
                    numpy_vector_cache=numpy_cache,  # Share the same cache
                )
            )
            nq_thres += topk
            
        super().__init__(data)
        
        # Use base class method for common attributes
        self._init_common_attributes(res, status, session_ts)
    
    def _create_numpy_vector_cache(self, fields_data: List[schema_pb2.FieldData]) -> Dict[str, np.ndarray]:
        """Phase 2: Create numpy views over bytes-based vector data ONCE.
        
        This avoids the Protobuf field access bottleneck where each access
        causes a full copy from C++ to Python.
        """
        cache = {}
        for fd in fields_data:
            field_name = fd.field_name
            if fd.type == DataType.BINARY_VECTOR:
                data = fd.vectors.binary_vector
                if data:
                    cache[field_name] = np.frombuffer(data, dtype=np.uint8)
            elif fd.type == DataType.FLOAT16_VECTOR:
                data = fd.vectors.float16_vector
                if data:
                    cache[field_name] = np.frombuffer(data, dtype=np.float16)
            elif fd.type == DataType.BFLOAT16_VECTOR:
                data = fd.vectors.bfloat16_vector
                if data:
                    # Note: numpy doesn't natively support bfloat16, use uint16
                    cache[field_name] = np.frombuffer(data, dtype=np.uint16)
            elif fd.type == DataType.INT8_VECTOR:
                data = fd.vectors.int8_vector
                if data:
                    cache[field_name] = np.frombuffer(data, dtype=np.int8)
        return cache
