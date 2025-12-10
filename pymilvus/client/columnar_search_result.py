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
    Compatible with original Hit dict-like interface.
    """
    __slots__ = ('_hits', '_idx')
    
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
        return self

    # ===== Dict-like compatibility methods =====
    
    def _get_dynamic_field_names(self):
        """Get field names from $meta JSON for dynamic fields."""
        import orjson
        meta_field = self._hits._fields_data_map.get("$meta")
        if meta_field is not None and meta_field.type == 23:  # DataType.JSON
            abs_idx = self._idx + self._hits.start
            json_bytes = meta_field.scalars.json_data.data[abs_idx]
            if json_bytes:
                return list(orjson.loads(json_bytes).keys())
        return []
    
    def keys(self):
        """Return field names (compatible with dict.keys())."""
        # Include dynamic fields from $meta, exclude $meta itself
        field_names = [f for f in self._hits.fields if f != "$meta"]
        field_names.extend(self._get_dynamic_field_names())
        return field_names
    
    def values(self):
        """Return field values (compatible with dict.values())."""
        return [self.get(f) for f in self.keys()]
    
    def items(self):
        """Return (field_name, value) pairs (compatible with dict.items())."""
        return [(f, self.get(f)) for f in self.keys()]
    
    def __contains__(self, key: str) -> bool:
        """Support 'field in hit' syntax."""
        if key in self._hits.fields or key == self._hits.pk_name:
            return True
        # Check dynamic fields
        return key in self._get_dynamic_field_names()
    
    def __iter__(self):
        """Iterate over field names (compatible with dict iteration)."""
        return iter(self.keys())

    def __str__(self):
        return str(self.to_dict())

    def __repr__(self):
        return self.__str__()


class ColumnarHits:
    """
    Holds search results in a columnar format (directly wrapping Protobuf structures).
    Avoids row-wise materialization during initialization.
    """
    __slots__ = ('start', 'end', '_all_pks', '_all_scores', 'pk_name', 'output_fields',
                 '_fields_data_map', '_fields', '_zero_copy_vectors', 
                 '_data_containers', '_vector_numpy_cache', '_ids_cache', '_distances_cache',
                 '_lazy_slicing')
    
    def __init__(
        self,
        start: int,
        end: int,
        all_pks: List[Union[str, int]],
        all_scores: List[float],
        fields_data_map: Dict[str, schema_pb2.FieldData],  # Shared from parent
        fields: List[str],  # Shared from parent
        output_fields: List[str],
        pk_name: str,
        zero_copy_vectors: bool = True,
        numpy_vector_cache: Optional[Dict[str, np.ndarray]] = None,
        lazy_slicing: bool = True,  # True=fast init, False=fast iterate
    ):
        self.start = start
        self.end = end
        self._lazy_slicing = lazy_slicing
        self.pk_name = pk_name
        self.output_fields = output_fields
        
        if lazy_slicing:
            # Lazy mode: store references, slice on first access
            self._all_pks = all_pks
            self._all_scores = all_scores
            self._ids_cache = None
            self._distances_cache = None
        else:
            # Eager mode: slice now for faster iteration
            self._all_pks = None
            self._all_scores = None
            self._ids_cache = all_pks[start:end]
            self._distances_cache = all_scores[start:end]
        
        # Use shared references from parent (no dict/list creation per instance)
        self._fields_data_map = fields_data_map
        self._fields = fields
        self._zero_copy_vectors = zero_copy_vectors
        
        # Cache for simple scalar containers to avoid repeated lookup
        self._data_containers = {}
        
        # Phase 2: Use shared numpy cache from parent (avoids repeated Protobuf access)
        self._vector_numpy_cache = numpy_vector_cache if numpy_vector_cache else {}

    @property
    def ids(self):
        """Return ids, slicing lazily if in lazy mode."""
        if self._ids_cache is None:
            self._ids_cache = self._all_pks[self.start:self.end]
        return self._ids_cache
    
    @property
    def distances(self):
        """Return distances, slicing lazily if in lazy mode."""
        if self._distances_cache is None:
            self._distances_cache = self._all_scores[self.start:self.end]
        return self._distances_cache

    @property
    def fields(self):
        return self._fields

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            # Support slice access for compatibility
            indices = range(*idx.indices(len(self)))
            return [RowProxy(self, i) for i in indices]
        if idx < 0:
            idx = len(self) + idx  # Support negative indexing
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")
        return RowProxy(self, idx)

    def __iter__(self):
        for i in range(len(self)):
            yield RowProxy(self, i)

    # ========== Batch Access API (high performance) ==========
    
    def get_all_ids(self) -> List[Union[str, int]]:
        """Return all IDs for this query as a list."""
        return self.ids
    
    def get_all_distances(self) -> List[float]:
        """Return all distances for this query as a list."""
        return self.distances
    
    def get_column(self, field_name: str) -> np.ndarray:
        """Return entire column of vector data as a numpy array.
        
        This is much faster than per-row access for batch processing.
        Returns shape (topk, dim) for vector fields.
        
        Args:
            field_name: Name of the vector field
            
        Returns:
            numpy array with shape (topk, dim) for vectors,
            or (topk,) for scalar fields
        """
        field_data = self._fields_data_map.get(field_name)
        if field_data is None:
            raise KeyError(f"Field {field_name} not found")
        
        dtype = field_data.type
        dim = field_data.vectors.dim if field_data.HasField("vectors") else 0
        topk = len(self)
        
        # Use numpy cache for bytes-based vectors
        if self._zero_copy_vectors and field_name in self._vector_numpy_cache:
            np_array = self._vector_numpy_cache[field_name]
            if dtype == DataType.FLOAT_VECTOR:
                return np_array[self.start * dim : self.end * dim].reshape(topk, dim)
            elif dtype == DataType.BINARY_VECTOR:
                bytes_per_vec = dim // 8
                return np_array[self.start * bytes_per_vec : self.end * bytes_per_vec].reshape(topk, bytes_per_vec)
            elif dtype in (DataType.FLOAT16_VECTOR, DataType.BFLOAT16_VECTOR, DataType.INT8_VECTOR):
                return np_array[self.start * dim : self.end * dim].reshape(topk, dim)
        
        # FLOAT_VECTOR: build numpy array from Protobuf container
        if dtype == DataType.FLOAT_VECTOR:
            container = field_data.vectors.float_vector.data
            return np.array(container[self.start * dim : self.end * dim], dtype=np.float32).reshape(topk, dim)
        
        # Scalar fields
        container = self._get_container(field_name)
        return np.array(container[self.start:self.end])

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
            elif field_data.type == DataType.SPARSE_FLOAT_VECTOR:
                self._data_containers[field_name] = field_data.vectors.sparse_float_vector
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
            elif field_data.type in (DataType.VARCHAR, DataType.STRING, DataType.TIMESTAMPTZ):
                self._data_containers[field_name] = field_data.scalars.string_data.data
            elif field_data.type == DataType.GEOMETRY:
                self._data_containers[field_name] = field_data.scalars.geometry_wkt_data.data
            elif field_data.type == DataType.JSON:
                 self._data_containers[field_name] = field_data.scalars.json_data.data
            elif field_data.type == DataType._ARRAY_OF_STRUCT:
                self._data_containers[field_name] = field_data.struct_arrays
            elif field_data.type == DataType._ARRAY_OF_VECTOR:
                self._data_containers[field_name] = field_data.vectors.vector_array
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
        
        # If field not found directly, check $meta for dynamic fields
        if field_data is None:
            meta_data = self._fields_data_map.get("$meta")
            if meta_data is not None and meta_data.type == DataType.JSON:
                import orjson
                json_bytes = meta_data.scalars.json_data.data[abs_idx]
                meta_dict = orjson.loads(json_bytes) if json_bytes else {}
                if field_name in meta_dict:
                    return meta_dict[field_name]
            raise KeyError(f"Field {field_name} not found")

        dtype = field_data.type
        dim = field_data.vectors.dim if field_data.HasField("vectors") else 0
        
        # Use numpy cache for bytes-based vectors (zero-copy slicing)
        if self._zero_copy_vectors and field_name in self._vector_numpy_cache:
            np_array = self._vector_numpy_cache[field_name]
            if dtype == DataType.BINARY_VECTOR:
                bytes_per_vector = dim // 8
                v_start = abs_idx * bytes_per_vector
                return np_array[v_start : v_start + bytes_per_vector]
            elif dtype in (DataType.FLOAT16_VECTOR, DataType.BFLOAT16_VECTOR):
                v_start = abs_idx * dim
                return np_array[v_start : v_start + dim]
            elif dtype == DataType.INT8_VECTOR:
                v_start = abs_idx * dim
                return np_array[v_start : v_start + dim]
        
        # Fallback: get container for non-cached access
        container = self._get_container(field_name)
        
        # FLOAT_VECTOR: direct Protobuf slicing (np.array copy too slow)
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
        
        # SPARSE_FLOAT_VECTOR: return sparse vector at index
        elif dtype == DataType.SPARSE_FLOAT_VECTOR:
            return container.contents[abs_idx]
        
        # GEOMETRY: return WKT string at index
        elif dtype == DataType.GEOMETRY:
            return container[abs_idx]
        
        # _ARRAY_OF_STRUCT: return struct array at index
        elif dtype == DataType._ARRAY_OF_STRUCT:
            return container[abs_idx]
        
        # _ARRAY_OF_VECTOR: return vector array at index
        elif dtype == DataType._ARRAY_OF_VECTOR:
            return container[abs_idx]
        
        # Simple scalar types - direct indexing
        elif hasattr(container, '__getitem__'):
            return container[abs_idx]

        raise NotImplementedError(f"Field type {dtype} access not fully implemented")


class ColumnarSearchResult(BaseSearchResult):
    """
    A drop-in replacement (mostly) for SearchResult that uses ColumnarHits.
    Optimized for zero-copy access patterns with lazy evaluation.
    
    Args:
        lazy_slicing: If True (default), ids/distances are sliced on first access (fast init).
                      If False, they are sliced during init (fast iteration).
    """
    def __init__(
        self,
        res: schema_pb2.SearchResultData,
        round_decimal: Optional[int] = None,
        status: Optional[Any] = None,
        session_ts: Optional[int] = 0,
        zero_copy_vectors: bool = True,
        lazy_slicing: bool = True,  # True=fast init, False=fast iterate
    ):
        self._res = res
        self._round_decimal = round_decimal
        self._zero_copy_vectors = zero_copy_vectors
        self._lazy_slicing = lazy_slicing
        _pk_name = res.primary_field_name or "id"
        
        # ids parsing
        if res.ids.HasField("int_id"):
            all_pks = res.ids.int_id.data
        elif res.ids.HasField("str_id"):
            all_pks = res.ids.str_id.data
        else:
            all_pks = []

        all_scores = res.scores
        
        # Create shared field map ONCE at this level (avoid per-ColumnarHits creation)
        fields_data_map = {fd.field_name: fd for fd in res.fields_data}
        fields = list(fields_data_map.keys())
        
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
                    fields_data_map,
                    fields,
                    res.output_fields,
                    _pk_name,
                    zero_copy_vectors=zero_copy_vectors,
                    numpy_vector_cache=numpy_cache,
                    lazy_slicing=lazy_slicing,
                )
            )
            nq_thres += topk
            
        super().__init__(data)
        
        # Use base class method for common attributes
        self._init_common_attributes(res, status, session_ts)
    
    def _create_numpy_vector_cache(self, fields_data: List[schema_pb2.FieldData]) -> Dict[str, np.ndarray]:
        """Create numpy arrays over vector data ONCE for zero-copy slicing.
        
        This avoids the Protobuf field access bottleneck where each access
        causes a full copy from C++ to Python.
        
        Note: FLOAT_VECTOR is NOT cached here because np.array() copy is too slow
        for large vectors. It uses Protobuf slicing directly.
        """
        cache = {}
        for fd in fields_data:
            field_name = fd.field_name
            # Only cache bytes-based vectors (np.frombuffer is zero-copy)
            # FLOAT_VECTOR uses RepeatedScalar which requires np.array() copy - too slow
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
