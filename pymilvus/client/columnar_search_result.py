"""
ColumnarSearchResult - A drop-in replacement for SearchResult with columnar storage.

Design Principles:
1. Reduce object creation: No pre-creation of nq x topk Hit objects
2. Read-only: Data is referenced from protobuf, not copied
3. Type compatible: All field return types match SearchResult exactly
4. Lazy access: Data is extracted on-demand, not at initialization

Performance Benefits:
- Initialization is O(1) instead of O(nq x topk)
- Memory usage is minimal (just references)
- Ideal for scenarios where only a subset of results is accessed
"""

import contextlib
from typing import Any, Callable, Dict, List, Optional, Union

import orjson

from pymilvus.client import entity_helper
from pymilvus.client.types import DataType
from pymilvus.exceptions import MilvusException
from pymilvus.grpc_gen import common_pb2, schema_pb2

# ==============================================================================
# Helper Accessor Classes (Performance Optimization)
# ==============================================================================


class BaseAccessor:
    __slots__ = ("data", "start")

    def __init__(self, data: Any, start: int):
        self.data = data
        self.start = start


class ScalarAccessor(BaseAccessor):
    __slots__ = ()

    def get(self, i: int) -> Any:
        return self.data[i + self.start]


class BytesVectorAccessor(BaseAccessor):
    __slots__ = ("bpv",)

    def __init__(self, data: Any, start: int, bpv: int):
        super().__init__(data, start)
        self.bpv = bpv

    def get(self, i: int) -> bytes:
        start_idx = (i + self.start) * self.bpv
        return self.data[start_idx : start_idx + self.bpv]


class FloatVectorAccessor(BaseAccessor):
    __slots__ = ("dim",)

    def __init__(self, data: Any, start: int, dim: int):
        super().__init__(data, start)
        self.dim = dim

    def get(self, i: int) -> List[float]:
        start_idx = (i + self.start) * self.dim
        return self.data[start_idx : start_idx + self.dim]


class Int8VectorAccessor(FloatVectorAccessor):  # Same logic different type hint
    __slots__ = ()

    def get(self, i: int) -> bytes:
        start_idx = (i + self.start) * self.dim
        return self.data[start_idx : start_idx + self.dim]


class JsonAccessor(BaseAccessor):
    __slots__ = ()

    def get(self, i: int) -> Any:
        val = self.data[i + self.start]
        return orjson.loads(val) if val else None


class NullableAccessor:
    __slots__ = ("raw_acc", "start", "valid_data")

    def __init__(self, raw_accessor: Callable[[int], Any], valid_data: Any, start: int):
        self.raw_acc = raw_accessor
        self.valid_data = valid_data
        self.start = start

    def get(self, i: int) -> Any:
        if self.valid_data[i + self.start]:
            return self.raw_acc(i)
        return None


# ==============================================================================


class RowProxy:
    """
    A lightweight read-only proxy that represents a single row in search results.

    It does not store data itself, but retrieves it from ColumnarHits on demand.
    Fully compatible with the original Hit dict-like interface.

    This is READ-ONLY - any attempt to modify will raise an error.
    """

    __slots__ = ("_hits", "_idx", "_pk_name")

    def __init__(self, hits: "ColumnarHits", idx: int, pk_name: str):
        self._hits = hits
        self._idx = idx
        self._pk_name = pk_name

    def __getitem__(self, key: str) -> Any:
        """Get field value by key. Supports both top-level and entity fields."""
        # Top-level keys
        if key in {self._pk_name, "id"}:
            return self.id
        if key == "distance":
            return self.distance
        if key == "entity":
            return self  # entity returns self for nested access like hit["entity"]["field"]

        # Entity field access
        return self._hits.get_value(key, self._idx)

    def get(self, key: str, default: Any = None) -> Any:
        """Get field value with default."""
        try:
            return self.__getitem__(key)
        except (KeyError, ValueError):
            return default

    def to_dict(self) -> Dict[str, Any]:
        """Materialize this row into a dictionary (creates a copy)."""
        entity = {}
        for field_name in self._hits.fields:
            if field_name == "$meta":
                continue
            with contextlib.suppress(KeyError, ValueError):
                entity[field_name] = self._hits.get_value(field_name, self._idx)

        # Add dynamic fields from $meta
        dynamic_names = self._get_dynamic_field_names()
        for name in dynamic_names:
            if name not in entity:
                entity[name] = self._hits.get_value(name, self._idx)

        return {self._pk_name: self.id, "distance": self.distance, "entity": entity}

    @property
    def id(self) -> Union[int, str]:
        """Primary key value."""
        return self._hits.ids[self._idx]

    @property
    def distance(self) -> float:
        """Distance/score value."""
        return self._hits.distances[self._idx]

    @property
    def entity(self):
        """Returns self for compatibility with hit.entity.field access."""
        return self

    @property
    def pk(self) -> Union[int, str]:
        """Alias for id."""
        return self.id

    @property
    def score(self) -> float:
        """Alias for distance."""
        return self.distance

    # ===== Dict-like compatibility methods =====

    def _get_dynamic_field_names(self) -> List[str]:
        """Get field names from $meta JSON for dynamic fields."""
        meta_field = self._hits._fields_data_map.get("$meta")
        if meta_field is not None and meta_field.type == DataType.JSON:
            abs_idx = self._idx + self._hits.start
            json_bytes = meta_field.scalars.json_data.data[abs_idx]
            if json_bytes:
                return list(orjson.loads(json_bytes).keys())
        return []

    def keys(self) -> List[str]:
        """Return field names (compatible with dict.keys())."""
        field_names = [f for f in self._hits.fields if f != "$meta"]
        field_names.extend(self._get_dynamic_field_names())
        return field_names

    def values(self) -> List[Any]:
        """Return field values (compatible with dict.values())."""
        return [self.get(f) for f in self.keys()]

    def items(self) -> List[tuple]:
        """Return (field_name, value) pairs (compatible with dict.items())."""
        return [(f, self.get(f)) for f in self.keys()]

    def __contains__(self, key: str) -> bool:
        """Support 'field in hit' syntax."""
        if key in (self._pk_name, "id", "distance", "entity"):
            return True
        if key in self._hits.fields:
            return True
        return key in self._get_dynamic_field_names()

    def __iter__(self):
        """Iterate over field names (compatible with dict iteration)."""
        return iter(self.keys())

    def __str__(self) -> str:
        return str(self.to_dict())

    def __repr__(self) -> str:
        return self.__str__()

    # Read-only enforcement
    def __setitem__(self, key: str, value: Any) -> None:
        msg = "RowProxy is read-only"
        raise TypeError(msg)


class ColumnarHits:
    """
    Holds search results for a single query in columnar format.

    Instead of creating topk Hit objects upfront, it stores references to
    the underlying protobuf data and creates lightweight RowProxy objects
    on demand.

    This class is READ-ONLY.
    """

    __slots__ = (
        "_accessor_cache",
        "_all_pks",
        "_all_scores",
        "_column_payload_cache",
        "_distances_cache",
        "_dynamic_fields",
        "_fields",
        "_fields_data_map",
        "_ids_cache",
        "end",
        "output_fields",
        "pk_name",
        "start",
    )

    def __init__(
        self,
        start: int,
        end: int,
        all_pks: List[Union[str, int]],
        all_scores: List[float],
        fields_data_map: Dict[str, schema_pb2.FieldData],
        fields: List[str],
        output_fields: List[str],
        pk_name: str,
        column_payload_cache: Dict[str, Any],
    ):
        self.start = start
        self.end = end
        self._all_pks = all_pks
        self._all_scores = all_scores
        self.pk_name = pk_name
        self.output_fields = output_fields

        # Shared references from parent (no dict/list creation per instance)
        self._fields_data_map = fields_data_map
        self._fields = fields

        # Shared cache for raw column payloads (avoids redundant extraction/copying)
        self._column_payload_cache = column_payload_cache

        # Dynamic fields = output_fields - actual fields
        self._dynamic_fields = set(output_fields) - set(fields)

        # Lazy caches
        self._ids_cache = None
        self._distances_cache = None
        # Accessor cache for O(1) field access (bypasses branching and map lookups)
        self._accessor_cache: Dict[str, Callable[[int], Any]] = {}

    @property
    def ids(self) -> List[Union[str, int]]:
        """Return ids for this query, slicing lazily."""
        if self._ids_cache is None:
            self._ids_cache = self._all_pks[self.start : self.end]
        return self._ids_cache

    @property
    def distances(self) -> List[float]:
        """Return distances for this query, slicing lazily."""
        if self._distances_cache is None:
            self._distances_cache = self._all_scores[self.start : self.end]
        return self._distances_cache

    @property
    def fields(self) -> List[str]:
        """Field names available in results."""
        return self._fields

    def __len__(self) -> int:
        return self.end - self.start

    def __getitem__(self, idx: int):
        if isinstance(idx, slice):
            indices = range(*idx.indices(len(self)))
            return [RowProxy(self, i, self.pk_name) for i in indices]
        if idx < 0:
            idx = len(self) + idx
        if idx < 0 or idx >= len(self):
            msg = "Index out of range"
            raise IndexError(msg)
        return RowProxy(self, idx, self.pk_name)

    def __iter__(self):
        for i in range(len(self)):
            yield RowProxy(self, i, self.pk_name)

    def __str__(self) -> str:
        """Only print at most 10 query results."""
        items = [str(self[i]) for i in range(min(10, len(self)))]
        reminder = f" ... and {len(self) - 10} entities remaining" if len(self) > 10 else ""
        return f"{items}{reminder}"

    __repr__ = __str__

    def get_value(self, field_name: str, idx: int) -> Any:
        "Retrieve a single value for a field at a specific relative index."
        accessor = self._accessor_cache.get(field_name)
        if accessor is not None:
            return accessor(idx)

        # Slow path: Bind accessor first
        return self._bind_accessor(field_name)(idx)

    def _bind_accessor(self, field_name: str) -> Callable[[int], Any]:
        "Determine field type and bind a fast accessor function for this field."
        field_data = self._fields_data_map.get(field_name)

        if field_data is None:
            # Check dynamic fields ($meta)
            meta_data = self._fields_data_map.get("$meta")
            if meta_data is not None and meta_data.type == DataType.JSON:
                if field_name not in self._column_payload_cache:
                    self._column_payload_cache[field_name] = meta_data.scalars.json_data.data

                json_data = self._column_payload_cache[field_name]
                start = self.start

                def meta_accessor(i: int) -> Any:
                    json_bytes = json_data[i + start]
                    meta_dict = orjson.loads(json_bytes) if json_bytes else {}
                    return meta_dict.get(field_name)

                self._accessor_cache[field_name] = meta_accessor
                return meta_accessor

            msg = f"Field '{field_name}' not found"
            raise KeyError(msg)

        dtype = field_data.type
        start = self.start
        valid_data = field_data.valid_data if len(field_data.valid_data) > 0 else None

        # Helper to get cached payload or extract it
        def get_payload(key: str, extractor_func: Callable[[], Any]) -> Any:
            if key in self._column_payload_cache:
                return self._column_payload_cache[key]
            payload = extractor_func()
            self._column_payload_cache[key] = payload
            return payload

        # Build raw accessor based on type
        # We prefer Bound Methods (obj.get) over Lambdas as they are faster to call
        accessor_obj = None

        if dtype == DataType.FLOAT_VECTOR:
            # Note: For RepeatedScalarFieldContainer, accessing .data is relatively cheap but
            # caching it avoids repeated property lookups.
            data = get_payload(field_name, lambda: field_data.vectors.float_vector.data)
            dim = field_data.vectors.dim
            accessor_obj = FloatVectorAccessor(data, start, dim)

        elif dtype == DataType.BINARY_VECTOR:
            # CRITICAL: Accessing .binary_vector creates a COPY of bytes.
            # We MUST cache this.
            data = get_payload(field_name, lambda: field_data.vectors.binary_vector)
            bpv = field_data.vectors.dim // 8
            accessor_obj = BytesVectorAccessor(data, start, bpv)

        elif dtype in (DataType.FLOAT16_VECTOR, DataType.BFLOAT16_VECTOR):
            # CRITICAL: Similar to BinaryVector, cache the bytes object.
            field_attr = "float16_vector" if dtype == DataType.FLOAT16_VECTOR else "bfloat16_vector"
            data = get_payload(field_name, lambda: getattr(field_data.vectors, field_attr))
            bpv = field_data.vectors.dim * 2
            accessor_obj = BytesVectorAccessor(data, start, bpv)

        elif dtype == DataType.INT8_VECTOR:
            data = get_payload(field_name, lambda: field_data.vectors.int8_vector)
            dim = field_data.vectors.dim
            accessor_obj = Int8VectorAccessor(data, start, dim)

        elif dtype == DataType.BOOL:
            data = get_payload(field_name, lambda: field_data.scalars.bool_data.data)
            accessor_obj = ScalarAccessor(data, start)

        elif dtype in (DataType.INT8, DataType.INT16, DataType.INT32):
            data = get_payload(field_name, lambda: field_data.scalars.int_data.data)
            accessor_obj = ScalarAccessor(data, start)

        elif dtype == DataType.INT64:
            data = get_payload(field_name, lambda: field_data.scalars.long_data.data)
            accessor_obj = ScalarAccessor(data, start)

        elif dtype == DataType.FLOAT:
            data = get_payload(field_name, lambda: field_data.scalars.float_data.data)
            accessor_obj = ScalarAccessor(data, start)

        elif dtype == DataType.DOUBLE:
            data = get_payload(field_name, lambda: field_data.scalars.double_data.data)
            accessor_obj = ScalarAccessor(data, start)

        elif dtype in (DataType.VARCHAR, DataType.STRING, DataType.TIMESTAMPTZ):
            data = get_payload(field_name, lambda: field_data.scalars.string_data.data)
            accessor_obj = ScalarAccessor(data, start)

        elif dtype == DataType.JSON:
            data = get_payload(field_name, lambda: field_data.scalars.json_data.data)
            accessor_obj = JsonAccessor(data, start)

        elif dtype == DataType.ARRAY:
            data = get_payload(field_name, lambda: field_data.scalars.array_data.data)
            elem_type = field_data.scalars.array_data.element_type

            # Keep lambda for complex types like ARRAY where logic is involved
            def array_accessor(i: int) -> Any:
                return self._extract_array_element(data[i + start], elem_type)

            raw_accessor = array_accessor

        elif dtype == DataType.SPARSE_FLOAT_VECTOR:
            # Sparse vector has a more complex structure, cached at higher level if possible
            # but usually accessing .sparse_float_vector is okay as it's a container.
            data = field_data.vectors.sparse_float_vector

            def sparse_accessor(i: int) -> Any:
                return entity_helper.sparse_proto_to_rows(data, i + start, i + start + 1)[0]

            raw_accessor = sparse_accessor

        elif dtype == DataType.GEOMETRY:
            data = get_payload(field_name, lambda: field_data.scalars.geometry_wkt_data.data)
            accessor_obj = ScalarAccessor(data, start)

        # Handle fallback for complex/special types if needed
        # If we created an optimized accessor object, use its .get method
        if accessor_obj is not None:
            raw_accessor = accessor_obj.get
        elif "raw_accessor" not in locals():
            raw_accessor = None

        # Handle fallback for complex/special types if needed
        if raw_accessor is None:
            # Special case fallback for struct arrays/vector arrays if they don't fit the lambda
            def fallback_accessor(i: int) -> Any:
                # We need to recalculate abs_idx in fallback
                abs_idx = i + start
                if dtype == DataType._ARRAY_OF_STRUCT:
                    if hasattr(field_data, "struct_arrays") and field_data.struct_arrays:
                        return entity_helper.extract_struct_array_from_column_data(
                            field_data.struct_arrays, abs_idx
                        )
                    return None
                if dtype == DataType._ARRAY_OF_VECTOR and (
                    hasattr(field_data, "vectors")
                    and hasattr(field_data.vectors, "vector_array")
                    and abs_idx < len(field_data.vectors.vector_array.data)
                ):
                    vector_data = field_data.vectors.vector_array.data[abs_idx]
                    v_dim = vector_data.dim
                    f_data = vector_data.float_vector.data
                    num_vecs = len(f_data) // v_dim
                    return [list(f_data[j * v_dim : (j + 1) * v_dim]) for j in range(num_vecs)]
                msg = f"Unsupported field type: {dtype}"
                raise MilvusException(message=msg)

            raw_accessor = fallback_accessor

        # Wrap for nullability if necessary
        if valid_data is not None:
            # NullableAccessor is also a class to keep things fast
            final_accessor = NullableAccessor(raw_accessor, valid_data, start).get
        else:
            final_accessor = raw_accessor

        self._accessor_cache[field_name] = final_accessor
        return final_accessor

    def _extract_array_element(self, array_data: Any, element_type: DataType) -> List:
        """Extract array data based on element type."""
        if element_type in (DataType.INT8, DataType.INT16, DataType.INT32):
            return list(array_data.int_data.data)
        if element_type == DataType.INT64:
            return list(array_data.long_data.data)
        if element_type == DataType.FLOAT:
            return list(array_data.float_data.data)
        if element_type == DataType.DOUBLE:
            return list(array_data.double_data.data)
        if element_type in (DataType.VARCHAR, DataType.STRING):
            return list(array_data.string_data.data)
        if element_type == DataType.BOOL:
            return list(array_data.bool_data.data)
        return []

    def get_column(self, field_name: str) -> Union[List, bytes, Any]:
        """
        Retrieve all values for a field in this query result column.
        
        This method returns the raw data efficiently:
        - For Scalars (INT, FLOAT, VARCHAR): Returns List[Any]
        - For FLOAT_VECTOR: Returns List[float] (FLATTENED for performance)
        - For BINARY/FLOAT16_VECTOR: Returns bytes (Concatenated)
        - For JSON: Returns List[bytes] (Raw JSON bytes)
        """
        field_data = self._fields_data_map.get(field_name)

        if field_data is None:
            # Check dynamic fields ($meta)
            meta_data = self._fields_data_map.get("$meta")
            if meta_data is not None and meta_data.type == DataType.JSON:
                if field_name not in self._column_payload_cache:
                    self._column_payload_cache[field_name] = meta_data.scalars.json_data.data
                
                json_data = self._column_payload_cache[field_name]
                # Dynamic fields require parsing $meta, so we must iterate.
                # But we can optimize by doing it in a tight loop.
                # Note: This is not as fast as native columns.
                res = []
                start, end = self.start, self.end
                for i in range(start, end):
                    json_bytes = json_data[i]
                    meta_dict = orjson.loads(json_bytes) if json_bytes else {}
                    res.append(meta_dict.get(field_name))
                return res

            msg = f"Field '{field_name}' not found"
            raise KeyError(msg)

        dtype = field_data.type
        start, end = self.start, self.end

        # Helper to get cached payload
        def get_payload(key: str, extractor_func: Callable[[], Any]) -> Any:
            if key in self._column_payload_cache:
                return self._column_payload_cache[key]
            payload = extractor_func()
            self._column_payload_cache[key] = payload
            return payload

        # Fast path for known types
        if dtype == DataType.FLOAT_VECTOR:
            data = get_payload(field_name, lambda: field_data.vectors.float_vector.data)
            dim = field_data.vectors.dim
            return data[start * dim : end * dim]

        elif dtype == DataType.BINARY_VECTOR:
            data = get_payload(field_name, lambda: field_data.vectors.binary_vector)
            bpv = field_data.vectors.dim // 8
            return data[start * bpv : end * bpv]

        elif dtype in (DataType.FLOAT16_VECTOR, DataType.BFLOAT16_VECTOR):
            field_attr = "float16_vector" if dtype == DataType.FLOAT16_VECTOR else "bfloat16_vector"
            data = get_payload(field_name, lambda: getattr(field_data.vectors, field_attr))
            bpv = field_data.vectors.dim * 2
            return data[start * bpv : end * bpv]

        elif dtype == DataType.INT8_VECTOR:
            data = get_payload(field_name, lambda: field_data.vectors.int8_vector)
            dim = field_data.vectors.dim
            return data[start * dim : end * dim]

        elif dtype == DataType.BOOL:
            data = get_payload(field_name, lambda: field_data.scalars.bool_data.data)
            return data[start:end]

        elif dtype in (DataType.INT8, DataType.INT16, DataType.INT32):
            data = get_payload(field_name, lambda: field_data.scalars.int_data.data)
            return data[start:end]

        elif dtype == DataType.INT64:
            data = get_payload(field_name, lambda: field_data.scalars.long_data.data)
            return data[start:end]

        elif dtype == DataType.FLOAT:
            data = get_payload(field_name, lambda: field_data.scalars.float_data.data)
            return data[start:end]

        elif dtype == DataType.DOUBLE:
            data = get_payload(field_name, lambda: field_data.scalars.double_data.data)
            return data[start:end]

        elif dtype in (DataType.VARCHAR, DataType.STRING, DataType.TIMESTAMPTZ):
            data = get_payload(field_name, lambda: field_data.scalars.string_data.data)
            return data[start:end]

        elif dtype == DataType.JSON:
            data = get_payload(field_name, lambda: field_data.scalars.json_data.data)
            # Return raw bytes for JSON to avoid expensive bulk deserialization
            return data[start:end]

        elif dtype == DataType.ARRAY:
            data = get_payload(field_name, lambda: field_data.scalars.array_data.data)
            elem_type = field_data.scalars.array_data.element_type
            # Arrays must be converted because the Proto representation is recursive/complex
            return [self._extract_array_element(data[i], elem_type) for i in range(start, end)]
            
        elif dtype == DataType.GEOMETRY:
            data = get_payload(field_name, lambda: field_data.scalars.geometry_wkt_data.data)
            return data[start:end]
            
        # Fallback for complex types
        accessor = self._bind_accessor(field_name)
        return [accessor(i) for i in range(len(self))]
    def get_all_ids(self) -> List[Union[str, int]]:
        """Return all IDs for this query."""
        return self.ids

    def get_all_distances(self) -> List[float]:
        """Return all distances for this query."""
        return self.distances


class ColumnarSearchResult(list):
    """
    A drop-in replacement for SearchResult that uses columnar storage.

    Key differences from SearchResult:
    1. Initialization is O(1) - no pre-creation of Hit objects
    2. Data is stored in columnar format (references to protobuf)
    3. RowProxy objects are created on-demand during iteration/access
    4. This is READ-ONLY - data cannot be modified

    API Compatibility:
    - Fully compatible with SearchResult iteration patterns
    - All field types return the same Python types as SearchResult
    - Supports indexing, slicing, iteration

    Usage:
        # Works exactly like SearchResult
        for hits in result:
            for hit in hits:
                print(hit.id, hit.distance, hit['field_name'])
    """

    def __init__(
        self,
        res: schema_pb2.SearchResultData,
        round_decimal: Optional[int] = None,
        status: Optional[common_pb2.Status] = None,
        session_ts: Optional[int] = 0,
    ):
        self._res = res
        self._round_decimal = round_decimal
        pk_name = res.primary_field_name or "id"

        # Parse IDs
        if res.ids.HasField("int_id"):
            all_pks = res.ids.int_id.data
        elif res.ids.HasField("str_id"):
            all_pks = res.ids.str_id.data
        else:
            all_pks = []

        # Parse scores with optional rounding
        if isinstance(round_decimal, int) and round_decimal > 0:
            all_scores = [round(x, round_decimal) for x in res.scores]
        else:
            all_scores = res.scores

        # Create shared field map ONCE (not per ColumnarHits)
        fields_data_map = {fd.field_name: fd for fd in res.fields_data}
        fields = list(fields_data_map.keys())

        # Shared payload cache for all hits/queries
        column_payload_cache = {}

        # Create ColumnarHits for each query
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
                    list(res.output_fields),
                    pk_name,
                    column_payload_cache,
                )
            )
            nq_thres += topk

        super().__init__(data)

        # Set recalls
        self.recalls = res.recalls if len(res.recalls) > 0 else None

        # Set extra info
        self.extra = {}
        if status and status.extra_info:
            if "report_value" in status.extra_info:
                self.extra["cost"] = int(status.extra_info["report_value"])
            if "scanned_remote_bytes" in status.extra_info:
                self.extra["scanned_remote_bytes"] = int(status.extra_info["scanned_remote_bytes"])
            if "scanned_total_bytes" in status.extra_info:
                self.extra["scanned_total_bytes"] = int(status.extra_info["scanned_total_bytes"])
            if "cache_hit_ratio" in status.extra_info:
                self.extra["cache_hit_ratio"] = float(status.extra_info["cache_hit_ratio"])

        # Iterator related
        self._session_ts = session_ts
        self._search_iterator_v2_results = res.search_iterator_v2_results

    def __str__(self) -> str:
        """Only print at most 10 results."""
        result_msg = f"data: {self[:10]}"
        recall_msg = f",recalls: {self.recalls[:10]}" if self.recalls else ""
        extra_msg = f",{self.extra}" if self.extra else ""
        reminder = f" ... and {len(self) - 10} results remaining" if len(self) > 10 else ""
        return f"{result_msg}{recall_msg}{reminder}{extra_msg}"

    __repr__ = __str__

    def materialize(self):
        """
        No-op for compatibility.

        ColumnarSearchResult doesn't need explicit materialization since
        data is accessed on-demand. This method exists for API compatibility
        with SearchResult.
        """

    def get_session_ts(self):
        """Iterator related inner method."""
        return self._session_ts

    def get_search_iterator_v2_results_info(self):
        """Iterator related inner method."""
        return self._search_iterator_v2_results
