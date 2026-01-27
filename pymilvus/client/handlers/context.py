"""
Context objects for pack and extract operations.

These provide shared state and utilities across handler calls,
including caches for performance optimization.
"""

from typing import Any, Dict, List, Optional

import numpy as np


class PackContext:
    """Context for packing operations (write path).

    Provides shared state across multiple pack_value calls,
    including byte caches for vector types.
    """

    def __init__(self) -> None:
        # Cache for accumulating vector bytes before final flush
        # Key: field_data id, Value: list of byte chunks
        self.vector_bytes_cache: Dict[int, List[bytes]] = {}

    def get_bytes_cache(self, field_data_id: int) -> List[bytes]:
        """Get or create a bytes cache for a field_data."""
        if field_data_id not in self.vector_bytes_cache:
            self.vector_bytes_cache[field_data_id] = []
        return self.vector_bytes_cache[field_data_id]

    def append_bytes(self, field_data_id: int, data: bytes) -> None:
        """Append bytes to the cache for a field_data."""
        cache = self.get_bytes_cache(field_data_id)
        cache.append(data)

    def pop_bytes_cache(self, field_data_id: int) -> Optional[List[bytes]]:
        """Pop and return the bytes cache for a field_data."""
        return self.vector_bytes_cache.pop(field_data_id, None)

    def flush_vector_bytes(self, field_data: Any, attr_name: str) -> None:
        """Flush accumulated bytes to the field_data vector attribute.

        This joins all cached byte chunks into a single bytes object
        and sets it on the appropriate vector field.

        Args:
            field_data: The protobuf FieldData object
            attr_name: The attribute name on field_data.vectors (e.g., 'binary_vector')
        """
        field_id = id(field_data)
        bytes_list = self.pop_bytes_cache(field_id)
        if bytes_list:
            setattr(field_data.vectors, attr_name, b"".join(bytes_list))


class ExtractContext:
    """Context for extraction operations (read path).

    Provides shared state across multiple extract calls,
    including caches for prefix sums (for nullable field indexing).
    """

    def __init__(
        self,
        dynamic_output_fields: Optional[List[str]] = None,
    ) -> None:
        # Fields to include from dynamic JSON fields
        self.dynamic_output_fields = set(dynamic_output_fields) if dynamic_output_fields else set()

        # Cache for prefix sums used in nullable vector physical index calculation
        # Key: field_data id, Value: prefix sum array or None
        self._prefix_sum_cache: Dict[int, Optional[np.ndarray]] = {}

    def get_physical_index(self, field_data: Any, logical_index: int) -> int:
        """Calculate physical index for nullable vectors with sparse storage.

        For nullable vectors, valid_data indicates which logical positions have valid data,
        and the actual data only contains valid values (sparse storage).
        Uses prefix sum for O(1) lookup instead of O(n) iteration.

        Args:
            field_data: The protobuf FieldData with valid_data
            logical_index: The logical row index

        Returns:
            The physical index in the dense data array
        """
        field_id = id(field_data)
        if field_id not in self._prefix_sum_cache:
            if len(field_data.valid_data) == 0:
                self._prefix_sum_cache[field_id] = None
            else:
                self._prefix_sum_cache[field_id] = np.cumsum(
                    [0] + [1 if v else 0 for v in field_data.valid_data]
                )

        prefix_sum = self._prefix_sum_cache[field_id]
        if prefix_sum is None:
            return logical_index
        return int(prefix_sum[logical_index])
