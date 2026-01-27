"""
Abstract base class for type handlers.

Each handler is responsible for:
1. Packing Python values into protobuf FieldData (write path)
2. Extracting values from protobuf FieldData back to Python (read path)
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from pymilvus.client.types import DataType
    from pymilvus.grpc_gen import schema_pb2

    from .context import ExtractContext, PackContext


class TypeHandler(ABC):
    """Abstract base class for all type handlers.

    Each handler is responsible for a set of related DataTypes and implements
    both the write path (packing values into protobuf) and read path
    (extracting values from protobuf).

    To add a new type:
    1. Create a handler class inheriting from TypeHandler
    2. Set supported_types to the tuple of DataTypes this handler supports
    3. Implement all abstract methods
    4. Register the handler in the registry
    """

    # Tuple of DataType values this handler supports
    supported_types: Tuple["DataType", ...] = ()

    # === Write Path Methods ===

    @abstractmethod
    def pack_value(
        self,
        value: Any,
        field_data: "schema_pb2.FieldData",
        field_info: Dict[str, Any],
        context: "PackContext",
    ) -> None:
        """Pack a single value into field_data.

        This is called when inserting data row-by-row.

        Args:
            value: The Python value to pack (can be None for nullable fields)
            field_data: The protobuf FieldData to populate
            field_info: Field metadata including name, type, params, etc.
            context: Pack context containing caches and utilities
        """

    @abstractmethod
    def pack_values(
        self,
        values: List[Any],
        field_data: "schema_pb2.FieldData",
        field_info: Dict[str, Any],
    ) -> None:
        """Pack multiple values into field_data at once.

        This is called for batch operations where all values are available.
        More efficient than calling pack_value repeatedly.

        Args:
            values: List of Python values to pack
            field_data: The protobuf FieldData to populate
            field_info: Field metadata including name, type, params, etc.
        """

    # === Read Path Methods ===

    @abstractmethod
    def get_raw_data(self, field_data: "schema_pb2.FieldData") -> Any:
        """Get the raw data container from field_data.

        Returns the underlying protobuf data structure (e.g., scalars.int_data.data).

        Args:
            field_data: The protobuf FieldData to extract from

        Returns:
            The raw data container (list, bytes, or other protobuf structure)
        """

    @abstractmethod
    def extract_value(
        self,
        field_data: "schema_pb2.FieldData",
        index: int,
        context: "ExtractContext",
    ) -> Any:
        """Extract a single value at the given index.

        Args:
            field_data: The protobuf FieldData to extract from
            index: The logical row index
            context: Extract context with caches and utilities

        Returns:
            The extracted Python value, or None for null values
        """

    def extract_range(
        self,
        field_data: "schema_pb2.FieldData",
        start: int,
        end: int,
    ) -> Tuple[List[Any], "schema_pb2.FieldData"]:
        """Extract a range of values.

        Default implementation calls extract_value repeatedly.
        Override for more efficient batch extraction.

        Args:
            field_data: The protobuf FieldData to extract from
            start: Start index (inclusive)
            end: End index (exclusive)

        Returns:
            Tuple of (extracted data list, field metadata)
        """
        from .context import ExtractContext

        context = ExtractContext()
        return [self.extract_value(field_data, i, context) for i in range(start, end)], field_data

    def extract_into_row(
        self,
        field_data: "schema_pb2.FieldData",
        row_dict: Dict[str, Any],
        index: int,
        context: "ExtractContext",
    ) -> bool:
        """Extract value at index and insert into row_dict.

        Args:
            field_data: The protobuf FieldData to extract from
            row_dict: The dictionary to insert the value into
            index: The logical row index
            context: Extract context with caches and utilities

        Returns:
            True if this field should be processed lazily, False otherwise
        """
        value = self.extract_value(field_data, index, context)
        row_dict[field_data.field_name] = value
        return False

    # === Utility Methods ===

    def is_nullable_null(
        self, field_data: "schema_pb2.FieldData", index: int
    ) -> bool:
        """Check if the value at index is null in a nullable field.

        Args:
            field_data: The protobuf FieldData
            index: The row index

        Returns:
            True if the value is null, False otherwise
        """
        if len(field_data.valid_data) > 0:
            return not field_data.valid_data[index]
        return False

    def get_field_name(self, field_info: Dict[str, Any]) -> str:
        """Get the field name from field_info."""
        return field_info.get("name", "")
