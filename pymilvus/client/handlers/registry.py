"""
Type handler registry - singleton that manages all type handlers.

The registry provides O(1) lookup of handlers by DataType.
"""

from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    from pymilvus.client.types import DataType

    from .base import TypeHandler


class TypeHandlerRegistry:
    """Singleton registry for all type handlers.

    Provides fast O(1) lookup of handlers by DataType value.
    Handlers are registered during module initialization.
    """

    _instance: Optional["TypeHandlerRegistry"] = None
    _initialized: bool = False

    def __new__(cls) -> "TypeHandlerRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if TypeHandlerRegistry._initialized:
            return
        TypeHandlerRegistry._initialized = True

        # Map from DataType value (int) to handler instance
        self._handlers: Dict[int, "TypeHandler"] = {}

    def register(self, handler: "TypeHandler") -> None:
        """Register a handler for its supported types.

        Args:
            handler: The TypeHandler instance to register
        """
        for dtype in handler.supported_types:
            self._handlers[int(dtype)] = handler

    def get(self, dtype: "DataType") -> Optional["TypeHandler"]:
        """Get the handler for a DataType.

        Args:
            dtype: The DataType to get a handler for

        Returns:
            The registered handler, or None if no handler is registered
        """
        return self._handlers.get(int(dtype))

    def has_handler(self, dtype: "DataType") -> bool:
        """Check if a handler is registered for a DataType."""
        return int(dtype) in self._handlers

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (for testing)."""
        cls._instance = None
        cls._initialized = False


# Global registry instance
_registry: Optional[TypeHandlerRegistry] = None


def _ensure_handlers_registered() -> TypeHandlerRegistry:
    """Ensure all handlers are registered and return the registry."""
    global _registry
    if _registry is None:
        _registry = TypeHandlerRegistry()
        _register_all_handlers(_registry)
    return _registry


def _register_all_handlers(registry: TypeHandlerRegistry) -> None:
    """Register all built-in handlers."""
    # Import handlers here to avoid circular imports
    from .complex import ArrayHandler, ArrayOfStructHandler, ArrayOfVectorHandler, JsonHandler
    from .scalars import (
        BoolHandler,
        DoubleHandler,
        FloatHandler,
        GeometryHandler,
        Int64Handler,
        IntHandler,
        TimestampTzHandler,
        VarCharHandler,
    )
    from .vectors import (
        BFloat16VectorHandler,
        BinaryVectorHandler,
        Float16VectorHandler,
        FloatVectorHandler,
        Int8VectorHandler,
        SparseFloatVectorHandler,
    )

    # Register scalar handlers
    registry.register(BoolHandler())
    registry.register(IntHandler())
    registry.register(Int64Handler())
    registry.register(FloatHandler())
    registry.register(DoubleHandler())
    registry.register(VarCharHandler())
    registry.register(GeometryHandler())
    registry.register(TimestampTzHandler())

    # Register vector handlers
    registry.register(FloatVectorHandler())
    registry.register(BinaryVectorHandler())
    registry.register(Float16VectorHandler())
    registry.register(BFloat16VectorHandler())
    registry.register(Int8VectorHandler())
    registry.register(SparseFloatVectorHandler())

    # Register complex handlers
    registry.register(JsonHandler())
    registry.register(ArrayHandler())
    registry.register(ArrayOfStructHandler())
    registry.register(ArrayOfVectorHandler())


def get_handler(dtype: "DataType") -> "TypeHandler":
    """Get the handler for a DataType.

    Args:
        dtype: The DataType to get a handler for

    Returns:
        The registered handler

    Raises:
        ValueError: If no handler is registered for the type
    """
    registry = _ensure_handlers_registered()
    handler = registry.get(dtype)
    if handler is None:
        raise ValueError(f"No handler registered for DataType: {dtype}")
    return handler


def has_handler(dtype: "DataType") -> bool:
    """Check if a handler is registered for a DataType."""
    registry = _ensure_handlers_registered()
    return registry.has_handler(dtype)
