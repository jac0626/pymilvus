"""
Insert Operations Module for Benchmark Kernels

Pure functions for benchmarking insert data preparation.
This module is pure Python with no pytest dependencies.

Note: This module benchmarks the CLIENT-SIDE data preparation for insert,
not the actual server round-trip.
"""

import random
import string
import json
from typing import Any, Dict, List, Optional

from pymilvus.client.types import DataType


# =============================================================================
# Helper Functions
# =============================================================================

def _gen_random_string(length: int) -> str:
    """Generate a random ASCII string of specified length."""
    return "".join(random.choices(string.ascii_letters + string.digits, k=length))


def _gen_random_vector(dim: int) -> List[float]:
    """Generate a random float vector."""
    return [random.random() for _ in range(dim)]


def _gen_random_binary_vector(dim: int) -> bytes:
    """Generate a random binary vector (dim is in bits)."""
    return bytes([random.randint(0, 255) for _ in range(dim // 8)])


# =============================================================================
# Insert Data Generators
# =============================================================================

def generate_insert_data(
    batch_size: int,
    field_configs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Generate insert data as a list of dictionaries (row-based format).
    
    Args:
        batch_size: Number of rows to generate.
        field_configs: List of field configurations, each with:
            - name: Field name
            - dtype: DataType enum
            - dim: (Optional) Vector dimension
            - length: (Optional) String length for VARCHAR
            - element_type: (Optional) Element type for ARRAY
    
    Returns:
        List of dictionaries, each representing a row.
    """
    rows = []
    for i in range(batch_size):
        row = {}
        for cfg in field_configs:
            row[cfg["name"]] = _generate_field_value(i, cfg)
        rows.append(row)
    return rows


def generate_insert_data_columnar(
    batch_size: int,
    field_configs: List[Dict[str, Any]],
) -> Dict[str, List[Any]]:
    """
    Generate insert data in columnar format (dict of lists).
    
    Args:
        batch_size: Number of rows to generate.
        field_configs: List of field configurations.
    
    Returns:
        Dictionary mapping field names to lists of values.
    """
    columns = {cfg["name"]: [] for cfg in field_configs}
    for i in range(batch_size):
        for cfg in field_configs:
            columns[cfg["name"]].append(_generate_field_value(i, cfg))
    return columns


def _generate_field_value(index: int, cfg: Dict[str, Any]) -> Any:
    """Generate a single field value based on configuration."""
    dtype = cfg["dtype"]
    
    # Scalars
    if dtype == DataType.BOOL:
        return random.choice([True, False])
    elif dtype in (DataType.INT8, DataType.INT16, DataType.INT32):
        return random.randint(0, 127)
    elif dtype == DataType.INT64:
        return index  # Use index as ID for predictability
    elif dtype == DataType.FLOAT:
        return random.random()
    elif dtype == DataType.DOUBLE:
        return random.random()
    elif dtype == DataType.VARCHAR:
        length = cfg.get("length", 256)
        return _gen_random_string(length)
    elif dtype == DataType.JSON:
        complexity = cfg.get("complexity", "simple")
        return _gen_json_value(index, complexity)
    elif dtype == DataType.ARRAY:
        elem_type = cfg.get("element_type", DataType.INT64)
        length = cfg.get("array_length", 5)
        return _gen_array_value(elem_type, length)
    
    # Vectors
    elif dtype == DataType.FLOAT_VECTOR:
        dim = cfg.get("dim", 128)
        return _gen_random_vector(dim)
    elif dtype == DataType.BINARY_VECTOR:
        dim = cfg.get("dim", 128)
        return _gen_random_binary_vector(dim)
    elif dtype in (DataType.FLOAT16_VECTOR, DataType.BFLOAT16_VECTOR):
        dim = cfg.get("dim", 128)
        return bytes([random.randint(0, 255) for _ in range(dim * 2)])
    elif dtype == DataType.INT8_VECTOR:
        dim = cfg.get("dim", 128)
        return bytes([random.randint(0, 255) for _ in range(dim)])
    
    # Advanced
    elif dtype == DataType._ARRAY_OF_VECTOR:
        dim = cfg.get("dim", 128)
        num_vecs = cfg.get("num_vecs", 3)
        return [_gen_random_vector(dim) for _ in range(num_vecs)]
    
    return None


def _gen_json_value(index: int, complexity: str) -> Dict[str, Any]:
    """Generate a JSON value with specified complexity."""
    if complexity == "simple":
        return {"id": index, "value": random.random()}
    elif complexity == "medium":
        return {
            "id": index,
            "name": f"item_{index}",
            "tags": [f"tag_{i}" for i in range(3)],
        }
    else:
        return {
            "user": {"id": index, "name": f"user_{index}"},
            "metadata": {"tags": [f"t_{i}" for i in range(5)]},
        }


def _gen_array_value(elem_type: DataType, length: int) -> List[Any]:
    """Generate an array value."""
    if elem_type == DataType.INT64:
        return [random.randint(0, 1000) for _ in range(length)]
    elif elem_type == DataType.FLOAT:
        return [random.random() for _ in range(length)]
    elif elem_type in (DataType.VARCHAR, DataType.STRING):
        return [_gen_random_string(16) for _ in range(length)]
    return []


# =============================================================================
# Pre-defined Field Configurations (Kitchen Sink)
# =============================================================================

def get_kitchen_sink_fields(dim: int = 128) -> List[Dict[str, Any]]:
    """Get a comprehensive list of fields covering all types."""
    return [
        {"name": "id", "dtype": DataType.INT64},
        {"name": "bool_field", "dtype": DataType.BOOL},
        {"name": "int8_field", "dtype": DataType.INT8},
        {"name": "int16_field", "dtype": DataType.INT16},
        {"name": "int32_field", "dtype": DataType.INT32},
        {"name": "float_field", "dtype": DataType.FLOAT},
        {"name": "double_field", "dtype": DataType.DOUBLE},
        {"name": "varchar_field", "dtype": DataType.VARCHAR, "length": 256},
        {"name": "json_field", "dtype": DataType.JSON, "complexity": "medium"},
        {"name": "array_field", "dtype": DataType.ARRAY, "element_type": DataType.INT64},
        {"name": "vector_field", "dtype": DataType.FLOAT_VECTOR, "dim": dim},
    ]


# =============================================================================
# Benchmark Functions
# =============================================================================

def run_insert_data_generation_benchmark(
    batch_size: int,
    field_configs: List[Dict[str, Any]],
    format: str = "row",
) -> int:
    """
    Benchmark insert data generation.
    
    Args:
        batch_size: Number of rows to generate.
        field_configs: Field configurations.
        format: "row" for list of dicts, "columnar" for dict of lists.
    
    Returns:
        Number of rows generated.
    """
    if format == "row":
        data = generate_insert_data(batch_size, field_configs)
    else:
        data = generate_insert_data_columnar(batch_size, field_configs)
    return batch_size
