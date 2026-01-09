"""
Data Generation Module for Benchmark Kernels

Generates mock SearchResultData protobuf messages for all supported data types.
This module is pure Python with no pytest dependencies.
"""

import json
import random
import string
from typing import Any, Dict, List, Optional, Tuple

from pymilvus.client.types import DataType
from pymilvus.grpc_gen import schema_pb2


# =============================================================================
# Helper Functions
# =============================================================================

def _gen_random_string(length: int) -> str:
    """Generate a random ASCII string of specified length."""
    return "".join(random.choices(string.ascii_letters + string.digits, k=length))


def _gen_mixed_length_strings(count: int, lengths: Tuple[int, ...] = (32, 256, 1024, 8192)) -> List[str]:
    """Generate strings with mixed/uneven lengths."""
    return [_gen_random_string(random.choice(lengths)) for _ in range(count)]


# =============================================================================
# Core Data Generators
# =============================================================================

def create_search_result_data(
    nq: int,
    topk: int,
    fields: List[Dict[str, Any]],
    pk_name: str = "id",
    pk_type: DataType = DataType.INT64,
) -> schema_pb2.SearchResultData:
    """
    Create a mock SearchResultData protobuf message.

    Args:
        nq: Number of queries.
        topk: Number of results per query.
        fields: List of field configurations, each is a dict with:
            - name: Field name
            - dtype: DataType enum
            - dim: (Optional) Vector dimension
            - length: (Optional) String length for VARCHAR
            - element_type: (Optional) Element type for ARRAY
            - is_dynamic: (Optional) Whether field is dynamic
        pk_name: Primary key field name.
        pk_type: Primary key data type.

    Returns:
        schema_pb2.SearchResultData
    """
    total_hits = nq * topk
    res = schema_pb2.SearchResultData()
    res.num_queries = nq
    res.top_k = topk
    res.topks.extend([topk] * nq)
    res.primary_field_name = pk_name

    # Generate IDs
    if pk_type == DataType.INT64:
        res.ids.int_id.data.extend(range(total_hits))
    else:
        res.ids.str_id.data.extend([f"id_{i}" for i in range(total_hits)])

    # Generate Scores
    res.scores.extend([random.random() for _ in range(total_hits)])

    # Generate Fields
    for field_cfg in fields:
        _add_field_data(res, field_cfg, total_hits)

    return res


def _add_field_data(
    res: schema_pb2.SearchResultData,
    field_cfg: Dict[str, Any],
    total_hits: int,
) -> None:
    """Add a single field's data to the SearchResultData."""
    name = field_cfg["name"]
    dtype = field_cfg["dtype"]

    field = res.fields_data.add()
    field.field_name = name
    field.type = dtype
    field.is_dynamic = field_cfg.get("is_dynamic", False)

    # --- Scalars ---
    if dtype == DataType.BOOL:
        field.scalars.bool_data.data.extend([random.choice([True, False]) for _ in range(total_hits)])
    elif dtype in (DataType.INT8, DataType.INT16, DataType.INT32):
        field.scalars.int_data.data.extend([random.randint(0, 127) for _ in range(total_hits)])
    elif dtype == DataType.INT64:
        field.scalars.long_data.data.extend([random.randint(0, 10**9) for _ in range(total_hits)])
    elif dtype == DataType.FLOAT:
        field.scalars.float_data.data.extend([random.random() for _ in range(total_hits)])
    elif dtype == DataType.DOUBLE:
        field.scalars.double_data.data.extend([random.random() for _ in range(total_hits)])
    elif dtype == DataType.VARCHAR:
        length = field_cfg.get("length", 256)
        if length == "mixed":
            field.scalars.string_data.data.extend(_gen_mixed_length_strings(total_hits))
        else:
            s = _gen_random_string(length)
            field.scalars.string_data.data.extend([s] * total_hits)
    elif dtype == DataType.JSON:
        complexity = field_cfg.get("complexity", "simple")
        for i in range(total_hits):
            obj = _gen_json_object(i, complexity)
            field.scalars.json_data.data.append(json.dumps(obj).encode())
    elif dtype == DataType.ARRAY:
        elem_type = field_cfg.get("element_type", DataType.INT64)
        field.scalars.array_data.element_type = elem_type
        for i in range(total_hits):
            arr = field.scalars.array_data.data.add()
            if elem_type == DataType.INT64:
                arr.long_data.data.extend([i + j for j in range(5)])
            elif elem_type == DataType.FLOAT:
                arr.float_data.data.extend([float(i + j) for j in range(5)])
            elif elem_type in (DataType.VARCHAR, DataType.STRING):
                arr.string_data.data.extend([f"arr_{i}_{j}" for j in range(5)])
    elif dtype == DataType.GEOMETRY:
        for i in range(total_hits):
            field.scalars.geometry_wkt_data.data.append(f"POINT({i} {i + 1})")

    # --- Vectors (Dense) ---
    elif dtype == DataType.FLOAT_VECTOR:
        dim = field_cfg.get("dim", 128)
        field.vectors.dim = dim
        field.vectors.float_vector.data.extend([random.random() for _ in range(total_hits * dim)])
    elif dtype == DataType.BINARY_VECTOR:
        dim = field_cfg.get("dim", 128)  # bits
        field.vectors.dim = dim
        bytes_per_vec = dim // 8
        field.vectors.binary_vector = bytes([random.randint(0, 255) for _ in range(total_hits * bytes_per_vec)])
    elif dtype == DataType.FLOAT16_VECTOR:
        dim = field_cfg.get("dim", 128)
        field.vectors.dim = dim
        bytes_per_vec = dim * 2
        field.vectors.float16_vector = bytes([random.randint(0, 255) for _ in range(total_hits * bytes_per_vec)])
    elif dtype == DataType.BFLOAT16_VECTOR:
        dim = field_cfg.get("dim", 128)
        field.vectors.dim = dim
        bytes_per_vec = dim * 2
        field.vectors.bfloat16_vector = bytes([random.randint(0, 255) for _ in range(total_hits * bytes_per_vec)])
    elif dtype == DataType.INT8_VECTOR:
        dim = field_cfg.get("dim", 128)
        field.vectors.dim = dim
        field.vectors.int8_vector = bytes([random.randint(0, 255) for _ in range(total_hits * dim)])
    elif dtype == DataType.SPARSE_FLOAT_VECTOR:
        # Sparse vectors are more complex, creating a simple representation
        dim = field_cfg.get("dim", 128)
        field.vectors.dim = dim
        # For simplicity, store dummy sparse data
        # In real usage, sparse vectors have special format
        pass  # TODO: Add proper sparse vector generation if needed

    # --- Advanced ---
    elif dtype == DataType._ARRAY_OF_VECTOR:
        # Embedding List: array of vectors per row
        dim = field_cfg.get("dim", 128)
        num_vecs_per_row = field_cfg.get("num_vecs", 3)
        for i in range(total_hits):
            vec_data = field.vectors.vector_array.data.add()
            vec_data.dim = dim
            vec_data.float_vector.data.extend([random.random() for _ in range(num_vecs_per_row * dim)])

    res.output_fields.append(name)


def _gen_json_object(index: int, complexity: str) -> Dict[str, Any]:
    """Generate a JSON object with varying complexity."""
    if complexity == "simple":
        return {"id": index, "value": random.random()}
    elif complexity == "medium":
        return {
            "id": index,
            "name": f"item_{index}",
            "tags": [f"tag_{i}" for i in range(3)],
            "score": random.random(),
        }
    else:  # complex
        return {
            "user": {
                "id": index,
                "name": f"user_{index}",
                "email": f"user_{index}@example.com",
            },
            "metadata": {
                "tags": [f"tag_{i}" for i in range(5)],
                "scores": [random.random() for _ in range(5)],
            },
            "timestamp": 1700000000 + index,
        }


# =============================================================================
# Pre-defined Field Configurations
# =============================================================================

# Core Scalar Fields
SCALAR_FIELDS_CORE = [
    {"name": "bool_field", "dtype": DataType.BOOL},
    {"name": "int8_field", "dtype": DataType.INT8},
    {"name": "int16_field", "dtype": DataType.INT16},
    {"name": "int32_field", "dtype": DataType.INT32},
    {"name": "int64_field", "dtype": DataType.INT64},
    {"name": "float_field", "dtype": DataType.FLOAT},
    {"name": "double_field", "dtype": DataType.DOUBLE},
]

# VARCHAR Fields with varying lengths
def get_varchar_field(length: int = 256) -> Dict[str, Any]:
    return {"name": f"varchar_{length}", "dtype": DataType.VARCHAR, "length": length}

# Vector Fields
def get_vector_field(dtype: DataType, dim: int = 128) -> Dict[str, Any]:
    return {"name": f"vector_{dtype.name.lower()}_{dim}", "dtype": dtype, "dim": dim}

# Dynamic Field (stored in $meta JSON)
DYNAMIC_FIELD = {"name": "$meta", "dtype": DataType.JSON, "is_dynamic": True}

# JSON Fields
def get_json_field(complexity: str = "simple") -> Dict[str, Any]:
    return {"name": f"json_{complexity}", "dtype": DataType.JSON, "complexity": complexity}

# Array Fields
def get_array_field(element_type: DataType = DataType.INT64) -> Dict[str, Any]:
    return {"name": f"array_{element_type.name.lower()}", "dtype": DataType.ARRAY, "element_type": element_type}

# Embedding List (Array of Vector)
def get_embedding_list_field(dim: int = 128, num_vecs: int = 3) -> Dict[str, Any]:
    return {"name": f"emb_list_{dim}", "dtype": DataType._ARRAY_OF_VECTOR, "dim": dim, "num_vecs": num_vecs}


# =============================================================================
# Convenience Functions for Common Test Scenarios
# =============================================================================

def create_kitchen_sink_result(nq: int = 10, topk: int = 100, dim: int = 128) -> schema_pb2.SearchResultData:
    """Create a SearchResultData with ALL field types (Kitchen Sink)."""
    fields = [
        *SCALAR_FIELDS_CORE,
        get_varchar_field(256),
        get_json_field("medium"),
        get_array_field(DataType.INT64),
        get_vector_field(DataType.FLOAT_VECTOR, dim),
        DYNAMIC_FIELD,
    ]
    return create_search_result_data(nq, topk, fields)


def create_vector_only_result(
    nq: int, topk: int, dtype: DataType = DataType.FLOAT_VECTOR, dim: int = 128
) -> schema_pb2.SearchResultData:
    """Create a SearchResultData with only a vector field."""
    fields = [get_vector_field(dtype, dim)]
    return create_search_result_data(nq, topk, fields)


def create_scalar_only_result(
    nq: int, topk: int, dtype: DataType, **kwargs
) -> schema_pb2.SearchResultData:
    """Create a SearchResultData with only a scalar field."""
    if dtype == DataType.VARCHAR:
        field = get_varchar_field(kwargs.get("length", 256))
    elif dtype == DataType.JSON:
        field = get_json_field(kwargs.get("complexity", "simple"))
    elif dtype == DataType.ARRAY:
        field = get_array_field(kwargs.get("element_type", DataType.INT64))
    else:
        field = {"name": f"scalar_{dtype.name.lower()}", "dtype": dtype}
    return create_search_result_data(nq, topk, [field])
