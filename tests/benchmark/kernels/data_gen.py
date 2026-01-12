#!/usr/bin/env python3
# Copyright (c) PyMilvus Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""
Unified mock data generation for benchmarking.

Consolidates all build_*_result functions into a single, composable API.
"""

import json
from dataclasses import dataclass
from typing import List, Optional, Tuple, Any, Dict
from enum import Enum

from pymilvus.grpc_gen import schema_pb2


# =============================================================================
# Configuration Types
# =============================================================================

class ScalarComplexity(Enum):
    """Complexity levels for scalar field data generation."""
    SMALL = "SMALL"      # ~20-50 chars for VARCHAR, simple JSON
    MEDIUM = "MEDIUM"    # ~200-500 chars for VARCHAR, nested JSON
    LARGE = "LARGE"      # ~1500-2500 chars for VARCHAR
    COMPLEX = "COMPLEX"  # Deep nested JSON structures
    UNEVEN = "UNEVEN"    # Heterogeneous: varying structure per row (realistic)


@dataclass
class ScalarFieldConfig:
    """Configuration for a scalar field."""
    name: str
    dtype: str  # "INT64", "VARCHAR", "JSON", "ARRAY", "BOOL", "FLOAT", "DOUBLE"
    complexity: Optional[ScalarComplexity] = None


@dataclass
class VectorFieldConfig:
    """Configuration for a vector field."""
    name: str
    dtype: str  # "FLOAT_VECTOR", "FLOAT16_VECTOR", "BFLOAT16_VECTOR", "BINARY_VECTOR", "INT8_VECTOR"
    dim: int = 128


# =============================================================================
# Core Builders (Low-Level API)
# =============================================================================

def build_search_result_base(nq: int, topk: int) -> schema_pb2.SearchResultData:
    """
    Create a base SearchResultData with IDs, scores, and topks.
    
    This is the foundation for all search result builders.
    """
    total = nq * topk
    res = schema_pb2.SearchResultData()
    res.num_queries = nq
    res.top_k = topk
    res.topks.extend([topk] * nq)
    res.ids.int_id.data.extend(list(range(total)))
    res.scores.extend([float(i) * 0.001 for i in range(total)])
    res.primary_field_name = "id"
    return res


def add_scalar_field(
    result: schema_pb2.SearchResultData,
    name: str,
    dtype: str,
    complexity: Optional[ScalarComplexity] = None,
) -> None:
    """
    Add a scalar field to an existing SearchResultData.
    
    Args:
        result: The SearchResultData to modify
        name: Field name
        dtype: One of "INT8", "INT16", "INT32", "INT64", "FLOAT", "DOUBLE", 
               "BOOL", "VARCHAR", "JSON", "ARRAY", "GEOMETRY", "TIMESTAMPTZ"
        complexity: For VARCHAR/JSON, controls data size (SMALL/MEDIUM/LARGE/COMPLEX)
    """
    total = result.num_queries * result.top_k
    field = result.fields_data.add()
    field.field_name = name
    
    if dtype == "INT8":
        field.type = schema_pb2.DataType.Int8
        field.scalars.int_data.data.extend([i % 128 for i in range(total)])
        
    elif dtype == "INT16":
        field.type = schema_pb2.DataType.Int16
        field.scalars.int_data.data.extend([i % 32768 for i in range(total)])
        
    elif dtype == "INT32":
        field.type = schema_pb2.DataType.Int32
        field.scalars.int_data.data.extend(list(range(total)))
        
    elif dtype == "INT64":
        field.type = schema_pb2.DataType.Int64
        field.scalars.long_data.data.extend([i * 100 for i in range(total)])
        
    elif dtype == "FLOAT":
        field.type = schema_pb2.DataType.Float
        field.scalars.float_data.data.extend([0.5 + i * 0.01 for i in range(total)])
        
    elif dtype == "DOUBLE":
        field.type = schema_pb2.DataType.Double
        field.scalars.double_data.data.extend([float(i) for i in range(total)])
        
    elif dtype == "BOOL":
        field.type = schema_pb2.DataType.Bool
        field.scalars.bool_data.data.extend([i % 2 == 0 for i in range(total)])
        
    elif dtype == "VARCHAR":
        field.type = schema_pb2.DataType.VarChar
        _fill_varchar_field(field, total, complexity or ScalarComplexity.SMALL)
        
    elif dtype == "JSON":
        field.type = schema_pb2.DataType.JSON
        _fill_json_field(field, total, complexity or ScalarComplexity.MEDIUM)
        
    elif dtype == "ARRAY":
        field.type = schema_pb2.DataType.Array
        field.scalars.array_data.element_type = schema_pb2.DataType.Int64
        for i in range(total):
            array_data = field.scalars.array_data.data.add()
            array_data.long_data.data.extend([100 + i + j for j in range(5)])
            
    elif dtype == "GEOMETRY":
        field.type = schema_pb2.DataType.Geometry
        # WKT format geometry data
        for i in range(total):
            field.scalars.geometry_wkt_data.data.append(f"POINT({i % 180} {i % 90})")
            
    elif dtype == "TIMESTAMPTZ":
        field.type = schema_pb2.DataType.Timestamptz
        # ISO 8601 timestamp format
        for i in range(total):
            field.scalars.string_data.data.append(f"2024-01-{(i % 28) + 1:02d}T12:00:00Z")
            
    elif dtype == "_ARRAY_OF_STRUCT":
        # Composite type: array of struct (used internally for STRUCT fields)
        field.type = schema_pb2.DataType.ArrayOfStruct
        # Create struct_arrays with sub-fields
        struct_arrays = field.struct_arrays
        # Add an INT64 sub-field
        int_field = struct_arrays.fields.add()
        int_field.field_name = "int_col"
        int_field.type = schema_pb2.DataType.Array
        int_field.scalars.array_data.element_type = schema_pb2.DataType.Int64
        # Add a VARCHAR sub-field
        str_field = struct_arrays.fields.add()
        str_field.field_name = "str_col"
        str_field.type = schema_pb2.DataType.Array
        str_field.scalars.array_data.element_type = schema_pb2.DataType.VarChar
        # Fill data for each row
        for i in range(total):
            # Each row has an array of structs, here we simulate 3 structs per row
            int_array = int_field.scalars.array_data.data.add()
            int_array.long_data.data.extend([i * 10 + j for j in range(3)])
            str_array = str_field.scalars.array_data.data.add()
            str_array.string_data.data.extend([f"item_{i}_{j}" for j in range(3)])
            
    elif dtype == "_ARRAY_OF_VECTOR":
        # Composite type: array of vectors (used internally for STRUCT with vector fields)
        field.type = schema_pb2.DataType.ArrayOfVector
        vector_array = field.vectors.vector_array
        # Each row contains multiple vectors
        for i in range(total):
            vec_data = vector_array.data.add()
            vec_data.dim = 4  # Small dimension for testing
            # 3 vectors per row, each with 4 floats
            for j in range(3):
                for k in range(4):
                    vec_data.float_vector.data.append(float(i * 100 + j * 10 + k))
    
    result.output_fields.append(name)


def _fill_varchar_field(
    field: schema_pb2.FieldData, 
    total: int, 
    complexity: ScalarComplexity
) -> None:
    """Fill VARCHAR field with realistic data based on complexity."""
    if complexity == ScalarComplexity.SMALL:
        # Realistic: product names, usernames, titles (~20-50 chars)
        templates = [
            "Premium Electronics Item {id}",
            "User_{id}@example.com",
            "Document Title {id}"
        ]
        for i in range(total):
            field.scalars.string_data.data.append(
                templates[i % 3].format(id=i)
            )
            
    elif complexity == ScalarComplexity.MEDIUM:
        # Realistic: descriptions, bios (~300-500 chars)
        template = (
            "This is a detailed description for item {id}. "
            "Features include high quality materials, excellent craftsmanship, and modern design. "
            "Perfect for daily use and special occasions. "
            "Available in multiple colors and sizes. "
            "Trusted by thousands of customers worldwide."
        )
        for i in range(total):
            field.scalars.string_data.data.append(template.format(id=i))
            
    elif complexity in (ScalarComplexity.LARGE, ScalarComplexity.COMPLEX):
        # Realistic: article content, long reviews (~1500-2500 chars)
        template = ("Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 25) + "_item_{}"
        for i in range(total):
            field.scalars.string_data.data.append(template.format(i))


def _fill_json_field(
    field: schema_pb2.FieldData, 
    total: int, 
    complexity: ScalarComplexity
) -> None:
    """Fill JSON field with realistic data based on complexity."""
    for i in range(total):
        if complexity == ScalarComplexity.SMALL:
            json_obj = {
                "author": f"user_{i}",
                "created_at": "2024-01-01",
                "category": "electronics"
            }
        elif complexity == ScalarComplexity.MEDIUM:
            json_obj = {
                "brand": f"Brand_{i % 10}",
                "category": "electronics",
                "tags": ["popular", "new", "sale"],
                "price": 99.99,
                "stock": 100
            }
        elif complexity == ScalarComplexity.UNEVEN:
            # Heterogeneous: each row has different structure/depth
            variant = i % 5
            if variant == 0:
                # Simple flat
                json_obj = {"id": i, "status": "active"}
            elif variant == 1:
                # Nested structure
                json_obj = {"level1": {"level2": {"level3": {"value": i}}}}
            elif variant == 2:
                # Array-heavy
                json_obj = {"items": list(range(i % 20)), "tags": [f"t{j}" for j in range(i % 10)]}
            elif variant == 3:
                # Mixed types
                json_obj = {
                    "int_val": i,
                    "float_val": i * 0.123,
                    "bool_val": i % 2 == 0,
                    "str_val": f"string_{i}",
                    "null_val": None
                }
            else:
                # Wide object (many keys)
                json_obj = {f"key_{j}": j * i for j in range(15)}
        else:  # LARGE or COMPLEX
            json_obj = {
                "user": {
                    "id": i,
                    "name": f"user_{i}",
                    "email": f"user_{i}@example.com",
                    "verified": True
                },
                "preferences": {
                    "notifications": True,
                    "privacy": "public"
                },
                "metadata": {
                    "tags": [f"tag_{j}" for j in range(5)],
                    "scores": [j * 0.1 for j in range(5)]
                }
            }
        
        field.scalars.json_data.data.append(json.dumps(json_obj).encode())


def add_vector_field(
    result: schema_pb2.SearchResultData,
    name: str,
    dtype: str,
    dim: int = 128,
) -> None:
    """
    Add a vector field to an existing SearchResultData.
    
    Args:
        result: The SearchResultData to modify
        name: Field name
        dtype: One of "FLOAT_VECTOR", "FLOAT16_VECTOR", "BFLOAT16_VECTOR", 
               "BINARY_VECTOR", "INT8_VECTOR", "SPARSE_FLOAT_VECTOR"
        dim: Vector dimension (for BINARY_VECTOR, this is in bits; ignored for SPARSE)
    """
    total = result.num_queries * result.top_k
    field = result.fields_data.add()
    field.field_name = name
    
    if dtype == "FLOAT_VECTOR":
        field.type = schema_pb2.DataType.FloatVector
        field.vectors.dim = dim
        dummy_data = [0.123] * (total * dim)
        field.vectors.float_vector.data.extend(dummy_data)
        
    elif dtype == "FLOAT16_VECTOR":
        field.type = schema_pb2.DataType.Float16Vector
        field.vectors.dim = dim
        field.vectors.float16_vector = bytes([i % 256 for i in range(total * dim * 2)])
        
    elif dtype == "BFLOAT16_VECTOR":
        field.type = schema_pb2.DataType.BFloat16Vector
        field.vectors.dim = dim
        field.vectors.bfloat16_vector = bytes([i % 256 for i in range(total * dim * 2)])
        
    elif dtype == "BINARY_VECTOR":
        field.type = schema_pb2.DataType.BinaryVector
        field.vectors.dim = dim
        bytes_per_vec = dim // 8
        field.vectors.binary_vector = bytes([i % 256 for i in range(total * bytes_per_vec)])
        
    elif dtype == "INT8_VECTOR":
        field.type = schema_pb2.DataType.Int8Vector
        field.vectors.dim = dim
        field.vectors.int8_vector = bytes([i % 256 for i in range(total * dim)])
        
    elif dtype == "SPARSE_FLOAT_VECTOR":
        import struct
        field.type = schema_pb2.DataType.SparseFloatVector
        # Sparse vectors: each row has variable number of (index, value) pairs
        # Format: 8 bytes per element (4 bytes index + 4 bytes float value)
        sparse_data = field.vectors.sparse_float_vector
        for i in range(total):
            # Create sparse vector with ~10 non-zero elements
            nnz = 10
            row_bytes = b''
            for j in range(nnz):
                idx = i * 100 + j * 10  # Sparse indices
                val = 0.1 * (j + 1)
                row_bytes += struct.pack('I', idx) + struct.pack('f', val)
            sparse_data.contents.append(row_bytes)
    
    result.output_fields.append(name)


# =============================================================================
# High-Level API
# =============================================================================

def build_search_result(
    nq: int,
    topk: int,
    scalar_fields: Optional[List[Tuple[str, str, Optional[str]]]] = None,
    vector_fields: Optional[List[Tuple[str, str, int]]] = None,
) -> schema_pb2.SearchResultData:
    """
    Build a complete SearchResultData with specified fields.
    
    Args:
        nq: Number of queries
        topk: Top K results per query
        scalar_fields: List of (name, type, complexity) tuples, e.g. [("age", "INT64", None)]
        vector_fields: List of (name, type, dim) tuples, e.g. [("vector", "FLOAT_VECTOR", 128)]
    
    Returns:
        A fully populated SearchResultData protobuf message
    
    Example:
        >>> result = build_search_result(
        ...     nq=10, topk=100,
        ...     scalar_fields=[("name", "VARCHAR", "MEDIUM"), ("meta", "JSON", "COMPLEX")],
        ...     vector_fields=[("embedding", "FLOAT_VECTOR", 768)]
        ... )
    """
    result = build_search_result_base(nq, topk)
    
    if scalar_fields:
        for field_tuple in scalar_fields:
            name, dtype = field_tuple[0], field_tuple[1]
            complexity_str = field_tuple[2] if len(field_tuple) > 2 else None
            complexity = ScalarComplexity(complexity_str) if complexity_str else None
            add_scalar_field(result, name, dtype, complexity)
    
    if vector_fields:
        for name, dtype, dim in vector_fields:
            add_vector_field(result, name, dtype, dim)
    
    return result


# =============================================================================
# Insert Data Generation
# =============================================================================

def build_insert_data(
    num_rows: int,
    schema: List[Tuple[str, str, Any]],
) -> List[Dict[str, Any]]:
    """
    Generate insert data as a list of dictionaries.
    
    Args:
        num_rows: Number of rows to generate
        schema: List of (field_name, field_type, type_param) tuples
                e.g. [("id", "INT64", None), ("embedding", "FLOAT_VECTOR", 128)]
    
    Returns:
        List of dictionaries suitable for MilvusClient.insert()
    """
    data = []
    for i in range(num_rows):
        row = {}
        for field_name, field_type, param in schema:
            row[field_name] = _generate_field_value(i, field_type, param)
        data.append(row)
    return data


def _generate_field_value(idx: int, field_type: str, param: Any) -> Any:
    """Generate a single field value based on type."""
    if field_type == "INT64":
        return idx
    elif field_type == "INT32":
        return idx % (2**31)
    elif field_type == "FLOAT":
        return float(idx) * 0.1
    elif field_type == "DOUBLE":
        return float(idx) * 0.01
    elif field_type == "BOOL":
        return idx % 2 == 0
    elif field_type == "VARCHAR":
        max_len = param or 100
        return f"text_{idx}_" + "x" * min(max_len - 10, 50)
    elif field_type == "JSON":
        return {"id": idx, "name": f"item_{idx}", "tags": [f"t{j}" for j in range(3)]}
    elif field_type == "FLOAT_VECTOR":
        dim = param or 128
        return [0.1 + (idx % 10) * 0.01] * dim
    elif field_type == "BINARY_VECTOR":
        dim = param or 128
        return bytes([idx % 256] * (dim // 8))
    else:
        raise ValueError(f"Unsupported field type: {field_type}")
