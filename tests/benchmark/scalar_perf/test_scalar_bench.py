#!/usr/bin/env python3
"""
Scalar Field Performance Benchmark

Tests the performance of scalar field access in SearchResult vs ColumnarSearchResult.
Covers: INT, VARCHAR, JSON, ARRAY types with varying data sizes.
"""

import pytest
from pymilvus.client.columnar_search_result import ColumnarSearchResult
from pymilvus.client.search_result import SearchResult
from pymilvus.grpc_gen import schema_pb2
import json

# =============================================================================
# Configuration - Dual-Layer Test Strategy
# =============================================================================

# ============= Layer 1: User-Centric Scenarios =============
# Real-world typical queries (not exhaustive matrix)
# Based on actual user patterns: TopK usually 10-100, NQ usually 1-100
TYPICAL_QUERIES = [
    (1, 10),      # Single query, top 10 results (most common)
    (10, 100),    # Batch 10 queries, top 100 each (common batch)
    (100, 100),   # Large batch (stress test)
]

# Real-world field scenarios
REAL_WORLD_SCENARIOS = [
    # Scenario 1: E-commerce product search
    ("product_name", "VARCHAR", "SMALL"),        # Product names ~50 chars
    ("description", "VARCHAR", "MEDIUM"),        # Descriptions ~500 chars
    ("metadata", "JSON", "MEDIUM"),              # {brand, category, tags}
    
    # Scenario 2: User profile search
    ("username", "VARCHAR", "SMALL"),            # Usernames ~20 chars
    ("bio", "VARCHAR", "MEDIUM"),                # User bios ~300 chars
    ("profile", "JSON", "COMPLEX"),              # Full user profile
    
    # Scenario 3: Document search
    ("title", "VARCHAR", "SMALL"),               # Document titles
    ("content_snippet", "VARCHAR", "LARGE"),     # Content preview ~2000 chars
    ("doc_metadata", "JSON", "SIMPLE"),          # {author, date, tags}
    
    # Scenario 4: Simple analytics
    ("count", "INT64", None),                    # Numeric aggregations
    
    # Scenario 5: Advanced types
    ("tag_ids", "ARRAY", None),                  # Array of integers
]

# ============= Layer 2: Full Parameter Coverage =============
# Comprehensive matrix for scientific analysis
NQ_VALUES = [10, 100, 1000]
TOPK_VALUES = [10, 100, 1000]

# All scalar types with complexity variants
FULL_MATRIX_TYPES = [
    # Simple types (no complexity)
    ("INT64", "int64_field", None),
    ("ARRAY", "array_field", None),
    
    # VARCHAR with all sizes
    ("VARCHAR", "varchar_field", "SMALL"),
    ("VARCHAR", "varchar_field", "MEDIUM"),
    ("VARCHAR", "varchar_field", "LARGE"),
    
    # JSON with all complexities
    ("JSON", "json_field", "SIMPLE"),
    ("JSON", "json_field", "MEDIUM"),
    ("JSON", "json_field", "COMPLEX"),
]

# =============================================================================
# Mock Data Builders
# =============================================================================

def build_scalar_result(nq: int, topk: int, field_name: str, field_type: str, complexity: str = None) -> schema_pb2.SearchResultData:
    """Build mock SearchResultData matching real-world scenarios."""
    total = nq * topk
    res = schema_pb2.SearchResultData()
    res.num_queries = nq
    res.top_k = topk
    res.topks.extend([topk] * nq)
    res.ids.int_id.data.extend(list(range(total)))
    res.scores.extend([float(i) * 0.001 for i in range(total)])
    res.primary_field_name = "id"
    
    # Add scalar field based on type
    if field_type == "INT64":
        field = res.fields_data.add()
        field.field_name = field_name
        field.type = schema_pb2.DataType.Int64
        field.scalars.long_data.data.extend([i * 100 for i in range(total)])
        res.output_fields.append(field_name)
        
    elif field_type == "VARCHAR":
        field = res.fields_data.add()
        field.field_name = field_name
        field.type = schema_pb2.DataType.VarChar
        
        if complexity == "SMALL":
            # Realistic: product names, usernames, titles (~20-50 chars)
            templates = [
                "Premium {category} Item {id}",
                "User_{id}@example.com",
                "Document Title {id}"
            ]
            for i in range(total):
                field.scalars.string_data.data.append(templates[i % 3].format(category="Electronics", id=i))
                
        elif complexity == "MEDIUM":
            # Realistic: descriptions, bios (~300-500 chars)
            template = "This is a detailed description for item {id}. " + \
                       "Features include high quality materials, excellent craftsmanship, and modern design. " + \
                       "Perfect for daily use and special occasions. " + \
                       "Available in multiple colors and sizes. " + \
                       "Trusted by thousands of customers worldwide."
            for i in range(total):
                field.scalars.string_data.data.append(template.format(id=i))
                
        elif complexity == "LARGE":
            # Realistic: article content, long reviews (~1500-2500 chars)
            template = ("Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 25) + f"_item_{{}}"
            for i in range(total):
                field.scalars.string_data.data.append(template.format(i))
        
        res.output_fields.append(field_name)
        
    elif field_type == "JSON":
        field = res.fields_data.add()
        field.field_name = field_name
        field.type = schema_pb2.DataType.JSON
        
        for i in range(total):
            if complexity == "SIMPLE":
                # Realistic: simple metadata
                json_obj = {
                    "author": f"user_{i}",
                    "created_at": "2024-01-01",
                    "category": "electronics"
                }
            elif complexity == "MEDIUM":
                # Realistic: E-commerce product metadata
                json_obj = {
                    "brand": f"Brand_{i % 10}",
                    "category": "electronics",
                    "tags": ["popular", "new", "sale"],
                    "price": 99.99,
                    "stock": 100
                }
            elif complexity == "COMPLEX":
                # Realistic: Full user profile or product details
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
        res.output_fields.append(field_name)
    
    elif field_type == "ARRAY":
        field = res.fields_data.add()
        field.field_name = field_name
        field.type = schema_pb2.DataType.Array
        # CRITICAL: Must set element_type for Legacy SearchResult compatibility
        field.scalars.array_data.element_type = schema_pb2.DataType.Int64
        
        for i in range(total):
            array_data = field.scalars.array_data.data.add()
            # Realistic: tag IDs, category IDs, etc. (5 items)
            array_data.long_data.data.extend([100 + i + j for j in range(5)])
        
        res.output_fields.append(field_name)
    
    return res
    """Build mock SearchResultData with specified scalar field type."""
    total = nq * topk
    res = schema_pb2.SearchResultData()
    res.num_queries = nq
    res.top_k = topk
    res.topks.extend([topk] * nq)
    res.ids.int_id.data.extend(list(range(total)))
    res.scores.extend([float(i) * 0.001 for i in range(total)])
    res.primary_field_name = "id"
    
    # Add scalar field based on type
    if field_type == "INT64":
        field = res.fields_data.add()
        field.field_name = "int64_field"
        field.type = schema_pb2.DataType.Int64
        field.scalars.long_data.data.extend([i * 100 for i in range(total)])
        res.output_fields.append("int64_field")
        
    elif field_type == "VARCHAR":
        field = res.fields_data.add()
        field.field_name = "varchar_field"
        field.type = schema_pb2.DataType.VarChar
        
        if complexity == "SMALL":
            for i in range(total):
                field.scalars.string_data.data.append(f"user_{i}@example.com")
        elif complexity == "MEDIUM":
            medium_text = "Medium text content. " * 10  # ~200 chars
            for i in range(total):
                field.scalars.string_data.data.append(f"{medium_text}_{i}")
        elif complexity == "LARGE":
            large_text = "Lorem ipsum dolor sit amet. " * 100  # ~2800 chars
            for i in range(total):
                field.scalars.string_data.data.append(f"{large_text}_{i}")
        
        res.output_fields.append("varchar_field")
        
    elif field_type == "JSON":
        field = res.fields_data.add()
        field.field_name = "json_field"
        field.type = schema_pb2.DataType.JSON
        
        for i in range(total):
            if complexity == "SIMPLE":
                json_obj = {"id": i, "name": f"user_{i}"}
            elif complexity == "MEDIUM":
                json_obj = {
                    "id": i,
                    "name": f"user_{i}",
                    "tags": [f"tag_{j}" for j in range(3)]
                }
            elif complexity == "COMPLEX":
                json_obj = {
                    "user": {"id": i, "name": f"user_{i}", "email": f"user_{i}@example.com"},
                    "metadata": {"tags": [f"tag_{j}" for j in range(5)], "scores": [j * 0.1 for j in range(10)]},
                    "timestamp": 1234567890 + i
                }
            
            field.scalars.json_data.data.append(json.dumps(json_obj).encode())
        res.output_fields.append("json_field")
    
    return res
    """Build mock SearchResultData with specified scalar field type."""
    total = nq * topk
    res = schema_pb2.SearchResultData()
    res.num_queries = nq
    res.top_k = topk
    res.topks.extend([topk] * nq)
    res.ids.int_id.data.extend(list(range(total)))
    res.scores.extend([float(i) * 0.001 for i in range(total)])
    res.primary_field_name = "id"
    
    # Add scalar field based on type
    if scalar_type == "INT64":
        field = res.fields_data.add()
        field.field_name = "int64_field"
        field.type = schema_pb2.DataType.Int64
        field.scalars.long_data.data.extend([i * 100 for i in range(total)])
        res.output_fields.append("int64_field")
        
    elif scalar_type == "VARCHAR_SMALL":
        field = res.fields_data.add()
        field.field_name = "varchar_field"
        field.type = schema_pb2.DataType.VarChar
        for i in range(total):
            field.scalars.string_data.data.append(f"user_{i}@example.com")
        res.output_fields.append("varchar_field")
        
    elif scalar_type == "VARCHAR_LARGE":
        field = res.fields_data.add()
        field.field_name = "varchar_field"
        field.type = schema_pb2.DataType.VarChar
        large_text = "Lorem ipsum dolor sit amet. " * 50  # ~1400 chars
        for i in range(total):
            field.scalars.string_data.data.append(f"{large_text}_{i}")
        res.output_fields.append("varchar_field")
        
    elif scalar_type == "VARCHAR_MIXED":
        field = res.fields_data.add()
        field.field_name = "varchar_field"
        field.type = schema_pb2.DataType.VarChar
        # Mix of small (20%), medium (50%), large (30%)
        for i in range(total):
            if i % 10 < 2:  # 20% small
                text = f"user_{i}@example.com"
            elif i % 10 < 7:  # 50% medium
                text = "Medium text. " * 10 + f"_{i}"  # ~140 chars
            else:  # 30% large
                text = "Lorem ipsum dolor sit amet. " * 50 + f"_{i}"  # ~1400 chars
            field.scalars.string_data.data.append(text)
        res.output_fields.append("varchar_field")
        
    elif scalar_type == "JSON_SIMPLE":
        field = res.fields_data.add()
        field.field_name = "json_field"
        field.type = schema_pb2.DataType.JSON
        for i in range(total):
            json_obj = {"id": i, "name": f"user_{i}", "score": i * 0.1}
            field.scalars.json_data.data.append(json.dumps(json_obj).encode())
        res.output_fields.append("json_field")
        
    elif scalar_type == "JSON_COMPLEX":
        field = res.fields_data.add()
        field.field_name = "json_field"
        field.type = schema_pb2.DataType.JSON
        for i in range(total):
            json_obj = {
                "user": {"id": i, "name": f"user_{i}", "email": f"user_{i}@example.com"},
                "metadata": {"tags": [f"tag_{j}" for j in range(5)], "scores": [j * 0.1 for j in range(10)]},
                "timestamp": 1234567890 + i
            }
            field.scalars.json_data.data.append(json.dumps(json_obj).encode())
        res.output_fields.append("json_field")
        
    elif scalar_type == "JSON_MIXED":
        field = res.fields_data.add()
        field.field_name = "json_field"
        field.type = schema_pb2.DataType.JSON
        # Mix of simple (40%), medium (40%), complex (20%)
        for i in range(total):
            if i % 10 < 4:  # 40% simple
                json_obj = {"id": i, "name": f"user_{i}"}
            elif i % 10 < 8:  # 40% medium
                json_obj = {"id": i, "name": f"user_{i}", "tags": [f"tag_{j}" for j in range(3)]}
            else:  # 20% complex
                json_obj = {
                    "user": {"id": i, "name": f"user_{i}", "email": f"user_{i}@example.com"},
                    "metadata": {"tags": [f"tag_{j}" for j in range(5)], "scores": [j * 0.1 for j in range(10)]},
                }
            field.scalars.json_data.data.append(json.dumps(json_obj).encode())
        res.output_fields.append("json_field")
        
        field = res.fields_data.add()
        field.field_name = "array_field"
        field.type = schema_pb2.DataType.Array
        for i in range(total):
            array_data = field.scalars.array_data.data.add()
            array_data.long_data.data.extend([i + j for j in range(10)])
        res.output_fields.append("array_field")
    
    return res

# =============================================================================
# Helper Functions
# =============================================================================

def iterate_scalar_result(results, field_name):
    """Iterate and access the specified scalar field."""
    count = 0
    for hits in results:
        for hit in hits:
            _ = hit[field_name]
            count += 1
    return count

# =============================================================================
# Benchmark Tests
# =============================================================================


# =============================================================================
# Helper Functions
# =============================================================================

def iterate_scalar_result(results, field_name):
    """Iterate and access the specified scalar field."""
    count = 0
    for hits in results:
        for hit in hits:
            _ = hit[field_name]
            count += 1
    return count

# =============================================================================
# Layer 1: Real-World User Scenario Tests
# =============================================================================

@pytest.mark.parametrize("nq, topk", TYPICAL_QUERIES)
@pytest.mark.parametrize("field_name, field_type, complexity", REAL_WORLD_SCENARIOS)
def test_real_world_columnar(benchmark, nq, topk, field_name, field_type, complexity):
    """Benchmark real-world scalar field scenarios - Columnar."""
    res_data = build_scalar_result(nq, topk, field_name, field_type, complexity)
    
    def run_columnar():
        cr = ColumnarSearchResult(res_data)
        iterate_scalar_result(cr, field_name)
    
    benchmark(run_columnar)


@pytest.mark.parametrize("nq, topk", TYPICAL_QUERIES)
@pytest.mark.parametrize("field_name, field_type, complexity", REAL_WORLD_SCENARIOS)
def test_real_world_legacy(benchmark, nq, topk, field_name, field_type, complexity):
    """Benchmark real-world scalar field scenarios - Legacy."""
    res_data = build_scalar_result(nq, topk, field_name, field_type, complexity)
    
    def run_legacy():
        sr = SearchResult(res_data)
        iterate_scalar_result(sr, field_name)
    
    benchmark(run_legacy)


# =============================================================================
# Layer 2: Full Parameter Coverage Tests
# =============================================================================

@pytest.mark.parametrize("nq", NQ_VALUES)
@pytest.mark.parametrize("topk", TOPK_VALUES)
@pytest.mark.parametrize("field_type, field_name, complexity", FULL_MATRIX_TYPES)
def test_full_matrix_columnar(benchmark, nq, topk, field_type, field_name, complexity):
    """Full parameter coverage for Columnar - All types × All NQ × All TopK."""
    res_data = build_scalar_result(nq, topk, field_name, field_type, complexity)
    
    def run_columnar():
        cr = ColumnarSearchResult(res_data)
        iterate_scalar_result(cr, field_name)
    
    benchmark(run_columnar)


@pytest.mark.parametrize("nq", NQ_VALUES)
@pytest.mark.parametrize("topk", TOPK_VALUES)
@pytest.mark.parametrize("field_type, field_name, complexity", FULL_MATRIX_TYPES)
def test_full_matrix_legacy(benchmark, nq, topk, field_type, field_name, complexity):
    """Full parameter coverage for Legacy - All types × All NQ × All TopK."""
    res_data = build_scalar_result(nq, topk, field_name, field_type, complexity)
    
    def run_legacy():
        sr = SearchResult(res_data)
        iterate_scalar_result(sr, field_name)
    
    benchmark(run_legacy)
