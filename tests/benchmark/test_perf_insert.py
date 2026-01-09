"""
Performance Benchmark: Insert Operations

This module provides pytest-benchmark tests for insert data generation
across all field types and batch sizes.

Usage:
    pytest tests/benchmark/test_perf_insert.py -v
    pytest tests/benchmark/test_perf_insert.py --benchmark-json=insert_results.json
"""

import pytest

from pymilvus.client.types import DataType

from .kernels.insert_ops import (
    generate_insert_data,
    generate_insert_data_columnar,
    get_kitchen_sink_fields,
    run_insert_data_generation_benchmark,
)
from .kernels.data_gen import (
    get_vector_field,
    get_varchar_field,
)


# =============================================================================
# Test Parameter Definitions
# =============================================================================

BATCH_SIZES = [100, 1000, 10000]
VECTOR_DIMS = [128, 768, 1536]


# =============================================================================
# Insert Tests - Kitchen Sink Schema
# =============================================================================

class TestInsertKitchenSink:
    """Test insert data generation with comprehensive schema."""
    
    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    @pytest.mark.parametrize("dim", VECTOR_DIMS)
    def test_insert_kitchen_sink_row_format(self, benchmark, batch_size, dim):
        """Generate insert data in row format (list of dicts)."""
        fields = get_kitchen_sink_fields(dim)
        
        def run():
            return generate_insert_data(batch_size, fields)
        
        result = benchmark(run)
        assert len(result) == batch_size
    
    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    @pytest.mark.parametrize("dim", VECTOR_DIMS)
    def test_insert_kitchen_sink_columnar_format(self, benchmark, batch_size, dim):
        """Generate insert data in columnar format (dict of lists)."""
        fields = get_kitchen_sink_fields(dim)
        
        def run():
            return generate_insert_data_columnar(batch_size, fields)
        
        result = benchmark(run)
        assert len(result[fields[0]["name"]]) == batch_size


# =============================================================================
# Insert Tests - Vector Only
# =============================================================================

class TestInsertVectorOnly:
    """Test insert data generation for vector-only scenarios."""
    
    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    @pytest.mark.parametrize("dim", VECTOR_DIMS)
    def test_insert_float_vector_only(self, benchmark, batch_size, dim):
        """Generate insert data with only FLOAT_VECTOR."""
        fields = [
            {"name": "id", "dtype": DataType.INT64},
            {"name": "vector", "dtype": DataType.FLOAT_VECTOR, "dim": dim},
        ]
        
        def run():
            return generate_insert_data(batch_size, fields)
        
        result = benchmark(run)
        assert len(result) == batch_size
    
    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    def test_insert_binary_vector_only(self, benchmark, batch_size):
        """Generate insert data with only BINARY_VECTOR."""
        dim = 1024  # bits
        fields = [
            {"name": "id", "dtype": DataType.INT64},
            {"name": "vector", "dtype": DataType.BINARY_VECTOR, "dim": dim},
        ]
        
        def run():
            return generate_insert_data(batch_size, fields)
        
        result = benchmark(run)
        assert len(result) == batch_size


# =============================================================================
# Insert Tests - Scalar Impact
# =============================================================================

class TestInsertScalarImpact:
    """Test how scalar fields impact insert performance."""
    
    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    def test_insert_with_large_varchar(self, benchmark, batch_size):
        """Insert with 8KB VARCHAR strings."""
        fields = [
            {"name": "id", "dtype": DataType.INT64},
            {"name": "content", "dtype": DataType.VARCHAR, "length": 8192},
            {"name": "vector", "dtype": DataType.FLOAT_VECTOR, "dim": 128},
        ]
        
        def run():
            return generate_insert_data(batch_size, fields)
        
        result = benchmark(run)
        assert len(result) == batch_size
    
    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    def test_insert_with_complex_json(self, benchmark, batch_size):
        """Insert with complex JSON objects."""
        fields = [
            {"name": "id", "dtype": DataType.INT64},
            {"name": "metadata", "dtype": DataType.JSON, "complexity": "complex"},
            {"name": "vector", "dtype": DataType.FLOAT_VECTOR, "dim": 128},
        ]
        
        def run():
            return generate_insert_data(batch_size, fields)
        
        result = benchmark(run)
        assert len(result) == batch_size


# =============================================================================
# Insert Tests - Format Comparison
# =============================================================================

class TestInsertFormatComparison:
    """Compare row vs columnar insert data generation."""
    
    def test_row_vs_columnar_small_batch(self, benchmark):
        """Small batch: row vs columnar."""
        batch_size = 100
        fields = get_kitchen_sink_fields(128)
        
        # Test row format in this test
        benchmark(generate_insert_data, batch_size, fields)
    
    def test_row_vs_columnar_large_batch(self, benchmark):
        """Large batch comparison."""
        batch_size = 10000
        fields = get_kitchen_sink_fields(128)
        
        benchmark(generate_insert_data, batch_size, fields)
