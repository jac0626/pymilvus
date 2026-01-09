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
    benchmark_insert_prepare,
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
    """Test insert data preparation (Dict to Protobuf) with comprehensive schema."""
    
    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    @pytest.mark.parametrize("dim", VECTOR_DIMS)
    def test_insert_prepare_kitchen_sink(self, benchmark, batch_size, dim):
        """Benchmark converting row data to InsertRequest (Protobuf)."""
        fields = get_kitchen_sink_fields(dim)
        data = generate_insert_data(batch_size, fields)
        
        benchmark(benchmark_insert_prepare, data, fields)


# =============================================================================
# Insert Tests - Vector Only
# =============================================================================

class TestInsertVectorOnly:
    """Test insert data preparation for vector-only scenarios."""
    
    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    @pytest.mark.parametrize("dim", VECTOR_DIMS)
    def test_insert_prepare_float_vector_only(self, benchmark, batch_size, dim):
        """Benchmark prepare insert: FLOAT_VECTOR."""
        fields = [
            {"name": "id", "dtype": DataType.INT64},
            {"name": "vector", "dtype": DataType.FLOAT_VECTOR, "dim": dim},
        ]
        data = generate_insert_data(batch_size, fields)
        benchmark(benchmark_insert_prepare, data, fields)
    
    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    def test_insert_prepare_binary_vector_only(self, benchmark, batch_size):
        """Benchmark prepare insert: BINARY_VECTOR."""
        dim = 1024  # bits
        fields = [
            {"name": "id", "dtype": DataType.INT64},
            {"name": "vector", "dtype": DataType.BINARY_VECTOR, "dim": dim},
        ]
        data = generate_insert_data(batch_size, fields)
        benchmark(benchmark_insert_prepare, data, fields)


# =============================================================================
# Insert Tests - Scalar Impact
# =============================================================================

class TestInsertScalarImpact:
    """Test how scalar fields impact insert preparation performance."""
    
    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    def test_insert_prepare_with_large_varchar(self, benchmark, batch_size):
        """Benchmark prepare insert: 8KB VARCHAR."""
        fields = [
            {"name": "id", "dtype": DataType.INT64},
            {"name": "content", "dtype": DataType.VARCHAR, "length": 8192},
            {"name": "vector", "dtype": DataType.FLOAT_VECTOR, "dim": 128},
        ]
        data = generate_insert_data(batch_size, fields)
        benchmark(benchmark_insert_prepare, data, fields)
    
    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    def test_insert_prepare_with_complex_json(self, benchmark, batch_size):
        """Benchmark prepare insert: Complex JSON."""
        fields = [
            {"name": "id", "dtype": DataType.INT64},
            {"name": "metadata", "dtype": DataType.JSON, "complexity": "complex"},
            {"name": "vector", "dtype": DataType.FLOAT_VECTOR, "dim": 128},
        ]
        data = generate_insert_data(batch_size, fields)
        benchmark(benchmark_insert_prepare, data, fields)


# =============================================================================
# Insert Tests - Format Comparison
# =============================================================================

class TestInsertFormatComparison:
    """Compare row vs columnar insert data preparation (TODO: Add columnar support)."""
    
    # Currently only row format is supported by benchmark_insert_prepare
    # because Prepare.row_insert_param works with list of dicts.
    pass
