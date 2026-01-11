#!/usr/bin/env python3
"""
Insert Performance Benchmarks

Tests the performance of insert data preparation (client-side serialization).
This benchmarks the core data generation and conversion logic.
"""

import pytest
from pymilvus import MilvusClient, DataType

from tests.benchmark.kernels import build_insert_data


# =============================================================================
# Test Configuration
# =============================================================================

# Standard insert schema from the performance report
INSERT_SCHEMA = [
    ("id", "INT64", None),
    ("embedding", "FLOAT_VECTOR", 128),
    ("name", "VARCHAR", 100),
    ("age", "INT32", None),
    ("json_field", "JSON", None),
    ("varchar_field", "VARCHAR", 100),
]

# Test sizes from the report
TEST_SIZES = [
    (1000, "basic"),      # Basic scenario
    (10000, "complex"),   # Complex scenario
]

# Extended sizes for stress testing
STRESS_SIZES = [
    (50000, "large"),
    (100000, "xlarge"),
]


# =============================================================================
# Fields Info for Prepare API (matching pymilvus schema format)
# =============================================================================

def get_fields_info():
    """Create fields_info list matching the schema for Prepare API."""
    return [
        {"name": "id", "type": DataType.INT64, "is_primary": True, "auto_id": False},
        {"name": "embedding", "type": DataType.FLOAT_VECTOR, "params": {"dim": 128}},
        {"name": "name", "type": DataType.VARCHAR, "params": {"max_length": 100}},
        {"name": "age", "type": DataType.INT32},
        {"name": "json_field", "type": DataType.JSON},
        {"name": "varchar_field", "type": DataType.VARCHAR, "params": {"max_length": 100}},
    ]


@pytest.fixture(scope="module")
def fields_info():
    """Fixture for fields_info."""
    return get_fields_info()


# =============================================================================
# Data Generation Benchmarks
# =============================================================================

@pytest.mark.parametrize("num_rows, scenario", TEST_SIZES)
def test_data_generation(benchmark, num_rows, scenario):
    """
    Benchmark data generation for insert.
    
    This tests the build_insert_data() kernel function.
    """
    def run_generate():
        return build_insert_data(num_rows, INSERT_SCHEMA)
    
    data = benchmark(run_generate)
    assert len(data) == num_rows


# =============================================================================
# Columnar Conversion Benchmarks
# =============================================================================

def _convert_to_columnar(rows: list) -> dict:
    """Convert row-based data to columnar format."""
    if not rows:
        return {}
    
    columns = {key: [] for key in rows[0].keys()}
    for row in rows:
        for key, value in row.items():
            columns[key].append(value)
    return columns


@pytest.mark.parametrize("num_rows, scenario", TEST_SIZES)
def test_columnar_conversion(benchmark, num_rows, scenario):
    """
    Benchmark converting row data to columnar format.
    
    This measures the overhead of row-to-columnar transformation.
    """
    # Pre-generate row data
    row_data = build_insert_data(num_rows, INSERT_SCHEMA)
    
    def run_convert():
        return _convert_to_columnar(row_data)
    
    columnar = benchmark(run_convert)
    assert len(columnar["id"]) == num_rows





# =============================================================================
# Stress Tests
# =============================================================================

@pytest.mark.parametrize("num_rows, scenario", STRESS_SIZES)
@pytest.mark.slow
def test_data_generation_stress(benchmark, num_rows, scenario):
    """Stress test data generation with large volumes."""
    def run_generate():
        return build_insert_data(num_rows, INSERT_SCHEMA)
    
    data = benchmark(run_generate)
    assert len(data) == num_rows
