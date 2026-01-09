"""
Performance Benchmark: Search Operations

This module provides comprehensive pytest-benchmark tests for SearchResult
and ColumnarSearchResult across all data types and access patterns.

Usage:
    pytest tests/benchmark/test_perf_search.py -v
    pytest tests/benchmark/test_perf_search.py --benchmark-json=results.json
"""

import pytest
from typing import Any, Dict, List

from pymilvus.client.types import DataType
from pymilvus.client.search_result import SearchResult
from pymilvus.client.columnar_search_result import ColumnarSearchResult

from .kernels.data_gen import (
    create_search_result_data,
    get_vector_field,
    get_varchar_field,
    get_json_field,
    get_array_field,
    get_embedding_list_field,
    SCALAR_FIELDS_CORE,
    DYNAMIC_FIELD,
)
from .kernels.search_ops import (
    benchmark_iteration_legacy,
    benchmark_iteration_columnar,
    benchmark_random_legacy,
    benchmark_random_columnar,
    benchmark_slice_legacy,
    benchmark_slice_columnar,
    benchmark_columnar_batch,
)


# =============================================================================
# Test Parameter Definitions
# =============================================================================

# NQ and TopK sweep values
NQ_VALUES = [1, 10, 100, 1000, 10000]
TOPK_VALUES = [1, 10, 100, 1000, 10000]

# Reduced set for matrix tests (to avoid combinatorial explosion)
NQ_REDUCED = [10, 100, 1000]
TOPK_REDUCED = [100, 1000]

# Vector types and dimensions
VECTOR_TYPES = [
    DataType.FLOAT_VECTOR,
    DataType.BINARY_VECTOR,
    DataType.FLOAT16_VECTOR,
    DataType.BFLOAT16_VECTOR,
    DataType.INT8_VECTOR,
]
VECTOR_DIMS = [128, 768, 1536]

# Scalar types
SCALAR_TYPES = [
    DataType.BOOL,
    DataType.INT8,
    DataType.INT16,
    DataType.INT32,
    DataType.INT64,
    DataType.FLOAT,
    DataType.DOUBLE,
]

# VARCHAR lengths
VARCHAR_LENGTHS = [32, 256, 1024, 8192, "mixed"]

# JSON complexities
JSON_COMPLEXITIES = ["simple", "medium", "complex"]


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="function")
def vector_data(request):
    """Generate vector-only SearchResultData."""
    nq = request.param.get("nq", 10)
    topk = request.param.get("topk", 100)
    vtype = request.param.get("vtype", DataType.FLOAT_VECTOR)
    dim = request.param.get("dim", 128)
    
    field = get_vector_field(vtype, dim)
    return create_search_result_data(nq, topk, [field])


@pytest.fixture(scope="function")
def scalar_data(request):
    """Generate scalar-only SearchResultData."""
    nq = request.param.get("nq", 10)
    topk = request.param.get("topk", 100)
    dtype = request.param.get("dtype", DataType.INT64)
    length = request.param.get("length", 256)
    complexity = request.param.get("complexity", "simple")
    
    if dtype == DataType.VARCHAR:
        field = get_varchar_field(length)
    elif dtype == DataType.JSON:
        field = get_json_field(complexity)
    elif dtype == DataType.ARRAY:
        field = get_array_field(DataType.INT64)
    else:
        field = {"name": f"scalar_{dtype.name.lower()}", "dtype": dtype}
    
    return create_search_result_data(nq, topk, [field]), field["name"]


# =============================================================================
# Vector Tests - Scaling (NQ x TopK)
# =============================================================================

class TestVectorScaling:
    """Test vector field access with varying NQ and TopK (includes init time)."""
    
    @pytest.mark.parametrize("nq", NQ_VALUES)
    @pytest.mark.parametrize("topk", TOPK_VALUES)
    def test_float_vector_iteration_legacy(self, benchmark, nq, topk):
        """Legacy SearchResult: Full iteration on FLOAT_VECTOR (cold start)."""
        if nq * topk > 10_000_000:
            pytest.skip("Skipping very large dataset")
        
        data = create_search_result_data(nq, topk, [get_vector_field(DataType.FLOAT_VECTOR, 128)])
        field_name = "vector_float_vector_128"
        
        benchmark(benchmark_iteration_legacy, data, field_name)
    
    @pytest.mark.parametrize("nq", NQ_VALUES)
    @pytest.mark.parametrize("topk", TOPK_VALUES)
    def test_float_vector_iteration_columnar(self, benchmark, nq, topk):
        """ColumnarSearchResult: Full iteration on FLOAT_VECTOR (cold start)."""
        if nq * topk > 10_000_000:
            pytest.skip("Skipping very large dataset")
        
        data = create_search_result_data(nq, topk, [get_vector_field(DataType.FLOAT_VECTOR, 128)])
        field_name = "vector_float_vector_128"
        
        benchmark(benchmark_iteration_columnar, data, field_name)


# =============================================================================
# Vector Tests - Type Comparison
# =============================================================================

class TestVectorTypes:
    """Test all vector types with fixed NQ/TopK (includes init time)."""
    
    @pytest.mark.parametrize("vtype", VECTOR_TYPES)
    @pytest.mark.parametrize("dim", VECTOR_DIMS)
    def test_vector_iteration_legacy(self, benchmark, vtype, dim):
        """Legacy SearchResult: Full iteration across vector types (cold start)."""
        nq, topk = 10, 1000
        field = get_vector_field(vtype, dim)
        data = create_search_result_data(nq, topk, [field])
        
        benchmark(benchmark_iteration_legacy, data, field["name"])
    
    @pytest.mark.parametrize("vtype", VECTOR_TYPES)
    @pytest.mark.parametrize("dim", VECTOR_DIMS)
    def test_vector_iteration_columnar(self, benchmark, vtype, dim):
        """ColumnarSearchResult: Full iteration across vector types (cold start)."""
        nq, topk = 10, 1000
        field = get_vector_field(vtype, dim)
        data = create_search_result_data(nq, topk, [field])
        
        benchmark(benchmark_iteration_columnar, data, field["name"])
    
    @pytest.mark.parametrize("vtype", VECTOR_TYPES)
    @pytest.mark.parametrize("dim", VECTOR_DIMS)
    def test_vector_columnar_batch(self, benchmark, vtype, dim):
        """ColumnarSearchResult only: Columnar batch access (cold start)."""
        nq, topk = 10, 1000
        field = get_vector_field(vtype, dim)
        data = create_search_result_data(nq, topk, [field])
        
        benchmark(benchmark_columnar_batch, data, field["name"])


# =============================================================================
# Scalar Tests - Type Comparison
# =============================================================================

class TestScalarTypes:
    """Test all scalar types with fixed NQ/TopK (includes init time)."""
    
    @pytest.mark.parametrize("dtype", SCALAR_TYPES)
    def test_scalar_iteration_legacy(self, benchmark, dtype):
        """Legacy SearchResult: Full iteration across scalar types (cold start)."""
        nq, topk = 10, 1000
        field = {"name": f"scalar_{dtype.name.lower()}", "dtype": dtype}
        data = create_search_result_data(nq, topk, [field])
        
        benchmark(benchmark_iteration_legacy, data, field["name"])
    
    @pytest.mark.parametrize("dtype", SCALAR_TYPES)
    def test_scalar_iteration_columnar(self, benchmark, dtype):
        """ColumnarSearchResult: Full iteration across scalar types (cold start)."""
        nq, topk = 10, 1000
        field = {"name": f"scalar_{dtype.name.lower()}", "dtype": dtype}
        data = create_search_result_data(nq, topk, [field])
        
        benchmark(benchmark_iteration_columnar, data, field["name"])
    
    @pytest.mark.parametrize("dtype", SCALAR_TYPES)
    def test_scalar_columnar_batch(self, benchmark, dtype):
        """ColumnarSearchResult only: Columnar batch access (cold start)."""
        nq, topk = 10, 1000
        field = {"name": f"scalar_{dtype.name.lower()}", "dtype": dtype}
        data = create_search_result_data(nq, topk, [field])
        
        benchmark(benchmark_columnar_batch, data, field["name"])


# =============================================================================
# VARCHAR Tests - Length Variation
# =============================================================================

class TestVarcharLength:
    """Test VARCHAR with varying string lengths (includes init time)."""
    
    @pytest.mark.parametrize("length", VARCHAR_LENGTHS)
    def test_varchar_iteration_legacy(self, benchmark, length):
        """Legacy SearchResult: Iteration with varying VARCHAR lengths (cold start)."""
        nq, topk = 10, 1000
        field = get_varchar_field(length)
        data = create_search_result_data(nq, topk, [field])
        
        benchmark(benchmark_iteration_legacy, data, field["name"])
    
    @pytest.mark.parametrize("length", VARCHAR_LENGTHS)
    def test_varchar_iteration_columnar(self, benchmark, length):
        """ColumnarSearchResult: Iteration with varying VARCHAR lengths (cold start)."""
        nq, topk = 10, 1000
        field = get_varchar_field(length)
        data = create_search_result_data(nq, topk, [field])
        
        benchmark(benchmark_iteration_columnar, data, field["name"])


# =============================================================================
# JSON Tests - Complexity Variation
# =============================================================================

class TestJsonComplexity:
    """Test JSON with varying complexity (includes init time)."""
    
    @pytest.mark.parametrize("complexity", JSON_COMPLEXITIES)
    def test_json_iteration_legacy(self, benchmark, complexity):
        """Legacy SearchResult: Iteration with varying JSON complexity (cold start)."""
        nq, topk = 10, 1000
        field = get_json_field(complexity)
        data = create_search_result_data(nq, topk, [field])
        
        benchmark(benchmark_iteration_legacy, data, field["name"])
    
    @pytest.mark.parametrize("complexity", JSON_COMPLEXITIES)
    def test_json_iteration_columnar(self, benchmark, complexity):
        """ColumnarSearchResult: Iteration with varying JSON complexity (cold start)."""
        nq, topk = 10, 1000
        field = get_json_field(complexity)
        data = create_search_result_data(nq, topk, [field])
        
        benchmark(benchmark_iteration_columnar, data, field["name"])


# =============================================================================
# Access Mode Tests
# =============================================================================

class TestAccessModes:
    """Compare all 4 access modes (includes init time)."""
    
    def _create_test_data(self, nq=10, topk=1000):
        """Helper to create consistent test data."""
        field = get_vector_field(DataType.FLOAT_VECTOR, 128)
        return create_search_result_data(nq, topk, [field]), field["name"]
    
    # Mode 1: Random Point Access
    def test_random_access_legacy(self, benchmark):
        """Legacy: Random point access (cold start)."""
        data, field_name = self._create_test_data()
        benchmark(benchmark_random_legacy, data, field_name, 1000)
    
    def test_random_access_columnar(self, benchmark):
        """Columnar: Random point access (cold start)."""
        data, field_name = self._create_test_data()
        benchmark(benchmark_random_columnar, data, field_name, 1000)
    
    # Mode 2: Columnar Batch (Columnar Only)
    def test_columnar_batch_access(self, benchmark):
        """Columnar only: Batch column access (cold start)."""
        data, field_name = self._create_test_data()
        benchmark(benchmark_columnar_batch, data, field_name)
    
    # Mode 3: Full Iteration
    def test_full_iteration_legacy(self, benchmark):
        """Legacy: Full iteration (cold start)."""
        data, field_name = self._create_test_data()
        benchmark(benchmark_iteration_legacy, data, field_name)
    
    def test_full_iteration_columnar(self, benchmark):
        """Columnar: Full iteration (cold start)."""
        data, field_name = self._create_test_data()
        benchmark(benchmark_iteration_columnar, data, field_name)
    
    # Mode 4: Slice Access
    def test_slice_access_legacy(self, benchmark):
        """Legacy: Slice access (cold start)."""
        data, field_name = self._create_test_data()
        benchmark(benchmark_slice_legacy, data, field_name, 100)
    
    def test_slice_access_columnar(self, benchmark):
        """Columnar: Slice access (cold start)."""
        data, field_name = self._create_test_data()
        benchmark(benchmark_slice_columnar, data, field_name, 100)


# =============================================================================
# Dynamic Field Tests
# =============================================================================

class TestDynamicField:
    """Test dynamic field access (includes init time)."""
    
    def test_dynamic_field_iteration_legacy(self, benchmark):
        """Legacy: Dynamic field access via expanded keys (cold start)."""
        nq, topk = 10, 1000
        data = create_search_result_data(nq, topk, [DYNAMIC_FIELD])
        # Legacy expands dynamic fields into entity, access 'id' key from JSON
        benchmark(benchmark_iteration_legacy, data, "id")
    
    def test_dynamic_field_iteration_columnar(self, benchmark):
        """Columnar: Dynamic field access via $meta (cold start)."""
        nq, topk = 10, 1000
        data = create_search_result_data(nq, topk, [DYNAMIC_FIELD])
        # Columnar can access $meta directly or expanded fields
        benchmark(benchmark_iteration_columnar, data, "id")


# =============================================================================
# Advanced Types Tests
# =============================================================================

class TestAdvancedTypes:
    """Test advanced types: ARRAY, Embedding List (includes init time)."""
    
    def test_array_iteration_legacy(self, benchmark):
        """Legacy: ARRAY field iteration (cold start)."""
        nq, topk = 10, 1000
        field = get_array_field(DataType.INT64)
        data = create_search_result_data(nq, topk, [field])
        benchmark(benchmark_iteration_legacy, data, field["name"])
    
    def test_array_iteration_columnar(self, benchmark):
        """Columnar: ARRAY field iteration (cold start)."""
        nq, topk = 10, 1000
        field = get_array_field(DataType.INT64)
        data = create_search_result_data(nq, topk, [field])
        benchmark(benchmark_iteration_columnar, data, field["name"])
    
    def test_embedding_list_iteration_columnar(self, benchmark):
        """Columnar: Embedding List (Array of Vector) iteration (cold start)."""
        nq, topk = 10, 100  # Smaller due to complexity
        field = get_embedding_list_field(dim=128, num_vecs=3)
        data = create_search_result_data(nq, topk, [field])
        benchmark(benchmark_iteration_columnar, data, field["name"])
