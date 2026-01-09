"""
Search Operations Module for Benchmark Kernels

Cold-start benchmark functions that include object initialization time.
This module is pure Python with no pytest dependencies.
"""

import random
from typing import Any, List

from pymilvus.client.columnar_search_result import ColumnarSearchResult
from pymilvus.client.search_result import SearchResult
from pymilvus.grpc_gen import schema_pb2


# =============================================================================
# Internal Access Functions (used by cold-start benchmarks)
# =============================================================================

def _run_iteration(result, field_name: str) -> int:
    """Iterate all hits and access a field."""
    count = 0
    for hits in result:
        for hit in hits:
            _ = hit[field_name]
            count += 1
    return count


def _run_random_access(result, field_name: str, num_accesses: int) -> int:
    """Randomly access items."""
    nq = len(result)
    count = 0
    for _ in range(num_accesses):
        q_idx = random.randint(0, nq - 1)
        if len(result[q_idx]) > 0:
            h_idx = random.randint(0, len(result[q_idx]) - 1)
            _ = result[q_idx][h_idx][field_name]
            count += 1
    return count


def _run_slice_access(result, field_name: str, slice_size: int) -> int:
    """Access slices of results."""
    count = 0
    for hits in result:
        if len(hits) >= slice_size:
            sliced = hits[0:slice_size]
            for hit in sliced:
                _ = hit[field_name]
                count += 1
    return count


def _run_columnar_access(result: ColumnarSearchResult, field_name: str) -> int:
    """Batch columnar access using get_column."""
    total = 0
    for hits in result:
        col = hits.get_column(field_name)
        if col is not None:
            total += len(col) if hasattr(col, "__len__") else 1
    return total


# =============================================================================
# Cold-Start Benchmarks: Legacy (SearchResult)
# =============================================================================

def benchmark_iteration_legacy(
    data: schema_pb2.SearchResultData,
    field_name: str,
) -> int:
    """Cold-start: Create SearchResult + iterate all fields."""
    result = SearchResult(data)
    return _run_iteration(result, field_name)


def benchmark_random_legacy(
    data: schema_pb2.SearchResultData,
    field_name: str,
    num_accesses: int = 1000,
) -> int:
    """Cold-start: Create SearchResult + random access."""
    result = SearchResult(data)
    return _run_random_access(result, field_name, num_accesses)


def benchmark_slice_legacy(
    data: schema_pb2.SearchResultData,
    field_name: str,
    slice_size: int = 100,
) -> int:
    """Cold-start: Create SearchResult + slice access."""
    result = SearchResult(data)
    return _run_slice_access(result, field_name, slice_size)


# =============================================================================
# Cold-Start Benchmarks: Columnar (ColumnarSearchResult)
# =============================================================================

def benchmark_iteration_columnar(
    data: schema_pb2.SearchResultData,
    field_name: str,
) -> int:
    """Cold-start: Create ColumnarSearchResult + iterate all fields."""
    result = ColumnarSearchResult(data)
    return _run_iteration(result, field_name)


def benchmark_random_columnar(
    data: schema_pb2.SearchResultData,
    field_name: str,
    num_accesses: int = 1000,
) -> int:
    """Cold-start: Create ColumnarSearchResult + random access."""
    result = ColumnarSearchResult(data)
    return _run_random_access(result, field_name, num_accesses)


def benchmark_slice_columnar(
    data: schema_pb2.SearchResultData,
    field_name: str,
    slice_size: int = 100,
) -> int:
    """Cold-start: Create ColumnarSearchResult + slice access."""
    result = ColumnarSearchResult(data)
    return _run_slice_access(result, field_name, slice_size)


def benchmark_columnar_batch(
    data: schema_pb2.SearchResultData,
    field_name: str,
) -> int:
    """Cold-start: Create ColumnarSearchResult + columnar batch access."""
    result = ColumnarSearchResult(data)
    return _run_columnar_access(result, field_name)
