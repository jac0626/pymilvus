"""
Search Operations Module for Benchmark Kernels

Pure functions implementing the 4 access modes for SearchResult and ColumnarSearchResult.
This module is pure Python with no pytest dependencies.
"""

from typing import Any, List, Type, Union

from pymilvus.client.columnar_search_result import ColumnarSearchResult
from pymilvus.client.search_result import SearchResult
from pymilvus.grpc_gen import schema_pb2


# Type alias for result types
ResultType = Union[SearchResult, ColumnarSearchResult]


# =============================================================================
# Result Construction
# =============================================================================

def create_search_result(data: schema_pb2.SearchResultData) -> SearchResult:
    """Wrap raw protobuf data in SearchResult (Legacy)."""
    return SearchResult(data)


def create_columnar_result(data: schema_pb2.SearchResultData) -> ColumnarSearchResult:
    """Wrap raw protobuf data in ColumnarSearchResult."""
    return ColumnarSearchResult(data)


# =============================================================================
# Access Mode 1: Random Point Access
# =============================================================================

def access_random_point(result: ResultType, query_idx: int = 0, hit_idx: int = 0) -> Any:
    """
    Access a single specific item: res[query_idx][hit_idx].
    
    Returns the Hit/RowProxy object.
    """
    return result[query_idx][hit_idx]


def access_random_point_field(
    result: ResultType, field_name: str, query_idx: int = 0, hit_idx: int = 0
) -> Any:
    """
    Access a specific field from a specific item: res[query_idx][hit_idx][field_name].
    
    Returns the field value.
    """
    return result[query_idx][hit_idx][field_name]


def run_random_access_benchmark(
    result: ResultType,
    field_name: str,
    num_accesses: int = 1000,
) -> int:
    """
    Benchmark random point access by accessing random items.
    
    Args:
        result: Search result object.
        field_name: Field to access.
        num_accesses: Number of random accesses to perform.
    
    Returns:
        Number of successful accesses.
    """
    import random
    nq = len(result)
    count = 0
    for _ in range(num_accesses):
        q_idx = random.randint(0, nq - 1)
        if len(result[q_idx]) > 0:
            h_idx = random.randint(0, len(result[q_idx]) - 1)
            _ = result[q_idx][h_idx][field_name]
            count += 1
    return count


# =============================================================================
# Access Mode 2: Columnar Batch Access (ColumnarSearchResult Only)
# =============================================================================

def access_columnar_batch(result: ColumnarSearchResult, field_name: str, query_idx: int = 0) -> Any:
    """
    Access an entire column for a query: res[query_idx].get_column(field_name).
    
    This is ONLY supported by ColumnarSearchResult.
    
    Returns the column data (list, bytes, or numpy array depending on type).
    """
    return result[query_idx].get_column(field_name)


def run_columnar_access_benchmark(
    result: ColumnarSearchResult,
    field_name: str,
) -> int:
    """
    Benchmark columnar batch access by iterating through all queries.
    
    Args:
        result: ColumnarSearchResult object.
        field_name: Field to access.
    
    Returns:
        Total number of elements accessed.
    """
    total = 0
    for hits in result:
        col = hits.get_column(field_name)
        if col is not None:
            total += len(col) if hasattr(col, "__len__") else 1
    return total


# =============================================================================
# Access Mode 3: Full Iteration
# =============================================================================

def run_full_iteration_benchmark(
    result: ResultType,
    field_name: str,
) -> int:
    """
    Benchmark full iteration: for hits in res: for hit in hits: hit[field_name].
    
    Args:
        result: Search result object.
        field_name: Field to access.
    
    Returns:
        Total number of hits accessed.
    """
    count = 0
    for hits in result:
        for hit in hits:
            _ = hit[field_name]
            count += 1
    return count


def run_full_iteration_all_fields_benchmark(
    result: ResultType,
    field_names: List[str],
) -> int:
    """
    Benchmark full iteration accessing multiple fields.
    
    Args:
        result: Search result object.
        field_names: List of fields to access.
    
    Returns:
        Total number of field accesses.
    """
    count = 0
    for hits in result:
        for hit in hits:
            for field_name in field_names:
                _ = hit[field_name]
                count += 1
    return count


# =============================================================================
# Access Mode 4: Range Slicing
# =============================================================================

def access_slice(
    result: ResultType,
    query_idx: int = 0,
    start: int = 0,
    end: int = 10,
) -> List[Any]:
    """
    Access a slice of results: res[query_idx][start:end].
    
    Returns a list of Hit/RowProxy objects.
    """
    return result[query_idx][start:end]


def run_slice_access_benchmark(
    result: ResultType,
    field_name: str,
    slice_size: int = 10,
) -> int:
    """
    Benchmark slicing by slicing each query result and accessing fields.
    
    Args:
        result: Search result object.
        field_name: Field to access from each sliced item.
        slice_size: Size of each slice.
    
    Returns:
        Total number of items accessed.
    """
    count = 0
    for hits in result:
        if len(hits) >= slice_size:
            sliced = hits[0:slice_size]
            for hit in sliced:
                _ = hit[field_name]
                count += 1
    return count


# =============================================================================
# Comparison Helpers
# =============================================================================

def compare_legacy_vs_columnar(
    data: schema_pb2.SearchResultData,
    field_name: str,
    mode: str = "iteration",
) -> dict:
    """
    Run both Legacy (SearchResult) and Columnar implementations and return timings.
    
    Args:
        data: Raw protobuf data.
        field_name: Field to access.
        mode: One of "random", "iteration", "slice".
    
    Returns:
        Dict with 'legacy_count', 'columnar_count' (for verification).
    """
    legacy_result = create_search_result(data)
    columnar_result = create_columnar_result(data)
    
    results = {}
    
    if mode == "random":
        results["legacy_count"] = run_random_access_benchmark(legacy_result, field_name, 1000)
        results["columnar_count"] = run_random_access_benchmark(columnar_result, field_name, 1000)
    elif mode == "iteration":
        results["legacy_count"] = run_full_iteration_benchmark(legacy_result, field_name)
        results["columnar_count"] = run_full_iteration_benchmark(columnar_result, field_name)
    elif mode == "slice":
        results["legacy_count"] = run_slice_access_benchmark(legacy_result, field_name, 10)
        results["columnar_count"] = run_slice_access_benchmark(columnar_result, field_name, 10)
    elif mode == "columnar":
        # Columnar batch access is only for ColumnarSearchResult
        results["columnar_count"] = run_columnar_access_benchmark(columnar_result, field_name)
        results["legacy_count"] = None  # Not supported
    
    return results
