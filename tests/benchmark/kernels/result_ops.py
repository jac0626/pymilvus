#!/usr/bin/env python3
# Copyright (c) PyMilvus Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""
Result iteration and access helpers for benchmarking.

Provides unified functions for iterating search results and accessing fields.
"""

from typing import List, Tuple, Optional, Any


def iterate_result(results, field_name: Optional[str] = None) -> int:
    """
    Iterate through all hits in a search result, optionally accessing a field.
    
    Args:
        results: SearchResult or ColumnarSearchResult
        field_name: If provided, access this field for each hit
    
    Returns:
        Total number of hits iterated
    """
    count = 0
    for hits in results:
        for hit in hits:
            if field_name:
                _ = hit[field_name]
            count += 1
    return count


def iterate_all_fields(results, field_names: List[str]) -> int:
    """
    Iterate through all hits and access multiple fields.
    
    Args:
        results: SearchResult or ColumnarSearchResult
        field_names: List of field names to access for each hit
    
    Returns:
        Total number of hits iterated
    """
    count = 0
    for hits in results:
        for hit in hits:
            for field_name in field_names:
                _ = hit[field_name]
            count += 1
    return count


def random_access(
    results,
    indices: List[Tuple[int, int]],
    field_name: Optional[str] = None
) -> List[Any]:
    """
    Perform random point access on search results (e.g., res[i][j]).
    
    Args:
        results: SearchResult or ColumnarSearchResult
        indices: List of (query_idx, hit_idx) tuples to access
        field_name: If provided, return the field value; otherwise return the hit
    
    Returns:
        List of accessed values
    
    Example:
        >>> values = random_access(results, [(0, 0), (0, 5), (1, 3)], "vector")
    """
    accessed = []
    for query_idx, hit_idx in indices:
        hit = results[query_idx][hit_idx]
        if field_name:
            accessed.append(hit[field_name])
        else:
            accessed.append(hit)
    return accessed


def slice_access(
    results,
    query_idx: int,
    start: int,
    end: int,
    field_name: Optional[str] = None
) -> List[Any]:
    """
    Access a slice of hits from a query result (e.g., res[0][0:10]).
    
    Args:
        results: SearchResult or ColumnarSearchResult
        query_idx: Which query to access
        start: Start index of slice (inclusive)
        end: End index of slice (exclusive)
        field_name: If provided, return field values; otherwise return hits
    
    Returns:
        List of accessed hits or field values
    
    Example:
        >>> first_10 = slice_access(results, 0, 0, 10, "vector")
    """
    hits = results[query_idx]
    sliced = hits[start:end]
    
    if field_name:
        return [hit[field_name] for hit in sliced]
    return list(sliced)


def get_column_batch(results, field_name: str) -> List[List[Any]]:
    """
    Get column data for all queries using batch access API.
    
    This uses the ColumnarSearchResult.get_column() API for optimal performance.
    Falls back to iteration for non-columnar results.
    
    Args:
        results: ColumnarSearchResult (ideally)
        field_name: Field name to get column for
    
    Returns:
        List of columns, one per query
    """
    columns = []
    for hits in results:
        if hasattr(hits, 'get_column'):
            # Columnar fast path
            columns.append(hits.get_column(field_name))
        else:
            # Legacy fallback
            columns.append([hit[field_name] for hit in hits])
    return columns
