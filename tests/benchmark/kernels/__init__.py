"""
Benchmark Kernels

This module contains the core benchmark logic, decoupled from test runners.

- `data_gen.py`: Generates mock protobuf data (SearchResultData).
- `search_ops.py`: Cold-start benchmark functions for search access patterns.
- `insert_ops.py`: Pure functions for insert operations.
"""

from .data_gen import (
    create_search_result_data,
    create_kitchen_sink_result,
    create_vector_only_result,
    create_scalar_only_result,
    get_vector_field,
    get_varchar_field,
    get_json_field,
    get_array_field,
    get_embedding_list_field,
    SCALAR_FIELDS_CORE,
    DYNAMIC_FIELD,
)

from .search_ops import (
    benchmark_iteration_legacy,
    benchmark_iteration_columnar,
    benchmark_random_legacy,
    benchmark_random_columnar,
    benchmark_slice_legacy,
    benchmark_slice_columnar,
    benchmark_columnar_batch,
)

from .insert_ops import (
    generate_insert_data,
    generate_insert_data_columnar,
    get_kitchen_sink_fields,
    run_insert_data_generation_benchmark,
)

__all__ = [
    # Data generation
    "create_search_result_data",
    "create_kitchen_sink_result",
    "create_vector_only_result",
    "create_scalar_only_result",
    "get_vector_field",
    "get_varchar_field",
    "get_json_field",
    "get_array_field",
    "get_embedding_list_field",
    "SCALAR_FIELDS_CORE",
    "DYNAMIC_FIELD",
    # Search benchmarks (cold-start)
    "benchmark_iteration_legacy",
    "benchmark_iteration_columnar",
    "benchmark_random_legacy",
    "benchmark_random_columnar",
    "benchmark_slice_legacy",
    "benchmark_slice_columnar",
    "benchmark_columnar_batch",
    # Insert ops
    "generate_insert_data",
    "generate_insert_data_columnar",
    "get_kitchen_sink_fields",
    "run_insert_data_generation_benchmark",
]
