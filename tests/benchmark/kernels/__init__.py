# Copyright (c) PyMilvus Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""
Shared kernel modules for PyMilvus benchmarking.

- data_gen: Mock data generation utilities
- result_ops: Result iteration and access helpers  
- profiling: CPU/Memory profiling utilities
"""

from .data_gen import (
    build_search_result,
    build_search_result_base,
    add_scalar_field,
    add_vector_field,
    build_insert_data,
    ScalarFieldConfig,
    VectorFieldConfig,
    ScalarComplexity,
)
from .result_ops import (
    iterate_result,
    iterate_all_fields,
    random_access,
    slice_access,
    get_column_batch,
)
from .profiling import (
    get_output_dir,
    profile_cpu,
    run_with_timing,
    compare_implementations,
)

__all__ = [
    # data_gen
    "build_search_result",
    "build_search_result_base",
    "add_scalar_field", 
    "add_vector_field",
    "build_insert_data",
    "ScalarFieldConfig",
    "VectorFieldConfig",
    "ScalarComplexity",
    # result_ops
    "iterate_result",
    "iterate_all_fields",
    "random_access",
    "slice_access",
    "get_column_batch",
    # profiling
    "get_output_dir",
    "profile_cpu",
    "run_with_timing",
    "compare_implementations",
]
