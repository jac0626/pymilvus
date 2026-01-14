# Copyright (C) 2019-2021 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.

import warnings

from pymilvus.client.types import (
    DataType,
    infer_dtype_by_scalar_data,
    infer_dtype_bydata,
    map_numpy_dtype_to_datatype,
)

# Constants were not strictly exported in __all__ but were available
# Need to check if anything else was consistently used.
# The original file had dtype_str_map, numpy_dtype_str_map, etc.
# but they were not in __all__ (which was not defined).
# We will just expose what seems public.

warnings.warn(
    "Importing from pymilvus.orm.types is deprecated. Please import from pymilvus.client.types instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "DataType",
    "infer_dtype_by_scalar_data",
    "infer_dtype_bydata",
    "map_numpy_dtype_to_datatype",
]
