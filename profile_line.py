#!/usr/bin/env python3
"""Line-by-line profiling of ColumnarHits.__init__"""

import os
os.chdir('/Users/zilliz/pymilvus')

from line_profiler import LineProfiler
from pymilvus.grpc_gen import schema_pb2
from pymilvus.client.types import DataType
from pymilvus.client.columnar_search_result import ColumnarSearchResult, ColumnarHits

# Create test data
nq, topk, dim = 100, 1000, 128
total = nq * topk

result_data = schema_pb2.SearchResultData()
result_data.ids.int_id.data.extend(list(range(total)))
result_data.scores.extend([float(i) * 0.01 for i in range(total)])
result_data.topks.extend([topk] * nq)
result_data.num_queries = nq
result_data.output_fields.extend(['float_vector', 'int_field'])

vec_field = result_data.fields_data.add()
vec_field.field_name = 'float_vector'
vec_field.type = DataType.FLOAT_VECTOR
vec_field.vectors.dim = dim
vec_field.vectors.float_vector.data.extend([float(i % 256) for i in range(total * dim)])

int_field = result_data.fields_data.add()
int_field.field_name = 'int_field'
int_field.type = DataType.INT64
int_field.scalars.long_data.data.extend(list(range(total)))

print(f"Test data: nq={nq}, topk={topk}, total={total:,}")
print()

def run_columnar():
    for _ in range(5):
        cr = ColumnarSearchResult(result_data, zero_copy_vectors=True)

# Profile ColumnarHits.__init__
profiler = LineProfiler()
profiler.add_function(ColumnarHits.__init__)
profiler.add_function(ColumnarSearchResult.__init__)
profiler.runctx('run_columnar()', globals(), locals())

print("=" * 70)
print("LINE-BY-LINE PROFILE:")
print("=" * 70)
profiler.print_stats()
