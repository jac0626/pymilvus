#!/usr/bin/env python3
"""Generate flame graph data for Phase 1+2 profiling."""

import cProfile
import pstats
import os
os.chdir('/Users/zilliz/pymilvus')

from pymilvus.grpc_gen import schema_pb2
from pymilvus.client.types import DataType
from pymilvus.client.search_result import SearchResult
from pymilvus.client.columnar_search_result import ColumnarSearchResult

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

def profile_original():
    for _ in range(3):
        sr = SearchResult(result_data)
        for i, hits in enumerate(sr):
            if i >= 10:
                break
            for j, hit in enumerate(hits):
                if j >= 10:
                    break
                _ = hit.id
                _ = hit.distance
                _ = hit.entity.get('float_vector')

def profile_columnar():
    for _ in range(3):
        cr = ColumnarSearchResult(result_data, zero_copy_vectors=True)
        for i, hits in enumerate(cr):
            if i >= 10:
                break
            for j, hit in enumerate(hits):
                if j >= 10:
                    break
                _ = hit.id
                _ = hit.distance
                _ = hit['float_vector']

import sys
if len(sys.argv) > 1 and sys.argv[1] == 'columnar':
    cProfile.run('profile_columnar()', 'columnar.prof')
else:
    cProfile.run('profile_original()', 'original.prof')
