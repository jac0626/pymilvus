#!/usr/bin/env python3
"""Profile Phase 1+2 Columnar Search Result implementation."""

import cProfile
import pstats
import io
from pstats import SortKey
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

# Add float vector field
vec_field = result_data.fields_data.add()
vec_field.field_name = 'float_vector'
vec_field.type = DataType.FLOAT_VECTOR
vec_field.vectors.dim = dim
vec_field.vectors.float_vector.data.extend([float(i % 256) for i in range(total * dim)])

# Add int field
int_field = result_data.fields_data.add()
int_field.field_name = 'int_field'
int_field.type = DataType.INT64
int_field.scalars.long_data.data.extend(list(range(total)))

print(f"Test data: nq={nq}, topk={topk}, dim={dim}, total={total:,}")
print()

def profile_original():
    """Profile Original SearchResult."""
    for _ in range(3):
        sr = SearchResult(result_data)
        # Access first 10 results from each query
        for i, hits in enumerate(sr):
            if i >= 10:
                break
            for j, hit in enumerate(hits):
                if j >= 10:
                    break
                _ = hit.id
                _ = hit.distance
                _ = hit.entity.get('float_vector')
                _ = hit.entity.get('int_field')

def profile_columnar():
    """Profile Columnar Phase 1+2."""
    for _ in range(3):
        cr = ColumnarSearchResult(result_data, zero_copy_vectors=True)
        # Access first 10 results from each query
        for i, hits in enumerate(cr):
            if i >= 10:
                break
            for j, hit in enumerate(hits):
                if j >= 10:
                    break
                _ = hit.id
                _ = hit.distance
                _ = hit['float_vector']
                _ = hit['int_field']

# Profile Original
print("=" * 60)
print("ORIGINAL SearchResult Profile:")
print("=" * 60)
pr = cProfile.Profile()
pr.enable()
profile_original()
pr.disable()
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats(SortKey.CUMULATIVE)
ps.print_stats(20)
print(s.getvalue())

# Profile Columnar
print("=" * 60)
print("COLUMNAR Phase 1+2 Profile:")
print("=" * 60)
pr = cProfile.Profile()
pr.enable()
profile_columnar()
pr.disable()
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats(SortKey.CUMULATIVE)
ps.print_stats(20)
print(s.getvalue())
