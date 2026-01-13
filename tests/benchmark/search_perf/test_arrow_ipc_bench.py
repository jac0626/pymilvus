#!/usr/bin/env python3
"""
Arrow IPC vs Protobuf Benchmark (E.7.4 Reproduction)

Measures the performance difference between standard Protobuf repeated float
and transmitting Arrow IPC binary data encapsulated in Protobuf bytes.
"""

import pytest
import numpy as np
import pyarrow as pa
import time
from pymilvus.grpc_gen import schema_pb2

# Configuration matching the report E.7.4
NQ = 1
TOPK = 10000
DIM = 768
TOTAL_VECTORS = NQ * TOPK

@pytest.fixture(scope="module")
def vector_data():
    """Generate mock vector data."""
    # 10,000 vectors of dim 768 float32
    return np.random.random((TOTAL_VECTORS, DIM)).astype(np.float32)

@pytest.fixture(scope="module")
def proto_standard(vector_data):
    """Standard Protobuf with repeated float."""
    f_vec = schema_pb2.FieldData(
        type=schema_pb2.FloatVector,
        field_name="embedding",
        vectors=schema_pb2.VectorField(
            dim=DIM,
            float_vector=schema_pb2.FloatArray(data=vector_data.flatten().tolist())
        )
    )
    result = schema_pb2.SearchResultData(
        num_queries=NQ,
        top_k=TOPK,
        fields_data=[f_vec],
        ids=schema_pb2.IDs(int_id=schema_pb2.LongArray(data=list(range(TOTAL_VECTORS)))),
        scores=[0.0] * TOTAL_VECTORS,
        topks=[TOPK]
    )
    return result

@pytest.fixture(scope="module")
def arrow_ipc_bytes(vector_data):
    """Arrow IPC serialized data."""
    # Convert numpy to Arrow Array
    # Flatten for simplicity if simulation, or keep structured.
    # To mimic report "Arrow IPC encapsulated in Protobuf bytes" with Zero Copy
    # We use FixedSizeList or similar.
    
    # 1. Create Arrow Table/RecordBatch
    # Use PyArrow to create a Tensor or just flat array
    # Report says "Arrow IPC data encapsulated in Protobuf bytes field"
    
    # Efficient way: Array of FixedSizeList
    tensor_type = pa.list_(pa.float32(), DIM)
    # This is slow in pyarrow conversion from numpy usually, optimizing:
    # fastest is pa.Tensor but IPC support varies.
    # Let's use simple RecordBatch with 1 column of FixedSizeList
    
    # Optimization for conversion:
    # pyarrow.FixedSizeListArray.from_numpy_ndarray is available in newer versions?
    # Fallback to simple list for benchmark prep (once)
    
    # Actually, for 10k vectors, conversion time is fine for setup fixture.
    # Using simple flat array for IPC to minimize structure overhead
    flat_data = vector_data.flatten()
    pa_array = pa.array(flat_data)
    batch = pa.RecordBatch.from_arrays([pa_array], names=['embedding'])
    
    # Serialize to IPC Stream
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, batch.schema) as writer:
        writer.write_batch(batch)
    
    return sink.getvalue()

@pytest.fixture(scope="module")
def proto_arrow(arrow_ipc_bytes):
    """Protobuf with Arrow IPC bytes in a binary field."""
    # We simulate passing this as a BINARY_VECTOR or just Raw Bytes field
    # In reality, Milvus might use a new field type or reuse BinaryVector
    f_vec = schema_pb2.FieldData(
        type=schema_pb2.Double, # Dummy type, using bytes actually
        field_name="embedding",
        scalars=schema_pb2.ScalarField(
            # Storing the whole IPC blob in one scalar string/bytes for simulation?
            # Or split per row? Report implies "Batch transmission".
            # Likely storing the whole blob in a sidecar field or hijacked field.
            # Let's put it in a single bytes field for the batch.
            string_data=schema_pb2.StringArray(data=[arrow_ipc_bytes.to_pybytes()]) 
        )
    )
     # Note: real implementation might be different, but this measures 
     # Protobuf overhead for carrying the blob vs parsing it.
    
    result = schema_pb2.SearchResultData(
        num_queries=NQ,
        top_k=TOPK,
        fields_data=[f_vec],
        ids=schema_pb2.IDs(int_id=schema_pb2.LongArray(data=list(range(TOTAL_VECTORS)))),
         scores=[0.0] * TOTAL_VECTORS,
        topks=[TOPK]
    )
    return result

# --- Benchmarks ---

def test_proto_standard_serialize(benchmark, proto_standard):
    """Baseline: Serialize Standard Protobuf (Repeated Float)"""
    def run():
        return proto_standard.SerializeToString()
    benchmark(run)

def test_proto_standard_deserialize(benchmark, proto_standard):
    """Baseline: Deserialize Standard Protobuf (Repeated Float)"""
    data = proto_standard.SerializeToString()
    def run():
        pb = schema_pb2.SearchResultData()
        pb.ParseFromString(data)
        return pb
    benchmark(run)

def test_proto_arrow_serialize(benchmark, proto_arrow):
    """Arrow IPC: Serialize Protobuf wrapper"""
    def run():
        return proto_arrow.SerializeToString()
    benchmark(run)

def test_proto_arrow_deserialize_roundtrip(benchmark, proto_arrow):
    """
    Arrow IPC: Deserialize Protobuf wrapper + Arrow IPC Open
    
    This measures the time to get from "Bytes on wire" to "Ready to read array".
    """
    data = proto_arrow.SerializeToString()
    
    def run():
        # 1. Parse Protobuf Wrapper
        pb = schema_pb2.SearchResultData()
        pb.ParseFromString(data)
        
        # 2. Extract Arrow IPC Bytes
        # (Simulating retrieval of the blob)
        arrow_bytes = pb.fields_data[0].scalars.string_data.data[0]
        
        # 3. Open Arrow IPC
        # Zero-copy reader
        reader = pa.ipc.open_stream(arrow_bytes)
        batch = reader.read_next_batch()
        return batch
        
    benchmark(run)
