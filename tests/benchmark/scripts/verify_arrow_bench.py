
import time
import numpy as np
import pyarrow as pa
from pymilvus.grpc_gen import schema_pb2
import struct

def benchmark_arrow_vs_proto():
    print("=== Arrow IPC vs Protobuf Performance Verification ===")
    
    NQ = 10000
    DIM = 768
    print(f"Data: {NQ} vectors, Dim={DIM} (Float32)")
    
    # 1. Generate Numpy Data
    print("Generating data...")
    # Create random float32 array
    data_np = np.random.rand(NQ, DIM).astype(np.float32)
    # Flatten for Protobuf
    data_flat = data_np.flatten().tolist()
    
    total_mb = data_np.nbytes / 1024 / 1024
    print(f"Payload Size: {total_mb:.2f} MB")
    
    # --- Protobuf Repeated ---
    print("\n[Protobuf] Serialize (Population)...")
    t0 = time.perf_counter()
    fd = schema_pb2.FieldData()
    fd.vectors.float_vector.data.extend(data_flat)
    pb_ser_time = (time.perf_counter() - t0) * 1000
    print(f"Protobuf Serialize: {pb_ser_time:.2f} ms")
    
    # Serialize to bytes (network simulation)
    pb_bytes = fd.SerializeToString()
    
    print("[Protobuf] Deserialize (Parse + List)...")
    t0 = time.perf_counter()
    fd_new = schema_pb2.FieldData()
    fd_new.ParseFromString(pb_bytes)
    # Simulate access: converting to list or numpy
    _ = list(fd_new.vectors.float_vector.data)
    pb_deser_time = (time.perf_counter() - t0) * 1000
    print(f"Protobuf Deserialize: {pb_deser_time:.2f} ms")
    
    # --- Arrow IPC (Embedded in Protobuf) ---
    print("\n[Arrow IPC over Protobuf] Serialize...")
    t0 = time.perf_counter()
    
    # 1. Create Arrow Bytes
    flat_arrow = pa.array(data_np.flatten())
    batch = pa.RecordBatch.from_arrays([flat_arrow], names=['vectors'])
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, batch.schema) as writer:
        writer.write_batch(batch)
    arrow_bytes = sink.getvalue().to_pybytes() # Convert to python bytes
    arrow_gen_time = (time.perf_counter() - t0) * 1000
    
    # 2. Wrap in Protobuf
    fd_arrow = schema_pb2.FieldData()
    fd_arrow.vectors.float16_vector = arrow_bytes # Store in bytes field
    
    # 3. Serialize Protobuf (Network Simulation)
    pb_arrow_bytes = fd_arrow.SerializeToString()
    
    arrow_over_proto_ser_time = (time.perf_counter() - t0) * 1000
    print(f"Arrow+Proto Serialize: {arrow_over_proto_ser_time:.2f} ms")
    
    print("[Arrow IPC over Protobuf] Deserialize...")
    t0 = time.perf_counter()
    
    # 1. Deserialize Protobuf
    fd_arrow_new = schema_pb2.FieldData()
    fd_arrow_new.ParseFromString(pb_arrow_bytes)
    
    # 2. Extract Bytes (Copy C++ -> Python)
    ipc_bytes_out = fd_arrow_new.vectors.float16_vector
    
    # 3. Arrow Deserialize (Zero Copy from bytes)
    reader = pa.ipc.open_stream(ipc_bytes_out)
    batch_read = reader.read_next_batch()
    np_out = batch_read.column(0).to_numpy()
    
    arrow_over_proto_deser_time = (time.perf_counter() - t0) * 1000
    print(f"Arrow+Proto Deserialize: {arrow_over_proto_deser_time:.2f} ms")
    
    # --- Summary ---
    print("\n=== Results ===")
    print(f"Standard Protobuf (Repeated) Roundtrip: {pb_ser_time + pb_deser_time:.2f} ms")
    print(f"Arrow IPC inside Protobuf Roundtrip:    {arrow_over_proto_ser_time + arrow_over_proto_deser_time:.2f} ms")
    
    speedup = (pb_ser_time + pb_deser_time) / (arrow_over_proto_ser_time + arrow_over_proto_deser_time)
    print(f"Speedup: {speedup:.1f}x")

if __name__ == "__main__":
    benchmark_arrow_vs_proto()
