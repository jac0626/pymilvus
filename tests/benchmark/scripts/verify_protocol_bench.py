
import sys
import time
import struct
import random
from pymilvus.grpc_gen import schema_pb2

def benchmark_protocol():
    print("=== Protocol Performance Verification (Repeated vs Bytes) ===")
    
    NQ = 10
    Dim = 768
    # Total floats: 10 * 768 = 7680
    # Ideally we want a larger payload to see the difference clearly, e.g. 10k vectors
    # User report used 30MB data which is ~10k vectors * 768 * 4 bytes
    
    NUM_VECTORS = 10000
    DIM = 768
    TOTAL_FLOATS = NUM_VECTORS * DIM
    PAYLOAD_SIZE = TOTAL_FLOATS * 4  # ~30 MB
    
    print(f"Data Size: {NUM_VECTORS} vectors (dim={DIM})")
    print(f"Payload: {PAYLOAD_SIZE / 1024 / 1024:.2f} MB")
    
    # 1. Setup Repeated Float (Legacy)
    print("\n[Setup] Populating Repeated Float Field...")
    t0 = time.time()
    fd_legacy = schema_pb2.FieldData()
    # Create a large list of floats
    floats = [random.random() for _ in range(TOTAL_FLOATS)]
    fd_legacy.vectors.float_vector.data.extend(floats)
    print(f"Setup Legacy took: {time.time() - t0:.4f}s")
    
    # 2. Setup Bytes (Arrow/Native)
    print("[Setup] Populating Bytes Field...")
    t0 = time.time()
    fd_bytes = schema_pb2.FieldData()
    # Create bytes equivalent
    # pack 'f' is float (4 bytes). We can just create random bytes
    byte_data = b'x' * PAYLOAD_SIZE
    fd_bytes.vectors.float16_vector = byte_data # Abuse float16_vector field which is 'bytes' type
    print(f"Setup Bytes took: {time.time() - t0:.4f}s")
    
    # 3. Benchmark Access (Legacy)
    print("\n[Run] Accessing Repeated Float (Materialize to list)...")
    start = time.perf_counter()
    # Simulate what SearchResult does: iterating the data
    # or converting to list
    _ = list(fd_legacy.vectors.float_vector.data)
    end = time.perf_counter()
    legacy_time = (end - start) * 1000
    print(f"Legacy Access Time: {legacy_time:.4f} ms")
    
    # 4. Benchmark Access (Bytes)
    print("[Run] Accessing Bytes...")
    start = time.perf_counter()
    # Accessing bytes field triggers C++ to Python bytes copy (but single object)
    _ = fd_bytes.vectors.float16_vector
    end = time.perf_counter()
    bytes_time = (end - start) * 1000
    print(f"Bytes Access Time:  {bytes_time:.4f} ms")
    
    # Summary
    print(f"\nSpeedup: {legacy_time / bytes_time:.1f}x")

if __name__ == "__main__":
    benchmark_protocol()
