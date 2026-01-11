#!/usr/bin/env python3
"""
Insert Performance Memory Profiling

For use with memray:
    memray run -o insert_mem.bin tests/benchmark/insert_perf/profile_insert_memory.py
    memray summary insert_mem.bin
    memray flamegraph insert_mem.bin
"""

import sys
from pymilvus import MilvusClient, DataType
from pymilvus.client.prepare import Prepare

# Add parent to path for kernel imports
sys.path.insert(0, str(__file__.rsplit('/', 3)[0]))
from tests.benchmark.kernels import build_insert_data


# =============================================================================
# Schema Definition
# =============================================================================

INSERT_SCHEMA = [
    ("id", "INT64", None),
    ("embedding", "FLOAT_VECTOR", 128),
    ("name", "VARCHAR", 100),
    ("age", "INT32", None),
    ("json_field", "JSON", None),
    ("varchar_field", "VARCHAR", 100),
]


def get_fields_info():
    """Create fields_info list matching the schema for Prepare API."""
    return [
        {"name": "id", "type": DataType.INT64, "is_primary": True, "auto_id": False},
        {"name": "embedding", "type": DataType.FLOAT_VECTOR, "params": {"dim": 128}},
        {"name": "name", "type": DataType.VARCHAR, "params": {"max_length": 100}},
        {"name": "age", "type": DataType.INT32},
        {"name": "json_field", "type": DataType.JSON},
        {"name": "varchar_field", "type": DataType.VARCHAR, "params": {"max_length": 100}},
    ]


def run_insert_workload(data, schema):
    """Run the insert preparation workload."""
    return Prepare.row_insert_param(
        collection_name="test_collection",
        entities=data,
        partition_name=None,
        fields_info=get_fields_info()
    )


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "complex"
    num_rows = 1000 if mode == "basic" else 10000
    
    print(f"=== Insert Memory Profiling ({mode}) ===")
    print(f"Rows: {num_rows}")
    
    # Generate data
    print("Generating test data...")
    data = build_insert_data(num_rows, INSERT_SCHEMA)
    schema = None
    
    # Run workload (memray will track memory)
    print("Running insert preparation...")
    result = run_insert_workload(data, schema)
    
    print("✅ Memory profiling complete!")
    print("Run 'memray summary <output.bin>' to view results")
