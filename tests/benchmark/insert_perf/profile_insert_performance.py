#!/usr/bin/env python3
"""
Insert Performance CPU Profiling

Profiles the insert data preparation path to identify bottlenecks.
Based on the performance report findings:
- pack_field_value_to_field_data: 88.9%
- convert_to_str_array: 40.6%
- Protobuf extend: 20.5%
- isinstance checks: 10.0%
"""

import sys
from pymilvus import MilvusClient, DataType
from pymilvus.client.prepare import Prepare

# Add parent to path for kernel imports
sys.path.insert(0, str(__file__.rsplit('/', 3)[0]))
from tests.benchmark.kernels import build_insert_data, profile_cpu, get_output_dir


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


# =============================================================================
# Workloads
# =============================================================================

def run_insert_workload(data, schema):
    """Run the insert preparation workload."""
    return Prepare.row_insert_param(
        collection_name="test_collection",
        entities=data,
        partition_name=None,
        fields_info=get_fields_info()
    )


# =============================================================================
# Main Profiling
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Profile Insert Performance")
    parser.add_argument("--rows", type=int, default=10000, help="Number of rows")
    parser.add_argument("--scenario", choices=["basic", "complex"], default="complex")
    args = parser.parse_args()
    
    num_rows = 1000 if args.scenario == "basic" else args.rows
    
    print(f"=== Insert Performance Profiling ===")
    print(f"Scenario: {args.scenario}")
    print(f"Rows: {num_rows}")
    print(f"Fields: {len(INSERT_SCHEMA)}")
    print()
    
    # Generate data
    print("Generating test data...")
    data = build_insert_data(num_rows, INSERT_SCHEMA)
    schema = None
    print(f"Generated {len(data)} rows")
    print()
    
    # Profile
    result, stats, elapsed = profile_cpu(
        f"insert_{args.scenario}",
        run_insert_workload,
        data,
        schema,
        top_n=20
    )
    
    # Summary
    print(f"\n=== Summary ===")
    print(f"Total time: {elapsed * 1000:.2f} ms")
    print(f"Throughput: {num_rows / elapsed:.0f} records/sec")
    print(f"Output dir: {get_output_dir()}")


if __name__ == "__main__":
    main()
