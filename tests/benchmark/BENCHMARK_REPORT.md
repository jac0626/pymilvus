# Benchmark Results Report
**Date:** 2026-01-10 19:22
**Tests Run:** 59 | **Passed:** 56 | **Failed:** 3

## Summary

| Category | Status | Key Findings |
|----------|--------|--------------|
| Access Patterns | ✅ All Pass | Columnar **10-100x faster** than Legacy |
| Insert Data Gen | ✅ All Pass | 100K rows in ~787ms |
| Scalar Batch | ✅ All Pass | JSON ~20ms for 10K items |
| Vector Batch | ⚠️ 3 Failed | `get_column` API not fully implemented |

---

## Key Performance Comparisons (Columnar vs Legacy)

### Random Access `res[i][j]`
| Scale (NQ×TopK) | Columnar | Legacy | **Speedup** |
|-----------------|----------|--------|-------------|
| 10×1000 | ~50 μs | ~78 ms | **1560x** |
| 100×1000 | ~80 μs | ~1.0 s | **12500x** |
| 1000×1000 | ~115 μs | ~1.4 s | **12000x** |

### Slice Access `res[0][0:100]`
| Scale | Columnar | Legacy | **Speedup** |
|-------|----------|--------|-------------|
| 10×10000 | ~100 μs | ~1.3 s | **13000x** |
| 100×1000 | ~170 μs | ~470 ms | **2800x** |

### Extended Scale (16384)
| Scenario | Columnar | Legacy | **Speedup** |
|----------|----------|--------|-------------|
| 1×16384 | 9.6 ms | 59 ms | **6x** |
| 128×128 | 10 ms | 67 ms | **7x** |
| 16384×1 | 75 ms | 260 ms | **3.5x** |

---

## Insert Data Generation

| Scenario | Records | Fields | Time | Throughput |
|----------|---------|--------|------|------------|
| Basic | 1,000 | 6 | 4.6 ms | 217K rec/s |
| Complex | 10,000 | 6 | 42 ms | 238K rec/s |
| Large | 50,000 | 6 | 331 ms | 151K rec/s |
| XLarge | 100,000 | 6 | 787 ms | 127K rec/s |

---

## Files Generated

- **JSON Data:** `/root/pymilvus/.benchmarks/results/benchmark_focused_20260110_191954.json`
- **Console Output:** `/root/pymilvus/.benchmarks/results/benchmark_focused.txt`

## Failed Tests (Expected)

The 3 failed tests are `test_vector_access_batch[]` - the `get_column()` API is not yet fully implemented for all vector types.
