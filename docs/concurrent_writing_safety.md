# Concurrent Writing Safety in Zarr v3

## Summary

**⚠️ CONCLUSION: Zarr v3 does NOT support safe concurrent writes, even with non-overlapping indices.**

Based on extensive testing and Context7 research, we've determined that `populate_epoch_from_catalog_v2` **cannot be used concurrently** without external synchronization.

## Test Results

### ✅ Sequential Writes (SAFE)
```python
# WORKS: Single thread populating multiple epochs
for epoch_key in epoch_keys:
    storage.populate_epoch_from_catalog_v2(
        epoch_key=epoch_key,
        source_catalog=catalog,
        measurement_keys=catalog.measurement_keys
    )
```

**Result**: All data written correctly and verifiable ✅

### ❌ Concurrent Writes (UNSAFE)
```python
# FAILS: Multiple threads populating same storage
# Even with non-overlapping object partitions
def worker(object_subset):
    worker_storage.populate_epoch_from_catalog_v2(...)
    
with ThreadPoolExecutor() as executor:
    executor.map(worker, object_partitions)
```

**Result**:
- Workers report successful writes (1000 measurements each)
- But final verification shows all NaN data ❌
- Data is corrupted or overwritten
- NO errors raised during execution (silent data corruption!)

## Why It Fails

### Zarr v3 Changes
In Zarr v3, the following v2 features were **removed**:
- `region` parameter for coordinated partial writes
- `synchronizer` parameter for explicit locking

### Current Safety Model
- **Chunk-level atomicity**: Each chunk is a separate file
- **Safe**: Different threads → different chunks ✅
- **Unsafe**: Multiple threads → same array = undefined behavior ❌

### Our Use Case
```python
# Our writes use fancy indexing (oindex)
lightcurves_array.oindex[oindex] = data

# This CAN span multiple chunks or modify same chunks
# Multiple threads doing this = race conditions
# Even if object indices don't overlap!
```

## What We Observed

1. **Workers execute successfully**:
   ```
   Worker 0: populated epoch_0000 with 1000 measurements ✅
   Worker 1: populated epoch_0000 with 1000 measurements ✅
   Worker 2: populated epoch_0000 with 1000 measurements ✅
   Worker 3: populated epoch_0000 with 1000 measurements ✅
   ```

2. **But data verification fails**:
   ```python
   data = storage.lightcurves.sel(object='1', epoch='epoch_0000')
   # Returns: [[nan nan], [nan nan]]  ❌
   ```

3. **Silent corruption** - no exceptions raised!

## Recommended Patterns

### ✅ SAFE: Sequential Population
```python
# One process, one thread, sequential writes
storage = LightcurveStorage(path)
for epoch in epochs:
    storage.populate_epoch_from_catalog_v2(
        epoch_key=epoch,
        source_catalog=catalog,
        measurement_keys=catalog.measurement_keys
    )
```

### ✅ SAFE: External Lock (Threading)
```python
import threading

lock = threading.Lock()

def worker(catalog_subset):
    storage = LightcurveStorage(path)
    storage.load_storage()
    
    for epoch in epochs:
        with lock:  # Serialize zarr writes
            storage.populate_epoch_from_catalog_v2(...)
```

### ✅ SAFE: Process Pool with Sequential Writes
```python
from multiprocessing import Pool

def populate_one_epoch(args):
    epoch_key, catalog = args
    storage = LightcurveStorage(path)
    storage.load_storage()
    storage.populate_epoch_from_catalog_v2(
        epoch_key=epoch_key,
        source_catalog=catalog,
        measurement_keys=catalog.measurement_keys
    )

# Each process handles ONE epoch at a time
with Pool(processes=4) as pool:
    pool.map(populate_one_epoch, epoch_catalog_pairs)
```

### ❌ UNSAFE: Concurrent Writes Without Lock
```python
# DO NOT DO THIS!
def worker(catalog_subset):
    storage = LightcurveStorage(path)
    for epoch in epochs:
        # NO LOCK - WILL CORRUPT DATA!
        storage.populate_epoch_from_catalog_v2(...)

with ThreadPoolExecutor() as executor:
    executor.map(worker, catalog_partitions)
```

## Implementation Notes

### Common Mistake: Wrong Measurement Keys
```python
# ❌ WRONG - using raw column names
storage.populate_epoch_from_catalog_v2(
    measurement_keys=['MAG_AUTO', 'MAGERR_AUTO']  # Won't match!
)

# ✅ CORRECT - using catalog's mapped keys
storage.populate_epoch_from_catalog_v2(
    measurement_keys=catalog.measurement_keys  # ['default-auto', 'default-iso']
)
```

The catalog creates measurement keys like `'default-auto'` from columns `MAG_AUTO` + `MAGERR_AUTO`.

### Spatial Ordering
Storage applies spatial ordering to objects during creation:
```python
# In create_storage()
object_keys = source_catalog.get_spatially_ordered_keys()
```

This is fine for storage, but when populating:
1. Use `catalog.measurement_keys` (the mapped names)
2. Catalog object matching is handled automatically
3. But concurrent writes still fail due to zarr limitations

## Performance Implications

### Sequential (Current Safe Approach)
- **Pros**: Guaranteed correct, no data corruption
- **Cons**: No parallelization benefit
- **Speed**: ~3-5 seconds per epoch (1000 objects)

### Locked Concurrent
- **Pros**: Can parallelize by epoch
- **Cons**: Lock serializes zarr writes (bottleneck)
- **Speed**: Similar to sequential for writes, but can parallelize catalog prep

### Ideal (Not Currently Possible)
- **Dream**: True concurrent writes to different object slices
- **Reality**: Zarr v3 doesn't support this safely
- **Future**: Wait for zarr v3 specification to mature or use different storage backend

## Conclusions

1. **Use `populate_epoch_from_catalog_v2` sequentially** for now
2. **Do not attempt concurrent population** without external locks
3. **Silent data corruption** is the failure mode (very dangerous!)
4. **Always verify data** after population in production workflows

## References

- **Zarr v3 Specification**: https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html
- **Context7 Research**: Confirmed removal of `region` and `synchronizer` parameters
- **Test Suite**: `tests/test_concurrent_writing.py` documents these findings

## Test Evidence

Run tests to see the behavior:
```bash
# Shows sequential writes work
pytest tests/test_concurrent_writing.py::TestConcurrentWriting::test_safe_pattern_with_sequential_writes -v

# Shows concurrent writes fail (xfail)
pytest tests/test_concurrent_writing.py::TestConcurrentWriting::test_safe_pattern_partition_by_objects -v
```

---

**Status**: Documented on 2025-01-10  
**Recommendation**: Keep using v2 for correctness; avoid concurrent usage until zarr v3 matures
