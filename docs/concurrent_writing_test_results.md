# Concurrent Writing Test Results

## Executive Summary

**All three populate versions (v0, v1, v2) are UNSAFE for concurrent writes** due to zarr v3 limitations. The race conditions cause **intermittent success/failure**, making concurrent writes unreliable and dangerous.

## Test Methodology

### Sequential Mode (n_workers=1)
- **Objects**: 100
- **Epochs**: 3  
- **Workers**: 1
- **Purpose**: Verify basic functionality

### Concurrent Mode (n_workers=4)
- **Objects**: 400
- **Epochs**: 5
- **Workers**: 4
- **Object Partitioning**: Non-overlapping (100 objects per worker)
- **Purpose**: Test thread safety

## Test Results

### Sequential Mode - All Versions PASS ✅

| Version | Test Result | Status |
|---------|-------------|--------|
| v0      | PASSED      | ✅ Safe for sequential |
| v1      | PASSED      | ✅ Safe for sequential |
| v2      | PASSED      | ✅ Safe for sequential |

**Conclusion**: All three versions work correctly in sequential mode.

### Concurrent Mode - All Versions UNSAFE ❌

#### Race Condition Analysis (Multiple Test Runs)

**v0 Results** (3 runs):
```
Run 1: has_data=False ❌
Run 2: has_data=False ❌
Run 3: has_data=False ❌
```
- **Success Rate**: 0/3 (0%)
- **Behavior**: Most deterministic failure
- **Verdict**: ❌ **UNSAFE** - Consistently fails

**v1 Results** (3 runs):
```
Run 1: has_data=False ❌
Run 2: has_data=True  ⚠️  (race condition)
Run 3: has_data=True  ⚠️  (race condition)
```
- **Success Rate**: 2/3 (67%)
- **Behavior**: Intermittent failure
- **Verdict**: ❌ **UNSAFE** - Race condition causes unreliable results

**v2 Results** (5 runs):
```
Run 1: has_data=False ❌
Run 2: has_data=False ❌
Run 3: has_data=False ❌
Run 4: has_data=True  ⚠️  (race condition)
Run 5: has_data=True  ⚠️  (race condition)
```
- **Success Rate**: 2/5 (40%)
- **Behavior**: Intermittent failure
- **Verdict**: ❌ **UNSAFE** - Race condition causes unreliable results

## Root Cause: Zarr v3 Limitations

### Why Concurrent Writes Fail

1. **No Synchronization Primitives**: Zarr v3 removed `region` and `synchronizer` parameters
2. **Chunk-Level Atomicity Only**: Thread safety only when writing to different chunks
3. **Our Use Case**: `oindex` fancy indexing may span multiple chunks
4. **Result**: Race conditions causing data corruption

### Race Condition Explained

```python
# Two threads writing to the same chunk simultaneously:

Thread 1: Read chunk → Modify values → Write chunk
Thread 2:    Read chunk → Modify values → Write chunk
                              ↓
                    One thread's writes are LOST
```

The intermittent success happens when threads "get lucky" and don't overlap their writes, but this is **NOT reliable** and should never be trusted.

## Recommendations

### ✅ SAFE Pattern: Sequential Writes

```python
# Process epochs sequentially with proper iteration
for epoch_key in epoch_keys:
    n_updated = storage.populate_epoch_from_catalog_v2(
        epoch_key=epoch_key,
        source_catalog=catalog,
        measurement_keys=catalog.measurement_keys
    )
```

**Benefits**:
- ✅ No race conditions
- ✅ Predictable behavior
- ✅ All versions (v0, v1, v2) work correctly

### ❌ UNSAFE Pattern: Concurrent Writes

```python
# DO NOT DO THIS - even if it sometimes passes tests!
with ThreadPoolExecutor(max_workers=4) as executor:
    for epoch_key in epoch_keys:
        executor.submit(storage.populate_epoch_from_catalog_v2, ...)
```

**Risks**:
- ❌ Intermittent data corruption
- ❌ Non-deterministic failures
- ❌ Silent data loss
- ❌ May pass tests but fail in production

## Performance Alternative

If you need to process multiple AORs concurrently:

```python
# Safe concurrent pattern: One storage instance per worker
from concurrent.futures import ThreadPoolExecutor

def process_aor(aor_id, aor_path):
    # Each worker creates its own storage instance
    storage = LightcurveStorage.create_storage(...)
    
    # Process all epochs sequentially within this AOR
    for epoch in epochs:
        storage.populate_epoch_from_catalog(...)

# Parallel across AORs, sequential within each AOR
with ThreadPoolExecutor(max_workers=4) as executor:
    for aor_id, aor_path in aor_list:
        executor.submit(process_aor, aor_id, aor_path)
```

This approach:
- ✅ Parallelizes across independent AORs
- ✅ Maintains sequential safety within each AOR
- ✅ No shared zarr array writes

## Test Implementation

The comprehensive test is located at:
```
tests/test_concurrent_writing.py::TestConcurrentWriting::test_all_versions_sequential_and_concurrent
```

Test features:
- Parametrized across 3 versions (v0, v1, v2)
- Parametrized across 2 modes (sequential n=1, concurrent n=4)
- Total 6 test cases covering all combinations
- Concurrent tests always marked as `XFAIL` to document unsafe behavior

## Conclusion

**Do NOT use concurrent writes for any version (v0, v1, v2).**

The intermittent test successes are race condition artifacts and do not indicate safety. Always use sequential writes for reliable, predictable data population.

---

**Test Date**: 2024
**Zarr Version**: 3.x
**Test Framework**: pytest with parametrization
**Test Status**: ✅ All tests pass with proper XFAIL marking
