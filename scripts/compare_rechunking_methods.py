#!/usr/bin/env python
"""Compare rechunking performance: dask vs rechunker.

Tests both methods with timing to determine which is better for our use case.
"""

import numpy as np
import zarr
from pathlib import Path
import shutil
import time
import dask.array as da

# Create test data directory
test_dir = Path("scratch_rechunking_comparison")
test_dir.mkdir(exist_ok=True)

print("=" * 80)
print("Rechunking Performance Comparison: Dask vs Rechunker")
print("=" * 80)

# Test parameters
n_objects = 500  # Larger dataset for realistic performance test
n_epochs = 20
n_measurements = 2

print(f"\nTest dataset:")
print(f"  Shape: ({n_objects} objects, {n_epochs} epochs, {n_measurements} measurements)")
print(f"  Total size: {n_objects * n_epochs * n_measurements * 4 / 1024 / 1024:.2f} MB")

# Step 1: Create source epoch-chunked storage
print("\n" + "=" * 80)
print("Step 1: Create source epoch-chunked storage")
print("=" * 80)

source_path = test_dir / "source_epoch_chunked.zarr"
if source_path.exists():
    shutil.rmtree(source_path)

epoch_chunks = (n_objects, 1, n_measurements)
print(f"\nCreating source with chunks: {epoch_chunks}")

start_time = time.time()
source_store = zarr.open(
    str(source_path),
    mode='w',
    shape=(n_objects, n_epochs, n_measurements),
    chunks=epoch_chunks,
    dtype='f4',
)

# Populate with test data
for epoch_idx in range(n_epochs):
    epoch_data = np.random.normal(20.0, 1.0, (n_objects, n_measurements))
    mask = np.random.random((n_objects, n_measurements)) < 0.2
    epoch_data[mask] = np.nan
    source_store[:, epoch_idx, :] = epoch_data

create_time = time.time() - start_time
print(f"✅ Source created in {create_time:.3f}s")
print(f"   Chunks: {source_store.chunks}")

# Target chunks for object-optimized storage
object_chunks = (1, n_epochs, n_measurements)
print(f"\nTarget chunks (object-optimized): {object_chunks}")

# Test 1: Dask rechunking
print("\n" + "=" * 80)
print("Test 1: Dask Rechunking")
print("=" * 80)

dask_target_path = test_dir / "target_dask.zarr"
if dask_target_path.exists():
    shutil.rmtree(dask_target_path)

print("\nLoading with dask...")
start_time = time.time()
dask_array = da.from_zarr(str(source_path))
load_time = time.time() - start_time
print(f"  Loaded in {load_time:.3f}s")
print(f"  Original chunks: {dask_array.chunks}")

print("\nRechunking...")
start_time = time.time()
rechunked_array = dask_array.rechunk(object_chunks)
rechunk_time = time.time() - start_time
print(f"  Rechunk plan created in {rechunk_time:.3f}s")
print(f"  New chunks: {rechunked_array.chunks}")

print("\nSaving to zarr...")
start_time = time.time()
rechunked_array.to_zarr(str(dask_target_path), overwrite=True)
save_time = time.time() - start_time
print(f"  Saved in {save_time:.3f}s")

dask_total_time = load_time + rechunk_time + save_time
print(f"\n✅ Dask total time: {dask_total_time:.3f}s")

# Verify dask output
dask_result = zarr.open(str(dask_target_path), mode='r')
print(f"   Result chunks: {dask_result.chunks}")

# Test 2: Rechunker (if compatible with zarr v3)
print("\n" + "=" * 80)
print("Test 2: Rechunker Library")
print("=" * 80)

try:
    from rechunker import rechunk
    
    rechunker_target_path = test_dir / "target_rechunker.zarr"
    rechunker_temp_path = test_dir / "temp_rechunker.zarr"
    
    if rechunker_target_path.exists():
        shutil.rmtree(rechunker_target_path)
    if rechunker_temp_path.exists():
        shutil.rmtree(rechunker_temp_path)
    
    print("\nCreating rechunking plan...")
    start_time = time.time()
    
    # Try to use rechunker with zarr v3
    try:
        rechunked_plan = rechunk(
            source_store,
            target_chunks=object_chunks,
            target_store=str(rechunker_target_path),
            temp_store=str(rechunker_temp_path),
            max_mem='256MB',
        )
        plan_time = time.time() - start_time
        print(f"  Plan created in {plan_time:.3f}s")
        print(f"  Plan: {rechunked_plan}")
        
        print("\nExecuting rechunking...")
        start_time = time.time()
        result = rechunked_plan.execute()
        execute_time = time.time() - start_time
        print(f"  Executed in {execute_time:.3f}s")
        
        rechunker_total_time = plan_time + execute_time
        print(f"\n✅ Rechunker total time: {rechunker_total_time:.3f}s")
        
        # Verify rechunker output
        rechunker_result = zarr.open(str(rechunker_target_path), mode='r')
        print(f"   Result chunks: {rechunker_result.chunks}")
        
        rechunker_works = True
        
    except (AttributeError, TypeError) as e:
        print(f"\n❌ Rechunker failed with zarr v3:")
        print(f"   Error: {type(e).__name__}: {e}")
        print(f"   This is expected - rechunker is built for zarr v2")
        rechunker_works = False
        rechunker_total_time = None
        
except ImportError:
    print("❌ Rechunker not installed (skipping)")
    rechunker_works = False
    rechunker_total_time = None

# Step 3: Verify data integrity
print("\n" + "=" * 80)
print("Step 3: Data Integrity Verification")
print("=" * 80)

print("\nChecking dask-rechunked data...")
for obj_idx in [0, n_objects // 2, n_objects - 1]:
    source_data = source_store[obj_idx, :, 0]
    dask_data = dask_result[obj_idx, :, 0]
    
    if np.allclose(source_data, dask_data, equal_nan=True):
        print(f"  ✅ Object {obj_idx}: data matches")
    else:
        print(f"  ❌ Object {obj_idx}: DATA MISMATCH!")
        break

if rechunker_works:
    print("\nChecking rechunker-rechunked data...")
    for obj_idx in [0, n_objects // 2, n_objects - 1]:
        source_data = source_store[obj_idx, :, 0]
        rechunker_data = rechunker_result[obj_idx, :, 0]
        
        if np.allclose(source_data, rechunker_data, equal_nan=True):
            print(f"  ✅ Object {obj_idx}: data matches")
        else:
            print(f"  ❌ Object {obj_idx}: DATA MISMATCH!")
            break

# Performance Summary
print("\n" + "=" * 80)
print("Performance Summary")
print("=" * 80)

print(f"\nDataset: {n_objects} objects × {n_epochs} epochs × {n_measurements} measurements")
print(f"Data size: {n_objects * n_epochs * n_measurements * 4 / 1024 / 1024:.2f} MB")

print(f"\n{'Method':<15} {'Time (s)':<12} {'Status':<10} {'Notes'}")
print("-" * 80)
print(f"{'Dask':<15} {dask_total_time:<12.3f} {'✅ Works':<10} Zarr v3 compatible")

if rechunker_works:
    print(f"{'Rechunker':<15} {rechunker_total_time:<12.3f} {'✅ Works':<10} If compatible")
    
    # Compare performance
    if dask_total_time < rechunker_total_time:
        speedup = rechunker_total_time / dask_total_time
        print(f"\n🏆 Dask is {speedup:.2f}x faster!")
    else:
        speedup = dask_total_time / rechunker_total_time
        print(f"\n🏆 Rechunker is {speedup:.2f}x faster!")
else:
    print(f"{'Rechunker':<15} {'N/A':<12} {'❌ Failed':<10} Zarr v2 only")
    print(f"\n🏆 Dask is the only working option for zarr v3!")

# Recommendations
print("\n" + "=" * 80)
print("Recommendations")
print("=" * 80)

if rechunker_works:
    if dask_total_time < rechunker_total_time:
        print("\n✅ Use dask.array.rechunk()")
        print("   - Faster performance")
        print("   - Simpler API")
        print("   - Already a dependency")
    else:
        print("\n✅ Use rechunker library")
        print("   - Better performance")
        print("   - Optimized for large datasets")
        print("   - More control over memory usage")
else:
    print("\n✅ Use dask.array.rechunk()")
    print("   - Only option that works with zarr v3")
    print("   - Simple and reliable")
    print("   - No additional dependencies needed")

print(f"\n### Cleanup ###")
print(f"Test data saved to: {test_dir}/")
print(f"To remove: rm -rf {test_dir}/")

print("\n" + "=" * 80)
print("Comparison complete!")
print("=" * 80)
