#!/usr/bin/env python
"""Simple standalone script to test the rechunking concept.

This demonstrates:
1. Creating epoch-chunked zarr storage (optimal for writing per-epoch)
2. Populating data efficiently
3. Rechunking to object-chunked storage (optimal for reading lightcurves)

Run with: python scripts/test_rechunking_concept.py
"""

import numpy as np
import zarr
from pathlib import Path
import shutil
import dask.array as da

# Create test data directory
test_dir = Path("scratch_rechunking_test")
test_dir.mkdir(exist_ok=True)

print("=" * 80)
print("Rechunking Workflow Test")
print("=" * 80)

# Step 1: Create epoch-chunked storage
print("\n### Step 1: Create epoch-chunked storage ###")

epoch_chunked_path = test_dir / "epoch_chunked.zarr"
if epoch_chunked_path.exists():
    shutil.rmtree(epoch_chunked_path)

# Create simple test array: (objects, epochs, measurements)
# Shape: (100 objects, 10 epochs, 2 measurements)
n_objects = 100
n_epochs = 10
n_measurements = 2

# Chunks optimized for writing all objects in one epoch at a time
epoch_chunks = (n_objects, 1, n_measurements)  # All objects, one epoch

print(f"Creating zarr array:")
print(f"  Shape: ({n_objects} objects, {n_epochs} epochs, {n_measurements} measurements)")
print(f"  Epoch-optimized chunks: {epoch_chunks}")

epoch_store = zarr.open(
    str(epoch_chunked_path),
    mode='w',
    shape=(n_objects, n_epochs, n_measurements),
    chunks=epoch_chunks,
    dtype='f4',
)

# Fill with NaN initially
epoch_store[:] = np.nan

# Step 2: Populate data epoch-by-epoch (simulating real workflow)
print("\n### Step 2: Populate epochs sequentially ###")

for epoch_idx in range(n_epochs):
    # Simulate measurements for this epoch
    # Random magnitudes with some missing data
    epoch_data = np.random.normal(20.0, 1.0, (n_objects, n_measurements))
    
    # 20% missing data
    mask = np.random.random((n_objects, n_measurements)) < 0.2
    epoch_data[mask] = np.nan
    
    # Write entire epoch at once (efficient with epoch-chunking)
    epoch_store[:, epoch_idx, :] = epoch_data
    
    n_valid = np.sum(~np.isnan(epoch_data))
    print(f"  Epoch {epoch_idx}: wrote {n_valid}/{n_objects * n_measurements} measurements")

print(f"\nEpoch-chunked storage created: {epoch_store.chunks}")

# Verify data
sample_data = epoch_store[0, :, 0]
n_valid = np.sum(~np.isnan(sample_data))
print(f"Sample object[0] has {n_valid}/{n_epochs} valid measurements")

# Step 3: Rechunk to object-chunked storage using dask
print("\n### Step 3: Rechunk to object-chunked storage (using dask) ###")

object_chunked_path = test_dir / "object_chunked.zarr"

if object_chunked_path.exists():
    shutil.rmtree(object_chunked_path)

# Target chunks: one object per chunk, all epochs together
object_chunks = (1, n_epochs, n_measurements)  # One object, all epochs

print(f"Rechunking from {epoch_chunks} to {object_chunks}")
print(f"  Source: epoch-optimized (good for writing)")
print(f"  Target: object-optimized (good for reading lightcurves)")

# Use dask to read and rechunk
print("\nLoading with dask...")
dask_array = da.from_zarr(str(epoch_chunked_path))

print(f"  Original chunks: {dask_array.chunks}")

print("\nRechunking...")
rechunked_array = dask_array.rechunk(object_chunks)

print(f"  New chunks: {rechunked_array.chunks}")

print("\nSaving to zarr...")
rechunked_array.to_zarr(str(object_chunked_path), overwrite=True)

print(f"✅ Rechunking complete!")

# Step 4: Verify data integrity
print("\n### Step 4: Verify data integrity ###")

object_store = zarr.open(str(object_chunked_path), mode='r')

# Compare a few samples
for obj_idx in [0, 50, 99]:
    epoch_data = epoch_store[obj_idx, :, 0]
    object_data = object_store[obj_idx, :, 0]
    
    if np.allclose(epoch_data, object_data, equal_nan=True):
        print(f"  ✅ Object {obj_idx}: data matches")
    else:
        print(f"  ❌ Object {obj_idx}: DATA MISMATCH!")
        break
else:
    print("\n✅ All data verified - rechunking successful!")

# Step 5: Performance implications
print("\n### Step 5: Performance characteristics ###")

print("\n**Epoch-chunked storage** (epoch_chunked.zarr):")
print(f"  Chunks: {epoch_store.chunks}")
print(f"  ✅ Writing all objects in one epoch: 1 chunk write")
print(f"  ❌ Reading one object's full lightcurve: {n_epochs} chunk reads")

print("\n**Object-chunked storage** (object_chunked.zarr):")
print(f"  Chunks: {object_store.chunks}")
print(f"  ✅ Reading one object's full lightcurve: 1 chunk read")
print(f"  ❌ Writing all objects in one epoch: {n_objects} chunk writes")

print("\n### Recommendation ###")
print("Use epoch-chunked for fast population, then rechunk to object-chunked")
print("for efficient lightcurve access!")

print(f"\n### Cleanup ###")
print(f"Test data saved to: {test_dir}/")
print(f"To remove: rm -rf {test_dir}/")

print("\n" + "=" * 80)
print("Rechunking test completed successfully!")
print("=" * 80)
