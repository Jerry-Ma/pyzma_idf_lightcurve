# Rechunking Investigation Results

## Summary

The `rechunker` library concept is **excellent** for our use case, but there's a **compatibility issue** with zarr v3.

## The Concept

The rechunking workflow would be perfect:

1. **Create epoch-chunked storage** (chunks: `{object: 1000, epoch: 1, ...}`)
   - Optimized for writing all objects in one epoch at a time
   - Sequential writes are fast and safe

2. **Populate epochs sequentially**
   - Each epoch write touches only 1 chunk
   - No concurrent writing needed
   - No data corruption risk

3. **Rechunk to object-chunked** (chunks: `{object: 1, epoch: 100, ...}`)
   - Optimized for reading individual lightcurves
   - Each lightcurve access touches only 1 chunk
   - Efficient for analysis

## The Problem

**Rechunker is not compatible with zarr v3**:

```
AttributeError: module 'zarr' has no attribute 'hierarchy'
```

The `rechunker` library (v0.5.4, latest as of 2024-10) was built for zarr v2 API, which has:
- `zarr.hierarchy.Group`
- `zarr.core.Array`  
- Different store abstractions

Zarr v3 has a completely different API:
- `zarr.Group`
- `zarr.Array`
- New store API (v3 spec)

## Test Results

The simple test script successfully:
- ✅ Created epoch-chunked zarr v3 array (100 objects, 10 epochs, 2 measurements)
- ✅ Populated 10 epochs with ~80% data coverage
- ✅ Verified data integrity
- ❌ **FAILED** at rechunking step due to zarr v2/v3 incompatibility

## Possible Solutions

### Option 1: Wait for rechunker zarr v3 support

Track issue: https://github.com/pangeo-data/rechunker/issues

**Pros**:
- Clean, well-tested solution
- Active maintenance from pangeo-data team
- Designed for distributed computing

**Cons**:
- Unclear timeline
- May require waiting months

### Option 2: Use zarr v2 for this project

Downgrade to zarr v2 and use rechunker as-is.

**Pros**:
- Rechunker works immediately
- Proven solution

**Cons**:
- Miss out on zarr v3 improvements
- Our entire codebase is already on zarr v3
- Would need to rewrite/test everything

### Option 3: Implement custom rechunking

Write our own rechunking function using zarr v3 API.

**Pros**:
- Full control
- Optimized for our specific use case
- Learns zarr v3 internals

**Cons**:
- Reinventing the wheel
- Need to handle edge cases ourselves
- More code to maintain

### Option 4: Hybrid approach - Manual copy

Since our data isn't huge (compared to climate datasets), use a simple copy loop:

```python
# Create object-chunked storage
object_store = zarr.open(
    'object_chunked.zarr',
    mode='w',
    shape=epoch_store.shape,
    chunks=(1, 100, 2),  # object-optimized
    dtype='f4',
)

# Copy data in batches
for obj_idx in range(n_objects):
    object_store[obj_idx, :, :] = epoch_store[obj_idx, :, :]
```

**Pros**:
- Simple and straightforward
- No dependencies
- Works with zarr v3

**Cons**:
- Not optimized for huge datasets
- Single-threaded (but could use dask)

### Option 5: Use dask for rechunking

Dask has built-in rechunking support and works with zarr v3:

```python
import dask.array as da

# Open with dask
epoch_arr = da.from_zarr('epoch_chunked.zarr')

# Rechunk
object_arr = epoch_arr.rechunk((1, 100, 2))

# Save to new location
object_arr.to_zarr('object_chunked.zarr')
```

**Pros**:
- Dask is already a dependency
- Works with zarr v3
- Handles large datasets well
- Can distribute across workers

**Cons**:
- Adds complexity if not already using dask
- Need to configure dask properly

## Recommendation

**Use Option 5 (dask rechunking)** for now:

1. Dask is already used in many astronomy projects
2. Works with zarr v3
3. Simple API for rechunking
4. Can scale if needed

If dask rechunking proves problematic, fallback to Option 4 (manual copy) which is simple and reliable for our data sizes.

## Next Steps

1. Test dask rechunking with zarr v3
2. Create example script showing the workflow
3. Document performance characteristics
4. Update LightcurveStorage to support optional rechunking after population

## Test Script Location

`scripts/test_rechunking_concept.py` - demonstrates the concept up to the rechunker incompatibility

## References

- rechunker: https://rechunker.readthedocs.io/
- zarr v3 spec: https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html
- dask rechunking: https://docs.dask.org/en/stable/array-chunks.html
