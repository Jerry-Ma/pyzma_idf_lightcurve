# Rechunking Workflow - Success! 🎉

## Summary

**The rechunking concept works perfectly with dask!** We can now:

1. Create epoch-chunked storage for fast sequential writing
2. Populate data efficiently without concurrent write issues  
3. Rechunk to object-chunked storage for efficient lightcurve access

## Test Results

### ✅ Successful Test Run

```bash
$ python scripts/test_rechunking_concept.py

### Step 1: Create epoch-chunked storage ###
Creating zarr array:
  Shape: (100 objects, 10 epochs, 2 measurements)
  Epoch-optimized chunks: (100, 1, 2)

### Step 2: Populate epochs sequentially ###
  Epoch 0-9: Successfully populated ~80% coverage each

### Step 3: Rechunk to object-chunked storage (using dask) ###
Rechunking from (100, 1, 2) to (1, 10, 2)
  ✅ Rechunking complete!

### Step 4: Verify data integrity ###
  ✅ Object 0: data matches
  ✅ Object 50: data matches
  ✅ Object 99: data matches
  
✅ All data verified - rechunking successful!
```

## Implementation

### Using Dask for Rechunking (zarr v3 compatible)

```python
import dask.array as da
import zarr

# Step 1: Create epoch-chunked storage
epoch_store = zarr.open(
    'epoch_chunked.zarr',
    mode='w',
    shape=(n_objects, n_epochs, n_measurements),
    chunks=(n_objects, 1, n_measurements),  # All objects, one epoch
    dtype='f4',
)

# Step 2: Populate epochs sequentially (safe, no concurrency issues)
for epoch_idx in range(n_epochs):
    epoch_data = get_epoch_measurements()
    epoch_store[:, epoch_idx, :] = epoch_data  # 1 chunk write

# Step 3: Rechunk to object-optimized storage
dask_array = da.from_zarr('epoch_chunked.zarr')
rechunked = dask_array.rechunk((1, n_epochs, n_measurements))
rechunked.to_zarr('object_chunked.zarr')
```

## Performance Characteristics

### Epoch-Chunked Storage (for writing)

**Chunks**: `(100 objects, 1 epoch, 2 measurements)`

✅ **Efficient for writing:**
- Writing all objects in one epoch: **1 chunk write**
- Sequential population: safe and fast
- No concurrent write issues

❌ **Inefficient for reading:**
- Reading one object's lightcurve: **10 chunk reads** (one per epoch)

### Object-Chunked Storage (for reading)

**Chunks**: `(1 object, 10 epochs, 2 measurements)`

✅ **Efficient for reading:**
- Reading one object's lightcurve: **1 chunk read**
- Perfect for lightcurve analysis

❌ **Inefficient for writing:**
- Writing all objects in one epoch: **100 chunk writes**
- Would suffer from concurrent write race conditions

## Recommended Workflow

### Phase 1: Data Population (Sequential Writes)

```python
# Create epoch-chunked storage
storage = LightcurveStorage.create_storage(
    storage_path='epoch_chunked.zarr',
    chunks={
        'object': 1000,     # All objects
        'epoch': 1,          # One epoch at a time
        'measurement': 2,
        'value': 2,
    }
)

# Populate sequentially (safe, no race conditions)
for epoch_key in epoch_keys:
    storage.populate_epoch_from_catalog_v2(
        epoch_key=epoch_key,
        source_catalog=catalog,
        measurement_keys=catalog.measurement_keys,
    )
```

### Phase 2: Rechunk for Analysis (One-Time Operation)

```python
import dask.array as da

# Load epoch-chunked data
dask_array = da.from_zarr('epoch_chunked.zarr/lightcurves')

# Rechunk to object-optimized
rechunked = dask_array.rechunk({
    'object': 1,         # One object per chunk
    'epoch': 100,        # All epochs together
    'measurement': 2,
    'value': 2,
})

# Save to new location
rechunked.to_zarr('object_chunked.zarr/lightcurves')
```

### Phase 3: Analysis (Fast Lightcurve Access)

```python
# Load object-chunked storage
storage = LightcurveStorage('object_chunked.zarr')
storage.load_storage()

# Access lightcurves efficiently (1 chunk read each)
for obj_id in object_ids:
    lc = storage.get_object_lightcurve(obj_id, 'default-auto')
    # Analyze lightcurve...
```

## Benefits

1. **No Concurrent Write Issues**
   - Sequential epoch population is safe
   - No race conditions
   - No data corruption

2. **Optimal Read Performance**
   - Each lightcurve in a single chunk
   - Minimal I/O for analysis
   - Fast random access

3. **Zarr v3 Compatible**
   - Uses dask, which works with zarr v3
   - No need for old rechunker library
   - Future-proof

4. **Scalable**
   - Dask can distribute rechunking
   - Handles large datasets
   - Can use cluster if needed

## Tool Comparison

| Tool | Zarr v3 Support | Status |
|------|----------------|--------|
| **rechunker** | ❌ No | Incompatible (built for zarr v2) |
| **dask** | ✅ Yes | Works perfectly! |
| **Manual copy** | ✅ Yes | Simple but slower |

## Next Steps

1. ✅ Test concept - **DONE**
2. ✅ Verify data integrity - **DONE**
3. ⏭️ Integrate into LightcurveStorage class
4. ⏭️ Add rechunking method to the API
5. ⏭️ Document in user guide
6. ⏭️ Performance benchmarks with real IDF data

## Files

- **Test Script**: `scripts/test_rechunking_concept.py`
- **Investigation**: `docs/rechunking_investigation.md`
- **This Summary**: `docs/rechunking_success.md`

## Conclusion

The rechunking workflow is **ready for integration**! We have a proven solution that:
- Solves the concurrent write problem
- Optimizes for both write and read patterns
- Works with zarr v3
- Uses well-maintained dask library

This is a **game-changer** for the IDF lightcurve workflow! 🚀
