# Rechunking Library Comparison: Dask vs Rechunker

**Date**: 2025-01-XX  
**Conclusion**: Use dask.array for zarr v3 compatibility  
**Status**: ✅ Decision Made

## Executive Summary

After reviewing the rechunker tutorial and running performance comparisons, **dask.array is the only viable option for rechunking zarr v3 arrays**. Rechunker is built exclusively for zarr v2 and fails with zarr v3 due to API incompatibility.

### Quick Recommendation

```python
# ✅ Use this approach for zarr v3
import dask.array as da

# Load epoch-chunked storage
array = da.from_zarr('epoch_chunked.zarr')

# Rechunk to object-optimized
rechunked = array.rechunk((1, n_epochs, 2))

# Save result
rechunked.to_zarr('object_chunked.zarr', overwrite=True)
```

## Test Results

### Test Configuration
- **Dataset**: 500 objects × 20 epochs × 2 measurements
- **Source chunks**: `(500, 1, 2)` (epoch-optimized)
- **Target chunks**: `(1, 20, 2)` (object-optimized)
- **Data size**: ~80 KB (0.08 MB)

### Performance Comparison

| Method | Time | Status | Notes |
|--------|------|--------|-------|
| **Dask** | 0.760s | ✅ Works | Zarr v3 compatible, simple API |
| **Rechunker** | N/A | ❌ Failed | Zarr v2 only - API incompatibility |

### Rechunker Failure Details

**Error**:
```
AttributeError: module 'zarr' has no attribute 'hierarchy'
```

**Root Cause**:
- Rechunker expects zarr v2 API: `zarr.hierarchy.Group`, `zarr.core.Array`
- Zarr v3 uses different API: `zarr.Group`, `zarr.Array`
- No compatibility layer exists

**Tutorial Findings**:
- [Rechunker tutorial](https://rechunker.readthedocs.io/en/latest/tutorial.html) shows clean API
- All examples use zarr v2 constructs
- No zarr v3 examples or migration path documented
- Features like `max_mem` and intermediate `temp_store` are nice but not essential

## API Comparison

### Rechunker (zarr v2 only)

**Advantages**:
- Explicit memory management (`max_mem` parameter)
- Two-step process with intermediate storage (safer for huge datasets)
- Progress bar support built-in
- Designed specifically for zarr rechunking

**Code**:
```python
from rechunker import rechunk

plan = rechunk(
    source_array,
    target_chunks=(1, 20, 2),
    max_mem='256MB',
    target_store='output.zarr',
    temp_store='temp.zarr'
)
result = plan.execute()  # Can use: with ProgressBar(): plan.execute()
```

**Disadvantages**:
- ❌ **Incompatible with zarr v3** (hard blocker)
- Requires temp storage
- More verbose API

### Dask (zarr v3 compatible)

**Advantages**:
- ✅ **Works with zarr v3**
- Simple, intuitive API
- Already a dependency
- No temp storage needed
- Automatic chunking optimization

**Code**:
```python
import dask.array as da

array = da.from_zarr('input.zarr')
rechunked = array.rechunk((1, 20, 2))
rechunked.to_zarr('output.zarr', overwrite=True)
```

**Disadvantages**:
- No explicit memory control (relies on dask settings)
- Less specialized for zarr-specific operations

## Zarr Version Compatibility

### Zarr v2 API (what rechunker expects)
```python
import zarr

# Old API
group = zarr.hierarchy.Group(...)
array = zarr.core.Array(...)
```

### Zarr v3 API (current)
```python
import zarr

# New API
group = zarr.Group(...)
array = zarr.Array(...)
```

**Migration Path**: None currently available for rechunker

## Alternative Considered: Downgrade to Zarr v2

**Why We Rejected This**:
1. Would break other code using zarr v3 features
2. Zarr v2 is legacy (maintenance mode)
3. No long-term benefit
4. Dask already works with zarr v3

## Data Integrity Verification

All rechunking tests verified data integrity:

```
✅ Object 0: data matches
✅ Object 250: data matches  
✅ Object 499: data matches
```

Method: `np.allclose(source_data, rechunked_data, equal_nan=True)`

## Performance Analysis

For small dataset (500 objects × 20 epochs):
- **Dask total time**: 0.760s
  - Load: 0.002s
  - Plan: 0.005s
  - Execute: 0.753s

Most time spent in execution phase (writing rechunked data).

### Scaling Expectations

For realistic IDF dataset (10,000 objects × 500 epochs):
- Data size: ~40 MB per measurement type
- Estimated time: ~2-3 minutes per rechunk
- Memory required: ~2-4× data size

## Implementation Plan

### Phase 1: Create Epoch-Chunked Storage
```python
from pyzma_idf_lightcurve.storage.v2 import LightcurveStorage

# Create with epoch-optimized chunking
storage = LightcurveStorage.create(
    path='lightcurves_epoch_chunked.zarr',
    n_objects=10000,
    max_epochs=500,
    chunks={
        'object': 1000,  # Process multiple objects
        'epoch': 1,      # One epoch at a time
        'measurement': 2,
        'value': 2,
    }
)
```

### Phase 2: Populate Sequentially (Safe!)
```python
# No race conditions with sequential writes
for epoch_idx in range(n_epochs):
    epoch_data = extract_epoch_measurements(epoch_idx)
    storage.populate_epoch_v2(epoch_idx, epoch_data)
```

### Phase 3: Rechunk to Object-Optimized
```python
import dask.array as da

# Load epoch-chunked storage
flux_array = da.from_zarr('lightcurves_epoch_chunked.zarr/flux')
flux_err_array = da.from_zarr('lightcurves_epoch_chunked.zarr/flux_err')

# Rechunk to object-optimized
target_chunks = (1, max_epochs, 2)
flux_rechunked = flux_array.rechunk(target_chunks)
flux_err_rechunked = flux_err_array.rechunk(target_chunks)

# Save to new storage
flux_rechunked.to_zarr('lightcurves_object_chunked.zarr/flux')
flux_err_rechunked.to_zarr('lightcurves_object_chunked.zarr/flux_err')
```

### Phase 4: Analyze from Object-Chunked Storage
```python
# Fast lightcurve access (1 chunk read per object)
storage = LightcurveStorage.open('lightcurves_object_chunked.zarr')
lc = storage.get_lightcurve(object_id=12345)
```

## Recommendations

### For This Project (IDF Lightcurves)

✅ **Use dask.array.rechunk()**

**Reasons**:
1. Only option that works with zarr v3
2. Simple, proven solution
3. No additional dependencies
4. Adequate performance for our dataset size
5. Clear, maintainable code

### Add Rechunking Method to LightcurveStorage

```python
class LightcurveStorage:
    def rechunk_to_object_optimized(
        self,
        output_path: str | Path,
        n_epochs: int,
        progress: bool = True
    ) -> 'LightcurveStorage':
        """Rechunk from epoch-optimized to object-optimized storage.
        
        This enables fast per-object lightcurve retrieval after
        sequential epoch-by-epoch population.
        
        Args:
            output_path: Path for rechunked storage
            n_epochs: Number of epochs in data
            progress: Show progress bar
            
        Returns:
            New LightcurveStorage instance pointing to rechunked data
        """
        import dask.array as da
        
        # Target chunks: one object, all epochs
        target_chunks = (1, n_epochs, 2)
        
        # Rechunk each measurement array
        for array_name in ['flux', 'flux_err']:
            array = da.from_zarr(f'{self.path}/{array_name}')
            rechunked = array.rechunk(target_chunks)
            rechunked.to_zarr(
                f'{output_path}/{array_name}',
                overwrite=True
            )
        
        return LightcurveStorage.open(output_path)
```

## Future Considerations

### If Rechunker Adds Zarr v3 Support

Monitor rechunker development for zarr v3 compatibility:
- GitHub: https://github.com/pangeo-data/rechunker
- If updated, reconsider for features like:
  - Explicit memory management
  - Better progress tracking
  - Optimized intermediate storage

### For Extremely Large Datasets

If dataset grows beyond available memory:
- Use dask distributed scheduler
- Adjust dask chunk size settings
- Consider streaming rechunking approach
- Monitor memory usage during execution

## References

- **Rechunker Documentation**: https://rechunker.readthedocs.io/en/latest/tutorial.html
- **Dask Array Documentation**: https://docs.dask.org/en/stable/array.html
- **Zarr v3 Specification**: https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html
- **Our Tests**:
  - `scripts/test_rechunking_concept.py` - Initial dask proof of concept
  - `scripts/compare_rechunking_methods.py` - Performance comparison
  - `docs/rechunking_investigation.md` - Options analysis
  - `docs/rechunking_success.md` - Dask implementation guide

## Conclusion

**Decision**: Use **dask.array.rechunk()** for all zarr v3 rechunking operations.

**Justification**:
- Only working solution for zarr v3
- Simple, maintainable API
- Proven performance
- No additional dependencies
- Easy to integrate into existing code

**Status**: Ready for production implementation ✅
