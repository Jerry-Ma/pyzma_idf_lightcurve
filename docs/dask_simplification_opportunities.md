# Dask Simplification Opportunities in datamodel.py

**Analysis Date**: 2025-10-11  
**Context**: After successfully implementing rechunking with dask, identify areas where dask can simplify the current implementation.

## Current State Analysis

The `LightcurveStorage` class currently uses:
- ✅ xarray for labeled access and metadata
- ✅ zarr for on-disk storage with chunking
- ✅ numpy for in-memory array operations
- ❌ **NO dask** - missing opportunity for simplification and distributed computing

## Identified Opportunities

### 1. **Add Rechunking Method** (High Priority) ⭐

**Current**: No rechunking capability in the class

**Proposed**: Add `rechunk_to_object_optimized()` method

```python
def rechunk_to_object_optimized(
    self,
    output_path: str | Path,
    progress: bool = True
) -> 'LightcurveStorage':
    """Rechunk from epoch-optimized to object-optimized storage.
    
    After sequential epoch-by-epoch population (safe from race conditions),
    rechunk to enable fast per-object lightcurve retrieval.
    
    Args:
        output_path: Path for rechunked storage
        progress: Show progress bar (requires tqdm)
        
    Returns:
        New LightcurveStorage instance pointing to rechunked data
    """
    import dask.array as da
    from pathlib import Path
    
    if self.lightcurves is None:
        self.load_storage(consolidated=True)
    assert self.lightcurves is not None
    
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get dimensions
    n_epochs = len(self.lightcurves.epoch)
    
    # Target chunks: one object, all epochs
    target_chunks = (1, len(self.lightcurves.measurement), 
                    len(self.lightcurves.value), n_epochs)
    
    # Load zarr as dask array
    zarr_path = str(self.storage_path / "lightcurves.zarr")
    dask_array = da.from_zarr(f'{zarr_path}/lightcurves')
    
    # Rechunk
    logger.info(f"Rechunking from {dask_array.chunks} to {target_chunks}")
    rechunked = dask_array.rechunk(target_chunks)
    
    # Save to new location
    output_zarr = str(output_path / "lightcurves.zarr")
    
    if progress:
        from dask.diagnostics import ProgressBar
        with ProgressBar():
            rechunked.to_zarr(output_zarr, overwrite=True)
    else:
        rechunked.to_zarr(output_zarr, overwrite=True)
    
    logger.info(f"Rechunked storage saved to {output_path}")
    
    # Copy metadata and coordinates
    # ... (copy xarray metadata, coordinates, attrs)
    
    return LightcurveStorage(output_path)
```

**Benefits**:
- ✅ Enables the recommended workflow: epoch-chunked → rechunk → object-chunked
- ✅ Solves concurrent write problem (write sequentially to epoch-chunked, then rechunk)
- ✅ Progress tracking built-in
- ✅ Memory-efficient for large datasets

**Effort**: ~50 lines of code

---

### 2. **Simplify Epoch Population** (Medium Priority)

**Current**: Three versions (v0, v1, v2) with manual zarr array manipulation

**Issue**: All versions have race conditions with concurrent writes (documented in tests)

**Proposed**: Use dask for chunked, parallel-safe epoch writing

```python
def populate_epoch_from_catalog_dask(
    self,
    epoch_key: str,
    source_catalog: SourceCatalog,
    measurement_keys: list[MeasurementKeyT]
) -> int:
    """Populate epoch using dask for better memory management and scheduling.
    
    Uses dask delayed operations to avoid loading entire epoch into memory.
    Still sequential (no race conditions), but better resource usage.
    """
    import dask.array as da
    from dask import delayed
    
    # Load storage
    if self.lightcurves is None:
        self.load_storage(consolidated=False)
    assert self.lightcurves is not None
    
    # Get indices
    epoch_idx = int(self.lightcurves.epoch.to_index().get_loc(epoch_key))
    
    # Build measurement data as dask delayed
    @delayed
    def extract_measurement_block(m_key):
        measurements = source_catalog.extract_measurements([m_key])
        if m_key not in measurements:
            return None
        values = measurements[m_key]
        return {
            'mag': values['mag'],
            'mag_err': values['mag_err']
        }
    
    # Schedule all extraction tasks
    delayed_tasks = [extract_measurement_block(mk) for mk in measurement_keys]
    
    # Compute in parallel (or sequentially with dask scheduler)
    results = da.compute(*delayed_tasks, scheduler='threads')
    
    # Write to zarr (still sequential for safety)
    zarr_path = str(self.storage_path / "lightcurves.zarr")
    zarr_group = zarr.open_group(zarr_path, mode='r+')
    lightcurves_array = zarr_group['lightcurves']
    
    # ... write results to epoch_idx ...
    
    return n_updated
```

**Benefits**:
- ✅ Better memory management (lazy evaluation)
- ✅ Can use dask scheduler for resource optimization
- ✅ Still sequential writes (no race conditions)
- ❌ More complex than current v1/v2

**Effort**: ~100 lines of code

**Decision**: **LOW PRIORITY** - Current v1 implementation works well for sequential writes. Only implement if memory becomes an issue.

---

### 3. **Lazy Loading and Computation** (Low Priority)

**Current**: `load_storage()` opens xarray with zarr backend (already lazy)

**Observation**: xarray with zarr backend is **already lazy**! Reading data only happens when you access `.values` or `.compute()`.

**Proposed**: No changes needed, but document the lazy behavior

```python
def load_storage(self, consolidated: bool = True):
    """Load existing xarray storage from zarr.
    
    Uses xr.open_zarr() which provides **lazy loading** by default.
    Data is only loaded from disk when accessed (e.g., .values, .compute()).
    
    For large datasets, use dask-backed xarray operations:
        - Use .sel(), .isel() for labeled/indexed selection (returns views)
        - Call .compute() only when you need actual values
        - Use .persist() to cache frequently-accessed subsets in memory
    
    Args:
        consolidated: Whether to use consolidated metadata.
    """
    # ... existing implementation ...
```

**Benefits**:
- ✅ No code changes needed
- ✅ Better documentation of existing lazy behavior

**Effort**: Documentation only

---

### 4. **Distributed Spatial Queries** (Future Enhancement)

**Current**: `get_objects_in_region()` uses vectorized numpy operations

**Proposed**: Use dask for distributed/parallel spatial filtering

```python
def get_objects_in_region_distributed(
    self, 
    ra_range: tuple, 
    dec_range: tuple,
    use_dask: bool = False
) -> list[str]:
    """Get object keys within a coordinate range.
    
    Args:
        ra_range: (min_ra, max_ra) in degrees
        dec_range: (min_dec, max_dec) in degrees
        use_dask: Use dask for distributed computation (for very large catalogs)
    """
    if self.lightcurves is None:
        self.load_storage()
    assert self.lightcurves is not None
    
    if not use_dask:
        # Fast path for small catalogs (existing implementation)
        ra_mask = (self.lightcurves.ra >= ra_range[0]) & (self.lightcurves.ra <= ra_range[1])
        dec_mask = (self.lightcurves.dec >= dec_range[0]) & (self.lightcurves.dec <= dec_range[1])
        region_mask = ra_mask & dec_mask
        return self.lightcurves.object.where(region_mask, drop=True).values.tolist()
    
    # Dask path for distributed computation
    import dask.array as da
    
    ra_da = da.from_array(self.lightcurves.ra.values, chunks=10000)
    dec_da = da.from_array(self.lightcurves.dec.values, chunks=10000)
    
    ra_mask = (ra_da >= ra_range[0]) & (ra_da <= ra_range[1])
    dec_mask = (dec_da >= dec_range[0]) & (dec_da <= dec_range[1])
    region_mask = (ra_mask & dec_mask).compute()
    
    return self.lightcurves.object.values[region_mask].tolist()
```

**Benefits**:
- ✅ Scales to very large catalogs (>1M objects)
- ✅ Can use distributed scheduler
- ❌ Overkill for IDF (~10K objects)

**Decision**: **NOT RECOMMENDED** for IDF project - current vectorized numpy is fast enough

**Effort**: ~30 lines of code

---

### 5. **Parallel Epoch Processing** (Architecture Change)

**Current**: Sequential epoch population to avoid race conditions

**Proposed**: Use dask delayed + final rechunking workflow

```python
def populate_all_epochs_parallel(
    self,
    epoch_catalog_pairs: list[tuple[str, SourceCatalog]],
    measurement_keys: list[MeasurementKeyT],
    n_workers: int = 4
) -> dict[str, int]:
    """Populate multiple epochs in parallel using dask delayed.
    
    IMPORTANT: This creates epoch-chunked storage where each epoch
    is in a separate chunk, avoiding race conditions.
    
    After completion, call rechunk_to_object_optimized() for
    efficient per-object lightcurve access.
    
    Args:
        epoch_catalog_pairs: List of (epoch_key, catalog) tuples
        measurement_keys: Measurement keys to extract
        n_workers: Number of parallel workers
        
    Returns:
        Dict of epoch_key -> n_measurements_stored
    """
    from dask import delayed, compute
    from dask.distributed import Client
    
    # Create delayed tasks for each epoch
    tasks = []
    for epoch_key, catalog in epoch_catalog_pairs:
        task = delayed(self.populate_epoch_from_catalog)(
            epoch_key, catalog, measurement_keys
        )
        tasks.append((epoch_key, task))
    
    # Compute in parallel
    with Client(n_workers=n_workers, threads_per_worker=1):
        results = compute(*[t[1] for t in tasks])
    
    return {epoch_key: result for (epoch_key, _), result in zip(tasks, results)}
```

**Benefits**:
- ✅ Faster population for many epochs
- ✅ Still safe (each epoch in separate chunk)
- ❌ Requires epoch-chunked storage (not object-chunked)
- ❌ More complex error handling

**Decision**: **DEFER** - Current sequential approach is working. Implement only if processing time becomes a bottleneck.

**Effort**: ~100 lines + testing

---

## Recommended Implementation Plan

### Phase 1: Essential (Do Now) ✅

1. **Add `rechunk_to_object_optimized()` method** (Opportunity #1)
   - **Why**: Solves the race condition problem permanently
   - **How**: Use dask.array.rechunk() as proven in tests
   - **Effort**: ~50 lines
   - **Benefit**: Enables safe sequential writes + fast reads

2. **Document lazy loading behavior** (Opportunity #3)
   - **Why**: Users should know xarray is already lazy
   - **How**: Update docstrings in `load_storage()` and class docs
   - **Effort**: 5 minutes
   - **Benefit**: Better API understanding

### Phase 2: Optional Enhancements (Consider Later)

3. **Dask-based epoch population** (Opportunity #2)
   - **When**: Only if memory usage becomes an issue
   - **Priority**: LOW - current v1 works well

4. **Distributed spatial queries** (Opportunity #4)
   - **When**: Only if catalog grows to >100K objects
   - **Priority**: VERY LOW - not needed for IDF

5. **Parallel epoch processing** (Opportunity #5)
   - **When**: Only if processing time is a bottleneck
   - **Priority**: MEDIUM - could speed up pipeline

### Phase 3: Not Recommended

- ❌ Replacing current populate methods with dask (unnecessary complexity)
- ❌ Distributed spatial queries for IDF scale (overkill)

## Concrete Next Steps

1. **Implement rechunking method** (from proven test code):
   ```python
   # Add to LightcurveStorage class
   def rechunk_to_object_optimized(self, output_path, progress=True):
       # ... implementation from tests ...
   ```

2. **Update documentation**:
   - Add "Recommended Workflow" section to class docstring
   - Document lazy loading in `load_storage()`
   - Add examples to `examples/rechunking_workflow.py`

3. **Update tests**:
   - Move `test_rechunking.py` tests into main test suite
   - Add integration test for full workflow

4. **Update pipeline assets** (if using dagster):
   - Add rechunking step after epoch population
   - Configure to use object-chunked storage for analysis

## Summary

**Biggest Win**: Add `rechunk_to_object_optimized()` method using dask ⭐

This single addition enables the recommended workflow:
1. Create epoch-chunked storage
2. Populate epochs sequentially (no race conditions)
3. Rechunk to object-chunked (one method call!)
4. Analyze from object-chunked storage (fast reads)

**Everything else**: Nice to have, but not essential for IDF project scale.

## Code Estimate

**Essential additions**:
- `rechunk_to_object_optimized()`: ~50 lines
- Documentation updates: ~20 lines
- **Total**: ~70 lines of production code

**Testing**:
- Integration test: ~50 lines
- Example script: ~100 lines
- **Total**: ~150 lines of test/example code

**Grand total**: ~220 lines to get full rechunking support! 🎉
