"""
IDF Lightcurve Storage with xarray-based labeled access and vectorized operations.

Uses xarray DataArray with labeled dimensions for intuitive access:
- object_key: Individual object string keys (e.g., 'I123' for IRAC source #123)
- measurement_key: IDF measurement keys (e.g., 'ch1_sci_clean-auto')
- value_key: 'mag' or 'mag_err'
- epoch_key: Temporal grouping keys (e.g., per-AOR group names, or per-week bins)

Eliminates spatial chunking complexity and enables efficient vectorized updates.
"""

import numpy as np
import xarray as xr
import zarr
import dask.array as da
from pathlib import Path
from typing import Any, cast, TypedDict
from astropy.table import Table
import re
from datetime import datetime
from loguru import logger
from ..utils.naming import NameTemplate
from .catalog import MeasurementKeyT, SourceCatalog, ValueKeyT

__all__ = ['LightcurveStorage', 'SourceCatalog']


class LightcurveStorage:
    """xarray-based lightcurve storage with labeled dimensions and vectorized operations.
    
    Follows xarray + Zarr best practices:
    - Uses consolidated metadata for faster opening
    - Optimized chunking strategy for spatial locality
    - Zarr v3 format for latest features
    - Support for region-based writes for distributed processing
    """
    
    def __init__(self, storage_path: Path, enable_spatial_chunking: bool = True, 
                 chunk_size: int = 1000):
        """
        Initialize lightcurve storage with xarray backend.
        
        Args:
            storage_path: Path to zarr storage directory
            enable_spatial_chunking: Whether to sort objects spatially for better chunk locality  
            chunk_size: Number of objects per spatial chunk
        """
        self.storage_path = Path(storage_path)
        self.enable_spatial_chunking = enable_spatial_chunking
        self.chunk_size = chunk_size
        self.lightcurves: xr.DataArray | None = None

    def create_storage(
        self,
        source_catalog: SourceCatalog,
        epoch_keys: list[str],
        measurement_keys: list[MeasurementKeyT] | None = None,
        value_keys: list[ValueKeyT] = ['mag', 'mag_err'],
        tbl_aors: Table | None = None
    ) -> None:
        """
        Create xarray-based storage with labeled dimensions.
        
        Args:
            source_catalog: SourceCatalog containing object coordinates and metadata
            epoch_keys: List of temporal grouping keys (e.g., group_name from tbl_aors)
            measurement_keys: List of measurement key strings. If None, uses all measurement keys from source_catalog
            value_keys: List of value keys (default: ['mag', 'mag_err'])
            tbl_aors: Optional AOR info table (sorted by reqkey) to attach as metadata
        """
        logger.info(f"Creating lightcurve storage at {self.storage_path}")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Use all measurement keys from catalog if not specified
        if measurement_keys is None:
            measurement_keys = source_catalog.measurement_keys
            logger.debug(f"Using all {len(measurement_keys)} measurement keys from source catalog")
        
        # Extract object keys with optional spatial ordering
        if self.enable_spatial_chunking:
            object_keys = source_catalog.get_spatially_ordered_keys()
            logger.info(f"Applied spatial ordering to {len(object_keys)} objects for better chunk locality")
        else:
            object_keys = source_catalog.object_keys
            logger.info(f"Using original object order for {len(object_keys)} objects")
        
        # Epoch keys are already strings (e.g., group_name from partition key)
        logger.debug(f"Processing {len(epoch_keys)} epochs: {epoch_keys[:5]}{'...' if len(epoch_keys) > 5 else ''}")
        
        # Use dask to create empty xarray Dataset without allocating memory
        # This is the recommended approach from xarray docs for large datasets
        import dask.array as da
        
        shape = (len(object_keys), len(measurement_keys), len(value_keys), len(epoch_keys))
        zarr_path = str(self.storage_path / "lightcurves.zarr")
        
        # Configure zarr chunking - only chunk on object_key dimension
        # Keep epochs together in the innermost dimension for efficient vector access
        zarr_chunks = (
            min(self.chunk_size, len(object_keys)),   # Spatial chunks for object_key
            len(measurement_keys),                    # Single chunk - all measurement keys together
            len(value_keys),                          # Single chunk - mag + err together  
            len(epoch_keys)                           # Single chunk - all epochs together (accessed as vectors)
        )
        
        # Create dask array with dummy values (values are irrelevant, only shape/dtype/chunks matter)
        # This doesn't allocate any memory!
        dummy_array = da.full(
            shape=shape,
            fill_value=np.nan,
            dtype=np.float32,
            chunks=zarr_chunks
        )
        
        # Create coordinates for each dimension
        coords = {
            'object': object_keys,
            'measurement': measurement_keys, 
            'value': value_keys,
            'epoch': epoch_keys
        }
        
        # Create xarray Dataset with dask-backed DataArray
        ds = xr.Dataset(
            {'lightcurves': (('object', 'measurement', 'value', 'epoch'), dummy_array)},
            coords=coords
        )
        
        # Assign coordinates directly to the DataArray in the Dataset
        ds['lightcurves'] = ds['lightcurves'].assign_coords(coords)
        ds['lightcurves'].attrs.update({
            'description': 'IDF lightcurve data',
            'created': str(np.datetime64('now')),
            'n_objects': len(object_keys),
            'n_measurement_keys': len(measurement_keys),
            'n_value_keys': len(value_keys), 
            'n_epochs': len(epoch_keys)
        })
        
        # Attach AOR info table to metadata if provided (sorted by reqkey for temporal ordering)
        if tbl_aors is not None:
            # Store as JSON-serializable dict in attrs
            ds['lightcurves'].attrs['tbl_aors_metadata'] = {
                'n_aors': len(tbl_aors),
                'columns': list(tbl_aors.colnames),
                'note': 'Full table stored separately; this confirms it was attached during creation'
            }
            # Save the actual table as a separate file in the zarr directory
            tbl_aors_path = self.storage_path / "aor_info_table.ecsv"
            tbl_aors.write(tbl_aors_path, format='ascii.ecsv', overwrite=True)
            logger.info(f"Saved AOR info table to {tbl_aors_path}")
        
        # Add coordinate arrays as non-dimension coordinates
        ra_vals, dec_vals, x_vals, y_vals = source_catalog.get_coordinate_arrays_for_objects(object_keys)
        
        # Add as non-dimension coordinates
        ds['lightcurves'] = ds['lightcurves'].assign_coords(
            ra=('object', ra_vals),
            dec=('object', dec_vals),
            x_image=('object', x_vals), 
            y_image=('object', y_vals)
        )
        
        # Write to zarr without computing array values (only metadata)
        # This is the key trick from xarray docs - creates the store structure without allocating data
        encoding = {
            'lightcurves': {
                'compressor': None,  # No compression for fast access
                'chunks': zarr_chunks
            }
        }
        logger.info(f"Writing zarr metadata (not computing array values): shape={shape}, chunks={zarr_chunks}")
        ds.to_zarr(zarr_path, mode='w', encoding=encoding, compute=False, consolidated=True)
        
        # Reopen with consolidated metadata enabled for optimal performance
        ds = xr.open_zarr(zarr_path, consolidated=True)
        self.lightcurves = ds['lightcurves']
        
        # Store chunk configuration for reference
        chunk_info = {
            'chunk_size': self.chunk_size,
            'chunks': {
                'object': zarr_chunks[0],
                'measurement': zarr_chunks[1],
                'value': zarr_chunks[2],
                'epoch': zarr_chunks[3]
            }
        }
        print(f"Saved with zarr chunking: {chunk_info} (consolidated metadata enabled)")
        
        logger.info(f"Created xarray lightcurve storage: {self.lightcurves.sizes}")
        logger.info("Storage created with optimizations:")
        logger.info("  • Used dask arrays to avoid memory allocation during creation")
        logger.info("  • Consolidated metadata enabled for faster opening")
        logger.info("  • Spatial chunking optimizes regional queries")
        logger.info("  • Lazy loading via .load_storage()")

    def load_storage(self, consolidated: bool = True):
        """Load existing xarray storage from zarr.
        
        Uses xr.open_zarr() which is the recommended way to open Zarr stores.
        
        Args:
            consolidated: Whether to use consolidated metadata.
                True (default): Requires consolidated metadata for fast reads (use after finalization).
                False: Opens without consolidated metadata (use during incremental writes).
        
        Raises:
            RuntimeError: If storage doesn't exist or consolidated metadata is missing
                when consolidated=True.
        """
        zarr_path = str(self.storage_path / "lightcurves.zarr")
        if not Path(zarr_path).exists():
            raise RuntimeError("Storage not found. Call create_storage() first.")
        
        ds = xr.open_zarr(zarr_path, consolidated=consolidated)
        
        # Extract the DataArray from the Dataset
        self.lightcurves = ds['lightcurves']
        
        logger.info(f"Loaded lightcurve storage from {self.storage_path}")
        return self.lightcurves

    def populate_epoch_from_catalog_v0(
        self, 
        epoch_key: str, 
        source_catalog: SourceCatalog, 
        measurement_keys: list[MeasurementKeyT]
    ) -> int:
        """
        Original implementation: Populate epoch data from SourceCatalog.
        
        This is the baseline version (v0) for performance comparison.
        Uses xarray .loc[] for updates then writes to zarr.
        
        Args:
            epoch_key: Temporal grouping key (e.g., group_name from partition)
            source_catalog: SourceCatalog containing measurement data
            measurement_keys: List of measurement keys to extract
            
        Returns:
            Number of measurements stored
        """
        logger.info(f"Populating epoch {epoch_key} with {len(measurement_keys)} measurement keys")
        
        # Ensure storage is loaded
        if self.lightcurves is None:
            self.load_storage(consolidated=False)
        assert self.lightcurves is not None, "Failed to load lightcurves storage"
        
        # Extract measurements for all requested keys
        logger.debug(f"Extracting measurements for keys: {measurement_keys}")
        measurements = source_catalog.extract_measurements(measurement_keys)
        valid_mask = source_catalog.get_valid_measurement_mask(measurement_keys)
        
        # Get object keys from catalog (strings, not ints)
        catalog_object_keys = source_catalog.object_keys
        
        # Filter to valid measurements only
        valid_object_keys = [catalog_object_keys[i] for i, v in enumerate(valid_mask) if v]
        logger.debug(f"Found {len(valid_object_keys)} objects with valid measurements out of {len(catalog_object_keys)} total")
        
        if len(valid_object_keys) == 0:
            logger.warning(f"No valid objects found for epoch {epoch_key}")
            return 0
        
        n_updated = 0
        
        # Update each measurement using xarray .loc[]
        for measurement_key in measurement_keys:
            if measurement_key not in measurements:
                logger.warning(f"No measurements found for key '{measurement_key}'")
                continue
                
            values = measurements[measurement_key]
            mag_data = values["mag"]
            err_data = values["mag_err"]

            valid_mag_data = mag_data[valid_mask]
            valid_err_data = err_data[valid_mask]
            
            if len(valid_mag_data) == 0:
                logger.warning(f"No valid measurements for key '{measurement_key}' in epoch {epoch_key}")
                continue
            
            # Update using xarray .loc[] (vectorized in-memory)
            self.lightcurves.loc[valid_object_keys, measurement_key, 'mag', epoch_key] = valid_mag_data
            self.lightcurves.loc[valid_object_keys, measurement_key, 'mag_err', epoch_key] = valid_err_data
            
            n_updated += len(valid_object_keys) * 2  # mag + err
            logger.debug(f"Updated {len(valid_object_keys)} measurements for key '{measurement_key}'")
        
        # Write the complete epoch to zarr storage
        if n_updated > 0:
            zarr_path = str(self.storage_path / "lightcurves.zarr")
            import zarr
            zarr_group = zarr.open_group(zarr_path, mode='r+')
            lightcurves_array = zarr_group['lightcurves']
            
            # Find epoch index and write the slice
            epoch_idx = int(self.lightcurves.epoch.to_index().get_loc(epoch_key))
            epoch_slice = self.lightcurves.isel(epoch=epoch_idx)
            lightcurves_array[:, :, :, epoch_idx] = epoch_slice.values  # type: ignore[index]
            
            logger.debug(f"Wrote epoch {epoch_key} to zarr ({n_updated} measurements)")
        
        logger.info(f"Updated {n_updated} measurements for epoch {epoch_key}")
        return n_updated

    def populate_epoch_from_catalog_v1(
        self, 
        epoch_key: str, 
        source_catalog: SourceCatalog, 
        measurement_keys: list[MeasurementKeyT]
    ) -> int:
        """
        Optimized v1 implementation: Single read-modify-write with pre-computed indices.
        
        Optimized approach:
        - Uses xarray coordinate lookups for robustness
        - Pre-computes object indices once (outside measurement loop)
        - Reads complete epoch slice once from zarr
        - Updates in memory using numpy fancy indexing
        - Writes complete epoch slice back to zarr once
        
        Args:
            epoch_key: Temporal grouping key (e.g., group_name from partition)
            source_catalog: SourceCatalog containing measurement data
            measurement_keys: List of measurement keys to extract
            
        Returns:
            Number of measurements stored
        """
        logger.info(f"Populating epoch {epoch_key} with {len(measurement_keys)} measurement keys")
        
        # Ensure storage is loaded
        if self.lightcurves is None:
            self.load_storage(consolidated=False)
        assert self.lightcurves is not None, "Failed to load lightcurves storage"
        
        # Get epoch index using xarray coordinate lookup
        epoch_idx = int(self.lightcurves.epoch.to_index().get_loc(epoch_key))
        
        # Get value indices using xarray coordinate lookup
        mag_idx = int(self.lightcurves.value.to_index().get_loc('mag'))
        err_idx = int(self.lightcurves.value.to_index().get_loc('mag_err'))
        
        # Extract measurements for all requested keys
        logger.debug(f"Extracting measurements for keys: {measurement_keys}")
        measurements = source_catalog.extract_measurements(measurement_keys)
        valid_mask = source_catalog.get_valid_measurement_mask(measurement_keys)
        
        # Get object keys from catalog (strings, not ints)
        catalog_object_keys = source_catalog.object_keys
        
        # Filter to valid measurements only
        valid_object_keys = [catalog_object_keys[i] for i, v in enumerate(valid_mask) if v]
        logger.debug(f"Found {len(valid_object_keys)} objects with valid measurements out of {len(catalog_object_keys)} total")
        
        if len(valid_object_keys) == 0:
            logger.warning(f"No valid objects found for epoch {epoch_key}")
            return 0
        
        # Build object indices ONCE using xarray coordinate lookups (OUTSIDE measurement loop!)
        # This is the key optimization - don't repeat this inside the loop
        object_index_array = self.lightcurves.object.to_index()
        object_indices = [int(object_index_array.get_loc(key)) for key in valid_object_keys]
        
        # Open zarr and read current epoch data
        zarr_path = str(self.storage_path / "lightcurves.zarr")
        import zarr
        zarr_group = zarr.open_group(zarr_path, mode='r+')
        lightcurves_array = zarr_group['lightcurves']
        
        # Read the full epoch slice once (shape: [n_objects, n_measurements, n_values])
        epoch_data = lightcurves_array[:, :, :, epoch_idx]  # type: ignore[index]
        
        n_updated = 0
        
        # Update the epoch data array in memory for each measurement
        for measurement_key in measurement_keys:
            if measurement_key not in measurements:
                logger.warning(f"No measurements found for key '{measurement_key}'")
                continue
            
            # Get measurement index using xarray coordinate lookup
            try:
                measurement_idx = int(self.lightcurves.measurement.to_index().get_loc(measurement_key))
            except KeyError:
                logger.warning(f"Measurement key '{measurement_key}' not found in storage coordinates")
                continue
                
            values = measurements[measurement_key]
            mag_data = values["mag"]
            err_data = values["mag_err"]

            valid_mag_data = mag_data[valid_mask]
            valid_err_data = err_data[valid_mask]
            
            if len(valid_mag_data) == 0:
                logger.warning(f"No valid measurements for key '{measurement_key}' in epoch {epoch_key}")
                continue
            
            # Use pre-computed object_indices (computed once outside this loop)
            # Update in-memory epoch array
            epoch_data[object_indices, measurement_idx, mag_idx] = valid_mag_data  # type: ignore[index]
            epoch_data[object_indices, measurement_idx, err_idx] = valid_err_data  # type: ignore[index]
            
            n_updated += len(object_indices) * 2  # mag + err
            logger.debug(f"Updated {len(object_indices)} measurements for key '{measurement_key}'")
        
        # Write the complete epoch slice back to zarr in ONE operation
        if n_updated > 0:
            lightcurves_array[:, :, :, epoch_idx] = epoch_data  # type: ignore[index]
            logger.debug(f"Wrote epoch {epoch_key} slice to zarr ({n_updated} measurements)")
        
        logger.info(f"Updated {n_updated} measurements for epoch {epoch_key}")
        return n_updated

    def populate_epoch_from_catalog_v2(
        self, 
        epoch_key: str, 
        source_catalog: SourceCatalog, 
        measurement_keys: list[MeasurementKeyT]
    ) -> int:
        logger.info(f"Populating epoch {epoch_key} with {len(measurement_keys)} measurement keys")
        
        # Ensure storage is loaded
        if self.lightcurves is None:
            self.load_storage(consolidated=False)
        assert self.lightcurves is not None, "Failed to load lightcurves storage"
        lightcurves = self.lightcurves

        # this is a dict of measurement_key -> measurements for this epoch
        measurements = source_catalog.extract_measurements(
            [k for k in measurement_keys if k in source_catalog.measurement_keys]
        )
        m_keys = list(measurements.keys())
        v_keys = source_catalog.value_keys
        logger.debug(f"Extracted measurements keys: {m_keys}")
        logger.debug(f"Value keys: {v_keys}")
        logger.debug(f"Number of catalog measurements: {len(source_catalog.measurement_keys)}")
        logger.debug(f"First measurement sample: {measurements[m_keys[0]] if m_keys else 'NO MEASUREMENTS'}")
        # m_arr shape is (n_measurements, n_values, n_objects)
        m_arr = np.array([
            [v[vk] for vk in v_keys]
            for v in measurements.values()
        ])
        # TODO revisit this this calls extract meaurements again, which is inefficient.
        m_valid = source_catalog.get_valid_measurement_mask(m_keys)
        logger.debug(f"m_valid mask: {m_valid[:10]} (first 10), sum={m_valid.sum()}")
        
        object_keys = source_catalog.object_keys
        o_keys_valid = [object_keys[i] for i, v in enumerate(m_valid) if v]
        logger.debug(f"Found {len(o_keys_valid)} objects with valid measurements out of {len(object_keys)} total")
        logger.debug(f"Catalog object_keys (first 10): {object_keys[:10]}")
        logger.debug(f"Storage object coords (first 10): {lightcurves.coords['object'].values[:10].tolist()}")
        logger.debug(f"Valid object keys to match (first 10): {o_keys_valid[:10]}")

        if len(o_keys_valid) == 0:
            logger.warning(f"No valid objects found for epoch {epoch_key}")
            return 0

        # Filter m_arr to only valid objects
        # m_arr is (n_measurements, n_values, n_objects), m_valid is (n_objects,)
        m_arr_valid = m_arr[:, :, m_valid]

        # use a helper function to build index from labels
        def get_oindex_from_xarray_sel(xarr, indexers):
            from xarray.core.indexing import map_index_queries
            dim_indexers = map_index_queries(xarr, indexers).dim_indexers
            return tuple(dim_indexers.get(dim, slice(None)) for dim in xarr.variable.dims)

        oindex = get_oindex_from_xarray_sel(
             lightcurves,
             {
                  "object": o_keys_valid,
                  "measurement": m_keys,
                  "value": v_keys,
                  "epoch": epoch_key
             }
             )

        # Open zarr and write data 
        zarr_path = str(self.storage_path / "lightcurves.zarr")
        import zarr
        zarr_group = zarr.open_group(zarr_path, mode='r+')
        lightcurves_array = zarr_group['lightcurves']
        assert type(lightcurves_array) is zarr.Array
        
        m_arr_transposed = np.moveaxis(m_arr_valid, -1, 0)
        
        lightcurves_array.oindex[oindex] = m_arr_transposed
        
        n_updated = m_arr_valid.size
        logger.debug(f"Wrote epoch {epoch_key} slice to zarr ({n_updated} measurements)")
        logger.info(f"Updated {n_updated} measurements for epoch {epoch_key}")
        return n_updated


    def populate_epoch_from_catalog(
        self, 
        epoch_key: str, 
        source_catalog: SourceCatalog, 
        measurement_keys: list[MeasurementKeyT]
    ) -> int:
        """
        Populate epoch data from SourceCatalog (currently uses v1 implementation).
        
        This is the default method that delegates to the current best implementation.
        See populate_epoch_from_catalog_v0, v1, v2 for specific versions.
        
        Args:
            epoch_key: Temporal grouping key (e.g., group_name from partition)
            source_catalog: SourceCatalog containing measurement data
            measurement_keys: List of measurement keys to extract
            
        Returns:
            Number of measurements stored
        """
        return self.populate_epoch_from_catalog_v1(epoch_key, source_catalog, measurement_keys)

    def get_object_lightcurve(self, object_key: str, measurement_key: str | None = None,
                            value_key: str | None = None) -> xr.DataArray:
        """
        Get lightcurve data for a specific object using optimized labeled indexing.
        
        Uses .sel() which returns a view when possible (fast, no copy).
        For frequently accessed data, consider calling .load() on the result.
        
        Args:
            object_key: Object key (string identifier)
            measurement_key: Optional measurement key filter
            value_key: Optional value key filter ('mag' or 'mag_err')
            
        Returns:
            xarray DataArray with requested lightcurve data (view when possible)
        """
        if self.lightcurves is None:
            self.load_storage()
        assert self.lightcurves is not None, "Failed to load lightcurves storage"
        
        # Use xarray's label-based selection with .sel()
        # This returns a view for single label selections (fast, efficient)
        selection: dict[str, str] = {'object': object_key}
        if measurement_key is not None:
            selection['measurement'] = measurement_key
        if value_key is not None:
            selection['value'] = value_key
            
        return self.lightcurves.sel(selection)

    def get_epoch_data(self, epoch_key: str, measurement_key: str | None = None,
                     value_key: str | None = None) -> xr.DataArray:
        """
        Get all object data for a specific epoch using optimized labeled indexing.
        
        Uses .sel() which returns a view when possible (fast, no copy).
        For large epoch datasets, consider calling .load() to bring data into memory.
        
        Args:
            epoch_key: Temporal grouping key
            measurement_key: Optional measurement key filter
            value_key: Optional value key filter ('mag' or 'mag_err')
            
        Returns:
            xarray DataArray with epoch data (view when possible)
        """
        if self.lightcurves is None:
            self.load_storage()
        assert self.lightcurves is not None, "Failed to load lightcurves storage"
            
        # Use xarray's label-based selection with .sel()
        # This returns a view for single label selections (fast, efficient)
        selection: dict[str, str] = {'epoch': epoch_key}
        if measurement_key is not None:
            selection['measurement'] = measurement_key
        if value_key is not None:
            selection['value'] = value_key
            
        return self.lightcurves.sel(selection)

    def get_objects_in_region(self, ra_range: tuple, dec_range: tuple) -> list[str]:
        """
        Get object keys within a coordinate range using optimized vectorized operations.
        
        Uses xarray's vectorized boolean indexing which is highly efficient.
        Coordinate arrays are 1D, so this operation is fast even for large catalogs.
        
        Args:
            ra_range: (min_ra, max_ra) in degrees
            dec_range: (min_dec, max_dec) in degrees
            
        Returns:
            List of object keys (strings) in the region
        """
        if self.lightcurves is None:
            self.load_storage()
        assert self.lightcurves is not None, "Failed to load lightcurves storage"
            
        # Check if coordinate data exists
        if 'ra' not in self.lightcurves.coords or 'dec' not in self.lightcurves.coords:
            print("Warning: No coordinate data available for spatial queries")
            return []
        
        # Vectorized coordinate filtering using xarray
        # These operations create boolean masks without copying data (efficient)
        ra_mask = (self.lightcurves.ra >= ra_range[0]) & (self.lightcurves.ra <= ra_range[1])
        dec_mask = (self.lightcurves.dec >= dec_range[0]) & (self.lightcurves.dec <= dec_range[1])
        
        # Combine masks and get object keys
        # .where() with drop=True efficiently filters coordinates
        region_mask = ra_mask & dec_mask
        object_keys = self.lightcurves.object.where(region_mask, drop=True)
        
        return object_keys.values.tolist()

    def get_storage_info(self) -> dict[str, Any]:
        """Get comprehensive storage information using xarray metadata."""
        if self.lightcurves is None:
            try:
                self.load_storage()
            except RuntimeError:
                return {"status": "not_created", "storage_path": str(self.storage_path)}
        
        assert self.lightcurves is not None, "Failed to load lightcurves storage"
        info = {
            "status": "ready",
            "storage_path": str(self.storage_path),
            "data_shape": dict(self.lightcurves.sizes),
            "dimensions": list(self.lightcurves.dims),
            "coordinates": list(self.lightcurves.coords.keys()),
            "measurement_keys": self.lightcurves.measurement.values.tolist(),
            "value_keys": self.lightcurves.value.values.tolist(),
            "n_epochs": len(self.lightcurves.epoch),
            "n_measurements": len(self.lightcurves.measurement),
            "n_objects": len(self.lightcurves.object),
            "created": self.lightcurves.attrs.get('created', 'unknown'),
            "description": self.lightcurves.attrs.get('description', ''),
            "data_type": str(self.lightcurves.dtype)
        }
        
        # Add coordinate ranges if available
        if 'ra' in self.lightcurves.coords:
            ra_vals = self.lightcurves.ra.values
            valid_ra = ra_vals[~np.isnan(ra_vals)]
            if len(valid_ra) > 0:
                info['ra_range'] = [float(valid_ra.min()), float(valid_ra.max())]
                
        if 'dec' in self.lightcurves.coords:
            dec_vals = self.lightcurves.dec.values  
            valid_dec = dec_vals[~np.isnan(dec_vals)]
            if len(valid_dec) > 0:
                info['dec_range'] = [float(valid_dec.min()), float(valid_dec.max())]
        
        return info



    def append_epoch_data(self, epoch_key: str, catalog: Table, measurement_keys: list[MeasurementKeyT]) -> int:
        """Append data for a new epoch to existing zarr storage using region writes."""
        if self.lightcurves is None:
            self.load_storage()
        assert self.lightcurves is not None, "Failed to load lightcurves storage"
        
        # Check if epoch already exists
        if epoch_key in self.lightcurves.epoch_key.values:
            logger.warning(f"Epoch {epoch_key} already exists, overwriting")
            source_catalog = SourceCatalog(catalog)
            return self.populate_epoch_from_catalog(epoch_key, source_catalog, measurement_keys)
        
        # For appending new epochs, we would need to extend the array along the epoch_key dimension
        # This is a placeholder for future implementation using append_dim
        print(f"Note: append_epoch_data not fully implemented. Use populate_epoch_from_catalog for existing epochs.")
        return 0

