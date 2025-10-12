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
from typing import Any, cast, TypedDict, Literal, ClassVar
from astropy.table import Table
import re
from datetime import datetime
from loguru import logger
from ..utils.naming import NameTemplate
from .catalog import SourceCatalog, SourceCatalogDataKey
import dataclasses

__all__ = ['LightcurveStorage']


DimNameT = Literal["object", "measurement", "value", "epoch"]


@dataclasses.dataclass
class LightcurveStorage:
    """xarray-based lightcurve storage with labeled dimensions and vectorized operations.
    
    Follows xarray + Zarr best practices:
    - Uses consolidated metadata for faster opening
    - Optimized chunking strategy for spatial locality
    - Zarr v3 format for latest features
    - Support for region-based writes for distributed processing
    """
    storage_path: Path
    sort_objects_by_position: bool = True
    lightcurves: xr.DataArray | None = dataclasses.field(init=False, default=None)
    _zarr_path_loaded : Path |None  = dataclasses.field(init=False, default=None)
    _zarr_path_for_write : Path  = dataclasses.field(init=False)
    _zarr_path_for_read : Path  = dataclasses.field(init=False)
    
    dim_names: ClassVar[tuple[DimNameT, ...]] = ("object", "measurement", "value", "epoch")
    lc_var_name: ClassVar[Literal["lightcurves"]] = "lightcurves"
    
    def __post_init__(self):
        self._zarr_path_for_write = self.storage_path / f"{self.lc_var_name}_write.zarr"
        self._zarr_path_for_read = self.storage_path / f"{self.lc_var_name}_read.zarr"
    
    @property
    def zarr_path_for_write(self):
        return self._zarr_path_for_write

    @property
    def zarr_path_for_read(self):
        return self._zarr_path_for_read

    def is_loaded(self):
        return self.lightcurves is not None

    def _validate_dim_args(self, source_catalog, dim_name, dim_keys, dim_vars):
        if dim_keys is None:
            logger.debug(f"use all {dim_name}_keys from source catalog {source_catalog}")
            dim_keys = getattr(source_catalog, f"{dim_name}_keys") 
            if not dim_keys:
                raise ValueError("no keys specified for {dim_name}")
        n_dim_keys = len(dim_keys)
        for var in dim_vars:
            if len(var) != n_dim_keys:
                raise ValueError(f"mismatch size between dim_keys ({n_dim_keys}) and dim_var {var} ({len(var)}).")
        return n_dim_keys, dim_keys, dim_vars

    def _create(
        self,
        zarr_path: Path,
        source_catalog: SourceCatalog,
        measurement_keys: list[str] | None = None,
        value_keys: list[str] | None = None,
        epoch_keys: list[str] | None = None,
        dim_vars: dict[DimNameT, dict[str, np.ndarray]] | None = None,
        zarr_chunks: dict[DimNameT, int] | None = None 
    ) -> None:
        """
        Create xarray-based storage with labeled dimensions.
        
        Args:
            source_catalog: SourceCatalog containing object coordinates and metadata
            epoch_keys: List of temporal grouping keys (e.g., group_name from tbl_aors)
            measurement_keys: List of measurement key strings. If None, uses all measurement keys from source_catalog
            value_keys: List of value keys (default: ['mag', 'mag_err'])
            dim_vars: additional variables to store for each dim.
        """
        logger.info(f"Creating lightcurve storage at {self.storage_path}")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Use all measurement keys from catalog if not specified
        if dim_vars is None:
            dim_vars = {}
        if zarr_chunks is None:
            zarr_chunks = {}

        n_measurements, measurement_keys, measurement_vars = self._validate_dim_args(source_catalog, "measurement", measurement_keys, dim_vars.get("measurement", {}))
        n_values, value_keys, value_vars = self._validate_dim_args(source_catalog, "value", value_keys, dim_vars.get("value", {}))
        n_epochs, epoch_keys, epoch_vars = self._validate_dim_args(source_catalog, "epoch", epoch_keys, dim_vars.get("epoch", {}))

        if self.sort_objects_by_position:
            source_catalog.sort_objects_by_position()
            logger.info(f"Using spatial ordering for objects.")
        else:
            logger.info(f"Using original ordering for objects")
        object_keys = source_catalog.object_keys
        n_objects = len(object_keys)
        logger.info(f"create storage for: {n_objects=} {n_measurements=} {n_values=} {n_epochs=}")
        
        import dask.array as da
        
        dim_names = self.dim_names
        dim_sizes : dict [DimNameT, int] = {
            "object": n_objects,
            "measurement": n_measurements,
            "value": n_values,
            "epoch": n_epochs,
        }
        shape = tuple(dim_sizes[dn] for dn in dim_names)
        
        arr_placeholder = da.full(
            shape=shape,
            fill_value=np.nan,
            dtype=np.float32,
        )
        arr_coords = {
            'object': object_keys,
            'measurement': measurement_keys, 
            'value': value_keys,
            'epoch': epoch_keys
        }
        lc_var_name = "lightcurves"
        ds = xr.Dataset(
            {lc_var_name: (dim_names, arr_placeholder)},
            coords=arr_coords
        )
        lc_var = ds[lc_var_name]
        lc_var.attrs.update({
            'description': 'IDF lightcurve data',
            'created': str(np.datetime64('now')),
        } | {f"n_{dn}s": dim_sizes[dn] for dn in dim_names})
        
        # attach dim vars
        for dn, vars in [
            ("measurement", measurement_vars),
            ("value", value_vars),
            ("epoch", epoch_vars),
                         ]:
            for name, array in vars.items():
                ds[f"{dn}_{name}"] = ((dn, ), array)

        # Add coordinate arrays as non-dimension coordinates
        # TODO change this to make it more concise
        ra_vals, dec_vals, x_vals, y_vals = source_catalog.get_coordinate_arrays_for_objects(object_keys.tolist())
        
        lc_var = lc_var.assign_coords(
            ra=('object', ra_vals),
            dec=('object', dec_vals),
            x_image=('object', x_vals),
            y_image=('object', y_vals)
        )
        # update ds with updated variables
        ds[lc_var_name] = lc_var

        # Write to zarr without computing array values (only metadata)
        # This is the key trick from xarray docs - creates the store structure without allocating data
        _zarr_chunks = tuple(
            zarr_chunks.get(dn, dim_sizes[dn])
            for dn in dim_names
        )
        logger.info(f"Writing zarr metadata: shape={shape}, chunk_shape={_zarr_chunks}")
        encoding = {
            "compressors": None,
            "chunks": _zarr_chunks,
        }
        ds.to_zarr(zarr_path, mode='w', encoding=encoding, compute=False, consolidated=True)

    def create_for_per_epoch_write(self, *args, **kwargs):
        # set epoch chunk size to 1 for parallel writing per-epoch.
        zarr_chunks = kwargs.pop("zarr_chunks", {})
        zarr_chunks["epoch"] = 1
        kwargs["zarr_chunks"] = zarr_chunks
        return self._create(self.zarr_path_for_write, *args, **kwargs)

    def rechunk_for_per_object_read(self, chunk_size=1000):
        zarr_path_for_write = self.zarr_path_for_write
        if not zarr_path_for_write.exists():
            raise RuntimeError(f"Storage for write not created yet.")
        arr = da.from_zarr(zarr_path_for_write)
        logger.info(f"storage chunks for write: {arr.chunks}")

        chunks_out = list(arr.shape)
        obj_dim_idx = self.dim_names.index("object")
        chunks_out[obj_dim_idx] = chunk_size
        chunks_out = tuple(chunks_out)
        rechunked = arr.rechunk(chunks_out)
        logger.info(f"rechunked storage chunks for read: {rechunked.chunks}")
        arr.to_zarr(self.zarr_path_for_read, overwrite=True)

    def _load(self, zarr_path, consolidated: bool = True):
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
        if not zarr_path.exists():
            raise RuntimeError(f"Storage not created yet: {zarr_path}")
        
        ds = xr.open_zarr(zarr_path, consolidated=consolidated)
        lc_var = self.lightcurves = ds['lightcurves']
        logger.info(f"Loaded lightcurve storage from {zarr_path}")
        self._zarr_path_loaded = zarr_path
        return lc_var

    def load_for_per_object_read(self):
        return self._load(self.zarr_path_for_read, consolidated=True)
        
    def load_for_per_epoch_write(self):
        return self._load(self.zarr_path_for_write, consolidated=False)

    def close(self):
        if self.lightcurves is not None:
            self.lightcurves.close()
            self.lightcurves = None
            self._zarr_path_loaded = None

    def populate_epoch(
        self, 
        source_catalog: SourceCatalog, 
        epoch_key: str, 
    ) -> int:
        logger.info(f"Populating epoch {epoch_key} with {source_catalog}")

        cat_object_keys = source_catalog.object_keys
        cat_measurement_keys = source_catalog.measurement_keys
        cat_value_keys = source_catalog.value_keys
        cat_epoch_keys = source_catalog.epoch_keys

        if not cat_epoch_keys:
            logger.debug(f"no epoch key found in {source_catalog}, assume {epoch_key=}")
        else:
            if epoch_key not in cat_epoch_keys:
                raise ValueError(f"no matching {epoch_key=} in {source_catalog}")
        if not self.is_loaded():
            self.load_for_per_epoch_write()
        lc_var = self.lightcurves
        assert lc_var is not None, "Failed to load lightcurves storage"

        lc_measurment_keys = lc_var.coords["measurement"].values
        lc_value_keys = lc_var.coords["value"].values
        lc_epoch_keys = lc_var.coords["epoch"].values

        data_measurement_keys = list(cat_measurement_keys.intersection(lc_measurment_keys))
        data_value_keys = list(cat_value_keys.intersection(lc_value_keys))
        data_epoch_keys = list({epoch_key}.intersection(lc_epoch_keys))
        data_shape = (len(data_measurement_keys), len(data_value_keys), len(data_epoch_keys))

        if 0 in data_shape:
            logger.error("no matching data in storage for update: {lc_var}")
            return 0

        data = source_catalog.data

        # pack data into orgthogonal indexing format (n_measurements, n_values, n_epochs, n_objs)
        data_packed = np.full(
            data_shape,
            None,
            dtype=object
            )
        for i, mk in enumerate(data_measurement_keys):
            for j, vk in enumerate(data_value_keys):
                if not cat_epoch_keys:
                    # use the data as epoch_key if cat is not epoch-aware
                    ek = None
                else:
                    ek = epoch_key
                data_packed[i, j, 1] = data[SourceCatalogDataKey(measurement=mk, value=vk, epoch=ek)]
        # check there is no None in it
        if None in data_packed:
            raise ValueError(f"cannot pack catalog data into oindex format: {source_catalog}")
        data_packed = np.array(data_packed.tolist())

        # use a helper function to build index from labels
        def get_oindex_from_xarray_sel(xarr, indexers):
            from xarray.core.indexing import map_index_queries
            dim_indexers = map_index_queries(xarr, indexers).dim_indexers
            return tuple(dim_indexers.get(dim, slice(None)) for dim in xarr.variable.dims)

        oindex = get_oindex_from_xarray_sel(
             lc_var,
             {
                  "object": source_catalog.object_keys.tolist(),
                  "measurement": data_measurement_keys,
                  "value": data_value_keys,
                  "epoch": data_epoch_keys,
             }
             )

        # close the store to open it with zarr directly.
        zarr_path = self._zarr_path_loaded
        self.close()
        zarr_group = zarr.open_group(zarr_path, mode='r+')
        lc_arr = zarr_group[self.lc_var_name]
        assert type(lc_arr) is zarr.Array
        
        # move the axis to the correct order (object, measurement, value, epoch)
        data_transposed = np.moveaxis(data_packed, -1, 0)
        
        lc_arr.oindex[oindex] = data_transposed
        
        n_updated = data_transposed.size
        logger.info(f"Updated {n_updated} measurements for epoch {epoch_key}")
        return n_updated

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
            self.load_for_per_object_read()
        assert self.lightcurves is not None, "Failed to load lightcurves storage"
        
        # Use xarray's label-based selection with .sel()
        # This returns a view for single label selections (fast, efficient)
        selection: dict[str, str] = {'object': object_key}
        if measurement_key is not None:
            selection['measurement'] = measurement_key
        # This gives (n_values, n_epochs)
        return self.lightcurves.sel(selection)

    def get_epoch_data(self, epoch_key: str, measurement_key: str | None = None) -> xr.DataArray:
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
            self.load_for_per_epoch_write()
        assert self.lightcurves is not None, "Failed to load lightcurves storage"
            
        # Use xarray's label-based selection with .sel()
        # This returns a view for single label selections (fast, efficient)
        selection: dict[str, str] = {'epoch': epoch_key}
        if measurement_key is not None:
            selection['measurement'] = measurement_key
        # This gives (n_objs, n_values)
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
            self.load_for_per_object_read()
        assert self.lightcurves is not None, "Failed to load lightcurves storage"
            
        # Check if coordinate data exists
        if 'ra' not in self.lightcurves.coords or 'dec' not in self.lightcurves.coords:
            print("Warning: No coordinate data available for spatial queries")
            return []
        
        # Vectorized coordinate filtering using xarray
        # These operations create boolean masks without copying data (efficient)
        # TODO handle wrapping of the ra coordiantes
        ra_mask = (self.lightcurves.ra >= ra_range[0]) & (self.lightcurves.ra <= ra_range[1])
        dec_mask = (self.lightcurves.dec >= dec_range[0]) & (self.lightcurves.dec <= dec_range[1])
        
        # Combine masks and get object keys
        # .where() with drop=True efficiently filters coordinates
        region_mask = ra_mask & dec_mask
        object_keys = self.lightcurves.object.where(region_mask, drop=True)
        
        return object_keys.values.tolist()

    def get_storage_info(self, which: Literal["read", "write"]) -> dict[str, Any]:
        """Get comprehensive storage information using xarray metadata."""
        if self.lightcurves is None:
            try:
                if which == "read":
                    self.load_for_per_object_read()
                elif which == "write":
                    self.load_for_per_epoch_write()
                else:
                    assert False
            except RuntimeError:
                return {"status": "not_created", "storage_path": str(self.storage_path)}
        
        assert self.lightcurves is not None, "Failed to load lightcurves storage"
        info = {
            "status": "ready",
            "storage_path": str(self.storage_path),
            "zarr_path": str(self._zarr_path_loaded),
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



    def append_epoch_data(self, epoch_key: str, catalog: Table, measurement_keys: list[str]) -> int:
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

