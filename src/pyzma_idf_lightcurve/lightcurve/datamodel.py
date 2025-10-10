"""
IDF Lightcurve Storage with xarray-based labeled access and vectorized operations.

Uses xarray DataArray with labeled dimensions for intuitive access:
- object_id: Individual object identifiers 
- measurement_type: IDF measurement types (e.g., 'ch1_sci_clean-auto')
- value_type: 'mag' or 'mag_err'
- aor: AOR identifiers

Eliminates spatial chunking complexity and enables efficient vectorized updates.
"""

import numpy as np
import xarray as xr
import zarr
from pathlib import Path
from typing import Any, cast, TypedDict
from astropy.table import Table
import re
from datetime import datetime
from loguru import logger
from ..utils.naming import NameTemplate
from .catalog import SourceCatalog

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
        aor_ids: list[str] | list[int],
        measurement_types: list[str],
        value_types: list[str] = ['mag', 'mag_err']
    ) -> None:
        """
        Create xarray-based storage with labeled dimensions.
        
        Args:
            source_catalog: SourceCatalog containing object coordinates and metadata
            aor_ids: List of AOR identifiers
            measurement_types: List of measurement type strings
            value_types: List of value types (default: ['mag', 'mag_err'])
        """
        logger.info(f"Creating lightcurve storage at {self.storage_path}")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Extract object IDs with optional spatial ordering
        if self.enable_spatial_chunking:
            object_ids = source_catalog.get_spatially_ordered_ids()
            logger.info(f"Applied spatial ordering to {len(object_ids)} objects for better chunk locality")
        else:
            object_ids = source_catalog.object_ids.tolist()
            logger.info(f"Using original object order for {len(object_ids)} objects")
        
        # Convert AOR IDs to consistent format
        aor_ids = [str(aor_id) for aor_id in aor_ids]
        logger.debug(f"Processing {len(aor_ids)} AORs: {aor_ids[:5]}{'...' if len(aor_ids) > 5 else ''}")
        
        # Create 4D xarray DataArray with labeled dimensions
        shape = (len(object_ids), len(measurement_types), len(value_types), len(aor_ids))
        
        # Initialize with NaN values
        data = np.full(shape, np.nan, dtype=np.float32)
        
        # Create coordinates for each dimension
        coords = {
            'object_id': object_ids,
            'measurement_type': measurement_types, 
            'value_type': value_types,
            'aor': aor_ids
        }
        
        # Create labeled DataArray
        self.lightcurves = xr.DataArray(
            data,
            coords=coords,
            dims=['object_id', 'measurement_type', 'value_type', 'aor'],
            name='lightcurves',
            attrs={
                'description': 'IDF lightcurve data',
                'created': str(np.datetime64('now')),
                'n_objects': len(object_ids),
                'n_measurement_types': len(measurement_types),
                'n_value_types': len(value_types), 
                'n_aors': len(aor_ids)
            }
        )
        
        # Add coordinate arrays as non-dimension coordinates
        ra_vals, dec_vals, x_vals, y_vals = source_catalog.get_coordinate_arrays_for_objects(object_ids)
        
        # Add as non-dimension coordinates
        self.lightcurves = self.lightcurves.assign_coords(
            ra=('object_id', ra_vals),
            dec=('object_id', dec_vals),
            x_image=('object_id', x_vals), 
            y_image=('object_id', y_vals)
        )
        
        # Save to zarr with optimized chunking strategy
        zarr_path = str(self.storage_path / "lightcurves.zarr")
        
        # Configure zarr chunking for spatial locality and memory efficiency
        # Zarr chunks must be a tuple matching the dimension order: (object_id, measurement_type, value_type, aor)
        zarr_chunks = (
            min(self.chunk_size, len(object_ids)),               # Spatial chunks for object_id
            len(measurement_types),                              # Single chunk - keep all types together
            len(value_types),                                    # Single chunk - mag + err together  
            min(len(aor_ids), 10) if len(aor_ids) > 10 else len(aor_ids)  # Limit AOR chunk size
        )
        
        encoding = {
            'lightcurves': {
                'compressors': None,      # Disable compression for fast access
                'chunks': zarr_chunks     # Optimized chunk sizes as tuple
            }
        }
        
        # Use consolidated=True for better performance (recommended in xarray docs)
        self.lightcurves.to_zarr(zarr_path, mode='w', encoding=encoding, consolidated=True)
        
        chunk_info = {
            'object_id': zarr_chunks[0],
            'measurement_type': zarr_chunks[1], 
            'value_type': zarr_chunks[2],
            'aor': zarr_chunks[3]
        }
        print(f"Saved with zarr chunking: {chunk_info} (consolidated metadata enabled)")
        
        logger.info(f"Created xarray lightcurve storage: {self.lightcurves.sizes}")
        logger.info("Storage created with optimizations:")
        logger.info("  • Consolidated metadata enabled for faster opening")
        logger.info("  • Spatial chunking optimizes regional queries")
        logger.info("  • Use .load_storage() for lazy loading")

    def load_storage(self):
        """Load existing xarray storage from zarr with optimized settings."""
        zarr_path = str(self.storage_path / "lightcurves.zarr")
        if not Path(zarr_path).exists():
            raise RuntimeError("Storage not found. Call create_storage() first.")
        
        # Try to open with consolidated metadata first (faster), fallback if needed
        try:
            self.lightcurves = xr.open_dataarray(zarr_path, engine='zarr', consolidated=True)
        except Exception:
            # Fallback to non-consolidated if consolidated metadata not available
            self.lightcurves = xr.open_dataarray(zarr_path, engine='zarr', consolidated=False)
            print("Warning: Opened zarr store without consolidated metadata (slower)")
        
        return self.lightcurves

    def populate_aor_from_catalog(
        self, 
        aor_id: str, 
        source_catalog: SourceCatalog, 
        measurement_types: list[str]
    ) -> int:
        """
        Populate AOR data from SourceCatalog with vectorized assignment.
        
        Args:
            aor_id: AOR identifier
            source_catalog: SourceCatalog containing measurement data
            measurement_types: List of measurement types to extract
            
        Returns:
            Number of measurements stored
        """
        logger.info(f"Populating AOR {aor_id} with {len(measurement_types)} measurement types")
        if self.lightcurves is None:
            self.load_storage()
        assert self.lightcurves is not None, "Failed to load lightcurves storage"
        
        # Extract measurements for all requested types
        logger.debug(f"Extracting measurements for types: {measurement_types}")
        measurements = source_catalog.extract_measurements(measurement_types)
        valid_mask = source_catalog.get_valid_measurement_mask(measurement_types)
        
        n_updated = 0
        
        # Get object IDs from catalog
        catalog_object_ids = source_catalog.object_ids
        
        # Filter to valid measurements only
        valid_object_ids = catalog_object_ids[valid_mask]
        logger.debug(f"Found {len(valid_object_ids)} objects with valid measurements out of {len(catalog_object_ids)} total")
        
        # Process each measurement type
        for measurement_type in measurement_types:
            if measurement_type not in measurements:
                logger.warning(f"No measurements found for type '{measurement_type}'")
                continue
                
            values = measurements[measurement_type]
            mag_data = values["mag"]
            err_data = values["mag_err"]

            valid_mag_data = mag_data[valid_mask]
            valid_err_data = err_data[valid_mask]
            
            if len(valid_mag_data) == 0:
                logger.warning(f"No valid measurements for type '{measurement_type}' in AOR {aor_id}")
                continue
                
            logger.debug(f"Processing {len(valid_mag_data)} valid measurements for {measurement_type}")
            
            # Vectorized assignment using xarray's label-based indexing
            # This is much more efficient than looping through individual assignments
            try:
                # Assign magnitude values
                self.lightcurves.loc[
                    dict(object_id=valid_object_ids, 
                         measurement_type=measurement_type,
                         value_type='mag',
                         aor=aor_id)
                ] = valid_mag_data
                
                # Assign error values  
                self.lightcurves.loc[
                    dict(object_id=valid_object_ids,
                         measurement_type=measurement_type, 
                         value_type='mag_err',
                         aor=aor_id)
                ] = valid_err_data
                
                n_updated += len(valid_object_ids) * 2  # mag + mag_err
                logger.debug(f"Successfully assigned {len(valid_object_ids)} measurements for {measurement_type}")
                
            except (KeyError, ValueError) as e:
                logger.error(f"Failed to assign {measurement_type} for AOR {aor_id}: {e}")
                continue
        
        logger.info(f"Updated {n_updated} measurements for AOR {aor_id}")
        return n_updated

    def save_storage(self):
        """Save current lightcurves to zarr storage with optimized settings."""
        if self.lightcurves is None:
            raise RuntimeError("No lightcurves to save. Create or load storage first.")
        
        zarr_path = str(self.storage_path / "lightcurves.zarr")
        
        # Use same chunking strategy as create_storage if available
        if hasattr(self, 'chunk_size'):
            zarr_chunks = (
                min(self.chunk_size, len(self.lightcurves.object_id)),
                len(self.lightcurves.measurement_type),
                len(self.lightcurves.value_type),
                min(len(self.lightcurves.aor), 10) if len(self.lightcurves.aor) > 10 else len(self.lightcurves.aor)
            )
            encoding = {
                'lightcurves': {
                    'compressors': None,
                    'chunks': zarr_chunks
                }
            }
        else:
            encoding = {'lightcurves': {'compressors': None}}
        
        # Use consolidated metadata for better performance
        self.lightcurves.to_zarr(zarr_path, mode='w', encoding=encoding, consolidated=True)

    def get_object_lightcurve(self, object_id: int, measurement_type: str | None = None,
                            value_type: str | None = None) -> xr.DataArray:
        """
        Get lightcurve data for a specific object using labeled indexing.
        
        Args:
            object_id: Object identifier
            measurement_type: Optional measurement type filter
            value_type: Optional value type filter ('mag' or 'mag_err')
            
        Returns:
            xarray DataArray with requested lightcurve data
        """
        if self.lightcurves is None:
            self.load_storage()
        assert self.lightcurves is not None, "Failed to load lightcurves storage"
        assert self.lightcurves is not None, "Failed to load lightcurves storage"
        
        # Use xarray's label-based selection  
        selection: dict[str, int | str] = {'object_id': object_id}
        if measurement_type is not None:
            selection['measurement_type'] = measurement_type
        if value_type is not None:
            selection['value_type'] = value_type
            
        return self.lightcurves.sel(selection)

    def get_aor_data(self, aor_id: str, measurement_type: str | None = None,
                    value_type: str | None = None) -> xr.DataArray:
        """
        Get all object data for a specific AOR using labeled indexing.
        
        Args:
            aor_id: AOR identifier
            measurement_type: Optional measurement type filter  
            value_type: Optional value type filter
            
        Returns:
            xarray DataArray with AOR data
        """
        if self.lightcurves is None:
            self.load_storage()
        assert self.lightcurves is not None, "Failed to load lightcurves storage"
            
        selection: dict[str, int | str] = {'aor': aor_id}
        if measurement_type is not None:
            selection['measurement_type'] = measurement_type
        if value_type is not None:
            selection['value_type'] = value_type
            
        return self.lightcurves.sel(selection)

    def get_objects_in_region(self, ra_range: tuple, dec_range: tuple) -> list[int]:
        """
        Get object IDs within a coordinate range using vectorized operations.
        
        Args:
            ra_range: (min_ra, max_ra) in degrees
            dec_range: (min_dec, max_dec) in degrees
            
        Returns:
            List of object IDs in the region
        """
        if self.lightcurves is None:
            self.load_storage()
        assert self.lightcurves is not None, "Failed to load lightcurves storage"
            
        # Check if coordinate data exists
        if 'ra' not in self.lightcurves.coords or 'dec' not in self.lightcurves.coords:
            print("Warning: No coordinate data available for spatial queries")
            return []
        
        # Vectorized coordinate filtering using xarray
        ra_mask = (self.lightcurves.ra >= ra_range[0]) & (self.lightcurves.ra <= ra_range[1])
        dec_mask = (self.lightcurves.dec >= dec_range[0]) & (self.lightcurves.dec <= dec_range[1])
        
        # Combine masks and get object_ids
        region_mask = ra_mask & dec_mask
        object_ids = self.lightcurves.object_id.where(region_mask, drop=True)
        
        return object_ids.values.tolist()

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
            "measurement_types": self.lightcurves.measurement_type.values.tolist(),
            "value_types": self.lightcurves.value_type.values.tolist(),
            "n_aors": len(self.lightcurves.aor),
            "n_objects": len(self.lightcurves.object_id),
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



    def append_aor_data(self, aor_id: str, catalog: Table, measurement_types: list[str]) -> int:
        """Append data for a new AOR to existing zarr storage using region writes."""
        if self.lightcurves is None:
            self.load_storage()
        assert self.lightcurves is not None, "Failed to load lightcurves storage"
        
        # Check if AOR already exists
        if aor_id in self.lightcurves.aor.values:
            logger.warning(f"AOR {aor_id} already exists, overwriting")
            source_catalog = SourceCatalog(catalog)
            return self.populate_aor_from_catalog(aor_id, source_catalog, measurement_types)
        
        # For appending new AORs, we would need to extend the array along the AOR dimension
        # This is a placeholder for future implementation using append_dim
        print(f"Note: append_aor_data not fully implemented. Use populate_aor_from_table for existing AORs.")
        return 0

