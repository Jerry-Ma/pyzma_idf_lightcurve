"""
Source catalog management for IDF lightcurve analysis.

This module provides the SourceCatalog class to encapsulate operations on
SExtractor catalog tables, including coordinate handling, column mapping,
and spatial sorting for efficient storage access patterns.
"""

import re
from duckdb import table
import numpy as np
from astropy.table import Table, Column
from pathlib import Path
from typing import TypedDict, ClassVar, Literal, get_args, cast
from ..utils.naming import NameTemplate, make_regex_stub_from_literal
from ..types import ChanT, ImageKindT, ImageSuffixT, IDFFilenameT, IDFFilename
from functools import cached_property


ColSuffixT = str
ValueTypeT = Literal["mag", "mag_err"]


class MeasurementTypeT(TypedDict):

    chan: ChanT
    kind: ImageKindT
    suffix: ImageSuffixT
    col_suffix: ColSuffixT


class MeasurementType(NameTemplate[MeasurementTypeT]):

    template = "{chan}_{kind}{suffix}-{col_suffix}"
    pattern = re.compile(
        rf"^{make_regex_stub_from_literal('chan', ChanT)}"
        rf"^{make_regex_stub_from_literal('kind', ImageKindT)}"
        rf"(?P<suffix>_[^-]+|)-(?P<col_suffix>auto|iso|aper_\d+)"
        )

class SourceCatalogInfo(IDFFilenameT):
    filepath: Path
    measurement_type_colnames: dict[str, dict[ValueTypeT, str]]
    value_types: list[ValueTypeT]
    

class SourceCatalog:
    """
    Encapsulates operations on SExtractor source catalogs.
    
    Provides clean interface for coordinate handling, column mapping,
    spatial sorting, and data extraction for lightcurve storage.
    """
    
    
    def __init__(self, tbl: Table):
        """
        Initialize with SExtractor catalog table.
        
        Args:
            catalog: Astropy Table containing SExtractor output
        """
        self._table = tbl
        _ = self.info  # trigger info property to cache

    @property
    def table(self) -> Table:
        return self._table

    @cached_property
    def filepath(self) -> Path:
        tbl = self.table
        if not tbl.meta or 'filepath' not in tbl.meta:
            raise ValueError("Table metadata does not contain 'filepath'")
        return Path(tbl.meta['filepath'])

    @cached_property
    def info(self) -> SourceCatalogInfo:
        filepath = self.filepath
        file_info = IDFFilename.parse(filepath.name)
        measurement_type_colnames = {}
        # TODO get this from the catalog columns
        for col_suffix in ["auto", "iso"]:
            mt = MeasurementType.make(col_suffix=col_suffix, **file_info)
            colnames = {
                "mag": f"MAG_{col_suffix.upper()}",
                "mag_err": f"MAGERR_{col_suffix.upper()}"
            }
            measurement_type_colnames[mt] = colnames
        return {
            "filepath": filepath,
            "measurement_type_colnames": measurement_type_colnames,
            "value_types": ["mag", "mag_err"],
            **file_info,
        }

    def _get_array(self, column_name: str) -> np.ndarray:
        return cast(Column, self.table[column_name]).data

    @cached_property
    def object_ids(self) -> np.ndarray:
        return self._get_array('object_id')

    @cached_property
    def ra_values(self) -> np.ndarray:
        return self._get_array('ALPHA_J2000')
        
    @cached_property
    def dec_values(self) -> np.ndarray:
        return self._get_array('DELTA_J2000')
        
    @cached_property
    def x_values(self) -> np.ndarray:
        return self._get_array("X_IMAGE")
        
    @cached_property
    def y_values(self) -> np.ndarray:
        return self._get_array("Y_IMAGE")
    
    def get_coordinate_dict(self) -> dict[int, dict[str, float]]:
        """
        Create coordinate lookup dictionary for all objects.
        
        Returns:
            Dictionary mapping object_id to coordinate dict with keys:
            'ra', 'dec', 'x', 'y'
        """
        coord_dict = {}
        obj_ids = self.object_ids
        ra_vals = self.ra_values
        dec_vals = self.dec_values  
        x_vals = self.x_values
        y_vals = self.y_values
        
        for i, obj_id in enumerate(obj_ids):
            coord_dict[int(obj_id)] = {
                'ra': float(ra_vals[i]),
                'dec': float(dec_vals[i]), 
                'x': float(x_vals[i]),
                'y': float(y_vals[i])
            }
            
        return coord_dict
    
    def get_spatially_ordered_ids(self, grid_divisions: int = 20) -> list[int]:
        """
        Sort object IDs by spatial location using grid-based approach.
        
        Args:
            grid_divisions: Number of divisions in each spatial dimension
            
        Returns:
            List of object IDs sorted by spatial grid position
        """
        ra_vals = self.ra_values
        dec_vals = self.dec_values
        object_ids = self.object_ids
        
        # Remove NaN coordinates
        valid_mask = ~(np.isnan(ra_vals) | np.isnan(dec_vals))
        if not np.any(valid_mask):
            print("Warning: No valid coordinates found, using original object order")
            return object_ids.tolist()
        
        ra_min, ra_max = ra_vals[valid_mask].min(), ra_vals[valid_mask].max()
        dec_min, dec_max = dec_vals[valid_mask].min(), dec_vals[valid_mask].max()
        
        def spatial_key(idx: int) -> int:
            """Calculate spatial grid position for sorting."""
            ra, dec = ra_vals[idx], dec_vals[idx]
            if np.isnan(ra) or np.isnan(dec):
                return 999999  # Put invalid coordinates at end
            
            # Calculate grid position (0 to grid_divisions-1)
            if ra_max == ra_min or dec_max == dec_min:
                return 0
                
            ra_bin = min(int((ra - ra_min) / (ra_max - ra_min + 1e-10) * grid_divisions), grid_divisions - 1)
            dec_bin = min(int((dec - dec_min) / (dec_max - dec_min + 1e-10) * grid_divisions), grid_divisions - 1)
            return ra_bin * grid_divisions + dec_bin
        
        # Sort by spatial grid cell
        sort_indices = np.argsort([spatial_key(i) for i in range(len(self.table))])
        sorted_object_ids = object_ids[sort_indices]
        
        print(f"Spatially sorted {len(sorted_object_ids)} objects using {grid_divisions}x{grid_divisions} grid")
        return sorted_object_ids.tolist()
    
    def get_coordinate_arrays_for_objects(self, object_ids: list[int]) -> tuple[list[float], list[float], list[float], list[float]]:
        """
        Get coordinate arrays for specific object IDs in given order.
        
        Args:
            object_ids: List of object IDs in desired order
            
        Returns:
            Tuple of (ra_vals, dec_vals, x_vals, y_vals) lists in object_id order
        """
        coord_dict = self.get_coordinate_dict()
        
        ra_vals = [coord_dict.get(obj_id, {}).get('ra', np.nan) for obj_id in object_ids]
        dec_vals = [coord_dict.get(obj_id, {}).get('dec', np.nan) for obj_id in object_ids]
        x_vals = [coord_dict.get(obj_id, {}).get('x', np.nan) for obj_id in object_ids]
        y_vals = [coord_dict.get(obj_id, {}).get('y', np.nan) for obj_id in object_ids]
        
        return ra_vals, dec_vals, x_vals, y_vals
    
    @property
    def measurement_types(self) -> list[str]:
        return list(self.info["measurement_type_colnames"].keys())

    def extract_measurements(self, measurement_types: list[str]) -> dict[str, dict[ValueTypeT, np.ndarray]]:
        """
        Extract magnitude and error measurements for given measurement types.
        
        Args:
            measurement_types: List of measurement type identifiers
            
        Returns:
            Dictionary mapping measurement types to (magnitude, error) arrays
        """
        measurements = {}
        
        for meas_type in measurement_types:
            colnames = self.info["measurement_type_colnames"].get(meas_type, None)
            if colnames is None:
                print(f"Warning: Could not find columns for measurement type '{meas_type}'")
                continue
            mag_col = colnames["mag"]
            err_col = colnames["mag_err"]
            mag_data = self._get_array(mag_col)
            err_data = self._get_array(err_col)

            measurements[meas_type] = {
                "mag": mag_data,
                "mag_err": err_data
            }

        return measurements
    
    def get_valid_measurement_mask(self, measurement_types: list[str]) -> np.ndarray:
        """
        Get boolean mask for objects with valid measurements.
        
        Args:
            measurement_types: List of measurement types to check
            
        Returns:
            Boolean array indicating which objects have valid measurements
        """
        tbl = self.table
        measurements = self.extract_measurements(measurement_types)
        
        if not measurements:
            return np.zeros(len(tbl), dtype=bool)

        # Combine validity across all measurement types
        valid_mask = np.ones(len(tbl), dtype=bool)
        
        for meas_type, values in measurements.items():
            mag_data = values["mag"]
            err_data = values["mag_err"]
            # Check for finite values and reasonable ranges
            print(mag_data)
            print(err_data)
            mag_valid = np.isfinite(mag_data) & (mag_data > 0) & (mag_data < 50)
            err_valid = np.isfinite(err_data) & (err_data > 0) & (err_data < 10)
            valid_mask &= mag_valid & err_valid
            
        return valid_mask
    
    def __len__(self) -> int:
        """Return number of objects in catalog."""
        return len(self.table)
        
    def __repr__(self) -> str:
        """String representation of catalog."""
        return f"SourceCatalog(n_objs={len(self)}, n_meas={len(self.info['measurement_type_colnames'])})"