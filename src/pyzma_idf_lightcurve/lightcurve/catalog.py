"""
Source catalog management for IDF lightcurve analysis.

This module provides the SourceCatalog class to encapsulate operations on
SExtractor catalog tables, including coordinate handling, column mapping,
and spatial sorting for efficient storage access patterns.
"""

import re
from duckdb import table
from loguru import logger
import numpy as np
from astropy.table import Table, Column
from pathlib import Path
from typing import TypedDict, ClassVar, Literal, get_args, cast, Callable, NamedTuple
from ..utils.naming import NameTemplate, make_regex_stub_from_literal, StrSepNameTemplate, DashSeparated, UnderscoreSeparated, StrSepSegment
from ..types import ChanT, ImageKindT, ImageSuffixT, IDFFilenameT, IDFFilename
from functools import cached_property
import dataclasses


# ValueKeyT = Literal["value", "uncertainty", "mag", "mag_err", "flux", "flux_err"]


class MeasurementKey(DashSeparated):
    pass


class MeasurementColname(UnderscoreSeparated):
    match_prefix: bool = False


TableColumnMapperT = None | str | tuple[str, Callable[[np.ndarray], np.ndarray]]
MeasurementKeySegmentMapperT = str | Callable[[StrSepSegment], str]


class SourceCatalogDataKey(NamedTuple):
    measurement: str
    value: str
    epoch: None | str


@dataclasses.dataclass
class SourceCatalogDataKeyInfo:
    colname_data_keys: dict[str, SourceCatalogDataKey]
    data_keys_colname: dict[SourceCatalogDataKey, str]
    measurement_keys: set[str]
    value_keys: set[str]
    epoch_keys: set[str]


@dataclasses.dataclass
class SourceCatalogTableTransform:
    ra_col: TableColumnMapperT = "ra"
    dec_col: TableColumnMapperT = "dec"
    x_col: TableColumnMapperT = "x"
    y_col: TableColumnMapperT = "y"
    obj_key_col: TableColumnMapperT = ("id", lambda col: col.astype(str))
    colname_template_cls: type[StrSepNameTemplate] = MeasurementColname
    data_colname_identify_regex: re.Pattern = re.compile(r"^(mag|magerr|flux|fluxerr)_.+", re.IGNORECASE)
    data_key_template_cls: type[StrSepNameTemplate] = MeasurementKey

    def data_keys_to_data_colname(self, keys: SourceCatalogDataKey) -> str:
        sep = self.colname_template_cls.sep
        return f"{keys.value}{sep}{keys.measurement}".lower()

    def data_colname_to_data_keys(self, colname: str) -> SourceCatalogDataKey:
        sep = self.colname_template_cls.sep
        parts = self.colname_template_cls.parse(colname.lower())
        return SourceCatalogDataKey(
            measurement=parts["suffix"].lstrip(sep),
            value=parts["stem"],
            epoch=None,
        )

    @staticmethod
    def _get_tbl_col_as_array(tbl, colname: str) -> np.ndarray:
        return cast(Column, tbl[colname]).data
    
    def get_or_create_mapped(self, tbl: Table, attr_name: str) -> np.ndarray:
        # this caches the mapped data into the table itself, to make sure we are immune to any sorting.
        mapper = getattr(self, attr_name)
        mapped_colname = f"_source_catalog_mapped_{attr_name}"
        if mapped_colname not in tbl.colnames:
            # create the mapped data
            if isinstance(mapper, str):
                mapped = self._get_tbl_col_as_array(tbl, mapper)
            elif isinstance(mapper, tuple):
                mapper, mapper_func = mapper
                mapped = mapper_func(self._get_tbl_col_as_array(tbl, mapper))
            else:
                assert False
            tbl[mapped_colname] = mapped
        return self._get_tbl_col_as_array(tbl, mapped_colname)

    def collect_data_key_info(self, tbl: Table):

        colname_data_keys: dict[str, SourceCatalogDataKey] = {}
        data_keys_colname: dict[SourceCatalogDataKey, str] = {}
        measurement_keys: set[str] = set()
        value_keys: set[str] = set()
        epoch_keys: set[str] = set()

        for colname in tbl.colnames:
            m = self.data_colname_identify_regex.match(colname)
            if not m:
                continue
            try:
                data_keys = self.data_colname_to_data_keys(colname)
            except ValueError:
                continue
            colname_data_keys[colname] = data_keys
            data_keys_colname[data_keys] = colname
            measurement_keys.add(data_keys.measurement)
            value_keys.add(data_keys.value)
            if data_keys.epoch is not None:
                epoch_keys.add(data_keys.epoch)
        return SourceCatalogDataKeyInfo(
            colname_data_keys=colname_data_keys,
            data_keys_colname=data_keys_colname,
            measurement_keys=measurement_keys,
            value_keys=value_keys,
            epoch_keys=epoch_keys,
        )


default_source_catalog_table_transform = SourceCatalogTableTransform()

class SourceCatalog:
    """
    Encapsulates operations on SExtractor source catalogs.
    
    Provides clean interface for coordinate handling, column mapping,
    spatial sorting, and data extraction for lightcurve storage.
    """

    _table_transform: SourceCatalogTableTransform

    def __init__(self, table: Table, table_transform: SourceCatalogTableTransform=default_source_catalog_table_transform, copy=True):
        """
        Initialize with SExtractor catalog table.
        
        Args:
            catalog: Astropy Table containing SExtractor output
        """
        if copy:
            table = table.copy()
        self._table = table
        tt = self._table_transform = table_transform
        self._data_key_info = tt.collect_data_key_info(self.table)

    @property
    def table(self) -> Table:
        return self._table
    
    @property
    def table_transform(self) -> SourceCatalogTableTransform:
        return self._table_transform

    def get_table_data(self, colname: str) -> np.ndarray:
        return self.table_transform._get_tbl_col_as_array(self.table, colname)

    @property
    def data_key_info(self) -> SourceCatalogDataKeyInfo:
        return self._data_key_info

    @property
    def measurement_keys(self) -> set[str]:
        return self.data_key_info.measurement_keys
   
    @property
    def value_keys(self) -> set[str]:
        return self.data_key_info.value_keys

    @property
    def epoch_keys(self) -> set[str]:
        return self._data_key_info.epoch_keys

    @cached_property
    def ra_values(self) -> np.ndarray:
        return self.table_transform.get_or_create_mapped(self.table, "ra_col")
        
    @property
    def dec_values(self) -> np.ndarray:
        return self.table_transform.get_or_create_mapped(self.table, "dec_col")
        
    @property
    def x_values(self) -> np.ndarray:
        return self.table_transform.get_or_create_mapped(self.table, "x_col")
        
    @property
    def y_values(self) -> np.ndarray:
        return self.table_transform.get_or_create_mapped(self.table, "y_col")
 
    @property
    def object_keys(self) -> np.ndarray:
        return self.table_transform.get_or_create_mapped(self.table, "obj_key_col")

    @cached_property
    def object_key_coordinates(self) -> dict[str, dict[str, float]]:
        """
        Create coordinate lookup dictionary for all objects.
        
        Returns:
            Dictionary mapping object_key (str) to coordinate dict with keys:
            'ra', 'dec', 'x', 'y'
        """
        coord_dict = {}
        obj_keys = self.object_keys
        ra_vals = self.ra_values
        dec_vals = self.dec_values  
        x_vals = self.x_values
        y_vals = self.y_values
        
        for i, obj_key in enumerate(obj_keys):
            coord_dict[obj_key] = {
                'ra': float(ra_vals[i]),
                'dec': float(dec_vals[i]), 
                'x': float(x_vals[i]),
                'y': float(y_vals[i])
            }
            
        return coord_dict
    
    def _get_spatially_ordered_keys(self, grid_divisions: int = 20) -> tuple[list[int], list[str]]:
        """
        Sort object keys by spatial location using grid-based approach.
        
        Args:
            grid_divisions: Number of divisions in each spatial dimension
            
        Returns:
            List of object keys (strings) sorted by spatial grid position
        """
        ra_vals = self.ra_values
        dec_vals = self.dec_values
        object_keys = self.object_keys
        
        # Remove NaN coordinates
        valid_mask = ~(np.isnan(ra_vals) | np.isnan(dec_vals))
        if not np.any(valid_mask):
            print("Warning: No valid coordinates found, using original object order")
            return object_keys.tolist()
        
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
        sort_keys = [spatial_key(i) for i in range(len(self.table))]
        sort_indices = np.argsort(sort_keys)
        sorted_object_keys = object_keys[sort_indices].tolist() 
        
        print(f"Spatially sorted {len(sorted_object_keys)} objects using {grid_divisions}x{grid_divisions} grid")
        return sort_keys, sorted_object_keys
    
    def sort_objects_by_position(self, **kwargs):
        sort_keys, _ = self._get_spatially_ordered_keys(**kwargs)
        self.table["_source_catalog_spatial_sort_key"] = sort_keys
        self.table.sort("_source_catalog_spatial_sort_key")
        logger.info(f"Sorted objects by spatial position")
        
    
    def get_coordinate_arrays_for_objects(self, object_keys: list[str]) -> tuple[list[float], list[float], list[float], list[float]]:
        """
        Get coordinate arrays for specific object keys in given order.
        
        Args:
            object_keys: List of object keys (strings) in desired order
            
        Returns:
            Tuple of (ra_vals, dec_vals, x_vals, y_vals) lists in object_key order
        """
        coord_dict = self.object_key_coordinates
        
        ra_vals = [coord_dict.get(obj_key, {}).get('ra', np.nan) for obj_key in object_keys]
        dec_vals = [coord_dict.get(obj_key, {}).get('dec', np.nan) for obj_key in object_keys]
        x_vals = [coord_dict.get(obj_key, {}).get('x', np.nan) for obj_key in object_keys]
        y_vals = [coord_dict.get(obj_key, {}).get('y', np.nan) for obj_key in object_keys]
        
        return ra_vals, dec_vals, x_vals, y_vals
    
    @cached_property
    def data(self) -> dict[SourceCatalogDataKey, np.ndarray]:
        data = {}
        for data_key, colname in self.data_key_info.data_keys_colname.items():
            data[data_key] = self.table_transform._get_tbl_col_as_array(self.table, colname)
        return data

    def __len__(self) -> int:
        """Return number of objects in catalog."""
        return len(self.table)

    def __repr__(self) -> str:
        """String representation of catalog."""
        n_objs = len(self.object_keys)
        n_measurements = len(self.measurement_keys),
        n_values = len(self.value_keys)
        n_epochs = len(self.epoch_keys)
        info_str = f"{n_objs}, {n_measurements}, {n_values}"
        if n_epochs > 0:
            info_str += f", {n_epochs}"
        return f"SourceCatalog({info_str})"


@dataclasses.dataclass
class SExtractorTableTransform(SourceCatalogTableTransform):
    ra_col: TableColumnMapperT = "ALPHA_J2000"
    dec_col: TableColumnMapperT = "DELTA_J2000"
    x_col: TableColumnMapperT = "X_IMAGE"
    y_col: TableColumnMapperT = "Y_IMAGE"
    obj_key_col: TableColumnMapperT = ("NUMBER", lambda col: col.astype(str))
    colname_template_cls: type[StrSepNameTemplate] = MeasurementColname
    data_colname_identify_regex: re.Pattern = re.compile(r"^(MAG|MAGERR|FLUX|FLUXERR)_.+")
    data_key_template_cls: type[StrSepNameTemplate] = MeasurementKey

    def data_keys_to_data_colname(self, keys: SourceCatalogDataKey) -> str:
        sep = self.colname_template_cls.sep
        return f"{keys.value}{sep}{keys.measurement}".upper()

    def data_colname_to_data_keys(self, colname: str) -> SourceCatalogDataKey:
        sep = self.colname_template_cls.sep
        parts = self.colname_template_cls.parse(colname.lower())
        return SourceCatalogDataKey(
            measurement=parts["suffix"].lstrip(sep),
            value = parts["stem"],
            epoch = None,
            )

