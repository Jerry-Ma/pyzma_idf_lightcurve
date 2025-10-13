"""
Unit tests for SourceCatalog and SourceCatalogTableTransform.

Tests the new customizable catalog transformation system including:
- Default transform behavior
- SExtractor-specific transform
- Custom transform configuration
- Measurement key inference
- Coordinate extraction and spatial sorting
- Data extraction and validation
"""

import dataclasses
import re

import numpy as np
import pytest
from astropy.table import Table

from pyzma_idf_lightcurve.lightcurve.catalog import (
    MeasurementColname,
    MeasurementKey,
    SExtractorTableTransform,
    SourceCatalog,
    SourceCatalogDataKey,
    SourceCatalogTableTransform,
    TableColumnMapperT,
    default_source_catalog_table_transform,
)


class TestMeasurementKey:
    """Test MeasurementKey template parsing and generation."""
    
    def test_parse_measurement_key(self):
        """Test parsing measurement key identifier with separator."""
        parsed = MeasurementKey.parse("auto")
        assert parsed["stem"] == "auto"
        assert parsed["prefix"] == ""
        assert parsed["suffix"] == ""
        
    def test_parse_measurement_key_with_prefix(self):
        """Test parsing measurement key with prefix (requires 3 parts)."""
        # 2 parts: stem="ch1", suffix="-auto" (no prefix)
        parsed = MeasurementKey.parse("ch1-auto")
        assert parsed["prefix"] == ""
        assert parsed["stem"] == "ch1"
        assert parsed["suffix"] == "-auto"
        
        # 3 parts: prefix="ch1-", stem="auto", suffix="-ap1"
        parsed = MeasurementKey.parse("ch1-auto-ap1")
        assert parsed["prefix"] == "ch1-"
        assert parsed["stem"] == "auto"
        assert parsed["suffix"] == "-ap1"
        
    def test_parse_measurement_key_with_suffix(self):
        """Test parsing measurement key with suffix."""
        parsed = MeasurementKey.parse("auto-1")
        assert parsed["stem"] == "auto"
        assert parsed["suffix"] == "-1"


class TestMeasurementColname:
    """Test MeasurementColname template parsing."""
    
    def test_parse_colname(self):
        """Test parsing column name with underscore separator."""
        parsed = MeasurementColname.parse("mag_auto")
        assert parsed["stem"] == "mag"
        assert parsed["suffix"] == "_auto"
        
    def test_parse_colname_complex(self):
        """Test parsing column name with match_prefix=False.
        
        MeasurementColname has match_prefix=False, so:
        - "magerr_aper_1" → stem="magerr", suffix="_aper_1" (prefix is not matched)
        - "mag_auto" → stem="mag", suffix="_auto"
        """
        # With match_prefix=False: stem captures everything before first suffix
        parsed = MeasurementColname.parse("magerr_aper_1")
        assert parsed["prefix"] == ""  # Not matched
        assert parsed["stem"] == "magerr"
        assert parsed["suffix"] == "_aper_1"
        
        # Two-part name
        parsed = MeasurementColname.parse("mag_auto")
        assert parsed["prefix"] == ""
        assert parsed["stem"] == "mag"
        assert parsed["suffix"] == "_auto"


class TestSourceCatalogDataKey:
    """Test SourceCatalogDataKey namedtuple."""
    
    def test_create_data_key(self):
        """Test creating data key."""
        key = SourceCatalogDataKey(measurement="auto", value="mag", epoch=None)
        assert key.measurement == "auto"
        assert key.value == "mag"
        assert key.epoch is None
        
    def test_create_data_key_with_epoch(self):
        """Test creating data key with epoch."""
        key = SourceCatalogDataKey(measurement="auto", value="flux", epoch="default")
        assert key.measurement == "auto"
        assert key.value == "flux"
        assert key.epoch == "default"


class TestSourceCatalogTableTransform:
    """Test the default SourceCatalogTableTransform."""
    
    def test_default_transform_columns(self):
        """Test default column mappings."""
        transform = SourceCatalogTableTransform()
        assert transform.ra_col == "ra"
        assert transform.dec_col == "dec"
        assert transform.x_col == "x"
        assert transform.y_col == "y"
        
    def test_default_obj_key_transformation(self):
        """Test default object key column transformation."""
        transform = SourceCatalogTableTransform()
        assert isinstance(transform.obj_key_col, tuple)
        col_name, mapper_func = transform.obj_key_col
        assert col_name == "id"
        # Test the lambda function
        test_arr = np.array([1, 2, 3])
        result = mapper_func(test_arr)
        assert all(isinstance(x, (str, np.str_)) for x in result)
        
    def test_data_keys_to_colname(self):
        """Test converting data keys to column name."""
        transform = SourceCatalogTableTransform()
        key = SourceCatalogDataKey(measurement="auto", value="mag", epoch=None)
        colname = transform.data_keys_to_data_colname(key)
        assert colname == "mag_auto"
        
    def test_data_colname_to_keys(self):
        """Test parsing column name to data keys."""
        transform = SourceCatalogTableTransform()
        key = transform.data_colname_to_data_keys("mag_auto")
        assert key.measurement == "auto"
        assert key.value == "mag"
        assert key.epoch is None
        
    def test_data_colname_to_keys_complex(self):
        """Test parsing complex column name."""
        transform = SourceCatalogTableTransform()
        key = transform.data_colname_to_data_keys("flux_aper_1")
        assert key.measurement == "aper_1"
        assert key.value == "flux"
        
    def test_collect_data_key_info(self):
        """Test collecting data key info from table."""
        tbl = Table()
        tbl['id'] = [1, 2, 3]
        tbl['ra'] = [10.0, 10.1, 10.2]
        tbl['dec'] = [20.0, 20.1, 20.2]
        tbl['x'] = [100.0, 200.0, 300.0]
        tbl['y'] = [150.0, 250.0, 350.0]
        tbl['mag_auto'] = [18.0, 18.5, 19.0]
        tbl['magerr_auto'] = [0.05, 0.06, 0.07]
        tbl['flux_iso'] = [1000.0, 900.0, 800.0]
        tbl['fluxerr_iso'] = [50.0, 45.0, 40.0]
        
        transform = SourceCatalogTableTransform()
        info = transform.collect_data_key_info(tbl)
        
        assert 'auto' in info.measurement_keys
        assert 'iso' in info.measurement_keys
        assert 'mag' in info.value_keys
        assert 'magerr' in info.value_keys
        assert 'flux' in info.value_keys
        assert 'fluxerr' in info.value_keys
        assert len(info.epoch_keys) == 0  # Default has no epochs


class TestSExtractorTableTransform:
    """Test the SExtractor-specific table transform."""
    
    def test_sextractor_columns(self):
        """Test SExtractor column mappings."""
        transform = SExtractorTableTransform()
        assert transform.ra_col == "ALPHA_J2000"
        assert transform.dec_col == "DELTA_J2000"
        assert transform.x_col == "X_IMAGE"
        assert transform.y_col == "Y_IMAGE"
        
    def test_sextractor_obj_key(self):
        """Test SExtractor object key transformation."""
        transform = SExtractorTableTransform()
        assert isinstance(transform.obj_key_col, tuple)
        col_name, mapper_func = transform.obj_key_col
        assert col_name == "NUMBER"
        test_arr = np.array([1001, 1002, 1003])
        result = mapper_func(test_arr)
        assert all(isinstance(x, (str, np.str_)) for x in result)
        
    def test_sextractor_data_keys_to_colname(self):
        """Test SExtractor column name generation (uppercase)."""
        transform = SExtractorTableTransform()
        key = SourceCatalogDataKey(measurement="auto", value="mag", epoch="default")
        colname = transform.data_keys_to_data_colname(key)
        assert colname == "MAG_AUTO"
        
    def test_sextractor_data_colname_to_keys(self):
        """Test SExtractor column name parsing."""
        transform = SExtractorTableTransform()
        key = transform.data_colname_to_data_keys("MAG_AUTO")
        assert key.measurement == "auto"
        assert key.value == "mag"
        assert key.epoch is None
        
    def test_sextractor_regex_pattern(self):
        """Test SExtractor column identification regex."""
        transform = SExtractorTableTransform()
        assert transform.data_colname_identify_regex.match("MAG_AUTO")
        assert transform.data_colname_identify_regex.match("MAGERR_ISO")
        assert transform.data_colname_identify_regex.match("FLUX_APER_1")
        assert not transform.data_colname_identify_regex.match("X_IMAGE")
        assert not transform.data_colname_identify_regex.match("NUMBER")


class TestSourceCatalogBasic:
    """Test basic SourceCatalog functionality with default transform."""
    
    def test_init_default_transform(self):
        """Test initialization with default transform."""
        tbl = Table()
        tbl['id'] = [1, 2, 3]
        tbl['ra'] = [10.0, 10.1, 10.2]
        tbl['dec'] = [20.0, 20.1, 20.2]
        tbl['x'] = [100.0, 200.0, 300.0]
        tbl['y'] = [150.0, 250.0, 350.0]
        
        catalog = SourceCatalog(tbl)
        assert len(catalog) == 3
        assert catalog.table_transform == default_source_catalog_table_transform
        
    def test_init_custom_transform(self):
        """Test initialization with custom transform."""
        tbl = Table()
        tbl['NUMBER'] = [1, 2, 3]
        tbl['ALPHA_J2000'] = [10.0, 10.1, 10.2]
        tbl['DELTA_J2000'] = [20.0, 20.1, 20.2]
        tbl['X_IMAGE'] = [100.0, 200.0, 300.0]
        tbl['Y_IMAGE'] = [150.0, 250.0, 350.0]
        
        transform = SExtractorTableTransform()
        catalog = SourceCatalog(tbl, table_transform=transform)
        assert len(catalog) == 3
        assert catalog.table_transform == transform
        
    def test_object_keys(self):
        """Test object key extraction."""
        tbl = Table()
        tbl['id'] = [1001, 1002, 1003]
        tbl['ra'] = [10.0, 10.1, 10.2]
        tbl['dec'] = [20.0, 20.1, 20.2]
        tbl['x'] = [100.0, 200.0, 300.0]
        tbl['y'] = [150.0, 250.0, 350.0]

        catalog = SourceCatalog(tbl)
        obj_keys = catalog.object_keys
        assert all(isinstance(k, (str, np.str_)) for k in obj_keys)
        assert list(obj_keys) == ['1001', '1002', '1003']
    
    def test_coordinate_extraction(self):
        """Test coordinate value extraction."""
        tbl = Table()
        tbl['id'] = [1, 2]
        tbl['ra'] = [15.5, 15.6]
        tbl['dec'] = [-30.0, -30.1]
        tbl['x'] = [100.0, 200.0]
        tbl['y'] = [150.0, 250.0]
        
        catalog = SourceCatalog(tbl)
        
        assert len(catalog.ra_values) == 2
        assert len(catalog.dec_values) == 2
        assert len(catalog.x_values) == 2
        assert len(catalog.y_values) == 2
        
        assert np.allclose(catalog.ra_values, [15.5, 15.6])
        assert np.allclose(catalog.dec_values, [-30.0, -30.1])
        assert np.allclose(catalog.x_values, [100.0, 200.0])
        assert np.allclose(catalog.y_values, [150.0, 250.0])


class TestMeasurementKeyInference:
    """Test automatic measurement key inference from columns."""
    
    def test_infer_auto_measurement_default_transform(self):
        """Test inference with default transform."""
        tbl = Table()
        tbl['id'] = [1, 2, 3]
        tbl['ra'] = [10.0, 10.1, 10.2]
        tbl['dec'] = [20.0, 20.1, 20.2]
        tbl['x'] = [100.0, 200.0, 300.0]
        tbl['y'] = [150.0, 250.0, 350.0]
        tbl['mag_auto'] = [18.0, 18.5, 19.0]
        tbl['magerr_auto'] = [0.05, 0.06, 0.07]
        
        catalog = SourceCatalog(tbl)
        
        assert "auto" in catalog.measurement_keys
        assert "mag" in catalog.value_keys
        assert "magerr" in catalog.value_keys
        
    def test_infer_measurements_sextractor_transform(self):
        """Test inference with SExtractor transform."""
        tbl = Table()
        tbl['NUMBER'] = [1, 2, 3]
        tbl['ALPHA_J2000'] = [10.0, 10.1, 10.2]
        tbl['DELTA_J2000'] = [20.0, 20.1, 20.2]
        tbl['X_IMAGE'] = [100.0, 200.0, 300.0]
        tbl['Y_IMAGE'] = [150.0, 250.0, 350.0]
        tbl['MAG_AUTO'] = [18.0, 18.5, 19.0]
        tbl['MAGERR_AUTO'] = [0.05, 0.06, 0.07]
        tbl['MAG_ISO'] = [17.5, 18.0, 18.5]
        tbl['MAGERR_ISO'] = [0.04, 0.05, 0.06]
        
        transform = SExtractorTableTransform()
        catalog = SourceCatalog(tbl, table_transform=transform)
        
        assert "auto" in catalog.measurement_keys
        assert "iso" in catalog.measurement_keys
        assert "mag" in catalog.value_keys
        assert "magerr" in catalog.value_keys
        
    def test_infer_aperture_measurements(self):
        """Test inference of aperture photometry."""
        tbl = Table()
        tbl['NUMBER'] = [1, 2]
        tbl['ALPHA_J2000'] = [10.0, 10.1]
        tbl['DELTA_J2000'] = [20.0, 20.1]
        tbl['X_IMAGE'] = [100.0, 200.0]
        tbl['Y_IMAGE'] = [150.0, 250.0]
        
        tbl['MAG_APER'] = [18.0, 18.5]
        tbl['MAGERR_APER'] = [0.05, 0.06]
        tbl['MAG_APER_1'] = [18.1, 18.6]
        tbl['MAGERR_APER_1'] = [0.05, 0.06]
        
        transform = SExtractorTableTransform()
        catalog = SourceCatalog(tbl, table_transform=transform)
        
        assert "aper" in catalog.measurement_keys
        assert "aper_1" in catalog.measurement_keys
        
    def test_infer_flux_measurements(self):
        """Test inference of flux measurements."""
        tbl = Table()
        tbl['id'] = [1, 2]
        tbl['ra'] = [10.0, 10.1]
        tbl['dec'] = [20.0, 20.1]
        tbl['x'] = [100.0, 200.0]
        tbl['y'] = [150.0, 250.0]
        tbl['flux_auto'] = [1000.0, 900.0]
        tbl['fluxerr_auto'] = [50.0, 45.0]
        
        catalog = SourceCatalog(tbl)
        
        assert "auto" in catalog.measurement_keys
        assert "flux" in catalog.value_keys
        assert "fluxerr" in catalog.value_keys


class TestMeasurementExtraction:
    """Test extracting measurements from catalog."""
    
    def test_data_property_structure(self):
        """Test that data property returns correct structure."""
        tbl = Table()
        tbl['NUMBER'] = [1, 2, 3]
        tbl['ALPHA_J2000'] = [10.0, 10.1, 10.2]
        tbl['DELTA_J2000'] = [20.0, 20.1, 20.2]
        tbl['X_IMAGE'] = [100.0, 200.0, 300.0]
        tbl['Y_IMAGE'] = [150.0, 250.0, 350.0]
        tbl['MAG_AUTO'] = np.array([18.0, 18.5, 19.0])
        tbl['MAGERR_AUTO'] = np.array([0.05, 0.06, 0.07])
        
        transform = SExtractorTableTransform()
        catalog = SourceCatalog(tbl, table_transform=transform)
        
        data = catalog.data
        
        # Should be a dict with SourceCatalogDataKey as keys
        assert isinstance(data, dict)
        assert len(data) > 0
        
        # All keys should be SourceCatalogDataKey tuples
        for key in data.keys():
            assert isinstance(key, tuple)
            assert len(key) == 3  # measurement, value, epoch
            assert isinstance(key[0], str)  # measurement
            assert isinstance(key[1], str)  # value
            assert key[2] is None or isinstance(key[2], str)  # epoch
        
        # All values should be numpy arrays
        for value in data.values():
            assert isinstance(value, np.ndarray)
            assert len(value) == 3  # Same length as table
    
    def test_data_property_content(self):
        """Test that data property contains correct content."""
        tbl = Table()
        tbl['NUMBER'] = [1, 2]
        tbl['ALPHA_J2000'] = [10.0, 10.1]
        tbl['DELTA_J2000'] = [20.0, 20.1]
        tbl['X_IMAGE'] = [100.0, 200.0]
        tbl['Y_IMAGE'] = [150.0, 250.0]
        tbl['MAG_AUTO'] = np.array([18.0, 18.5])
        tbl['MAGERR_AUTO'] = np.array([0.05, 0.06])
        tbl['FLUX_ISO'] = np.array([1000.0, 900.0])
        tbl['FLUXERR_ISO'] = np.array([50.0, 45.0])
        
        transform = SExtractorTableTransform()
        catalog = SourceCatalog(tbl, table_transform=transform)
        
        data = catalog.data
        
        # Check that expected keys are present
        measurements = {key[0] for key in data.keys()}
        assert 'auto' in measurements
        assert 'iso' in measurements
        
        values = {key[1] for key in data.keys()}
        assert 'mag' in values
        assert 'magerr' in values
        assert 'flux' in values
        assert 'fluxerr' in values
        
        # Check actual data values for MAG_AUTO
        from pyzma_idf_lightcurve.lightcurve.catalog import SourceCatalogDataKey
        mag_auto_key = SourceCatalogDataKey(measurement='auto', value='mag', epoch=None)
        assert mag_auto_key in data
        np.testing.assert_array_equal(data[mag_auto_key], np.array([18.0, 18.5]))
        
        # Check MAGERR_AUTO
        magerr_auto_key = SourceCatalogDataKey(measurement='auto', value='magerr', epoch=None)
        assert magerr_auto_key in data
        np.testing.assert_array_equal(data[magerr_auto_key], np.array([0.05, 0.06]))
        
        # Check FLUX_ISO
        flux_iso_key = SourceCatalogDataKey(measurement='iso', value='flux', epoch=None)
        assert flux_iso_key in data
        np.testing.assert_array_equal(data[flux_iso_key], np.array([1000.0, 900.0]))


class TestCoordinateMethods:
    """Test coordinate-related methods."""
    
    def test_object_key_coordinates(self):
        """Test object_key_coordinates property."""
        tbl = Table()
        tbl['id'] = [1, 2, 3]
        tbl['ra'] = [10.0, 10.1, 10.2]
        tbl['dec'] = [20.0, 20.1, 20.2]
        tbl['x'] = [100.0, 200.0, 300.0]
        tbl['y'] = [150.0, 250.0, 350.0]
        
        catalog = SourceCatalog(tbl)
        
        coord_dict = catalog.object_key_coordinates
        
        assert len(coord_dict) == 3
        assert '1' in coord_dict
        assert '2' in coord_dict
        assert '3' in coord_dict
        
        assert coord_dict['1']['ra'] == 10.0
        assert coord_dict['1']['dec'] == 20.0
        assert coord_dict['1']['x'] == 100.0
        assert coord_dict['1']['y'] == 150.0
        
    def test_get_coordinate_arrays_for_objects(self):
        """Test getting coordinate arrays for specific objects."""
        tbl = Table()
        tbl['id'] = [1, 2, 3, 4, 5]
        tbl['ra'] = [10.0, 10.1, 10.2, 10.3, 10.4]
        tbl['dec'] = [20.0, 20.1, 20.2, 20.3, 20.4]
        tbl['x'] = [100.0, 200.0, 300.0, 400.0, 500.0]
        tbl['y'] = [150.0, 250.0, 350.0, 450.0, 550.0]
        
        catalog = SourceCatalog(tbl)
        
        # Get coordinates for subset in specific order
        object_keys = ['3', '1', '5']
        ra_vals, dec_vals, x_vals, y_vals = catalog.get_coordinate_arrays_for_objects(object_keys)
        
        assert len(ra_vals) == 3
        assert ra_vals == [10.2, 10.0, 10.4]
        assert dec_vals == [20.2, 20.0, 20.4]
        assert x_vals == [300.0, 100.0, 500.0]
        assert y_vals == [350.0, 150.0, 550.0]


class TestCustomTransform:
    """Test creating and using custom table transforms."""
    
    def test_custom_column_mapping(self):
        """Test custom column name mapping."""
        @dataclasses.dataclass
        class CustomTransform(SourceCatalogTableTransform):
            ra_col: TableColumnMapperT = "RA_DEG"
            dec_col: TableColumnMapperT = "DEC_DEG"
            x_col: TableColumnMapperT = "PIX_X"
            y_col: TableColumnMapperT = "PIX_Y"
            obj_key_col: TableColumnMapperT = ("OBJ_ID", lambda col: col.astype(str))
        tbl = Table()
        tbl['OBJ_ID'] = [1, 2, 3]
        tbl['RA_DEG'] = [10.0, 10.1, 10.2]
        tbl['DEC_DEG'] = [20.0, 20.1, 20.2]
        tbl['PIX_X'] = [100.0, 200.0, 300.0]
        tbl['PIX_Y'] = [150.0, 250.0, 350.0]
        
        transform = CustomTransform()
        catalog = SourceCatalog(tbl, table_transform=transform)
        
        assert np.allclose(catalog.ra_values, [10.0, 10.1, 10.2])
        assert np.allclose(catalog.dec_values, [20.0, 20.1, 20.2])
        assert list(catalog.object_keys) == ['1', '2', '3']
    
    def test_custom_data_key_mapping(self):
        """Test custom data key to column name mapping."""
        @dataclasses.dataclass
        class CustomTransform(SourceCatalogTableTransform):
            data_colname_identify_regex: re.Pattern = re.compile(r"^(m|merr|f|ferr)_.+")
            
            def data_keys_to_data_colname(self, keys):
                # Use 'm' instead of 'mag', 'f' instead of 'flux'
                value_map = {"mag": "m", "magerr": "merr", "flux": "f", "fluxerr": "ferr"}
                short_val = value_map.get(keys.value, keys.value)
                return f"{short_val}_{keys.measurement}".lower()
            
            def data_colname_to_data_keys(self, colname):
                parts = self.colname_template_cls.parse(colname.lower())
                # Reverse mapping
                value_map = {"m": "mag", "merr": "magerr", "f": "flux", "ferr": "fluxerr"}
                long_val = value_map.get(parts["stem"], parts["stem"])
                return SourceCatalogDataKey(
                    measurement=parts["suffix"].lstrip("_"),
                    value=long_val,
                    epoch=None
                )
        
        tbl = Table()
        tbl['id'] = [1, 2]
        tbl['ra'] = [10.0, 10.1]
        tbl['dec'] = [20.0, 20.1]
        tbl['x'] = [100.0, 200.0]
        tbl['y'] = [150.0, 250.0]
        tbl['m_auto'] = [18.0, 18.5]
        tbl['merr_auto'] = [0.05, 0.06]
        
        transform = CustomTransform()
        catalog = SourceCatalog(tbl, table_transform=transform)
        
        assert "auto" in catalog.measurement_keys
        assert "mag" in catalog.value_keys
        assert "magerr" in catalog.value_keys


class TestSortObjectsByPosition:
    """Test sort_objects_by_position functionality."""
    
    def test_sort_objects_by_position_basic(self):
        """Test that sort_objects_by_position correctly reorders the internal table."""
        # Create a test catalog with unsorted spatial positions
        test_table = Table()
        test_table['ra'] = [100.5, 100.1, 100.9, 100.2, 100.7]
        test_table['dec'] = [20.5, 20.1, 20.9, 20.2, 20.7]
        test_table['x'] = [50.0, 10.0, 90.0, 20.0, 70.0]
        test_table['y'] = [50.0, 10.0, 90.0, 20.0, 70.0]
        test_table['id'] = ['obj_a', 'obj_b', 'obj_c', 'obj_d', 'obj_e']
        test_table['flux_auto'] = [100.0, 200.0, 300.0, 400.0, 500.0]
        
        # Create catalog
        catalog = SourceCatalog(test_table, copy=True)
        
        # Store original order
        original_ids = list(catalog.object_keys)
        original_ra = list(catalog.ra_values)
        
        # Sort by position
        catalog.sort_objects_by_position(grid_divisions=10)
        
        # Verify that table was updated
        sorted_ids = list(catalog.object_keys)
        sorted_ra = list(catalog.ra_values)
        
        # Verify the spatial sort key column exists and is sorted
        assert "_source_catalog_spatial_sort_key" in catalog.table.colnames
        sort_keys = list(catalog.table["_source_catalog_spatial_sort_key"])  # type: ignore
        assert sort_keys == sorted(sort_keys), "Sort keys should be in ascending order"
        
        # Verify that all properties are consistent with the new table order
        for i in range(len(catalog)):
            assert catalog.object_keys[i] == catalog.table['id'][i]
            assert catalog.ra_values[i] == catalog.table['ra'][i]
            assert catalog.dec_values[i] == catalog.table['dec'][i]
        
        # Verify that data access through properties matches table directly
        assert list(catalog.object_keys) == list(catalog.table['id']) # type: ignore
    
    def test_sort_preserves_data_integrity(self):
        """Test that sorting preserves all data columns correctly."""
        test_table = Table()
        test_table['id'] = [1, 2, 3, 4, 5]
        test_table['ra'] = [100.5, 100.1, 100.9, 100.2, 100.7]
        test_table['dec'] = [20.5, 20.1, 20.9, 20.2, 20.7]
        test_table['x'] = [50.0, 10.0, 90.0, 20.0, 70.0]
        test_table['y'] = [50.0, 10.0, 90.0, 20.0, 70.0]
        test_table['mag_auto'] = [18.0, 18.1, 18.2, 18.3, 18.4]
        test_table['flux_auto'] = [100.0, 200.0, 300.0, 400.0, 500.0]
        
        catalog = SourceCatalog(test_table, copy=True)
        
        # Create mapping of id to mag before sorting
        id_to_mag = {str(catalog.table['id'][i]): catalog.table['mag_auto'][i] 
                     for i in range(len(catalog))}
        
        # Sort
        catalog.sort_objects_by_position(grid_divisions=10)
        
        # Verify that each id still maps to the same mag value
        for i in range(len(catalog)):
            obj_id = catalog.object_keys[i]
            expected_mag = id_to_mag[obj_id]
            actual_mag = catalog.table['mag_auto'][i]
            assert actual_mag == expected_mag, f"Data integrity lost for object {obj_id}"
    
    def test_sort_with_different_grid_divisions(self):
        """Test sorting with different grid division values."""
        test_table = Table()
        test_table['id'] = list(range(1, 21))
        test_table['ra'] = np.random.RandomState(42).uniform(100.0, 101.0, 20)
        test_table['dec'] = np.random.RandomState(42).uniform(20.0, 21.0, 20)
        test_table['x'] = np.random.RandomState(42).uniform(0, 1000, 20)
        test_table['y'] = np.random.RandomState(42).uniform(0, 1000, 20)
        
        # Test with grid_divisions=5
        catalog1 = SourceCatalog(test_table.copy(), copy=True)
        catalog1.sort_objects_by_position(grid_divisions=5)
        assert "_source_catalog_spatial_sort_key" in catalog1.table.colnames
        sort_keys1 = list(catalog1.table["_source_catalog_spatial_sort_key"])  # type: ignore
        assert sort_keys1 == sorted(sort_keys1)
        
        # Test with grid_divisions=20
        catalog2 = SourceCatalog(test_table.copy(), copy=True)
        catalog2.sort_objects_by_position(grid_divisions=20)
        sort_keys2 = list(catalog2.table["_source_catalog_spatial_sort_key"])  # type: ignore
        assert sort_keys2 == sorted(sort_keys2)
        

class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_catalog(self):
        """Test catalog with no objects."""
        tbl = Table()
        tbl['id'] = []
        tbl['ra'] = []
        tbl['dec'] = []
        tbl['x'] = []
        tbl['y'] = []
        
        catalog = SourceCatalog(tbl)
        
        assert len(catalog) == 0
        assert len(catalog.measurement_keys) == 0
        
    def test_catalog_with_no_measurements(self):
        """Test catalog with coordinates but no measurement columns."""
        tbl = Table()
        tbl['id'] = [1, 2, 3]
        tbl['ra'] = [10.0, 10.1, 10.2]
        tbl['dec'] = [20.0, 20.1, 20.2]
        tbl['x'] = [100.0, 200.0, 300.0]
        tbl['y'] = [150.0, 250.0, 350.0]
        
        catalog = SourceCatalog(tbl)
        
        assert len(catalog) == 3
        assert len(catalog.measurement_keys) == 0
    
    def test_catalog_repr(self):
        """Test catalog string representation."""
        tbl = Table()
        tbl['id'] = [1, 2, 3]
        tbl['ra'] = [10.0, 10.1, 10.2]
        tbl['dec'] = [20.0, 20.1, 20.2]
        tbl['x'] = [100.0, 200.0, 300.0]
        tbl['y'] = [150.0, 250.0, 350.0]
        tbl['mag_auto'] = [18.0, 18.5, 19.0]
        tbl['magerr_auto'] = [0.05, 0.06, 0.07]
        
        catalog = SourceCatalog(tbl)
        repr_str = repr(catalog)
        
        assert "SourceCatalog" in repr_str
        assert "(3," in repr_str  # n_objs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
