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

import pytest
import numpy as np
from astropy.table import Table
import re
import dataclasses

from pyzma_idf_lightcurve.lightcurve.catalog import (
    SourceCatalog,
    SourceCatalogTableTransform,
    SExtractorTableTransform,
    SourceCatalogDataKey,
    MeasurementKey,
    MeasurementColname,
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
    
    def test_extract_measurements_default(self):
        """Test extracting measurements with default transform."""
        tbl = Table()
        tbl['id'] = [1, 2, 3]
        tbl['ra'] = [10.0, 10.1, 10.2]
        tbl['dec'] = [20.0, 20.1, 20.2]
        tbl['x'] = [100.0, 200.0, 300.0]
        tbl['y'] = [150.0, 250.0, 350.0]
        tbl['mag_auto'] = np.array([18.0, 18.5, 19.0])
        tbl['magerr_auto'] = np.array([0.05, 0.06, 0.07])
        
        catalog = SourceCatalog(tbl)
        measurements = catalog.extract_measurements(["auto"])
        
        assert "auto" in measurements
        # Note: extract_measurements returns dict[measurement, array]
        # The actual structure may vary based on implementation
        
    def test_extract_multiple_measurements(self):
        """Test extracting multiple measurement types."""
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
        
        measurements = catalog.extract_measurements(["auto", "iso"])
        
        assert "auto" in measurements
        assert "iso" in measurements


class TestSpatialOrdering:
    """Test spatial ordering functionality."""
    
    def test_spatial_ordering_basic(self):
        """Test basic spatial ordering."""
        n_obj = 20
        np.random.seed(42)
        tbl = Table()
        tbl['id'] = list(range(1, n_obj + 1))
        tbl['ra'] = np.random.uniform(10.0, 11.0, n_obj)
        tbl['dec'] = np.random.uniform(20.0, 21.0, n_obj)
        tbl['x'] = np.random.uniform(0, 1000, n_obj)
        tbl['y'] = np.random.uniform(0, 1000, n_obj)
        
        catalog = SourceCatalog(tbl)
        
        ordered_keys = catalog.get_spatially_ordered_keys(grid_divisions=5)
        
        assert len(ordered_keys) == n_obj
        # Object keys are strings
        assert set(ordered_keys) == set(str(i) for i in range(1, n_obj + 1))
    
    def test_spatial_ordering_with_nan(self):
        """Test spatial ordering handles NaN coordinates."""
        tbl = Table()
        tbl['id'] = [1, 2, 3, 4]
        tbl['ra'] = [10.0, 10.1, np.nan, 10.3]
        tbl['dec'] = [20.0, 20.1, 20.2, np.nan]
        tbl['x'] = [100.0, 200.0, 300.0, 400.0]
        tbl['y'] = [150.0, 250.0, 350.0, 450.0]
        
        catalog = SourceCatalog(tbl)
        
        ordered_keys = catalog.get_spatially_ordered_keys()
        
        # Should still return all keys (NaN go to end)
        assert len(ordered_keys) == 4
        assert set(ordered_keys) == {'1', '2', '3', '4'}


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
            ra_col: str = "RA_DEG"
            dec_col: str = "DEC_DEG"
            x_col: str = "PIX_X"
            y_col: str = "PIX_Y"
            obj_key_col: tuple = ("OBJ_ID", lambda col: col.astype(str))
        
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
