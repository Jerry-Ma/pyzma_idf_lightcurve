"""
Unit tests for SourceCatalog and measurement type handling.

Tests:
- SourceCatalog initialization with and without name parameter
- Measurement type inference from column names
- Column suffix parsing (AUTO, ISO, APER, APER_1, etc.)
- Coordinate extraction and spatial sorting
- Measurement extraction and validation
"""

import pytest
import numpy as np
from astropy.table import Table

from pyzma_idf_lightcurve.lightcurve.catalog import SourceCatalog, MeasurementKey


class TestMeasurementKey:
    """Test MeasurementKey template parsing and generation."""
    
    def test_make_measurement_key(self):
        """Test creating measurement key identifier."""
        mt = MeasurementKey.make(tbl_name="ch1_sci", col_suffix="auto")
        assert mt == "ch1_sci-auto"
        
    def test_parse_measurement_key(self):
        """Test parsing measurement key identifier."""
        parsed = MeasurementKey.parse("ch2_sci_clean-iso")
        assert parsed["tbl_name"] == "ch2_sci_clean"
        assert parsed["col_suffix"] == "iso"
        
    def test_parse_measurement_key_with_underscore(self):
        """Test parsing measurement key with aperture number."""
        parsed = MeasurementKey.parse("ch1_mosaic-aper_3")
        assert parsed["tbl_name"] == "ch1_mosaic"
        assert parsed["col_suffix"] == "aper_3"


class TestSourceCatalogBasic:
    """Test basic SourceCatalog functionality."""
    
    def test_init_with_name(self):
        """Test initialization with custom name."""
        tbl = Table()
        tbl['NUMBER'] = [1, 2, 3]
        tbl['ALPHA_J2000'] = [10.0, 10.1, 10.2]
        tbl['DELTA_J2000'] = [20.0, 20.1, 20.2]
        tbl['X_IMAGE'] = [100.0, 200.0, 300.0]
        tbl['Y_IMAGE'] = [150.0, 250.0, 350.0]
        
        catalog = SourceCatalog(tbl, name="test_catalog")
        assert catalog.name == "test_catalog"
        assert len(catalog) == 3
        
    def test_init_default_name(self):
        """Test initialization with default name."""
        tbl = Table()
        tbl['NUMBER'] = [1, 2, 3]
        tbl['ALPHA_J2000'] = [10.0, 10.1, 10.2]
        tbl['DELTA_J2000'] = [20.0, 20.1, 20.2]
        tbl['X_IMAGE'] = [100.0, 200.0, 300.0]
        tbl['Y_IMAGE'] = [150.0, 250.0, 350.0]
        
        catalog = SourceCatalog(tbl)  # No name parameter
        assert catalog.name == "default"
        
    def test_object_ids(self):
        """Test object key extraction (converted NUMBER column to strings)."""
        tbl = Table()
        tbl['NUMBER'] = [1001, 1002, 1003]
        tbl['ALPHA_J2000'] = [10.0, 10.1, 10.2]
        tbl['DELTA_J2000'] = [20.0, 20.1, 20.2]
        tbl['X_IMAGE'] = [100.0, 200.0, 300.0]
        tbl['Y_IMAGE'] = [150.0, 250.0, 350.0]

        catalog = SourceCatalog(tbl, name="test")
        assert list(catalog.object_keys) == ['1001', '1002', '1003']
    
    def test_coordinate_extraction(self):
        """Test coordinate value extraction."""
        tbl = Table()
        tbl['NUMBER'] = [1, 2]
        tbl['ALPHA_J2000'] = [15.5, 15.6]
        tbl['DELTA_J2000'] = [-30.0, -30.1]
        tbl['X_IMAGE'] = [100.0, 200.0]
        tbl['Y_IMAGE'] = [150.0, 250.0]
        
        catalog = SourceCatalog(tbl, name="test")
        
        assert len(catalog.ra_values) == 2
        assert len(catalog.dec_values) == 2
        assert len(catalog.x_values) == 2
        assert len(catalog.y_values) == 2
        
        assert np.allclose(catalog.ra_values, [15.5, 15.6])
        assert np.allclose(catalog.dec_values, [-30.0, -30.1])


class TestMeasurementTypeInference:
    """Test automatic measurement type inference from columns."""
    
    def test_infer_auto_measurement(self):
        """Test inference of AUTO measurement type."""
        tbl = Table()
        tbl['NUMBER'] = [1, 2, 3]
        tbl['ALPHA_J2000'] = [10.0, 10.1, 10.2]
        tbl['DELTA_J2000'] = [20.0, 20.1, 20.2]
        tbl['X_IMAGE'] = [100.0, 200.0, 300.0]
        tbl['Y_IMAGE'] = [150.0, 250.0, 350.0]
        tbl['MAG_AUTO'] = [18.0, 18.5, 19.0]
        tbl['MAGERR_AUTO'] = [0.05, 0.06, 0.07]
        
        catalog = SourceCatalog(tbl, name="test_table")
        
        measurement_keys = catalog.measurement_keys
        assert "test_table-auto" in measurement_keys
        
        # Check column mapping
        colnames = catalog.measurement_key_colnames["test_table-auto"]
        assert colnames["mag"] == "MAG_AUTO"
        assert colnames["mag_err"] == "MAGERR_AUTO"
        
    def test_infer_iso_measurement(self):
        """Test inference of ISO measurement type."""
        tbl = Table()
        tbl['NUMBER'] = [1, 2]
        tbl['ALPHA_J2000'] = [10.0, 10.1]
        tbl['DELTA_J2000'] = [20.0, 20.1]
        tbl['X_IMAGE'] = [100.0, 200.0]
        tbl['Y_IMAGE'] = [150.0, 250.0]
        tbl['MAG_ISO'] = [17.5, 18.0]
        tbl['MAGERR_ISO'] = [0.04, 0.05]
        
        catalog = SourceCatalog(tbl, name="my_catalog")
        
        assert "my_catalog-iso" in catalog.measurement_keys
        
        colnames = catalog.measurement_key_colnames["my_catalog-iso"]
        assert colnames["mag"] == "MAG_ISO"
        assert colnames["mag_err"] == "MAGERR_ISO"
        
    def test_infer_aperture_measurements(self):
        """Test inference of aperture photometry measurement types."""
        tbl = Table()
        tbl['NUMBER'] = [1, 2]
        tbl['ALPHA_J2000'] = [10.0, 10.1]
        tbl['DELTA_J2000'] = [20.0, 20.1]
        tbl['X_IMAGE'] = [100.0, 200.0]
        tbl['Y_IMAGE'] = [150.0, 250.0]
        
        # Add aperture columns
        tbl['MAG_APER'] = [18.0, 18.5]
        tbl['MAGERR_APER'] = [0.05, 0.06]
        tbl['MAG_APER_1'] = [18.1, 18.6]
        tbl['MAGERR_APER_1'] = [0.05, 0.06]
        tbl['MAG_APER_2'] = [18.2, 18.7]
        tbl['MAGERR_APER_2'] = [0.06, 0.07]
        
        catalog = SourceCatalog(tbl, name="aperture_test")
        
        # Should detect all three aperture types
        assert "aperture_test-aper" in catalog.measurement_keys
        assert "aperture_test-aper_1" in catalog.measurement_keys
        assert "aperture_test-aper_2" in catalog.measurement_keys
        
    def test_infer_mixed_measurements(self):
        """Test inference when multiple measurement types present."""
        tbl = Table()
        tbl['NUMBER'] = [1, 2, 3]
        tbl['ALPHA_J2000'] = [10.0, 10.1, 10.2]
        tbl['DELTA_J2000'] = [20.0, 20.1, 20.2]
        tbl['X_IMAGE'] = [100.0, 200.0, 300.0]
        tbl['Y_IMAGE'] = [150.0, 250.0, 350.0]
        
        # Add AUTO, ISO, and aperture columns
        tbl['MAG_AUTO'] = [18.0, 18.5, 19.0]
        tbl['MAGERR_AUTO'] = [0.05, 0.06, 0.07]
        tbl['MAG_ISO'] = [17.5, 18.0, 18.5]
        tbl['MAGERR_ISO'] = [0.04, 0.05, 0.06]
        tbl['MAG_APER_1'] = [18.1, 18.6, 19.1]
        tbl['MAGERR_APER_1'] = [0.05, 0.06, 0.07]
        
        catalog = SourceCatalog(tbl, name="mixed")
        
        measurement_keys = catalog.measurement_keys
        assert "mixed-auto" in measurement_keys
        assert "mixed-iso" in measurement_keys
        assert "mixed-aper_1" in measurement_keys
        assert len(measurement_keys) == 3
        
    def test_skip_measurement_without_error_column(self):
        """Test that measurements without error columns are skipped."""
        tbl = Table()
        tbl['NUMBER'] = [1, 2]
        tbl['ALPHA_J2000'] = [10.0, 10.1]
        tbl['DELTA_J2000'] = [20.0, 20.1]
        tbl['X_IMAGE'] = [100.0, 200.0]
        tbl['Y_IMAGE'] = [150.0, 250.0]
        
        # MAG_AUTO but no MAGERR_AUTO
        tbl['MAG_AUTO'] = [18.0, 18.5]
        # MAG_ISO with MAGERR_ISO
        tbl['MAG_ISO'] = [17.5, 18.0]
        tbl['MAGERR_ISO'] = [0.04, 0.05]
        
        catalog = SourceCatalog(tbl, name="incomplete")
        
        # Should only detect ISO (has error column)
        measurement_keys = catalog.measurement_keys
        assert "incomplete-iso" in measurement_keys
        assert "incomplete-auto" not in measurement_keys


class TestMeasurementExtraction:
    """Test extracting measurements from catalog."""
    
    def test_extract_single_measurement(self):
        """Test extracting a single measurement type."""
        tbl = Table()
        tbl['NUMBER'] = [1, 2, 3]
        tbl['ALPHA_J2000'] = [10.0, 10.1, 10.2]
        tbl['DELTA_J2000'] = [20.0, 20.1, 20.2]
        tbl['X_IMAGE'] = [100.0, 200.0, 300.0]
        tbl['Y_IMAGE'] = [150.0, 250.0, 350.0]
        tbl['MAG_AUTO'] = np.array([18.0, 18.5, 19.0])
        tbl['MAGERR_AUTO'] = np.array([0.05, 0.06, 0.07])
        
        catalog = SourceCatalog(tbl, name="extract_test")
        
        measurements = catalog.extract_measurements(["extract_test-auto"])
        
        assert "extract_test-auto" in measurements
        assert "mag" in measurements["extract_test-auto"]
        assert "mag_err" in measurements["extract_test-auto"]
        
        mag_data = measurements["extract_test-auto"]["mag"]
        err_data = measurements["extract_test-auto"]["mag_err"]
        
        assert np.allclose(mag_data, [18.0, 18.5, 19.0])
        assert np.allclose(err_data, [0.05, 0.06, 0.07])
        
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
        tbl['MAG_ISO'] = np.array([17.5, 18.0])
        tbl['MAGERR_ISO'] = np.array([0.04, 0.05])
        
        catalog = SourceCatalog(tbl, name="multi")
        
        measurements = catalog.extract_measurements(["multi-auto", "multi-iso"])
        
        assert "multi-auto" in measurements
        assert "multi-iso" in measurements
        assert len(measurements) == 2


class TestSpatialOrdering:
    """Test spatial ordering functionality."""
    
    def test_spatial_ordering_basic(self):
        """Test basic spatial ordering."""
        n_obj = 20
        tbl = Table()
        tbl['NUMBER'] = list(range(1, n_obj + 1))
        tbl['ALPHA_J2000'] = np.random.uniform(10.0, 11.0, n_obj)
        tbl['DELTA_J2000'] = np.random.uniform(20.0, 21.0, n_obj)
        tbl['X_IMAGE'] = np.random.uniform(0, 1000, n_obj)
        tbl['Y_IMAGE'] = np.random.uniform(0, 1000, n_obj)
        
        catalog = SourceCatalog(tbl, name="spatial")
        
        ordered_ids = catalog.get_spatially_ordered_keys(grid_divisions=5)
        
        assert len(ordered_ids) == n_obj
        # Object keys are now strings
        assert set(ordered_ids) == set(str(i) for i in range(1, n_obj + 1))
    
    def test_spatial_ordering_with_nan(self):
        """Test spatial ordering handles NaN coordinates."""
        tbl = Table()
        tbl['NUMBER'] = [1, 2, 3, 4]
        tbl['ALPHA_J2000'] = [10.0, 10.1, np.nan, 10.3]
        tbl['DELTA_J2000'] = [20.0, 20.1, 20.2, np.nan]
        tbl['X_IMAGE'] = [100.0, 200.0, 300.0, 400.0]
        tbl['Y_IMAGE'] = [150.0, 250.0, 350.0, 450.0]
        
        catalog = SourceCatalog(tbl, name="nan_test")
        
        ordered_ids = catalog.get_spatially_ordered_keys()
        
        # Should still return all IDs (NaN go to end)
        # Object keys are now strings
        assert len(ordered_ids) == 4
        assert set(ordered_ids) == {'1', '2', '3', '4'}


class TestCoordinateDictionary:
    """Test coordinate dictionary creation."""
    
    def test_get_coordinate_dict(self):
        """Test creating coordinate dictionary."""
        tbl = Table()
        tbl['NUMBER'] = [1, 2, 3]
        tbl['ALPHA_J2000'] = [10.0, 10.1, 10.2]
        tbl['DELTA_J2000'] = [20.0, 20.1, 20.2]
        tbl['X_IMAGE'] = [100.0, 200.0, 300.0]
        tbl['Y_IMAGE'] = [150.0, 250.0, 350.0]
        
        catalog = SourceCatalog(tbl, name="coord_dict")
        
        coord_dict = catalog.get_coordinate_dict()
        
        assert len(coord_dict) == 3
        # Object keys are now strings
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
        tbl['NUMBER'] = [1, 2, 3, 4, 5]
        tbl['ALPHA_J2000'] = [10.0, 10.1, 10.2, 10.3, 10.4]
        tbl['DELTA_J2000'] = [20.0, 20.1, 20.2, 20.3, 20.4]
        tbl['X_IMAGE'] = [100.0, 200.0, 300.0, 400.0, 500.0]
        tbl['Y_IMAGE'] = [150.0, 250.0, 350.0, 450.0, 550.0]
        
        catalog = SourceCatalog(tbl, name="coord_arrays")
        
        # Get coordinates for subset in specific order (use string keys)
        object_ids = ['3', '1', '5']
        ra_vals, dec_vals, x_vals, y_vals = catalog.get_coordinate_arrays_for_objects(object_ids)
        
        assert len(ra_vals) == 3
        assert ra_vals == [10.2, 10.0, 10.4]  # Order matches requested object_ids
        assert dec_vals == [20.2, 20.0, 20.4]
        assert x_vals == [300.0, 100.0, 500.0]
        assert y_vals == [350.0, 150.0, 550.0]


class TestValidMeasurementMask:
    """Test valid measurement mask generation."""
    
    def test_valid_measurement_mask(self):
        """Test creating mask for valid measurements."""
        tbl = Table()
        tbl['NUMBER'] = [1, 2, 3, 4]
        tbl['ALPHA_J2000'] = [10.0, 10.1, 10.2, 10.3]
        tbl['DELTA_J2000'] = [20.0, 20.1, 20.2, 20.3]
        tbl['X_IMAGE'] = [100.0, 200.0, 300.0, 400.0]
        tbl['Y_IMAGE'] = [150.0, 250.0, 350.0, 450.0]
        
        # Mix of valid and invalid measurements
        tbl['MAG_AUTO'] = np.array([18.0, np.nan, 18.5, 99.0])  # One NaN, one out of range
        tbl['MAGERR_AUTO'] = np.array([0.05, 0.06, np.inf, 0.08])  # One inf
        
        catalog = SourceCatalog(tbl, name="valid_mask")
        
        mask = catalog.get_valid_measurement_mask(["valid_mask-auto"])
        
        # Only first object should be valid
        assert mask[0] == True
        assert mask[1] == False  # NaN mag
        assert mask[2] == False  # inf error
        assert mask[3] == False  # mag out of range


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_catalog(self):
        """Test catalog with no objects."""
        tbl = Table()
        tbl['NUMBER'] = []
        tbl['ALPHA_J2000'] = []
        tbl['DELTA_J2000'] = []
        tbl['X_IMAGE'] = []
        tbl['Y_IMAGE'] = []
        
        catalog = SourceCatalog(tbl, name="empty")
        
        assert len(catalog) == 0
        assert len(catalog.measurement_keys) == 0
        
    def test_catalog_with_no_measurements(self):
        """Test catalog with coordinates but no magnitude columns."""
        tbl = Table()
        tbl['NUMBER'] = [1, 2, 3]
        tbl['ALPHA_J2000'] = [10.0, 10.1, 10.2]
        tbl['DELTA_J2000'] = [20.0, 20.1, 20.2]
        tbl['X_IMAGE'] = [100.0, 200.0, 300.0]
        tbl['Y_IMAGE'] = [150.0, 250.0, 350.0]
        
        catalog = SourceCatalog(tbl, name="no_meas")
        
        assert len(catalog) == 3
        assert len(catalog.measurement_keys) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
