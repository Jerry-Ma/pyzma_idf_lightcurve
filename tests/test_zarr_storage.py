"""Test zarr storage persistence and consolidation.

This module tests critical data persistence features:
- Data persists to disk (not just in memory)
- Zarr file structure and metadata
- Region writes actually persist
- Storage can be closed and reopened
"""

import pytest
import numpy as np
from pathlib import Path
from astropy.table import Table

from pyzma_idf_lightcurve.lightcurve.datamodel import LightcurveStorage
from pyzma_idf_lightcurve.lightcurve.catalog import SourceCatalog


@pytest.fixture
def sample_catalog():
    """Create a sample catalog for testing."""
    tbl = Table()
    tbl['NUMBER'] = [1, 2, 3, 4, 5]
    tbl['ALPHA_J2000'] = [10.0, 10.1, 10.2, 10.3, 10.4]
    tbl['DELTA_J2000'] = [20.0, 20.1, 20.2, 20.3, 20.4]
    tbl['X_IMAGE'] = [100.0, 200.0, 300.0, 400.0, 500.0]
    tbl['Y_IMAGE'] = [150.0, 250.0, 350.0, 450.0, 550.0]
    tbl['MAG_AUTO'] = [18.0, 18.5, 19.0, 19.5, 20.0]
    tbl['MAGERR_AUTO'] = [0.05, 0.06, 0.07, 0.08, 0.09]
    tbl['MAG_ISO'] = [18.1, 18.6, 19.1, 19.6, 20.1]
    tbl['MAGERR_ISO'] = [0.06, 0.07, 0.08, 0.09, 0.10]
    return SourceCatalog(tbl, name="test")


@pytest.fixture
def epoch_keys():
    """Epoch keys for testing."""
    return ['epoch_1', 'epoch_2', 'epoch_3']


class TestZarrPersistence:
    """Test data persistence to disk."""
    
    def test_data_persists_after_close_and_reload(self, tmp_path, sample_catalog, epoch_keys):
        """Verify data persists when storage is closed and reopened."""
        storage_path = tmp_path / "test_storage"
        measurement_keys = sample_catalog.measurement_keys
        
        # Create and populate storage
        storage1 = LightcurveStorage(storage_path)
        storage1.create_storage(
            source_catalog=sample_catalog,
            epoch_keys=epoch_keys
        )
        
        # Populate first epoch with known values
        storage1.populate_epoch_from_catalog('epoch_1', sample_catalog, measurement_keys)
        
        # Get expected values before closing
        expected_lc_obj1 = storage1.get_object_lightcurve('1', 'test-auto').values.copy()
        expected_lc_obj3 = storage1.get_object_lightcurve('3', 'test-iso').values.copy()
        expected_epoch1 = storage1.get_epoch_data('epoch_1').values.copy()
        
        # Explicitly close storage (delete object to force flush)
        del storage1
        
        # Verify zarr directory structure exists
        assert storage_path.exists()
        assert (storage_path / 'lightcurves.zarr').exists()
        
        # Reopen from disk using instance method
        storage2 = LightcurveStorage(storage_path)
        storage2.load_storage(consolidated=True)
        
        # Verify data matches
        actual_lc_obj1 = storage2.get_object_lightcurve('1', 'test-auto').values
        actual_lc_obj3 = storage2.get_object_lightcurve('3', 'test-iso').values
        actual_epoch1 = storage2.get_epoch_data('epoch_1').values
        
        np.testing.assert_array_equal(actual_lc_obj1, expected_lc_obj1)
        np.testing.assert_array_equal(actual_lc_obj3, expected_lc_obj3)
        np.testing.assert_array_equal(actual_epoch1, expected_epoch1)
    
    def test_multiple_epoch_persistence(self, tmp_path, sample_catalog, epoch_keys):
        """Verify multiple epochs persist correctly."""
        storage_path = tmp_path / "test_storage"
        measurement_keys = sample_catalog.measurement_keys
        
        # Create storage and populate all epochs
        storage1 = LightcurveStorage(storage_path)
        storage1.create_storage(
            source_catalog=sample_catalog,
            epoch_keys=epoch_keys
        )
        
        for epoch_key in epoch_keys:
            storage1.populate_epoch_from_catalog(epoch_key, sample_catalog, measurement_keys)
        
        # Get all data
        expected_data = {}
        for epoch_key in epoch_keys:
            expected_data[epoch_key] = storage1.get_epoch_data(epoch_key).values.copy()
        
        # Close and reopen
        del storage1
        storage2 = LightcurveStorage(storage_path)
        storage2.load_storage(consolidated=True)
        
        # Verify all epochs persisted
        for epoch_key in epoch_keys:
            actual = storage2.get_epoch_data(epoch_key).values
            expected = expected_data[epoch_key]
            np.testing.assert_array_equal(actual, expected,
                                         err_msg=f"Epoch {epoch_key} data mismatch")
    
    def test_coordinates_persist(self, tmp_path, sample_catalog, epoch_keys):
        """Verify coordinate data persists via lightcurves DataArray."""
        storage_path = tmp_path / "test_storage"
        
        # Create storage
        storage1 = LightcurveStorage(storage_path)
        storage1.create_storage(
            source_catalog=sample_catalog,
            epoch_keys=epoch_keys
        )
        assert storage1.lightcurves is not None  # Type guard for linter
        
        # Get coordinate data from lightcurves (coordinates are stored as non-dimension coords)
        expected_ra = storage1.lightcurves.coords['ra'].values.copy()
        expected_dec = storage1.lightcurves.coords['dec'].values.copy()
        expected_x_image = storage1.lightcurves.coords['x_image'].values.copy()
        expected_y_image = storage1.lightcurves.coords['y_image'].values.copy()
        
        # Close and reopen
        del storage1
        storage2 = LightcurveStorage(storage_path)
        storage2.load_storage(consolidated=True)
        assert storage2.lightcurves is not None  # Type guard for linter
        
        # Verify coordinates
        np.testing.assert_array_equal(storage2.lightcurves.coords['ra'].values, expected_ra)
        np.testing.assert_array_equal(storage2.lightcurves.coords['dec'].values, expected_dec)
        np.testing.assert_array_equal(storage2.lightcurves.coords['x_image'].values, expected_x_image)
        np.testing.assert_array_equal(storage2.lightcurves.coords['y_image'].values, expected_y_image)
    
    def test_storage_info_persists(self, tmp_path, sample_catalog, epoch_keys):
        """Verify storage metadata persists."""
        storage_path = tmp_path / "test_storage"
        
        # Create storage
        storage1 = LightcurveStorage(storage_path)
        storage1.create_storage(
            source_catalog=sample_catalog,
            epoch_keys=epoch_keys
        )
        
        # Get info
        info1 = storage1.get_storage_info()
        
        # Close and reopen
        del storage1
        storage2 = LightcurveStorage(storage_path)
        storage2.load_storage(consolidated=True)
        
        # Verify info matches (use keys that actually exist)
        info2 = storage2.get_storage_info()
        assert info1['n_objects'] == info2['n_objects']
        assert info1['n_epochs'] == info2['n_epochs']
        assert info1['n_measurements'] == info2['n_measurements']
        assert info1['measurement_keys'] == info2['measurement_keys']
        assert info1['value_keys'] == info2['value_keys']
        assert info1['data_shape'] == info2['data_shape']


class TestZarrStructure:
    """Test zarr file structure and metadata."""
    
    def test_zarr_metadata_structure(self, tmp_path, sample_catalog, epoch_keys):
        """Verify zarr creates proper metadata structure."""
        storage_path = tmp_path / "test_storage"
        
        # Create storage
        storage = LightcurveStorage(storage_path)
        storage.create_storage(
            source_catalog=sample_catalog,
            epoch_keys=epoch_keys
        )
        
        # Check zarr v3 metadata structure exists
        zarr_path = storage_path / 'lightcurves.zarr'
        assert zarr_path.exists()
        # Zarr v3 uses zarr.json instead of .zarray/.zattrs
        assert (zarr_path / 'zarr.json').exists()
        # Check that some dimension directories exist
        assert (zarr_path / 'lightcurves').exists()
    
    def test_zarr_chunks_created(self, tmp_path, sample_catalog, epoch_keys):
        """Verify zarr creates chunk files when data is written."""
        storage_path = tmp_path / "test_storage"
        measurement_keys = sample_catalog.measurement_keys
        
        # Create and populate storage
        storage = LightcurveStorage(storage_path)
        storage.create_storage(
            source_catalog=sample_catalog,
            epoch_keys=epoch_keys
        )
        storage.populate_epoch_from_catalog('epoch_1', sample_catalog, measurement_keys)
        
        # Close to ensure flush
        del storage
        
        # Check that chunk files exist
        zarr_path = storage_path / 'lightcurves.zarr'
        
        # Zarr v3 uses different directory structure - check for data subdirectory
        # or just verify the directory has files
        assert len(list(zarr_path.iterdir())) > 2  # More than just .zarray and .zattrs


class TestRegionWrites:
    """Test that region writes persist data correctly."""
    
    def test_region_write_persistence(self, tmp_path, sample_catalog, epoch_keys):
        """Verify region writes actually persist to disk."""
        storage_path = tmp_path / "test_storage"
        measurement_keys = sample_catalog.measurement_keys
        
        # Create storage
        storage1 = LightcurveStorage(storage_path)
        storage1.create_storage(
            source_catalog=sample_catalog,
            epoch_keys=epoch_keys
        )
        
        # Populate using method that should use region writes
        storage1.populate_epoch_from_catalog('epoch_1', sample_catalog, measurement_keys)
        
        # Force close without explicit save call
        del storage1
        
        # Reopen and verify data exists
        storage2 = LightcurveStorage(storage_path)
        storage2.load_storage(consolidated=True)
        epoch_data = storage2.get_epoch_data('epoch_1')
        
        # Check that we have actual magnitude values, not all NaN
        mag_data = epoch_data.sel(value='mag').values
        assert not np.all(np.isnan(mag_data)), "All magnitude data is NaN - region write failed"
        
        # Verify specific values match catalog
        obj1_mag = storage2.get_object_lightcurve('1', 'test-auto').sel(value='mag', epoch='epoch_1').values
        assert obj1_mag == pytest.approx(18.0), f"Expected 18.0, got {obj1_mag}"
    
    def test_incremental_population_persistence(self, tmp_path, sample_catalog, epoch_keys):
        """Verify incremental population (epoch by epoch) persists correctly."""
        storage_path = tmp_path / "test_storage"
        measurement_keys = sample_catalog.measurement_keys
        
        # Create storage
        storage1 = LightcurveStorage(storage_path)
        storage1.create_storage(
            source_catalog=sample_catalog,
            epoch_keys=epoch_keys
        )
        
        # Populate first epoch
        storage1.populate_epoch_from_catalog('epoch_1', sample_catalog, measurement_keys)
        del storage1
        
        # Reopen and populate second epoch
        storage2 = LightcurveStorage(storage_path)
        storage2.load_storage(consolidated=False)  # Use non-consolidated for writes
        storage2.populate_epoch_from_catalog('epoch_2', sample_catalog, measurement_keys)
        del storage2
        
        # Reopen and verify both epochs have data
        storage3 = LightcurveStorage(storage_path)
        storage3.load_storage(consolidated=True)
        
        # Check epoch_1 data
        epoch1_data = storage3.get_epoch_data('epoch_1').sel(value='mag').values
        assert not np.all(np.isnan(epoch1_data)), "Epoch 1 data lost after reopening"
        
        # Check epoch_2 data
        epoch2_data = storage3.get_epoch_data('epoch_2').sel(value='mag').values
        assert not np.all(np.isnan(epoch2_data)), "Epoch 2 data not persisted"
        
        # Check epoch_3 is still NaN (not populated)
        epoch3_data = storage3.get_epoch_data('epoch_3').sel(value='mag').values
        assert np.all(np.isnan(epoch3_data)), "Epoch 3 should still be NaN"


class TestStorageReload:
    """Test loading existing storage from disk."""
    
    def test_load_storage_without_create(self, tmp_path, sample_catalog, epoch_keys):
        """Verify we can load existing storage without calling create_storage."""
        storage_path = tmp_path / "test_storage"
        measurement_keys = sample_catalog.measurement_keys
        
        # Create and populate storage
        storage1 = LightcurveStorage(storage_path)
        storage1.create_storage(
            source_catalog=sample_catalog,
            epoch_keys=epoch_keys
        )
        storage1.populate_epoch_from_catalog('epoch_1', sample_catalog, measurement_keys)
        
        # Get expected value
        expected = storage1.get_object_lightcurve('2', 'test-auto').sel(value='mag', epoch='epoch_1').values
        del storage1
        
        # Load existing storage (should not need create_storage)
        storage2 = LightcurveStorage(storage_path)
        storage2.load_storage(consolidated=True)
        
        # Verify we can access data
        actual = storage2.get_object_lightcurve('2', 'test-auto').sel(value='mag', epoch='epoch_1').values
        assert actual == pytest.approx(expected)
    
    def test_load_nonexistent_storage_fails(self, tmp_path):
        """Verify loading non-existent storage raises appropriate error."""
        storage_path = tmp_path / "nonexistent_storage"
        
        storage = LightcurveStorage(storage_path)
        with pytest.raises(RuntimeError, match="Storage not found"):
            storage.load_storage()
    
    def test_storage_path_validation(self, tmp_path, sample_catalog, epoch_keys):
        """Verify storage path is validated correctly."""
        storage_path = tmp_path / "test_storage"
        
        # Create storage
        storage1 = LightcurveStorage(storage_path)
        storage1.create_storage(
            source_catalog=sample_catalog,
            epoch_keys=epoch_keys
        )
        
        # Path should exist
        assert storage_path.exists()
        assert (storage_path / 'lightcurves.zarr').exists()


class TestDataIntegrity:
    """Test data integrity across save/load cycles."""
    
    def test_nan_values_preserved(self, tmp_path, sample_catalog, epoch_keys):
        """Verify NaN values are preserved across save/load."""
        storage_path = tmp_path / "test_storage"
        
        # Create storage (all values start as NaN)
        storage1 = LightcurveStorage(storage_path)
        storage1.create_storage(
            source_catalog=sample_catalog,
            epoch_keys=epoch_keys
        )
        
        # Check initial NaN state
        epoch2_data = storage1.get_epoch_data('epoch_2').sel(value='mag').values
        assert np.all(np.isnan(epoch2_data))
        
        del storage1
        
        # Reload and verify NaN preserved
        storage2 = LightcurveStorage(storage_path)
        storage2.load_storage(consolidated=True)
        epoch2_data_reloaded = storage2.get_epoch_data('epoch_2').sel(value='mag').values
        assert np.all(np.isnan(epoch2_data_reloaded))
    
    def test_exact_value_preservation(self, tmp_path, sample_catalog, epoch_keys):
        """Verify exact floating point values are preserved."""
        storage_path = tmp_path / "test_storage"
        measurement_keys = sample_catalog.measurement_keys
        
        # Create and populate storage
        storage1 = LightcurveStorage(storage_path)
        storage1.create_storage(
            source_catalog=sample_catalog,
            epoch_keys=epoch_keys
        )
        storage1.populate_epoch_from_catalog('epoch_1', sample_catalog, measurement_keys)
        
        # Get all magnitude values
        expected_mags = []
        expected_errs = []
        for obj_key in ['1', '2', '3', '4', '5']:
            lc = storage1.get_object_lightcurve(obj_key, 'test-auto')
            expected_mags.append(lc.sel(value='mag', epoch='epoch_1').values)
            expected_errs.append(lc.sel(value='mag_err', epoch='epoch_1').values)
        
        del storage1
        
        # Reload and verify exact values
        storage2 = LightcurveStorage(storage_path)
        storage2.load_storage(consolidated=True)
        for i, obj_key in enumerate(['1', '2', '3', '4', '5']):
            lc = storage2.get_object_lightcurve(obj_key, 'test-auto')
            actual_mag = lc.sel(value='mag', epoch='epoch_1').values
            actual_err = lc.sel(value='mag_err', epoch='epoch_1').values
            
            # Use exact equality for floating point (should be bit-exact)
            assert actual_mag == expected_mags[i]
            assert actual_err == expected_errs[i]
