"""
Test correctness of populate_epoch_from_catalog implementations.

Verifies that v0, v1, and v2 produce identical results.
"""

import numpy as np
import pytest
import shutil
from pathlib import Path

from pyzma_idf_lightcurve.lightcurve.catalog import SourceCatalog
from pyzma_idf_lightcurve.lightcurve.datamodel import LightcurveStorage


@pytest.fixture
def temp_storage_dir(tmp_path):
    """Create temporary directory for test storage."""
    storage_dir = tmp_path / "correctness_test"
    storage_dir.mkdir(exist_ok=True)
    yield storage_dir
    if storage_dir.exists():
        shutil.rmtree(storage_dir)


@pytest.fixture
def test_catalog():
    """Create a small test catalog with known values."""
    from astropy.table import Table
    
    # Create simple test data with 100 objects, 3 measurements
    n_objects = 100
    
    table = Table()
    # Use SExtractor-compatible column names
    table['NUMBER'] = np.arange(1, n_objects + 1)
    table['ALPHA_J2000'] = np.linspace(150.0, 150.5, n_objects)
    table['DELTA_J2000'] = np.linspace(2.0, 2.5, n_objects)
    table['X_IMAGE'] = np.linspace(100.0, 1000.0, n_objects)
    table['Y_IMAGE'] = np.linspace(100.0, 1000.0, n_objects)
    
    # Add measurements with predictable values
    # mag values: 20.0 + object_index * 0.01
    # err values: 0.05 + object_index * 0.0001
    # Use uppercase format as expected by SourceCatalog (MAG_*, MAGERR_*)
    for meas_idx, meas_name in enumerate(['AUTO', 'ISO', 'APER']):
        table[f'MAG_{meas_name}'] = 20.0 + np.arange(n_objects) * 0.01 + meas_idx * 0.1
        table[f'MAGERR_{meas_name}'] = 0.05 + np.arange(n_objects) * 0.0001 + meas_idx * 0.01
    
    return SourceCatalog(table)


@pytest.fixture
def measurement_keys(test_catalog):
    """Return measurement keys from the test catalog."""
    # Use the actual measurement keys from the catalog, not plain strings
    return test_catalog.measurement_keys


@pytest.fixture
def epoch_key():
    """Test epoch key."""
    return 'r58520832'


class TestPopulateCorrectness:
    """Test that all populate versions produce identical results."""
    
    def create_and_populate_storage(self, storage_path, catalog, epoch_key, measurement_keys, version):
        """Helper to create and populate storage with specific version."""
        # Clear existing storage
        if storage_path.exists():
            shutil.rmtree(storage_path)
        storage_path.mkdir(exist_ok=True)
        
        # Create storage
        storage = LightcurveStorage(storage_path)
        storage.create_storage(
            source_catalog=catalog,
            epoch_keys=[epoch_key],
        )
        
        # Populate using specified version
        if version == 'v0':
            storage.populate_epoch_from_catalog_v0(
                epoch_key=epoch_key,
                source_catalog=catalog,
                measurement_keys=measurement_keys,
            )
        elif version == 'v1':
            storage.populate_epoch_from_catalog_v1(
                epoch_key=epoch_key,
                source_catalog=catalog,
                measurement_keys=measurement_keys,
            )
        elif version == 'v2':
            storage.populate_epoch_from_catalog_v2(
                epoch_key=epoch_key,
                source_catalog=catalog,
                measurement_keys=measurement_keys,
            )
        else:
            raise ValueError(f"Unknown version: {version}")
        
        # Reload to ensure data is from disk
        storage.load_storage(consolidated=False)
        
        return storage
    
    def test_v0_v1_identical(self, temp_storage_dir, test_catalog, epoch_key, measurement_keys):
        """Test that v0 and v1 produce identical results."""
        # Create storages with v0 and v1
        storage_v0 = self.create_and_populate_storage(
            temp_storage_dir / "v0",
            test_catalog,
            epoch_key,
            measurement_keys,
            version='v0'
        )
        
        storage_v1 = self.create_and_populate_storage(
            temp_storage_dir / "v1",
            test_catalog,
            epoch_key,
            measurement_keys,
            version='v1'
        )
        
        # Compare data arrays
        data_v0 = storage_v0.lightcurves.values
        data_v1 = storage_v1.lightcurves.values
        
        # Check shapes match
        assert data_v0.shape == data_v1.shape, \
            f"Shape mismatch: v0={data_v0.shape}, v1={data_v1.shape}"
        
        # Check values are identical (using allclose to handle floating point)
        # For NaN values, they should be in the same positions
        nan_mask_v0 = np.isnan(data_v0)
        nan_mask_v1 = np.isnan(data_v1)
        
        assert np.array_equal(nan_mask_v0, nan_mask_v1), \
            "NaN positions differ between v0 and v1"
        
        # Compare non-NaN values
        valid_v0 = data_v0[~nan_mask_v0]
        valid_v1 = data_v1[~nan_mask_v1]
        
        assert np.allclose(valid_v0, valid_v1, rtol=1e-10, atol=1e-10), \
            "Data values differ between v0 and v1"
        
        print(f"✓ v0 and v1 produce identical results")
        print(f"  Shape: {data_v0.shape}")
        print(f"  Non-NaN values: {(~nan_mask_v0).sum()}")
        print(f"  NaN values: {nan_mask_v0.sum()}")
    
    def test_v1_v2_identical(self, temp_storage_dir, test_catalog, epoch_key, measurement_keys):
        """Test that v1 and v2 produce identical results."""
        # Create storages with v1 and v2
        storage_v1 = self.create_and_populate_storage(
            temp_storage_dir / "v1",
            test_catalog,
            epoch_key,
            measurement_keys,
            version='v1'
        )
        
        storage_v2 = self.create_and_populate_storage(
            temp_storage_dir / "v2",
            test_catalog,
            epoch_key,
            measurement_keys,
            version='v2'
        )
        
        # Compare data arrays
        data_v1 = storage_v1.lightcurves.values
        data_v2 = storage_v2.lightcurves.values
        
        # Check shapes match
        assert data_v1.shape == data_v2.shape, \
            f"Shape mismatch: v1={data_v1.shape}, v2={data_v2.shape}"
        
        # Check values are identical
        nan_mask_v1 = np.isnan(data_v1)
        nan_mask_v2 = np.isnan(data_v2)
        
        assert np.array_equal(nan_mask_v1, nan_mask_v2), \
            "NaN positions differ between v1 and v2"
        
        # Compare non-NaN values
        valid_v1 = data_v1[~nan_mask_v1]
        valid_v2 = data_v2[~nan_mask_v2]
        
        assert np.allclose(valid_v1, valid_v2, rtol=1e-10, atol=1e-10), \
            "Data values differ between v1 and v2"
        
        print(f"✓ v1 and v2 produce identical results")
        print(f"  Shape: {data_v1.shape}")
        print(f"  Non-NaN values: {(~nan_mask_v1).sum()}")
        print(f"  NaN values: {nan_mask_v1.sum()}")
    
    def test_all_versions_identical(self, temp_storage_dir, test_catalog, epoch_key, measurement_keys):
        """Test that all three versions produce identical results."""
        # Create storages with all versions
        storages = {}
        for version in ['v0', 'v1', 'v2']:
            storages[version] = self.create_and_populate_storage(
                temp_storage_dir / version,
                test_catalog,
                epoch_key,
                measurement_keys,
                version=version
            )
        
        # Extract data arrays
        data_arrays = {v: s.lightcurves.values for v, s in storages.items()}
        
        # All shapes should match
        shapes = {v: d.shape for v, d in data_arrays.items()}
        assert len(set(shapes.values())) == 1, \
            f"Shape mismatch across versions: {shapes}"
        
        # All NaN masks should match
        nan_masks = {v: np.isnan(d) for v, d in data_arrays.items()}
        for v1, v2 in [('v0', 'v1'), ('v1', 'v2'), ('v0', 'v2')]:
            assert np.array_equal(nan_masks[v1], nan_masks[v2]), \
                f"NaN positions differ between {v1} and {v2}"
        
        # All non-NaN values should match
        nan_mask = nan_masks['v0']  # Same for all
        for v1, v2 in [('v0', 'v1'), ('v1', 'v2'), ('v0', 'v2')]:
            valid_v1 = data_arrays[v1][~nan_mask]
            valid_v2 = data_arrays[v2][~nan_mask]
            assert np.allclose(valid_v1, valid_v2, rtol=1e-10, atol=1e-10), \
                f"Data values differ between {v1} and {v2}"
        
        print(f"✓ All versions (v0, v1, v2) produce identical results")
        print(f"  Shape: {data_arrays['v0'].shape}")
        print(f"  Non-NaN values: {(~nan_mask).sum()}")
        print(f"  NaN values: {nan_mask.sum()}")
    
    def test_expected_values(self, temp_storage_dir, test_catalog, epoch_key, measurement_keys):
        """Test that populated values match expected catalog values."""
        # Use v2 (fastest) for this test
        storage = self.create_and_populate_storage(
            temp_storage_dir / "expected_test",
            test_catalog,
            epoch_key,
            measurement_keys,
            version='v2'
        )
        
        # Check a few specific values
        # Object 'obj_0050' should have:
        # - auto mag: 20.0 + 50 * 0.01 = 20.50
        # - auto err: 0.05 + 50 * 0.0001 = 0.055
        
        obj_idx = 50
        meas_idx = 0  # 'auto'
        epoch_idx = 0
        
        mag_value = storage.lightcurves.values[obj_idx, meas_idx, 0, epoch_idx]
        err_value = storage.lightcurves.values[obj_idx, meas_idx, 1, epoch_idx]
        
        expected_mag = 20.0 + obj_idx * 0.01 + meas_idx * 0.1
        expected_err = 0.05 + obj_idx * 0.0001 + meas_idx * 0.01
        
        assert np.isclose(mag_value, expected_mag, rtol=1e-10), \
            f"Magnitude mismatch: got {mag_value}, expected {expected_mag}"
        
        assert np.isclose(err_value, expected_err, rtol=1e-10), \
            f"Error mismatch: got {err_value}, expected {expected_err}"
        
        print(f"✓ Values match catalog expectations")
        print(f"  Object {obj_idx}: mag={mag_value:.4f}, err={err_value:.6f}")
        print(f"  Expected: mag={expected_mag:.4f}, err={expected_err:.6f}")
    
    def test_coordinate_preservation(self, temp_storage_dir, test_catalog, epoch_key, measurement_keys):
        """Test that coordinates are preserved correctly in all versions."""
        storages = {}
        for version in ['v0', 'v1', 'v2']:
            storages[version] = self.create_and_populate_storage(
                temp_storage_dir / f"coords_{version}",
                test_catalog,
                epoch_key,
                measurement_keys,
                version=version
            )
        
        # Check that coordinates match catalog
        catalog_ra = test_catalog.table['ALPHA_J2000']
        catalog_dec = test_catalog.table['DELTA_J2000']
        
        for version, storage in storages.items():
            storage_ra = storage.lightcurves.ra.values
            storage_dec = storage.lightcurves.dec.values
            
            # Should be identical
            assert np.allclose(storage_ra, catalog_ra, rtol=1e-10), \
                f"{version}: RA coordinates don't match catalog"
            
            assert np.allclose(storage_dec, catalog_dec, rtol=1e-10), \
                f"{version}: Dec coordinates don't match catalog"
        
        print(f"✓ Coordinates preserved correctly in all versions")
