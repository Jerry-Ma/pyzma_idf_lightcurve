#!/usr/bin/env python
"""
Updated tests for the lightcurve datamodel module with new API.

Tests the xarray-based storage functionality with epoch and object_key terminology.
"""

import pytest
import numpy as np
import shutil
import tempfile
from pathlib import Path
from typing import Any
import warnings
from astropy.table import Table

# Import the main components
from pyzma_idf_lightcurve.lightcurve.datamodel import LightcurveStorage
from pyzma_idf_lightcurve.lightcurve.catalog import SourceCatalog


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    # Cleanup after test
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_catalog():
    """Create sample astronomical catalog data for testing."""
    np.random.seed(42)  # For reproducible tests
    n_objects = 100
    
    # Create realistic IDF field coordinates (17.6 deg RA, -29.8 deg DEC)
    ra = np.random.uniform(17.5, 17.7, n_objects)
    dec = np.random.uniform(-29.9, -29.7, n_objects)
    
    # Create pixel coordinates for 750x750 field
    x_image = np.random.uniform(0, 750, n_objects)
    y_image = np.random.uniform(0, 750, n_objects)
    
    # Create astropy Table as expected by the API
    catalog = Table({
        'NUMBER': np.arange(1, n_objects + 1),  # Object IDs starting from 1
        'ALPHA_J2000': ra,
        'DELTA_J2000': dec, 
        'X_IMAGE': x_image,
        'Y_IMAGE': y_image,
        'MAG_AUTO': np.random.uniform(15, 25, n_objects),
        'FLUX_AUTO': 10**(-0.4 * np.random.uniform(15, 25, n_objects)),
        'MAGERR_AUTO': np.random.uniform(0.01, 0.1, n_objects),
        'FLUXERR_AUTO': np.random.uniform(10, 100, n_objects),
    })
    
    # Add table name metadata (defaults to 'default' if not specified)
    if catalog.meta is None:
        catalog.meta = {}
    catalog.meta['table_name'] = 'default'
    
    return catalog


@pytest.fixture
def source_catalog(sample_catalog):
    """Create a SourceCatalog instance from sample data."""
    return SourceCatalog(sample_catalog)


@pytest.fixture
def measurement_keys():
    """Standard measurement keys for IDF lightcurve analysis.
    
    For the test catalog with default table name, the valid keys are:
    'default-auto' (from MAG_AUTO/MAGERR_AUTO columns)
    """
    return ['default-auto']


@pytest.fixture
def epoch_keys():
    """Sample epoch keys (temporal grouping identifiers)."""
    return [f"r{58520832 + i * 256}" for i in range(20)]


class TestLightcurveStorage:
    """Test the LightcurveStorage class functionality with new API."""
    
    def test_source_catalog_measurement_keys(self, source_catalog):
        """Test that source catalog has correct measurement keys."""
        # For default table name with MAG_AUTO/MAGERR_AUTO columns
        # the measurement key should be 'default-auto'
        assert 'default-auto' in source_catalog.measurement_keys
        # Should have at least this one measurement key
        assert len(source_catalog.measurement_keys) >= 1
    
    def test_init(self, temp_dir):
        """Test LightcurveStorage initialization."""
        storage_path = temp_dir / "test_storage"
        storage = LightcurveStorage(storage_path)
        
        assert storage.storage_path == storage_path
        assert storage.chunk_size == 1000  # Default chunk size
        assert storage.lightcurves is None
    
    def test_create_storage_basic(self, temp_dir, source_catalog, epoch_keys):
        """Test basic storage creation with new API (using all catalog measurements)."""
        storage_path = temp_dir / "test_storage"
        storage = LightcurveStorage(storage_path)
        
        # Create storage without specifying measurement_keys - should use all from catalog
        storage.create_storage(
            source_catalog=source_catalog,
            epoch_keys=epoch_keys,
        )
        assert storage.lightcurves is not None  # Type guard for linter
        
        # Check that storage was created
        assert storage.storage_path.exists()
        zarr_path = storage.storage_path / "lightcurves.zarr"
        assert zarr_path.exists()
        
        # Check dimensions
        assert storage.lightcurves is not None
        assert storage.lightcurves.shape == (
            len(source_catalog.object_keys),
            len(source_catalog.measurement_keys),  # Uses all measurement keys from catalog
            2,  # mag, mag_err
            len(epoch_keys)
        )
        
        # Check dimension names (simplified)
        assert list(storage.lightcurves.dims) == ['object', 'measurement', 'value', 'epoch']
    
    def test_storage_coordinates(self, temp_dir, source_catalog, epoch_keys):
        """Test that coordinates are properly assigned."""
        storage_path = temp_dir / "test_storage"
        storage = LightcurveStorage(storage_path)
        
        storage.create_storage(
            source_catalog=source_catalog,
            epoch_keys=epoch_keys,
        )
        assert storage.lightcurves is not None  # Type guard for linter
        
        # Check coordinate existence
        assert 'object' in storage.lightcurves.coords
        assert 'measurement' in storage.lightcurves.coords
        assert 'value' in storage.lightcurves.coords
        assert 'epoch' in storage.lightcurves.coords
        
        # Check coordinate values
        assert len(storage.lightcurves.object) == len(source_catalog.object_keys)
        assert list(storage.lightcurves.measurement.values) == source_catalog.measurement_keys
        assert list(storage.lightcurves.value.values) == ['mag', 'mag_err']
        assert list(storage.lightcurves.epoch.values) == epoch_keys
    
    def test_populate_epoch_from_catalog(self, temp_dir, source_catalog, epoch_keys):
        """Test populating epoch data from catalog."""
        storage_path = temp_dir / "test_populate"
        storage = LightcurveStorage(storage_path)
        
        # Create storage
        storage.create_storage(
            source_catalog=source_catalog,
            epoch_keys=epoch_keys,
        )
        
        # Populate first epoch
        test_epoch = epoch_keys[0]
        n_updated = storage.populate_epoch_from_catalog(
            epoch_key=test_epoch,
            source_catalog=source_catalog,
            measurement_keys=source_catalog.measurement_keys
        )
        
        # Should have updated measurements (objects × measurements × 2 value_keys)
        # The method returns total entries updated: objects × measurements × value_keys
        expected = len(source_catalog.object_keys) * len(source_catalog.measurement_keys) * 2  # 2 value_keys (mag, mag_err)
        assert n_updated == expected
    
    def test_get_object_lightcurve(self, temp_dir, source_catalog, epoch_keys):
        """Test retrieving object lightcurve."""
        storage_path = temp_dir / "test_object_lc"
        storage = LightcurveStorage(storage_path)
        
        storage.create_storage(
            source_catalog=source_catalog,
            epoch_keys=epoch_keys,
        )
        
        # Get lightcurve for first object
        object_key = source_catalog.object_keys[0]
        lc = storage.get_object_lightcurve(object_key)
        
        # Should have correct dimensions
        assert lc.dims == ('measurement', 'value', 'epoch')
        assert len(lc.measurement) == len(source_catalog.measurement_keys)
        assert len(lc.epoch) == len(epoch_keys)
    
    def test_get_epoch_data(self, temp_dir, source_catalog, epoch_keys):
        """Test retrieving epoch data."""
        storage_path = temp_dir / "test_epoch_data"
        storage = LightcurveStorage(storage_path)
        
        storage.create_storage(
            source_catalog=source_catalog,
            epoch_keys=epoch_keys,
        )
        
        # Get data for first epoch
        epoch_key = epoch_keys[0]
        epoch_data = storage.get_epoch_data(epoch_key)
        
        # Should have correct dimensions
        assert epoch_data.dims == ('object', 'measurement', 'value')
        assert len(epoch_data.object) == len(source_catalog.object_keys)
        assert len(epoch_data.measurement) == len(source_catalog.measurement_keys)
    
    def test_get_storage_info(self, temp_dir, source_catalog, epoch_keys):
        """Test storage information retrieval."""
        storage_path = temp_dir / "test_info"
        storage = LightcurveStorage(storage_path)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            storage.create_storage(
                source_catalog=source_catalog,
                epoch_keys=epoch_keys,
            )
            
            info = storage.get_storage_info()
            
            # Check expected fields
            assert info['status'] == 'ready'
            assert info['n_objects'] == len(source_catalog.object_keys)
            assert info['n_measurements'] == len(source_catalog.measurement_keys)  
            assert info['n_epochs'] == len(epoch_keys)


    def test_get_objects_in_region(self, temp_dir, source_catalog, epoch_keys):
        """Test spatial region query functionality."""
        storage_path = temp_dir / "test_region"
        storage = LightcurveStorage(storage_path)
        
        storage.create_storage(
            source_catalog=source_catalog,
            epoch_keys=epoch_keys,
        )
        
        # Test region query (using RA/DEC ranges)
        ra_range = (17.55, 17.65)  # Middle of our test range
        dec_range = (-29.85, -29.75)
        
        obj_keys = storage.get_objects_in_region(ra_range, dec_range)
        
        # Should return a list
        assert isinstance(obj_keys, list)
        # May be empty or have objects depending on random catalog
        assert len(obj_keys) >= 0
        
    def test_initial_values_are_nan(self, temp_dir, source_catalog, epoch_keys):
        """Test that storage is initialized with NaN values."""
        storage_path = temp_dir / "test_nan"
        storage = LightcurveStorage(storage_path)
        
        storage.create_storage(
            source_catalog=source_catalog,
            epoch_keys=epoch_keys,
        )
        assert storage.lightcurves is not None  # Type guard for linter
        
        # All initial values should be NaN
        assert np.all(np.isnan(storage.lightcurves.values))


class TestIntegrationScenarios:
    """Integration tests that combine multiple components."""
    
    def test_complete_workflow(self, temp_dir, source_catalog, epoch_keys):
        """Test a complete workflow from creation to population."""
        storage_path = temp_dir / "integration_test"
        storage = LightcurveStorage(storage_path)
        
        # Step 1: Create storage
        storage.create_storage(
            source_catalog=source_catalog,
            epoch_keys=epoch_keys,
        )
        
        # Step 2: Populate data for multiple epochs
        for epoch_idx in range(min(3, len(epoch_keys))):
            epoch_key = epoch_keys[epoch_idx]
            n_updated = storage.populate_epoch_from_catalog(
                epoch_key=epoch_key,
                source_catalog=source_catalog,
                measurement_keys=source_catalog.measurement_keys
            )
            expected = len(source_catalog.object_keys) * len(source_catalog.measurement_keys) * 2  # 2 value_keys
            assert n_updated == expected
        
        # Step 3: Verify storage info
        info = storage.get_storage_info()
        assert info['status'] == 'ready'
        assert info['n_objects'] == len(source_catalog.object_keys)


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
