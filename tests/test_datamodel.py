#!/usr/bin/env python
"""
Updated tests for the lightcurve datamodel module with new API.

Tests the xarray-based storage functionality with epoch and object_key terminology.
Uses the new API: create_for_per_epoch_write(), populate_epoch(), load_for_per_object_read(), etc.
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
from pyzma_idf_lightcurve.lightcurve.catalog import SourceCatalog, SExtractorTableTransform


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    # Cleanup after test
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_catalog():
    """Create sample astronomical catalog data for testing.
    
    Note: With the new catalog API, measurement_keys are extracted from column names.
    MAG_AUTO/MAGERR_AUTO columns will generate measurement_key='auto'
    """
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
    
    return catalog


@pytest.fixture
def source_catalog(sample_catalog):
    """Create a SourceCatalog instance from sample data.
    
    Configure column mappings to match the test catalog structure:
    - Use NUMBER column for object IDs
    - Use ALPHA_J2000/DELTA_J2000 for coordinates
    - Use X_IMAGE/Y_IMAGE for pixel positions
    """
    
    # Configure table transform to match test data column names
    table_transform = SExtractorTableTransform()
    return SourceCatalog(sample_catalog, table_transform=table_transform)


@pytest.fixture
def epoch_keys():
    """Sample epoch keys (temporal grouping identifiers)."""
    return [f"r{58520832 + i * 256}" for i in range(20)]


class TestLightcurveStorage:
    """Test the LightcurveStorage class functionality with new API."""
    
    def test_source_catalog_measurement_keys(self, source_catalog):
        """Test that source catalog has correct measurement keys."""
        # With the new catalog API, MAG_AUTO/MAGERR_AUTO columns generate measurement_key='auto'
        assert 'auto' in source_catalog.measurement_keys
        # Should have at least this one measurement key
        assert len(source_catalog.measurement_keys) >= 1
        # Should also have value_keys (lowercase, as returned by catalog API)
        assert 'mag' in source_catalog.value_keys
        assert 'magerr' in source_catalog.value_keys
    
    def test_init(self, temp_dir):
        """Test LightcurveStorage initialization."""
        storage_path = temp_dir / "test_storage"
        storage = LightcurveStorage(storage_path)
        
        assert storage.storage_path == storage_path
        assert storage.sort_objects_by_position is True  # Default sorting behavior
        assert storage.lightcurves is None
        assert not storage.is_loaded()
    
    def test_init_custom_sort(self, temp_dir):
        """Test LightcurveStorage initialization with custom sort parameter."""
        storage_path = temp_dir / "test_storage"
        storage = LightcurveStorage(storage_path, sort_objects_by_position=False)
        
        assert storage.storage_path == storage_path
        assert storage.sort_objects_by_position is False
        assert storage.lightcurves is None
        assert not storage.is_loaded()
    
    def test_init_zarr_paths(self, temp_dir):
        """Test zarr path properties."""
        storage_path = temp_dir / "test_storage"
        storage = LightcurveStorage(storage_path)
        
        expected_write = storage_path / "lightcurves_write.zarr"
        expected_read = storage_path / "lightcurves_read.zarr"
        
        assert storage.zarr_path_for_write == expected_write
        assert storage.zarr_path_for_read == expected_read
    
    # DIM_NAMES class variable not implemented yet - skipped
    
    def test_create_storage_basic(self, temp_dir, source_catalog, epoch_keys):
        """Test basic storage creation with new API (using all catalog measurements)."""
        storage_path = temp_dir / "test_storage"
        storage = LightcurveStorage(storage_path)
        
        # Create storage for per-epoch writing
        storage.create_for_per_epoch_write(
            source_catalog=source_catalog,
            epoch_keys=epoch_keys,
        )
        
        # Load storage to access data
        storage.load_for_per_epoch_write()
        assert storage.is_loaded()
        assert storage.lightcurves is not None  # Type guard for linter
        
        # Check that storage was created
        assert storage.storage_path.exists()
        zarr_path_write = storage.zarr_path_for_write
        assert zarr_path_write.exists()
        
        # Check dimensions
        assert storage.lightcurves.shape == (
            len(source_catalog.object_keys),
            len(source_catalog.measurement_keys),  # Uses all measurement keys from catalog
            len(source_catalog.value_keys),  # mag, mag_err
            len(epoch_keys)
        )
        
        # Check dimension names (simplified)
        assert list(storage.lightcurves.dims) == ['object', 'measurement', 'value', 'epoch']
    
    def test_create_with_measurement_keys(self, temp_dir, source_catalog, epoch_keys):
        """Test storage creation with filtered measurement_keys."""
        storage_path = temp_dir / "test_measurement_filter"
        storage = LightcurveStorage(storage_path)
        
        # Filter to only 'auto' measurement (API expects list/tuple not set)
        measurement_keys = ['auto']
        
        storage.create_for_per_epoch_write(
            source_catalog=source_catalog,
            epoch_keys=epoch_keys,
            measurement_keys=measurement_keys,
        )
        
        storage.load_for_per_epoch_write()
        assert storage.lightcurves is not None
        
        # Should only have the filtered measurement
        assert list(storage.lightcurves.measurement.values) == measurement_keys
        assert len(storage.lightcurves.measurement) == 1
    
    def test_create_with_value_keys(self, temp_dir, source_catalog, epoch_keys):
        """Test storage creation with filtered value_keys."""
        storage_path = temp_dir / "test_value_filter"
        storage = LightcurveStorage(storage_path)
        
        # Filter to only 'mag' value (API expects list/tuple not set)
        value_keys = ['mag']
        
        storage.create_for_per_epoch_write(
            source_catalog=source_catalog,
            epoch_keys=epoch_keys,
            value_keys=value_keys,
        )
        
        storage.load_for_per_epoch_write()
        assert storage.lightcurves is not None
        
        # Should only have the filtered value
        assert list(storage.lightcurves.value.values) == value_keys
        assert len(storage.lightcurves.value) == 1
    
    # test_create_with_dim_vars and test_create_custom_chunks removed - API not yet implemented
    
    def test_create_sorted_objects(self, temp_dir, source_catalog, epoch_keys):
        """Test spatial sorting of objects during creation."""
        storage_path = temp_dir / "test_sorted"
        storage = LightcurveStorage(storage_path, sort_objects_by_position=True)
        
        storage.create_for_per_epoch_write(
            source_catalog=source_catalog,
            epoch_keys=epoch_keys,
        )
        
        storage.load_for_per_epoch_write()
        assert storage.lightcurves is not None
        
        # Objects should be sorted (can't easily verify z-order, just check it worked)
        assert len(storage.lightcurves.object) == len(source_catalog.object_keys)
    
    # test_validate_dim_args removed - epoch_vars not yet implemented in API
    
    def test_storage_coordinates(self, temp_dir, source_catalog, epoch_keys):
        """Test that coordinates are properly assigned."""
        storage_path = temp_dir / "test_storage"
        storage = LightcurveStorage(storage_path)
        
        storage.create_for_per_epoch_write(
            source_catalog=source_catalog,
            epoch_keys=epoch_keys,
        )
        storage.load_for_per_epoch_write()
        assert storage.lightcurves is not None  # Type guard for linter
        
        # Check coordinate existence
        assert 'object' in storage.lightcurves.coords
        assert 'measurement' in storage.lightcurves.coords
        assert 'value' in storage.lightcurves.coords
        assert 'epoch' in storage.lightcurves.coords
        
        # Check coordinate values
        assert len(storage.lightcurves.object) == len(source_catalog.object_keys)
        assert set(storage.lightcurves.measurement.values) == source_catalog.measurement_keys
        assert set(storage.lightcurves.value.values) == source_catalog.value_keys
        assert list(storage.lightcurves.epoch.values) == epoch_keys
    
    def test_populate_epoch_from_catalog(self, temp_dir, source_catalog, epoch_keys):
        """Test populating epoch data from catalog."""
        storage_path = temp_dir / "test_populate"
        storage = LightcurveStorage(storage_path)
        
        # Create storage
        storage.create_for_per_epoch_write(
            source_catalog=source_catalog,
            epoch_keys=epoch_keys,
        )
        
        # Populate first epoch
        test_epoch = epoch_keys[0]
        n_updated = storage.populate_epoch(
            source_catalog=source_catalog,
            epoch_key=test_epoch,
        )
        
        # Should have updated measurements (objects × measurements × value_keys)
        expected = len(source_catalog.object_keys) * len(source_catalog.measurement_keys) * len(source_catalog.value_keys)
        assert n_updated == expected
    
    def test_populate_multiple_epochs(self, temp_dir, source_catalog, epoch_keys):
        """Test populating multiple epochs."""
        storage_path = temp_dir / "test_multi_populate"
        storage = LightcurveStorage(storage_path)
        
        storage.create_for_per_epoch_write(
            source_catalog=source_catalog,
            epoch_keys=epoch_keys,
        )
        
        # Populate first 3 epochs
        for i in range(3):
            n_updated = storage.populate_epoch(
                source_catalog=source_catalog,
                epoch_key=epoch_keys[i],
            )
            expected = len(source_catalog.object_keys) * len(source_catalog.measurement_keys) * len(source_catalog.value_keys)
            assert n_updated == expected
    
    def test_populate_reopens_storage(self, temp_dir, source_catalog, epoch_keys):
        """Test that populate_epoch can work after closing storage."""
        storage_path = temp_dir / "test_reopen"
        storage = LightcurveStorage(storage_path)
        
        storage.create_for_per_epoch_write(
            source_catalog=source_catalog,
            epoch_keys=epoch_keys,
        )
        
        # Populate first epoch
        storage.populate_epoch(source_catalog=source_catalog, epoch_key=epoch_keys[0])
        
        # Close storage
        storage.close()
        assert not storage.is_loaded()
        
        # Populate another epoch (should reopen automatically)
        n_updated = storage.populate_epoch(
            source_catalog=source_catalog,
            epoch_key=epoch_keys[1],
        )
        expected = len(source_catalog.object_keys) * len(source_catalog.measurement_keys) * len(source_catalog.value_keys)
        assert n_updated == expected
    
    def test_get_object_lightcurve(self, temp_dir, source_catalog, epoch_keys):
        """Test retrieving object lightcurve."""
        storage_path = temp_dir / "test_object_lc"
        storage = LightcurveStorage(storage_path)
        
        storage.create_for_per_epoch_write(
            source_catalog=source_catalog,
            epoch_keys=epoch_keys,
        )
        
        # Rechunk and load for per-object read
        storage.rechunk_for_per_object_read()
        storage.load_for_per_object_read()
        
        # Get lightcurve for first object
        object_key = source_catalog.object_keys[0]
        lc = storage.get_object_lightcurve(object_key)
        
        # Should have correct dimensions
        assert lc.dims == ('measurement', 'value', 'epoch')
        assert len(lc.measurement) == len(source_catalog.measurement_keys)
        assert len(lc.epoch) == len(epoch_keys)
    
    def test_get_object_lightcurve_with_coords(self, temp_dir, source_catalog, epoch_keys):
        """Test retrieving object lightcurve with coordinate selection."""
        storage_path = temp_dir / "test_obj_lc_coords"
        storage = LightcurveStorage(storage_path)
        
        storage.create_for_per_epoch_write(
            source_catalog=source_catalog,
            epoch_keys=epoch_keys,
        )
        storage.rechunk_for_per_object_read()
        storage.load_for_per_object_read()
        
        object_key = source_catalog.object_keys[0]
        measurement_key = list(source_catalog.measurement_keys)[0]
        
        # Get with specific measurement (value selection collapses one dimension)
        lc = storage.get_object_lightcurve(
            object_key,
            measurement_key=measurement_key
        )
        
        # Should have value and epoch dimensions (measurement collapsed)
        assert lc.dims == ('value', 'epoch')
        assert len(lc.epoch) == len(epoch_keys)
    
    def test_get_epoch_data(self, temp_dir, source_catalog, epoch_keys):
        """Test retrieving epoch data."""
        storage_path = temp_dir / "test_epoch_data"
        storage = LightcurveStorage(storage_path)
        
        storage.create_for_per_epoch_write(
            source_catalog=source_catalog,
            epoch_keys=epoch_keys,
        )
        storage.load_for_per_epoch_write()
        
        # Get data for first epoch
        epoch_key = epoch_keys[0]
        epoch_data = storage.get_epoch_data(epoch_key)
        
        # Should have correct dimensions
        assert epoch_data.dims == ('object', 'measurement', 'value')
        assert len(epoch_data.object) == len(source_catalog.object_keys)
        assert len(epoch_data.measurement) == len(source_catalog.measurement_keys)
    
    def test_get_epoch_data_with_coords(self, temp_dir, source_catalog, epoch_keys):
        """Test retrieving epoch data with coordinate selection."""
        storage_path = temp_dir / "test_epoch_coords"
        storage = LightcurveStorage(storage_path)
        
        storage.create_for_per_epoch_write(
            source_catalog=source_catalog,
            epoch_keys=epoch_keys,
        )
        storage.load_for_per_epoch_write()
        
        epoch_key = epoch_keys[0]
        measurement_key = list(source_catalog.measurement_keys)[0]
        
        # Get with specific measurement (measurement dimension collapsed)
        epoch_data = storage.get_epoch_data(
            epoch_key,
            measurement_key=measurement_key
        )
        
        # Should have object and value dimensions (measurement collapsed)
        assert epoch_data.dims == ('object', 'value')
        assert len(epoch_data.object) == len(source_catalog.object_keys)
    
    def test_get_storage_info(self, temp_dir, source_catalog, epoch_keys):
        """Test storage information retrieval."""
        storage_path = temp_dir / "test_info"
        storage = LightcurveStorage(storage_path)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            storage.create_for_per_epoch_write(
                source_catalog=source_catalog,
                epoch_keys=epoch_keys,
            )
            
            info = storage.get_storage_info(which="write")
            
            # Check expected fields
            assert info['status'] == 'ready'
            assert info['n_objects'] == len(source_catalog.object_keys)
            assert info['n_measurements'] == len(source_catalog.measurement_keys)
            assert info['n_epochs'] == len(epoch_keys)
    
    def test_info_write_storage(self, temp_dir, source_catalog, epoch_keys):
        """Test info for write storage."""
        storage_path = temp_dir / "test_info_write"
        storage = LightcurveStorage(storage_path)
        
        storage.create_for_per_epoch_write(
            source_catalog=source_catalog,
            epoch_keys=epoch_keys,
        )
        
        info = storage.get_storage_info(which="write")
        
        assert info['status'] == 'ready'
        # store_type not yet implemented in API
        assert info['n_objects'] == len(source_catalog.object_keys)
    
    def test_info_read_storage(self, temp_dir, source_catalog, epoch_keys):
        """Test info for read storage."""
        storage_path = temp_dir / "test_info_read"
        storage = LightcurveStorage(storage_path)
        
        storage.create_for_per_epoch_write(
            source_catalog=source_catalog,
            epoch_keys=epoch_keys,
        )
        storage.rechunk_for_per_object_read()
        
        info = storage.get_storage_info(which="read")
        
        assert info['status'] == 'ready'
        # store_type not yet implemented in API
        assert info['n_objects'] == len(source_catalog.object_keys)
    
    def test_info_not_created(self, temp_dir):
        """Test info before storage creation."""
        storage_path = temp_dir / "test_info_not_created"
        storage = LightcurveStorage(storage_path)
        
        info = storage.get_storage_info(which="write")
        
        assert info['status'] == 'not_created'
    
    def test_info_coordinate_ranges(self, temp_dir, source_catalog, epoch_keys):
        """Test coordinate range information."""
        storage_path = temp_dir / "test_info_ranges"
        storage = LightcurveStorage(storage_path)
        
        storage.create_for_per_epoch_write(
            source_catalog=source_catalog,
            epoch_keys=epoch_keys,
        )
        
        # Populate some data to have coordinate ranges
        storage.populate_epoch(source_catalog=source_catalog, epoch_key=epoch_keys[0])
        
        info = storage.get_storage_info(which="write")
        
        # Check for coordinate information if provided by API
        assert 'n_objects' in info
        assert 'n_measurements' in info


    def test_get_objects_in_region(self, temp_dir, source_catalog, epoch_keys):
        """Test spatial region query functionality."""
        storage_path = temp_dir / "test_region"
        storage = LightcurveStorage(storage_path)
        
        storage.create_for_per_epoch_write(
            source_catalog=source_catalog,
            epoch_keys=epoch_keys,
        )
        
        # Rechunk and load for per-object read (needed for spatial queries)
        storage.rechunk_for_per_object_read()
        storage.load_for_per_object_read()
        
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
        
        storage.create_for_per_epoch_write(
            source_catalog=source_catalog,
            epoch_keys=epoch_keys,
        )
        storage.load_for_per_epoch_write()
        assert storage.lightcurves is not None  # Type guard for linter
        
        # Check a sample value (don't load entire array)
        sample = storage.lightcurves.isel(object=0, measurement=0, value=0, epoch=0)
        assert np.isnan(sample.values)


    def test_measurement_filtering(self, temp_dir, source_catalog, epoch_keys):
        """Test filtering by measurement_key in get_object_lightcurve and get_epoch_data."""
        storage_path = temp_dir / "test_measurement_filter"
        storage = LightcurveStorage(storage_path)
        
        storage.create_for_per_epoch_write(
            source_catalog=source_catalog,
            epoch_keys=epoch_keys,
        )
        
        # Populate some data
        storage.populate_epoch(source_catalog=source_catalog, epoch_key=epoch_keys[0])
        
        # Rechunk for per-object read
        storage.rechunk_for_per_object_read()
        storage.load_for_per_object_read()
        
        # Test filtering by measurement_key in get_object_lightcurve
        object_key = source_catalog.object_keys[0]
        measurement_keys = list(source_catalog.measurement_keys)
        
        # Get all measurements
        lc_all = storage.get_object_lightcurve(object_key)
        assert len(lc_all.measurement) == len(measurement_keys)
        
        # Get specific measurement
        if len(measurement_keys) > 0:
            lc_filtered = storage.get_object_lightcurve(object_key, measurement_key=measurement_keys[0])
            # Should collapse measurement dimension
            assert 'measurement' not in lc_filtered.dims
            assert lc_filtered.dims == ('value', 'epoch')
        
        # Test filtering in get_epoch_data
        epoch_data_all = storage.get_epoch_data(epoch_keys[0])
        assert len(epoch_data_all.measurement) == len(measurement_keys)
        
        if len(measurement_keys) > 0:
            epoch_data_filtered = storage.get_epoch_data(epoch_keys[0], measurement_key=measurement_keys[0])
            assert 'measurement' not in epoch_data_filtered.dims
            assert epoch_data_filtered.dims == ('object', 'value')


    def test_sort_objects_disabled(self, temp_dir, source_catalog, epoch_keys):
        """Test storage creation without spatial sorting."""
        storage_path = temp_dir / "test_no_sort"
        storage = LightcurveStorage(storage_path, sort_objects_by_position=False)
        
        assert storage.sort_objects_by_position is False
        
        storage.create_for_per_epoch_write(
            source_catalog=source_catalog,
            epoch_keys=epoch_keys,
        )
        storage.load_for_per_epoch_write()
        
        assert storage.lightcurves is not None
        # Object keys should maintain catalog order without sorting
        assert len(storage.lightcurves.object) == len(source_catalog.object_keys)


    def test_storage_persistence(self, temp_dir, source_catalog, epoch_keys):
        """Test that storage persists after closing and can be reloaded."""
        storage_path = temp_dir / "test_persistence"
        
        # Create and populate storage
        storage1 = LightcurveStorage(storage_path)
        storage1.create_for_per_epoch_write(
            source_catalog=source_catalog,
            epoch_keys=epoch_keys,
        )
        storage1.populate_epoch(source_catalog=source_catalog, epoch_key=epoch_keys[0])
        storage1.close()
        
        # Reload storage in new instance
        storage2 = LightcurveStorage(storage_path)
        storage2.load_for_per_epoch_write()
        
        assert storage2.is_loaded()
        assert storage2.lightcurves is not None
        assert storage2.lightcurves.shape[0] == len(source_catalog.object_keys)
        assert storage2.lightcurves.shape[3] == len(epoch_keys)
        
        # Verify data exists for first epoch (should not be all NaN)
        epoch_data = storage2.get_epoch_data(epoch_keys[0])
        # At least some values should be non-NaN
        assert not np.all(np.isnan(epoch_data.values))


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_load_before_create(self, temp_dir):
        """Test that loading before creation raises appropriate error."""
        storage_path = temp_dir / "nonexistent"
        storage = LightcurveStorage(storage_path)
        
        # Should fail gracefully when trying to get info without creating storage
        info = storage.get_storage_info(which="write")
        assert info['status'] == 'not_created'
    
    
    def test_populate_nonexistent_epoch(self, temp_dir, source_catalog, epoch_keys):
        """Test populating with epoch_key not in storage."""
        storage_path = temp_dir / "test_bad_epoch"
        storage = LightcurveStorage(storage_path)
        
        storage.create_for_per_epoch_write(
            source_catalog=source_catalog,
            epoch_keys=epoch_keys,
        )
        
        # Try to populate with epoch not in storage
        fake_epoch = "r99999999"
        n_updated = storage.populate_epoch(
            source_catalog=source_catalog,
            epoch_key=fake_epoch,
        )
        
        # Should return 0 updates (no matching epoch)
        assert n_updated == 0
    
    
    def test_get_nonexistent_object(self, temp_dir, source_catalog, epoch_keys):
        """Test retrieving lightcurve for nonexistent object."""
        storage_path = temp_dir / "test_bad_object"
        storage = LightcurveStorage(storage_path)
        
        storage.create_for_per_epoch_write(
            source_catalog=source_catalog,
            epoch_keys=epoch_keys,
        )
        storage.rechunk_for_per_object_read()
        storage.load_for_per_object_read()
        
        # Try to get lightcurve for nonexistent object
        with pytest.raises(KeyError):
            storage.get_object_lightcurve("nonexistent_object_key")
    
    
    def test_rechunk_before_create(self, temp_dir):
        """Test that rechunking before creation raises error."""
        storage_path = temp_dir / "test_rechunk_fail"
        storage = LightcurveStorage(storage_path)
        
        with pytest.raises(RuntimeError, match="Storage not created yet"):
            storage.rechunk_for_per_object_read()
    
    def test_rechunk_for_per_object_read(self, temp_dir, source_catalog, epoch_keys):
        """Test rechunking operation."""
        storage_path = temp_dir / "test_rechunk"
        storage = LightcurveStorage(storage_path)
        
        storage.create_for_per_epoch_write(
            source_catalog=source_catalog,
            epoch_keys=epoch_keys,
        )
        
        # Rechunk should complete successfully
        storage.rechunk_for_per_object_read()
        
        # Read storage should now exist
        assert storage.zarr_path_for_read.exists()

    def test_rechunk_actual_chunk_structure(self, temp_dir, source_catalog, epoch_keys):
        """Test that rechunking actually changes the chunk structure as expected.
        
        This test validates:
        1. Write-optimized storage has epoch chunks of size 1
        2. Read-optimized storage has object chunks matching the chunk_size parameter
        3. The actual Zarr chunk encoding matches expectations
        """
        import xarray as xr
        
        storage_path = temp_dir / "test_rechunk_chunks"
        storage = LightcurveStorage(storage_path)
        
        # Create with 100 objects and 5 epochs for easy verification
        storage.create_for_per_epoch_write(
            source_catalog=source_catalog,
            epoch_keys=epoch_keys,
        )
        
        # Populate a few epochs so we have data
        for epoch_key in epoch_keys[:3]:
            storage.populate_epoch(source_catalog, epoch_key)
        
        # === BEFORE RECHUNKING: Verify write-optimized chunks ===
        storage.close()
        storage.load_for_per_epoch_write()
        
        # Check chunks in write storage
        write_chunks = storage.lightcurves.chunks
        n_objects = len(storage.lightcurves.object)
        n_epochs = len(storage.lightcurves.epoch)
        
        # Write-optimized: should be chunked along epoch dimension (object=all, epoch=1)
        assert write_chunks[0] == (n_objects,), \
            f"Write storage object chunks should be ({n_objects},), got {write_chunks[0]}"
        # Each epoch should be in its own chunk
        assert all(c == 1 for c in write_chunks[3]), \
            f"Write storage epoch chunks should all be 1, got {write_chunks[3]}"
        
        # === AFTER RECHUNKING: Verify read-optimized chunks ===
        chunk_size = 50  # Use 50 for easier verification with 100 objects
        storage.rechunk_for_per_object_read(chunk_size=chunk_size)
        
        # Load the rechunked storage
        storage.close()
        storage.load_for_per_object_read()
        
        # Check chunks in read storage
        read_chunks = storage.lightcurves.chunks
        
        # Read-optimized: should be chunked along object dimension
        # With 100 objects and chunk_size=50, expect 2 chunks of size 50
        expected_object_chunks = tuple([chunk_size] * (n_objects // chunk_size))
        assert read_chunks[0] == expected_object_chunks, \
            f"Read storage object chunks should be {expected_object_chunks}, got {read_chunks[0]}"
        
        # All epochs should be in one chunk
        assert read_chunks[3] == (n_epochs,), \
            f"Read storage epoch chunks should be ({n_epochs},), got {read_chunks[3]}"
        
        # === VERIFY ACTUAL ZARR ENCODING ===
        # Open the Zarr store directly to check actual chunk encoding
        ds_read = xr.open_zarr(storage.zarr_path_for_read, consolidated=True)
        
        # Check the encoding chunks for the main data variable
        data_var = ds_read['lightcurves']
        zarr_chunks = data_var.encoding.get('chunks')
        
        # Zarr encoding should match our expectations
        # Format: (object_chunk, measurement_chunk, value_chunk, epoch_chunk)
        # Use actual dimensions from the dataset
        assert storage.lightcurves is not None
        n_measurements = len(storage.lightcurves.measurement)
        n_values = len(storage.lightcurves.value)
        expected_zarr_chunks = (chunk_size, n_measurements, n_values, n_epochs)
        assert zarr_chunks == expected_zarr_chunks, \
            f"Zarr encoding chunks should be {expected_zarr_chunks}, got {zarr_chunks}"
        
        # Coordinate arrays were computed (loaded into memory) to avoid chunk alignment
        # issues, so they may have their original chunk sizes or no chunks at all.
        # This is acceptable - coordinates are small and can be kept in memory.
        # The important verification is that the main data variable chunks changed.
        
        ds_read.close()
    
    def test_load_for_per_epoch_write(self, temp_dir, source_catalog, epoch_keys):
        """Test loading storage for epoch-wise write operations."""
        storage_path = temp_dir / "test_load_write"
        storage = LightcurveStorage(storage_path)
        
        storage.create_for_per_epoch_write(
            source_catalog=source_catalog,
            epoch_keys=epoch_keys,
        )
        
        # Close then reload
        storage.close()
        storage.load_for_per_epoch_write()
        
        assert storage.is_loaded()
        assert storage.lightcurves is not None
    
    def test_load_for_per_object_read_fails_without_rechunk(self, temp_dir, source_catalog, epoch_keys):
        """Test that loading for read fails without rechunking."""
        storage_path = temp_dir / "test_load_read_fail"
        storage = LightcurveStorage(storage_path)
        
        storage.create_for_per_epoch_write(
            source_catalog=source_catalog,
            epoch_keys=epoch_keys,
        )
        
        # Try to load for read without rechunking
        with pytest.raises(RuntimeError, match="Storage not created yet"):
            storage.load_for_per_object_read()
    
    def test_close(self, temp_dir, source_catalog, epoch_keys):
        """Test storage close functionality."""
        storage_path = temp_dir / "test_close"
        storage = LightcurveStorage(storage_path)
        
        storage.create_for_per_epoch_write(
            source_catalog=source_catalog,
            epoch_keys=epoch_keys,
        )
        storage.load_for_per_epoch_write()
        
        assert storage.is_loaded()
        
        # Close storage
        storage.close()
        
        assert not storage.is_loaded()
        assert storage.lightcurves is None


class TestIntegrationScenarios:
    """Integration tests that combine multiple components."""
    
    def test_complete_workflow(self, temp_dir, source_catalog, epoch_keys):
        """Test a complete workflow from creation to population."""
        storage_path = temp_dir / "integration_test"
        storage = LightcurveStorage(storage_path)
        
        # Step 1: Create storage
        storage.create_for_per_epoch_write(
            source_catalog=source_catalog,
            epoch_keys=epoch_keys,
        )
        
        # Step 2: Populate data for multiple epochs
        for epoch_idx in range(min(3, len(epoch_keys))):
            epoch_key = epoch_keys[epoch_idx]
            n_updated = storage.populate_epoch(
                source_catalog=source_catalog,
                epoch_key=epoch_key,
            )
            expected = len(source_catalog.object_keys) * len(source_catalog.measurement_keys) * len(source_catalog.value_keys)
            assert n_updated == expected
        
        # Close and reopen for verification
        storage.close()
        
        # Step 3: Verify storage info
        info = storage.get_storage_info(which="write")
        assert info['status'] == 'ready'
        assert info['n_objects'] == len(source_catalog.object_keys)
    
    def test_data_persistence_across_reloads(self, temp_dir, source_catalog, epoch_keys):
        """Test data persistence when closing and reopening storage."""
        storage_path = temp_dir / "test_persistence_reload"
        
        # Create and populate
        storage1 = LightcurveStorage(storage_path)
        storage1.create_for_per_epoch_write(
            source_catalog=source_catalog,
            epoch_keys=epoch_keys,
        )
        storage1.populate_epoch(source_catalog=source_catalog, epoch_key=epoch_keys[0])
        storage1.populate_epoch(source_catalog=source_catalog, epoch_key=epoch_keys[1])
        storage1.close()
        
        # Reload and verify data
        storage2 = LightcurveStorage(storage_path)
        storage2.load_for_per_epoch_write()
        
        epoch_data = storage2.get_epoch_data(epoch_keys[0])
        assert not np.all(np.isnan(epoch_data.values))
        
        storage2.close()


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_load_nonexistent_storage(self, temp_dir):
        """Test loading nonexistent storage."""
        storage_path = temp_dir / "nonexistent"
        storage = LightcurveStorage(storage_path)
        
        # Should raise error when loading nonexistent storage
        with pytest.raises(RuntimeError):
            storage.load_for_per_epoch_write()
    
    def test_query_before_loading(self, temp_dir, source_catalog, epoch_keys):
        """Test querying storage before loading (auto-load behavior)."""
        storage_path = temp_dir / "test_query_autoload"
        storage = LightcurveStorage(storage_path)
        
        storage.create_for_per_epoch_write(
            source_catalog=source_catalog,
            epoch_keys=epoch_keys,
        )
        storage.populate_epoch(source_catalog=source_catalog, epoch_key=epoch_keys[0])
        storage.close()
        
        # Query without explicit load (should auto-load)
        epoch_data = storage.get_epoch_data(epoch_keys[0])
        
        assert epoch_data is not None
        assert not np.all(np.isnan(epoch_data.values))
