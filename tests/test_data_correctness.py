#!/usr/bin/env python
"""
Comprehensive tests for data correctness in lightcurve storage.

This module tests that:
1. Array data (measurements) are stored correctly
2. Ancillary data (non-measurement columns) are preserved
3. Coordinates (RA, Dec, X, Y) are stored in correct order
4. Data integrity is maintained after spatial sorting
5. Object key mapping remains correct throughout operations
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
from astropy.table import Table

from pyzma_idf_lightcurve.lightcurve.catalog import (
    SExtractorTableTransform,
    SourceCatalog,
)
from pyzma_idf_lightcurve.lightcurve.datamodel import LightcurveStorage


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def deterministic_catalog():
    """
    Create a deterministic catalog with known values for testing.
    
    This catalog has:
    - 10 objects with unique, trackable IDs
    - Known RA/Dec positions
    - Known pixel positions  
    - Known magnitude/flux values
    - Objects arranged in a specific non-spatial order initially
    """
    # Create objects with known IDs and properties
    object_ids = [f"obj_{i:03d}" for i in range(10)]
    
    # Positions arranged non-spatially (to test sorting)
    # Objects zigzag across the field
    ra_values = [17.50, 17.70, 17.51, 17.69, 17.52, 17.68, 17.53, 17.67, 17.54, 17.66]
    dec_values = [-29.90, -29.70, -29.89, -29.71, -29.88, -29.72, -29.87, -29.73, -29.86, -29.74]
    
    # Pixel positions corresponding to RA/Dec
    x_values = [10.0, 740.0, 20.0, 730.0, 30.0, 720.0, 40.0, 710.0, 50.0, 700.0]
    y_values = [10.0, 740.0, 20.0, 730.0, 30.0, 720.0, 40.0, 710.0, 50.0, 700.0]
    
    # Measurement values - each object has unique mag/flux
    # mag = 18.0 + object_index * 0.5
    mag_values = [18.0 + i * 0.5 for i in range(10)]
    flux_values = [10**(-0.4 * mag) for mag in mag_values]
    magerr_values = [0.01 + i * 0.001 for i in range(10)]
    fluxerr_values = [flux * magerr / 1.0857 for flux, magerr in zip(flux_values, magerr_values)]
    
    # Create flux_aper and fluxerr_aper for second measurement
    mag_aper_values = [m + 0.1 for m in mag_values]
    flux_aper_values = [10**(-0.4 * mag) for mag in mag_aper_values]
    magerr_aper_values = [e + 0.005 for e in magerr_values]
    fluxerr_aper_values = [flux * magerr / 1.0857 for flux, magerr in zip(flux_aper_values, magerr_aper_values)]
    
    catalog = Table({
        'NUMBER': list(range(1, 11)),
        'ID_STR': object_ids,  # String IDs for tracking
        'ALPHA_J2000': ra_values,
        'DELTA_J2000': dec_values,
        'X_IMAGE': x_values,
        'Y_IMAGE': y_values,
        'MAG_AUTO': mag_values,
        'FLUX_AUTO': flux_values,
        'MAGERR_AUTO': magerr_values,
        'FLUXERR_AUTO': fluxerr_values,
        'MAG_APER': mag_aper_values,  # Second measurement
        'FLUX_APER': flux_aper_values,
        'MAGERR_APER': magerr_aper_values,
        'FLUXERR_APER': fluxerr_aper_values,
    })
    
    return catalog


@pytest.fixture
def deterministic_source_catalog(deterministic_catalog):
    """Create SourceCatalog from deterministic catalog data."""
    # Use ID_STR column directly (no transformation needed since already strings)
    table_transform = SExtractorTableTransform(
        obj_key_col='ID_STR',  # Use string IDs for easier tracking
    )
    return SourceCatalog(deterministic_catalog, table_transform=table_transform)


@pytest.fixture
def epoch_keys():
    """Sample epoch keys for testing."""
    return [f"r{58520832 + i * 256}" for i in range(5)]


class TestArrayDataCorrectness:
    """Test that measurement array data is stored correctly."""
    
    def test_measurement_values_stored_correctly(self, temp_dir, deterministic_source_catalog, epoch_keys):
        """
        Test that measurement values from catalog are stored correctly in storage.
        
        Verifies:
        - Each measurement value matches the catalog source
        - Values are at the correct object/measurement/value/epoch indices
        """
        storage_path = temp_dir / "test_array_correctness"
        storage = LightcurveStorage(storage_path)
        
        catalog = deterministic_source_catalog
        
        # Create storage
        storage.create_for_per_epoch_write(
            source_catalog=catalog,
            epoch_keys=epoch_keys,
        )
        
        # Populate first epoch
        epoch_key = epoch_keys[0]
        storage.populate_epoch(source_catalog=catalog, epoch_key=epoch_key)
        
        # Verify data for each object
        storage.load_for_per_epoch_write()
        assert storage.lightcurves is not None
        
        # Check that stored values match catalog values
        # Get catalog data
        catalog_data = catalog.data
        
        # For each object, verify measurements match
        for obj_idx, obj_key in enumerate(catalog.object_keys):
            for meas_idx, meas_key in enumerate(catalog.measurement_keys):
                for val_idx, val_key in enumerate(catalog.value_keys):
                    # Get stored value
                    stored_val = storage.lightcurves.isel(
                        object=obj_idx,
                        measurement=meas_idx,
                        value=val_idx,
                        epoch=0
                    ).values
                    
                    # Get catalog value
                    from pyzma_idf_lightcurve.lightcurve.catalog import SourceCatalogDataKey
                    data_key = SourceCatalogDataKey(
                        measurement=meas_key,
                        value=val_key,
                        epoch=None
                    )
                    catalog_val = catalog_data[data_key][obj_idx]
                    
                    # Compare (handle potential NaN)
                    if np.isnan(catalog_val):
                        assert np.isnan(stored_val)
                    else:
                        # Use tolerance appropriate for float32 precision
                        np.testing.assert_allclose(stored_val, catalog_val, rtol=1e-6)
    
    def test_multiple_epochs_data_separation(self, temp_dir, deterministic_source_catalog, epoch_keys):
        """
        Test that data from different epochs is stored separately and correctly.
        
        Verifies:
        - Each epoch's data is independent
        - Populating one epoch doesn't affect others
        """
        storage_path = temp_dir / "test_epoch_separation"
        storage = LightcurveStorage(storage_path)
        
        catalog = deterministic_source_catalog
        
        storage.create_for_per_epoch_write(
            source_catalog=catalog,
            epoch_keys=epoch_keys,
        )
        
        # Populate epochs 0 and 2, leaving 1 empty
        storage.populate_epoch(source_catalog=catalog, epoch_key=epoch_keys[0])
        storage.populate_epoch(source_catalog=catalog, epoch_key=epoch_keys[2])
        
        storage.load_for_per_epoch_write()
        assert storage.lightcurves is not None
        
        # Check epoch 0 has data
        epoch_0_data = storage.get_epoch_data(epoch_keys[0])
        assert not np.all(np.isnan(epoch_0_data.values))
        
        # Check epoch 1 is all NaN (not populated)
        epoch_1_data = storage.get_epoch_data(epoch_keys[1])
        assert np.all(np.isnan(epoch_1_data.values))
        
        # Check epoch 2 has data
        epoch_2_data = storage.get_epoch_data(epoch_keys[2])
        assert not np.all(np.isnan(epoch_2_data.values))


class TestCoordinateCorrectness:
    """Test that coordinates are stored in correct order."""
    
    def test_coordinate_arrays_match_object_keys(self, deterministic_source_catalog):
        """
        Test that coordinate arrays align with object keys.
        
        Verifies:
        - RA values are in same order as object_keys
        - Dec values are in same order as object_keys
        - X, Y pixel values are in same order as object_keys
        """
        catalog = deterministic_source_catalog
        
        # Get coordinates
        ra_vals = catalog.ra_values
        dec_vals = catalog.dec_values
        x_vals = catalog.x_values
        y_vals = catalog.y_values
        
        # Get expected values from table
        expected_ra = list(catalog.table['ALPHA_J2000'])
        expected_dec = list(catalog.table['DELTA_J2000'])
        expected_x = list(catalog.table['X_IMAGE'])
        expected_y = list(catalog.table['Y_IMAGE'])
        
        # Verify alignment
        np.testing.assert_allclose(ra_vals, expected_ra, rtol=1e-10)
        np.testing.assert_allclose(dec_vals, expected_dec, rtol=1e-10)
        np.testing.assert_allclose(x_vals, expected_x, rtol=1e-10)
        np.testing.assert_allclose(y_vals, expected_y, rtol=1e-10)
    
    def test_coordinate_object_key_mapping(self, deterministic_source_catalog):
        """
        Test that object_key_coordinates dictionary maps correctly.
        
        Verifies:
        - Each object key has correct coordinates
        - Coordinates match the catalog table values
        """
        catalog = deterministic_source_catalog
        coord_dict = catalog.object_key_coordinates
        
        for obj_idx, obj_key in enumerate(catalog.object_keys):
            # Get coordinates from dict
            coords = coord_dict[obj_key]
            
            # Get expected from table
            expected_ra = catalog.table['ALPHA_J2000'][obj_idx]
            expected_dec = catalog.table['DELTA_J2000'][obj_idx]
            expected_x = catalog.table['X_IMAGE'][obj_idx]
            expected_y = catalog.table['Y_IMAGE'][obj_idx]
            
            # Verify
            np.testing.assert_allclose(coords['ra'], expected_ra, rtol=1e-10)
            np.testing.assert_allclose(coords['dec'], expected_dec, rtol=1e-10)
            np.testing.assert_allclose(coords['x'], expected_x, rtol=1e-10)
            np.testing.assert_allclose(coords['y'], expected_y, rtol=1e-10)
    
    def test_coordinates_stored_with_lightcurves(self, temp_dir, deterministic_source_catalog, epoch_keys):
        """
        Test that coordinates remain accessible after creating storage.
        
        Verifies:
        - Catalog coordinates are used to create storage
        - Object keys in storage match catalog order
        """
        storage_path = temp_dir / "test_coord_storage"
        storage = LightcurveStorage(storage_path)
        
        catalog = deterministic_source_catalog
        
        storage.create_for_per_epoch_write(
            source_catalog=catalog,
            epoch_keys=epoch_keys,
        )
        storage.load_for_per_epoch_write()
        assert storage.lightcurves is not None
        
        # Verify storage object dimension matches catalog object keys
        stored_object_keys = list(storage.lightcurves.coords['object'].values)
        catalog_object_keys = list(catalog.object_keys)
        
        assert stored_object_keys == catalog_object_keys
        
        # Verify we can still get coordinates from catalog for each stored object
        for obj_key in stored_object_keys:
            assert obj_key in catalog.object_key_coordinates
            coords = catalog.object_key_coordinates[obj_key]
            # Just verify coordinates exist and are numeric
            assert isinstance(coords['ra'], (int, float))
            assert isinstance(coords['dec'], (int, float))
            assert isinstance(coords['x'], (int, float))
            assert isinstance(coords['y'], (int, float))


class TestSpatialSortingDataIntegrity:
    """Test that data integrity is maintained after spatial sorting."""
    
    def test_sorting_preserves_object_data_mapping(self, deterministic_source_catalog):
        """
        Test that sorting preserves the mapping between object keys and their data.
        
        Verifies:
        - Each object key still maps to its original data after sorting
        - Measurement values follow the object through the sort
        """
        catalog = deterministic_source_catalog
        
        # Create mapping of object_key -> measurement values before sorting
        pre_sort_mapping = {}
        for obj_idx, obj_key in enumerate(catalog.object_keys):
            # Get measurement values for this object
            meas_dict = {}
            for meas_key in catalog.measurement_keys:
                for val_key in catalog.value_keys:
                    from pyzma_idf_lightcurve.lightcurve.catalog import SourceCatalogDataKey
                    data_key = SourceCatalogDataKey(
                        measurement=meas_key,
                        value=val_key,
                        epoch=None
                    )
                    meas_dict[f"{meas_key}_{val_key}"] = catalog.data[data_key][obj_idx]
            pre_sort_mapping[obj_key] = meas_dict
        
        # Sort by position
        catalog.sort_objects_by_position(grid_divisions=10)
        
        # Verify mapping is preserved after sorting
        for obj_idx, obj_key in enumerate(catalog.object_keys):
            for meas_key in catalog.measurement_keys:
                for val_key in catalog.value_keys:
                    from pyzma_idf_lightcurve.lightcurve.catalog import SourceCatalogDataKey
                    data_key = SourceCatalogDataKey(
                        measurement=meas_key,
                        value=val_key,
                        epoch=None
                    )
                    post_sort_val = catalog.data[data_key][obj_idx]
                    pre_sort_val = pre_sort_mapping[obj_key][f"{meas_key}_{val_key}"]
                    
                    np.testing.assert_allclose(post_sort_val, pre_sort_val, rtol=1e-10,
                                               err_msg=f"Data mismatch for {obj_key} {meas_key}_{val_key}")
    
    def test_sorting_preserves_coordinate_mapping(self, deterministic_source_catalog):
        """
        Test that sorting preserves object key to coordinate mapping.
        
        Verifies:
        - Each object key still has the same coordinates after sorting
        - Coordinate arrays are reordered correctly with the sort
        """
        catalog = deterministic_source_catalog
        
        # Store pre-sort coordinates for each object
        pre_sort_coords = {
            obj_key: {
                'ra': catalog.object_key_coordinates[obj_key]['ra'],
                'dec': catalog.object_key_coordinates[obj_key]['dec'],
                'x': catalog.object_key_coordinates[obj_key]['x'],
                'y': catalog.object_key_coordinates[obj_key]['y'],
            }
            for obj_key in catalog.object_keys
        }
        
        # Sort
        catalog.sort_objects_by_position(grid_divisions=10)
        
        # Verify each object still has same coordinates
        for obj_key in catalog.object_keys:
            post_sort_coords = catalog.object_key_coordinates[obj_key]
            pre_coords = pre_sort_coords[obj_key]
            
            np.testing.assert_allclose(post_sort_coords['ra'], pre_coords['ra'], rtol=1e-10)
            np.testing.assert_allclose(post_sort_coords['dec'], pre_coords['dec'], rtol=1e-10)
            np.testing.assert_allclose(post_sort_coords['x'], pre_coords['x'], rtol=1e-10)
            np.testing.assert_allclose(post_sort_coords['y'], pre_coords['y'], rtol=1e-10)
    
    def test_sorted_catalog_storage_correctness(self, temp_dir, deterministic_source_catalog, epoch_keys):
        """
        Test that sorted catalog data is stored correctly.
        
        Verifies:
        - Sorted catalog can be stored
        - Retrieved data matches the sorted catalog
        - Object keys are in spatial order
        """
        catalog = deterministic_source_catalog
        
        # Store pre-sort object keys
        original_object_keys = list(catalog.object_keys)
        
        # Sort catalog
        catalog.sort_objects_by_position(grid_divisions=10)
        sorted_object_keys = list(catalog.object_keys)
        
        # Verify sort actually changed the order
        assert original_object_keys != sorted_object_keys
        
        # Create storage with sorted catalog
        storage_path = temp_dir / "test_sorted_storage"
        storage = LightcurveStorage(storage_path)
        storage.create_for_per_epoch_write(
            source_catalog=catalog,
            epoch_keys=epoch_keys,
        )
        storage.populate_epoch(source_catalog=catalog, epoch_key=epoch_keys[0])
        
        # Reload and verify
        storage.close()
        storage.load_for_per_epoch_write()
        assert storage.lightcurves is not None
        
        # Verify object keys in storage match sorted order
        stored_object_keys = list(storage.lightcurves.coords['object'].values)
        assert stored_object_keys == sorted_object_keys
        
        # Verify data correctness for a few objects
        for obj_idx in [0, len(sorted_object_keys) // 2, len(sorted_object_keys) - 1]:
            obj_key = sorted_object_keys[obj_idx]
            
            # Get measurement value from catalog
            meas_key = list(catalog.measurement_keys)[0]
            val_key = list(catalog.value_keys)[0]
            
            from pyzma_idf_lightcurve.lightcurve.catalog import SourceCatalogDataKey
            data_key = SourceCatalogDataKey(
                measurement=meas_key,
                value=val_key,
                epoch=None
            )
            catalog_val = catalog.data[data_key][obj_idx]
            
            # Get stored value
            meas_idx = list(catalog.measurement_keys).index(meas_key)
            val_idx = list(catalog.value_keys).index(val_key)
            stored_val = storage.lightcurves.isel(
                object=obj_idx,
                measurement=meas_idx,
                value=val_idx,
                epoch=0
            ).values
            
            # Use tolerance appropriate for float32 storage precision
            np.testing.assert_allclose(stored_val, catalog_val, rtol=1e-6)


class TestAncillaryDataPreservation:
    """Test that ancillary (non-measurement) data is preserved."""
    
    def test_all_table_columns_preserved_after_operations(self, deterministic_source_catalog):
        """
        Test that all catalog table columns are preserved through operations.
        
        Verifies:
        - All columns exist after catalog creation
        - All columns preserved after sorting
        - Column data values remain correct
        """
        catalog = deterministic_source_catalog
        
        # Get initial columns
        initial_columns = set(catalog.table.colnames)
        initial_data = {col: list(catalog.table[col]) for col in initial_columns}
        
        # Perform sort operation
        catalog.sort_objects_by_position(grid_divisions=10)
        
        # Check all columns still exist (plus sort key)
        post_sort_columns = set(catalog.table.colnames)
        assert initial_columns.issubset(post_sort_columns)
        assert "_source_catalog_spatial_sort_key" in post_sort_columns
        
        # Verify data integrity: each ID_STR should map to same values
        for row_idx, obj_id in enumerate(catalog.table['ID_STR']):
            for col in initial_columns:
                if col == 'ID_STR':
                    continue
                # Find this object in initial data
                initial_idx = initial_data['ID_STR'].index(obj_id)
                expected_val = initial_data[col][initial_idx]
                actual_val = catalog.table[col][row_idx]
                
                # Compare
                if isinstance(expected_val, (int, float)):
                    np.testing.assert_allclose(actual_val, expected_val, rtol=1e-10)
                else:
                    assert actual_val == expected_val


class TestCompleteWorkflow:
    """Test complete workflow with all operations."""
    
    def test_full_pipeline_data_correctness(self, temp_dir, deterministic_catalog, epoch_keys):
        """
        Test complete pipeline: create catalog -> sort -> store -> retrieve -> verify.
        
        This is the most comprehensive test covering:
        - Catalog creation
        - Spatial sorting
        - Storage creation
        - Multi-epoch population
        - Data retrieval
        - Coordinate verification
        - Value verification
        """
        # Step 1: Create and sort catalog
        table_transform = SExtractorTableTransform(obj_key_col='ID_STR')
        catalog = SourceCatalog(deterministic_catalog, table_transform=table_transform)
        
        # Store pre-sort mappings
        pre_sort_data = {}
        for obj_key in catalog.object_keys:
            obj_idx = list(catalog.object_keys).index(obj_key)
            pre_sort_data[obj_key] = {
                'ra': catalog.ra_values[obj_idx],
                'dec': catalog.dec_values[obj_idx],
                'measurements': {}
            }
            for meas_key in catalog.measurement_keys:
                for val_key in catalog.value_keys:
                    from pyzma_idf_lightcurve.lightcurve.catalog import SourceCatalogDataKey
                    data_key = SourceCatalogDataKey(
                        measurement=meas_key,
                        value=val_key,
                        epoch=None
                    )
                    pre_sort_data[obj_key]['measurements'][f"{meas_key}_{val_key}"] = \
                        catalog.data[data_key][obj_idx]
        
        # Sort catalog
        catalog.sort_objects_by_position(grid_divisions=10)
        
        # Step 2: Create storage
        storage_path = temp_dir / "test_full_pipeline"
        storage = LightcurveStorage(storage_path)
        storage.create_for_per_epoch_write(
            source_catalog=catalog,
            epoch_keys=epoch_keys,
        )
        
        # Step 3: Populate multiple epochs
        for epoch_idx in [0, 2, 4]:
            storage.populate_epoch(
                source_catalog=catalog,
                epoch_key=epoch_keys[epoch_idx]
            )
        
        # Step 4: Reload and verify
        storage.close()
        storage.load_for_per_epoch_write()
        assert storage.lightcurves is not None
        
        # Step 5: Verify coordinates (from catalog)
        for obj_key in catalog.object_keys:
            catalog_coords = catalog.object_key_coordinates[obj_key]
            pre_coords = pre_sort_data[obj_key]
            
            np.testing.assert_allclose(catalog_coords['ra'], pre_coords['ra'], rtol=1e-10)
            np.testing.assert_allclose(catalog_coords['dec'], pre_coords['dec'], rtol=1e-10)
        
        # Step 6: Verify measurement data
        for obj_idx, obj_key in enumerate(catalog.object_keys):
            for epoch_idx in [0, 2, 4]:
                for meas_idx, meas_key in enumerate(catalog.measurement_keys):
                    for val_idx, val_key in enumerate(catalog.value_keys):
                        # Get stored value
                        stored_val = storage.lightcurves.isel(
                            object=obj_idx,
                            measurement=meas_idx,
                            value=val_idx,
                            epoch=epoch_idx
                        ).values
                        
                        # Get expected value
                        expected_val = pre_sort_data[obj_key]['measurements'][f"{meas_key}_{val_key}"]
                        
                        # Use tolerance appropriate for float32 storage precision
                        np.testing.assert_allclose(stored_val, expected_val, rtol=1e-6,
                                                   err_msg=f"Mismatch for {obj_key} epoch {epoch_idx} {meas_key}_{val_key}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
