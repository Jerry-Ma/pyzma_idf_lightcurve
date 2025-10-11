"""
Scaling benchmarks for populate_epoch_from_catalog implementations.

Tests how v0, v1, v2 perform with different dataset sizes to verify
they scale similarly and identify any performance characteristics.
"""

import pytest
import numpy as np
from pathlib import Path
from astropy.table import Table

from pyzma_idf_lightcurve.lightcurve.catalog import SourceCatalog
from pyzma_idf_lightcurve.lightcurve.datamodel import LightcurveStorage


def create_test_catalog_with_size(n_objects: int, n_measurements: int = 3) -> SourceCatalog:
    """Create test catalog with specified number of objects and measurements."""
    table = Table()
    table['NUMBER'] = np.arange(1, n_objects + 1)
    table['ALPHA_J2000'] = np.linspace(150.0, 150.5, n_objects)
    table['DELTA_J2000'] = np.linspace(2.0, 2.5, n_objects)
    table['X_IMAGE'] = np.linspace(100.0, 1000.0, n_objects)
    table['Y_IMAGE'] = np.linspace(100.0, 1000.0, n_objects)
    
    # Add measurements
    for meas_idx in range(n_measurements):
        meas_name = ['AUTO', 'ISO', 'APER'][meas_idx]
        table[f'MAG_{meas_name}'] = 20.0 + np.arange(n_objects) * 0.01 + meas_idx * 0.1
        table[f'MAGERR_{meas_name}'] = 0.05 + np.arange(n_objects) * 0.0001 + meas_idx * 0.01
    
    return SourceCatalog(table)


@pytest.fixture(params=[10, 50, 100, 500, 1000])
def catalog_size(request):
    """Parametrize catalog sizes for scaling tests."""
    return request.param


@pytest.fixture
def temp_storage_dir(tmp_path):
    """Create temporary directory for test storage."""
    return tmp_path / "scaling_test"


class TestPopulateScaling:
    """Test scaling behavior of populate implementations."""
    
    def create_and_populate_storage(
        self, 
        storage_dir: Path, 
        catalog: SourceCatalog,
        epoch_key: str,
        version: str = 'v1'
    ):
        """Helper to create storage and populate with specified version."""
        measurement_keys = catalog.measurement_keys
        
        # Create storage
        storage = LightcurveStorage.create_storage(
            storage_dir,
            source_catalog=catalog,
            measurement_keys=measurement_keys,
            epoch_keys=[epoch_key]
        )
        
        # Populate based on version
        if version == 'v0':
            storage.populate_epoch_from_catalog_v0(epoch_key, catalog, measurement_keys)
        elif version == 'v1':
            storage.populate_epoch_from_catalog_v1(epoch_key, catalog, measurement_keys)
        elif version == 'v2':
            storage.populate_epoch_from_catalog_v2(epoch_key, catalog, measurement_keys)
        else:
            raise ValueError(f"Unknown version: {version}")
        
        return storage
    
    def test_v0_scaling(self, benchmark, temp_storage_dir, catalog_size):
        """Benchmark v0 with different catalog sizes."""
        catalog = create_test_catalog_with_size(catalog_size)
        epoch_key = 'r58520832'
        
        storage_dir = temp_storage_dir / f"v0_{catalog_size}"
        
        benchmark(
            self.create_and_populate_storage,
            storage_dir,
            catalog,
            epoch_key,
            version='v0'
        )
    
    def test_v1_scaling(self, benchmark, temp_storage_dir, catalog_size):
        """Benchmark v1 with different catalog sizes."""
        catalog = create_test_catalog_with_size(catalog_size)
        epoch_key = 'r58520832'
        
        storage_dir = temp_storage_dir / f"v1_{catalog_size}"
        
        benchmark(
            self.create_and_populate_storage,
            storage_dir,
            catalog,
            epoch_key,
            version='v1'
        )
    
    def test_v2_scaling(self, benchmark, temp_storage_dir, catalog_size):
        """Benchmark v2 with different catalog sizes."""
        catalog = create_test_catalog_with_size(catalog_size)
        epoch_key = 'r58520832'
        
        storage_dir = temp_storage_dir / f"v2_{catalog_size}"
        
        benchmark(
            self.create_and_populate_storage,
            storage_dir,
            catalog,
            epoch_key,
            version='v2'
        )


@pytest.mark.parametrize('n_objects', [100, 1000, 5000])
@pytest.mark.parametrize('version', ['v0', 'v1', 'v2'])
def test_populate_scaling_comparison(benchmark, tmp_path, n_objects, version):
    """
    Compare scaling of all versions with different object counts.
    
    This test allows grouping by version or by size for easier comparison.
    """
    catalog = create_test_catalog_with_size(n_objects)
    epoch_key = 'r58520832'
    measurement_keys = catalog.measurement_keys
    
    storage_dir = tmp_path / f"scale_{version}_{n_objects}"
    
    def populate_once():
        storage = LightcurveStorage(storage_dir)
        storage.create_storage(
            source_catalog=catalog,
            measurement_keys=measurement_keys,
            epoch_keys=[epoch_key]
        )
        
        if version == 'v0':
            storage.populate_epoch_from_catalog_v0(epoch_key, catalog, measurement_keys)
        elif version == 'v1':
            storage.populate_epoch_from_catalog_v1(epoch_key, catalog, measurement_keys)
        elif version == 'v2':
            storage.populate_epoch_from_catalog_v2(epoch_key, catalog, measurement_keys)
        
        return storage
    
    benchmark(populate_once)
