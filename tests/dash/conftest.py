"""Shared pytest fixtures for Dash tests."""

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
    # Cleanup after test
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_catalog():
    """Create sample astronomical catalog data for testing.
    
    Creates a small realistic catalog with 20 objects.
    """
    np.random.seed(42)  # For reproducible tests
    n_objects = 20
    
    # Create realistic IDF field coordinates (17.6 deg RA, -29.8 deg DEC)
    ra = np.random.uniform(17.5, 17.7, n_objects)
    dec = np.random.uniform(-29.9, -29.7, n_objects)
    
    # Create pixel coordinates for 750x750 field
    x_image = np.random.uniform(0, 750, n_objects)
    y_image = np.random.uniform(0, 750, n_objects)
    
    # Create astropy Table
    catalog = Table({
        'NUMBER': np.arange(1, n_objects + 1),
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
    """Create a SourceCatalog instance from sample data."""
    table_transform = SExtractorTableTransform()
    return SourceCatalog(sample_catalog, table_transform=table_transform)


@pytest.fixture
def epoch_keys():
    """Sample epoch keys (5 epochs for testing)."""
    return [f"r{58520832 + i * 256}" for i in range(5)]


@pytest.fixture
def populated_storage_read(temp_dir, source_catalog, epoch_keys):
    """Create a populated LightcurveStorage with sample data in read mode.
    
    Creates storage, populates with sample data from source_catalog,
    and returns loaded storage in read mode (consolidated).
    """
    storage_path = temp_dir / "test_storage"
    storage = LightcurveStorage(storage_path)
    
    # Create storage for per-epoch writing
    storage.create_for_per_epoch_write(
        source_catalog=source_catalog,
        epoch_keys=epoch_keys,
    )
    
    # Populate with data from all epochs
    for epoch_key in epoch_keys:
        storage.populate_epoch(
            source_catalog=source_catalog,
            epoch_key=epoch_key,
        )
    
    # Rechunk for per-object reading (creates lightcurves_read.zarr)
    storage.rechunk_for_per_object_read(chunk_size=1000)
    
    # Load in read mode
    storage.load_for_per_object_read()
    
    return storage


@pytest.fixture
def populated_storage_write(temp_dir, source_catalog, epoch_keys):
    """Create a populated LightcurveStorage with sample data in write mode.
    
    Creates storage, populates with sample data from source_catalog,
    and returns loaded storage in write mode (non-consolidated).
    """
    storage_path = temp_dir / "test_storage"
    storage = LightcurveStorage(storage_path)
    
    # Create storage for per-epoch writing
    storage.create_for_per_epoch_write(
        source_catalog=source_catalog,
        epoch_keys=epoch_keys,
    )
    
    # Populate with data from all epochs
    for epoch_key in epoch_keys:
        storage.populate_epoch(
            source_catalog=source_catalog,
            epoch_key=epoch_key,
        )
    
    # Load in write mode
    storage.load_for_per_epoch_write()
    
    return storage


@pytest.fixture
def empty_storage_path(temp_dir, source_catalog, epoch_keys):
    """Create an empty (uninitialized) storage path.
    
    Just creates storage structure without populating data.
    Returns the path to the storage.
    """
    storage_path = temp_dir / "empty_storage"
    storage = LightcurveStorage(storage_path)
    
    # Create but don't populate
    storage.create_for_per_epoch_write(
        source_catalog=source_catalog,
        epoch_keys=epoch_keys,
    )
    storage.close()
    
    return storage_path
