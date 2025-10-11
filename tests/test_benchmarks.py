#!/usr/bin/env python
"""
Performance benchmarks for lightcurve storage system.

Uses pytest-benchmark for accurate performance measurement.

Run benchmarks:
    # Run only benchmarks
    uv run pytest tests/test_benchmarks.py --benchmark-only
    
    # Save baseline
    uv run pytest tests/test_benchmarks.py --benchmark-autosave
    
    # Compare against baseline
    uv run pytest tests/test_benchmarks.py --benchmark-compare=0001
    
    # Fail if regression >10%
    uv run pytest tests/test_benchmarks.py --benchmark-compare-fail=mean:10%
"""

import pytest
import numpy as np
import shutil
import tempfile
from pathlib import Path
from astropy.table import Table

from pyzma_idf_lightcurve.lightcurve.datamodel import LightcurveStorage
from pyzma_idf_lightcurve.lightcurve.catalog import SourceCatalog


# ============================================================================
# FIXTURES - Large Dataset Creation
# ============================================================================


@pytest.fixture(scope="module")
def large_temp_dir():
    """Module-scoped temporary directory for benchmark data."""
    temp_path = Path(tempfile.mkdtemp(prefix="benchmark_"))
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture(scope="module")
def benchmark_catalog_10k():
    """Create large catalog with 10,000 objects for benchmarking.
    
    Simulates realistic IDF field catalog with:
    - 10,000 detected objects
    - Spatial coordinates (RA, DEC, X, Y)
    - Multiple measurement types (AUTO, ISO, APER)
    """
    np.random.seed(42)
    n_objects = 10_000
    
    # IDF field coordinates (17.6 deg RA, -29.8 deg DEC)
    ra = np.random.uniform(17.5, 17.7, n_objects)
    dec = np.random.uniform(-29.9, -29.7, n_objects)
    
    # Pixel coordinates for 750x750 field
    x_image = np.random.uniform(0, 750, n_objects)
    y_image = np.random.uniform(0, 750, n_objects)
    
    # Create catalog with multiple measurement types
    catalog = Table({
        'NUMBER': np.arange(1, n_objects + 1),
        'ALPHA_J2000': ra,
        'DELTA_J2000': dec, 
        'X_IMAGE': x_image,
        'Y_IMAGE': y_image,
        # AUTO measurements
        'MAG_AUTO': np.random.uniform(15, 25, n_objects),
        'MAGERR_AUTO': np.random.uniform(0.01, 0.1, n_objects),
        # ISO measurements
        'MAG_ISO': np.random.uniform(15, 25, n_objects),
        'MAGERR_ISO': np.random.uniform(0.01, 0.1, n_objects),
        # Aperture measurements
        'MAG_APER': np.random.uniform(15, 25, n_objects),
        'MAGERR_APER': np.random.uniform(0.01, 0.1, n_objects),
    })
    
    if catalog.meta is None:
        catalog.meta = {}
    catalog.meta['table_name'] = 'benchmark'
    
    return catalog


@pytest.fixture(scope="module")
def benchmark_source_catalog(benchmark_catalog_10k):
    """SourceCatalog instance for benchmarking."""
    return SourceCatalog(benchmark_catalog_10k, name='benchmark')


@pytest.fixture(scope="module")
def benchmark_epoch_keys_100():
    """100 epoch keys for realistic temporal coverage."""
    return [f"r{58520832 + i * 256}" for i in range(100)]


@pytest.fixture(scope="module")
def benchmark_measurement_keys():
    """Measurement keys for benchmark catalog."""
    return ['benchmark-auto', 'benchmark-iso', 'benchmark-aper']


# ============================================================================
# BENCHMARK TESTS - Storage Creation
# ============================================================================


class TestBenchmarkCreation:
    """Benchmark storage creation performance."""
    
    @pytest.mark.benchmark(group="creation")
    def test_benchmark_storage_creation_10k_100epochs(
        self,
        benchmark,
        large_temp_dir,
        benchmark_source_catalog,
        benchmark_epoch_keys_100,
    ):
        """Benchmark creating storage with 10K objects × 100 epochs.
        
        Tests the storage initialization and zarr array allocation
        for a realistic production dataset size.
        """
        storage_path = large_temp_dir / "creation_benchmark"
        
        def create_storage():
            storage = LightcurveStorage(storage_path)
            storage.create_storage(
                source_catalog=benchmark_source_catalog,
                epoch_keys=benchmark_epoch_keys_100,
            )
            return storage
        
        result = benchmark(create_storage)
        assert result.lightcurves is not None
        
        # Cleanup for next run
        if storage_path.exists():
            shutil.rmtree(storage_path)


# ============================================================================
# BENCHMARK TESTS - Data Population
# ============================================================================


class TestBenchmarkPopulation:
    """Benchmark epoch population performance."""
    
    @pytest.fixture
    def empty_storage_10k_100epochs(
        self,
        large_temp_dir,
        benchmark_source_catalog,
        benchmark_epoch_keys_100,
    ):
        """Create empty storage ready for population benchmarks."""
        storage_path = large_temp_dir / "population_benchmark"
        if storage_path.exists():
            shutil.rmtree(storage_path)
        
        storage = LightcurveStorage(storage_path)
        storage.create_storage(
            source_catalog=benchmark_source_catalog,
            epoch_keys=benchmark_epoch_keys_100,
        )
        return storage
    
    @pytest.mark.benchmark(group="population")
    def test_benchmark_populate_single_epoch_10k_objects(
        self,
        benchmark,
        empty_storage_10k_100epochs,
        benchmark_source_catalog,
        benchmark_measurement_keys,
    ):
        """Benchmark populating one epoch with 10K measurements.
        
        Tests the vectorized assignment performance for a single
        epoch with all measurement types.
        """
        storage = empty_storage_10k_100epochs
        
        result = benchmark(
            storage.populate_epoch_from_catalog,
            epoch_key="r58520832",
            source_catalog=benchmark_source_catalog,
            measurement_keys=benchmark_measurement_keys,
        )
        assert result > 0

    @pytest.mark.benchmark(group="populate_versions")
    def test_benchmark_populate_v0_10k_objects(
        self,
        benchmark,
        empty_storage_10k_100epochs,
        benchmark_source_catalog,
        benchmark_measurement_keys,
    ):
        """Benchmark v0: Original xarray .loc[] + zarr write."""
        storage = empty_storage_10k_100epochs
        
        result = benchmark(
            storage.populate_epoch_from_catalog_v0,
            epoch_key="r58520832",
            source_catalog=benchmark_source_catalog,
            measurement_keys=benchmark_measurement_keys,
        )
        assert result > 0

    @pytest.mark.benchmark(group="populate_versions")
    def test_benchmark_populate_v1_10k_objects(
        self,
        benchmark,
        empty_storage_10k_100epochs,
        benchmark_source_catalog,
        benchmark_measurement_keys,
    ):
        """Benchmark v1: Read-modify-write with pre-computed indices."""
        storage = empty_storage_10k_100epochs
        
        result = benchmark(
            storage.populate_epoch_from_catalog_v1,
            epoch_key="r58520832",
            source_catalog=benchmark_source_catalog,
            measurement_keys=benchmark_measurement_keys,
        )
        assert result > 0

    @pytest.mark.benchmark(group="populate_versions")
    def test_benchmark_populate_v2_10k_objects(
        self,
        benchmark,
        empty_storage_10k_100epochs,
        benchmark_source_catalog,
        benchmark_measurement_keys,
    ):
        """Benchmark v2: User's handwritten implementation."""
        storage = empty_storage_10k_100epochs
        
        result = benchmark(
            storage.populate_epoch_from_catalog_v2,
            epoch_key="r58520832",
            source_catalog=benchmark_source_catalog,
            measurement_keys=benchmark_measurement_keys,
        )
        assert result > 0
    
    @pytest.mark.benchmark(group="population")
    def test_benchmark_populate_multiple_epochs(
        self,
        benchmark,
        large_temp_dir,
        benchmark_source_catalog,
        benchmark_measurement_keys,
    ):
        """Benchmark populating 10 epochs sequentially.
        
        Tests the performance of incremental epoch population,
        simulating a typical workflow.
        """
        storage_path = large_temp_dir / "multi_epoch_benchmark"
        if storage_path.exists():
            shutil.rmtree(storage_path)
        
        # Create storage with 10 epochs
        epoch_keys = [f"r{58520832 + i * 256}" for i in range(10)]
        
        def populate_all_epochs():
            storage = LightcurveStorage(storage_path)
            storage.create_storage(
                source_catalog=benchmark_source_catalog,
                epoch_keys=epoch_keys,
            )
            
            # Populate all epochs
            for epoch_key in epoch_keys:
                storage.populate_epoch_from_catalog(
                    epoch_key=epoch_key,
                    source_catalog=benchmark_source_catalog,
                    measurement_keys=benchmark_measurement_keys,
                )
            
            return storage
        
        result = benchmark(populate_all_epochs)
        assert result.lightcurves is not None


# ============================================================================
# BENCHMARK TESTS - I/O Operations
# ============================================================================


class TestBenchmarkIO:
    """Benchmark I/O performance (load/save/reload)."""
    
    @pytest.fixture(scope="class")
    def populated_storage_path(
        self,
        large_temp_dir,
        benchmark_source_catalog,
        benchmark_measurement_keys,
    ):
        """Create and populate storage for I/O benchmarks."""
        storage_path = large_temp_dir / "io_benchmark"
        if storage_path.exists():
            shutil.rmtree(storage_path)
        
        # Create storage with 100 epochs
        epoch_keys = [f"r{58520832 + i * 256}" for i in range(100)]
        storage = LightcurveStorage(storage_path)
        storage.create_storage(
            source_catalog=benchmark_source_catalog,
            epoch_keys=epoch_keys,
        )
        
        # Populate first 10 epochs to have realistic data
        for i in range(10):
            storage.populate_epoch_from_catalog(
                epoch_key=epoch_keys[i],
                source_catalog=benchmark_source_catalog,
                measurement_keys=benchmark_measurement_keys,
            )
        
        return storage_path
    
    @pytest.mark.benchmark(group="io")
    def test_benchmark_storage_reload_consolidated(
        self,
        benchmark,
        populated_storage_path,
    ):
        """Benchmark loading storage with consolidated metadata.
        
        Tests the performance of loading a large storage from disk
        with zarr consolidated metadata enabled.
        """
        def load_storage():
            storage = LightcurveStorage(populated_storage_path)
            storage.load_storage(consolidated=True)
            return storage
        
        result = benchmark(load_storage)
        assert result.lightcurves is not None
    
    @pytest.mark.benchmark(group="io")
    def test_benchmark_storage_reload_unconsolidated(
        self,
        benchmark,
        populated_storage_path,
    ):
        """Benchmark loading storage without consolidated metadata.
        
        Compares performance with unconsolidated metadata loading.
        """
        def load_storage():
            storage = LightcurveStorage(populated_storage_path)
            storage.load_storage(consolidated=False)
            return storage
        
        result = benchmark(load_storage)
        assert result.lightcurves is not None


# ============================================================================
# BENCHMARK TESTS - Query Operations
# ============================================================================


class TestBenchmarkQueries:
    """Benchmark query performance."""
    
    @pytest.fixture(scope="class")
    def populated_storage_for_queries(
        self,
        large_temp_dir,
        benchmark_source_catalog,
        benchmark_measurement_keys,
    ):
        """Create populated storage for query benchmarks."""
        storage_path = large_temp_dir / "query_benchmark"
        if storage_path.exists():
            shutil.rmtree(storage_path)
        
        epoch_keys = [f"r{58520832 + i * 256}" for i in range(20)]
        storage = LightcurveStorage(storage_path)
        storage.create_storage(
            source_catalog=benchmark_source_catalog,
            epoch_keys=epoch_keys,
        )
        
        # Populate all epochs
        for epoch_key in epoch_keys:
            storage.populate_epoch_from_catalog(
                epoch_key=epoch_key,
                source_catalog=benchmark_source_catalog,
                measurement_keys=benchmark_measurement_keys,
            )
        
        return storage
    
    @pytest.mark.benchmark(group="query")
    def test_benchmark_object_lightcurve_retrieval(
        self,
        benchmark,
        populated_storage_for_queries,
    ):
        """Benchmark retrieving a single object's lightcurve.
        
        Tests the performance of accessing one object's time series
        from a large dataset.
        """
        storage = populated_storage_for_queries
        object_key = "1"  # First object
        measurement_key = "benchmark-auto"
        
        result = benchmark(
            storage.get_object_lightcurve,
            object_key=object_key,
            measurement_key=measurement_key,
        )
        
        assert result is not None
    
    @pytest.mark.benchmark(group="query")
    def test_benchmark_epoch_data_retrieval(
        self,
        benchmark,
        populated_storage_for_queries,
    ):
        """Benchmark retrieving all objects for a single epoch.
        
        Tests the performance of accessing one epoch's measurements
        across all objects.
        """
        storage = populated_storage_for_queries
        epoch_key = "r58520832"
        measurement_key = "benchmark-auto"
        
        result = benchmark(
            storage.get_epoch_data,
            epoch_key=epoch_key,
            measurement_key=measurement_key,
        )
        
        assert result is not None
    
    # TODO: Add benchmarks for spatial region queries when functionality is available
    # Spatial region query methods not yet implemented in LightcurveStorage


# ============================================================================
# BENCHMARK TESTS - Memory Usage
# ============================================================================


class TestBenchmarkMemory:
    """Benchmark memory-related operations.
    
    Note: These tests measure time, not memory directly.
    For actual memory profiling, use pytest-memray plugin.
    """
    
    @pytest.mark.benchmark(group="memory")
    def test_benchmark_coordinate_extraction(
        self,
        benchmark,
        benchmark_source_catalog,
    ):
        """Benchmark extracting coordinates from large catalog.
        
        Tests memory-efficient coordinate extraction performance.
        """
        def extract_coordinates():
            # Get coordinate dictionary for all objects
            coords_dict = benchmark_source_catalog.get_coordinate_dict()
            return coords_dict
        
        result = benchmark(extract_coordinates)
        assert result is not None
    
    @pytest.mark.benchmark(group="memory")
    def test_benchmark_storage_info_access(
        self,
        benchmark,
        large_temp_dir,
        benchmark_source_catalog,
    ):
        """Benchmark accessing storage metadata.
        
        Tests the performance of metadata access operations.
        """
        storage_path = large_temp_dir / "info_benchmark"
        if storage_path.exists():
            shutil.rmtree(storage_path)
        
        storage = LightcurveStorage(storage_path)
        storage.create_storage(
            source_catalog=benchmark_source_catalog,
            epoch_keys=[f"r{58520832 + i * 256}" for i in range(100)],
        )
        
        result = benchmark(storage.get_storage_info)
        assert result is not None
