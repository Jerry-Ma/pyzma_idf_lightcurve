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


def _create_benchmark_catalog(n_objects, seed=42):
    """Helper function to create benchmark catalogs of different sizes."""
    np.random.seed(seed)
    
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
    return catalog


@pytest.fixture(scope="module")
def benchmark_catalog_10k():
    """Create large catalog with 10,000 objects for benchmarking.
    
    Simulates realistic IDF field catalog with:
    - 10,000 detected objects
    - Spatial coordinates (RA, DEC, X, Y)
    - Multiple measurement types (AUTO, ISO, APER)
    """
    return _create_benchmark_catalog(10_000)


@pytest.fixture(scope="module")
def benchmark_catalog_50k():
    """Create large catalog with 50,000 objects for scaling tests."""
    return _create_benchmark_catalog(50_000)


@pytest.fixture(scope="module")
def benchmark_catalog_100k():
    """Create large catalog with 100,000 objects for scaling tests."""
    return _create_benchmark_catalog(100_000)


@pytest.fixture(scope="module")
def benchmark_source_catalog(benchmark_catalog_10k):
    """SourceCatalog instance for benchmarking."""
    from pyzma_idf_lightcurve.lightcurve.catalog import SExtractorTableTransform
    table_transform = SExtractorTableTransform()
    return SourceCatalog(benchmark_catalog_10k, table_transform=table_transform)


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
            storage.create_for_per_epoch_write(
                source_catalog=benchmark_source_catalog,
                epoch_keys=benchmark_epoch_keys_100,
            )
            return storage
        
        result = benchmark(create_storage)
        assert result.load_for_per_epoch_write() is not None
        
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
        storage.create_for_per_epoch_write(
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
            storage.populate_epoch,
            epoch_key="r58520832",
            source_catalog=benchmark_source_catalog,
        )
        assert result > 0

    # v0, v1, v2 populate methods removed - only current implementation remains
    
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
            storage.create_for_per_epoch_write(
                source_catalog=benchmark_source_catalog,
                epoch_keys=epoch_keys,
            )
            
            # Populate all epochs
            for epoch_key in epoch_keys:
                storage.populate_epoch(
                    epoch_key=epoch_key,
                    source_catalog=benchmark_source_catalog,
                )
            
            return storage
        
        result = benchmark(populate_all_epochs)
        assert result.load_for_per_epoch_write() is not None


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
        storage.create_for_per_epoch_write(
            source_catalog=benchmark_source_catalog,
            epoch_keys=epoch_keys,
        )
        
        # Populate first 10 epochs to have realistic data
        for i in range(10):
            storage.populate_epoch(
                epoch_key=epoch_keys[i],
                source_catalog=benchmark_source_catalog,
            )
        storage.rechunk_for_per_object_read() 
        return storage_path
    
    @pytest.mark.benchmark(group="io")
    def test_benchmark_storage_load_write_optimized(
        self,
        benchmark,
        populated_storage_path,
    ):
        def load_storage():
            storage = LightcurveStorage(populated_storage_path)
            storage.load_for_per_epoch_write()
            return storage
        
        result = benchmark(load_storage)
        assert result.lightcurves is not None
    
    @pytest.mark.benchmark(group="io")
    def test_benchmark_storage_load_read_optimized(
        self,
        benchmark,
        populated_storage_path,
    ):
        def load_storage():
            storage = LightcurveStorage(populated_storage_path)
            storage.load_for_per_object_read()
            return storage
        
        result = benchmark(load_storage)
        assert result.lightcurves is not None


# ============================================================================
# BENCHMARK TESTS - Rechunking Operations
# ============================================================================


class TestBenchmarkRechunking:
    """Benchmark rechunking performance."""
    
    @pytest.fixture(scope="class")
    def populated_storage_for_rechunk(self, large_temp_dir, benchmark_source_catalog):
        """Create populated storage for rechunking benchmarks."""
        storage_path = large_temp_dir / "rechunk_benchmark"
        if storage_path.exists():
            shutil.rmtree(storage_path)
        
        epoch_keys = [f"r{58520832 + i * 256}" for i in range(20)]
        storage = LightcurveStorage(storage_path)
        storage.create_for_per_epoch_write(
            source_catalog=benchmark_source_catalog,
            epoch_keys=epoch_keys,
        )
        
        # Populate all epochs to have realistic data
        for epoch_key in epoch_keys:
            storage.populate_epoch(
                epoch_key=epoch_key,
                source_catalog=benchmark_source_catalog,
            )
        
        return storage
    
    @pytest.mark.benchmark(group="rechunking")
    def test_benchmark_rechunk_for_per_object_read(self, benchmark, populated_storage_for_rechunk):
        """Benchmark rechunking from per-epoch-write to per-object-read layout.

        Tests the performance of reorganizing a 10K objects × 20 epochs dataset
        from epoch-optimized chunks to object-optimized chunks.
        """
        storage = populated_storage_for_rechunk

        result = benchmark(storage.rechunk_for_per_object_read)

        # Verify rechunking completed
        assert result is None  # Method returns None on success
        assert storage.zarr_path_for_read.exists()


# ============================================================================
# BENCHMARK TESTS - Read Performance Comparison
# ============================================================================


class TestBenchmarkReadComparison:
    """Compare read performance between write-optimized and read-optimized storage.
    
    These tests validate that rechunking actually improves per-object read performance.
    Uses a large dataset (50K objects × 100 epochs) to demonstrate real performance differences.
    """
    
    @pytest.fixture(scope="class")
    def dual_storage_setup(self, large_temp_dir, benchmark_catalog_50k):
        """Create both write-optimized and read-optimized storage for comparison.
        
        Uses 50K objects × 100 epochs (15M measurements) to demonstrate
        clear performance differences between access patterns.
        """
        from pyzma_idf_lightcurve.lightcurve.catalog import SExtractorTableTransform
        
        write_storage_path = large_temp_dir / "read_comparison"
        
        # Clean up if exists
        for path in [write_storage_path]:
            if path.exists():
                shutil.rmtree(path)
        
        # Create SourceCatalog from the 50K benchmark catalog
        table_transform = SExtractorTableTransform()
        source_catalog = SourceCatalog(benchmark_catalog_50k, table_transform=table_transform)
        
        # Create and populate write-optimized storage with 100 epochs
        epoch_keys = [f"r{58520832 + i * 256}" for i in range(100)]
        write_storage = LightcurveStorage(write_storage_path)
        write_storage.create_for_per_epoch_write(
            source_catalog=source_catalog,
            epoch_keys=epoch_keys,
            zarr_chunks={"object": 50000},  # Explicit chunk size for large dataset
        )
        
        # Populate all epochs
        for epoch_key in epoch_keys:
            write_storage.populate_epoch(
                epoch_key=epoch_key,
                source_catalog=source_catalog,
            )
        
        # Create read-optimized version
        write_storage.rechunk_for_per_object_read(chunk_size=5000)
        
        # Load both storages
        write_storage.load_for_per_epoch_write()

        # Load read-optimized storage
        read_storage = LightcurveStorage(write_storage_path)
        read_storage.load_for_per_object_read()
        assert write_storage._zarr_path_loaded != read_storage._zarr_path_loaded
        
        return {
            'write_storage': write_storage,
            'read_storage': read_storage,
            'epoch_keys': epoch_keys,
        }
    
    @pytest.mark.benchmark(group="read_comparison_object")
    def test_benchmark_object_read_write_optimized(self, benchmark, dual_storage_setup):
        """Benchmark per-object read on write-optimized (epoch-chunked) storage.

        Dataset: 50K objects × 100 epochs
        This should be baseline performance, reading across many epoch chunks.
        """
        storage = dual_storage_setup['write_storage']
        object_key = "25000"  # Middle object
        measurement_key = "auto"

        result = benchmark(
            storage.get_object_lightcurve,
            object_key=object_key,
            measurement_key=measurement_key,
        )
        
        assert result is not None
    
    @pytest.mark.benchmark(group="read_comparison_object")
    def test_benchmark_object_read_read_optimized(self, benchmark, dual_storage_setup):
        """Benchmark per-object read on read-optimized (object-chunked) storage.

        Dataset: 50K objects × 100 epochs
        This should be faster as object data is stored contiguously.
        """
        storage = dual_storage_setup['read_storage']
        object_key = "25000"  # Middle object
        measurement_key = "auto"

        result = benchmark(
            storage.get_object_lightcurve,
            object_key=object_key,
            measurement_key=measurement_key,
        )
        
        assert result is not None
    
    @pytest.mark.benchmark(group="read_comparison_epoch")
    def test_benchmark_epoch_read_write_optimized(self, benchmark, dual_storage_setup):
        """Benchmark per-epoch read on write-optimized (epoch-chunked) storage.

        Dataset: 50K objects × 100 epochs
        This should be faster as epoch data is stored contiguously.
        """
        storage = dual_storage_setup['write_storage']
        epoch_key = "r58533632"  # Middle epoch (index 50)
        measurement_key = "auto"

        result = benchmark(
            storage.get_epoch_data,
            epoch_key=epoch_key,
            measurement_key=measurement_key,
        )
        
        assert result is not None
    
    @pytest.mark.benchmark(group="read_comparison_epoch")
    def test_benchmark_epoch_read_read_optimized(self, benchmark, dual_storage_setup):
        """Benchmark per-epoch read on read-optimized (object-chunked) storage.

        Dataset: 50K objects × 100 epochs
        This should be slower as it needs to read across multiple object chunks.
        """
        storage = dual_storage_setup['read_storage']
        epoch_key = "r58533632"  # Middle epoch (index 50)
        measurement_key = "auto"

        result = benchmark(
            storage.get_epoch_data,
            epoch_key=epoch_key,
            measurement_key=measurement_key,
        )
        
        assert result is not None


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
            # Get sky coordinates (ra, dec) for all objects
            ra = benchmark_source_catalog.ra_values
            dec = benchmark_source_catalog.dec_values
            return ra, dec
        
        result = benchmark(extract_coordinates)
        assert result is not None
        assert len(result) == 2  # (ra, dec)
    
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
        storage.create_for_per_epoch_write(
            source_catalog=benchmark_source_catalog,
            epoch_keys=[f"r{58520832 + i * 256}" for i in range(100)],
        )
        
        result = benchmark(storage.get_storage_info, which="write")
        assert result is not None


# ============================================================================
# BENCHMARK TESTS - Scaling with Large Datasets
# ============================================================================


class TestBenchmarkScaling:
    """Test performance scaling with increasingly large datasets.
    
    These tests help identify performance bottlenecks and verify
    that the clear_encoding() approach scales well.
    
    Note: Tests with >10K objects currently fail due to Dask's automatic
    chunking of coordinate arrays conflicting with Zarr encoding. This is
    a known limitation that needs to be addressed in datamodel.py by
    explicitly chunking the Dask array at creation time.
    """
    
    def _create_source_catalog(self, catalog_table):
        """Helper to create SourceCatalog from table."""
        from pyzma_idf_lightcurve.lightcurve.catalog import SExtractorTableTransform
        table_transform = SExtractorTableTransform()
        return SourceCatalog(catalog_table, table_transform=table_transform)
    
    @pytest.mark.benchmark(group="scaling_rechunk")
    def test_benchmark_rechunk_10k_objects(
        self,
        benchmark,
        large_temp_dir,
        benchmark_catalog_10k,
    ):
        """Benchmark rechunking with 10K objects (baseline)."""
        storage_path = large_temp_dir / "scaling_rechunk_10k"
        if storage_path.exists():
            shutil.rmtree(storage_path)
        
        source_catalog = self._create_source_catalog(benchmark_catalog_10k)
        epoch_keys = [f"r{58520832 + i * 256}" for i in range(100)]
        
        storage = LightcurveStorage(storage_path)
        storage.create_for_per_epoch_write(
            source_catalog=source_catalog,
            epoch_keys=epoch_keys,
        )
        
        # Populate a few epochs to have realistic data
        for i in range(10):
            storage.populate_epoch(
                epoch_key=epoch_keys[i],
                source_catalog=source_catalog,
            )
        
        result = benchmark(storage.rechunk_for_per_object_read, chunk_size=1000)
        assert result is None
    
    @pytest.mark.benchmark(group="scaling_rechunk")
    def test_benchmark_rechunk_50k_objects(
        self,
        benchmark,
        large_temp_dir,
        benchmark_catalog_50k,
    ):
        """Benchmark rechunking with 50K objects (5x scaling test)."""
        storage_path = large_temp_dir / "scaling_rechunk_50k"
        if storage_path.exists():
            shutil.rmtree(storage_path)
        
        source_catalog = self._create_source_catalog(benchmark_catalog_50k)
        epoch_keys = [f"r{58520832 + i * 256}" for i in range(100)]
        
        storage = LightcurveStorage(storage_path)
        # Use explicit zarr_chunks matching object count to avoid Dask chunking issues
        storage.create_for_per_epoch_write(
            source_catalog=source_catalog,
            epoch_keys=epoch_keys,
            zarr_chunks={"object": 50000},  # Explicit chunk size to match array
        )
        
        # Populate a few epochs to have realistic data
        for i in range(10):
            storage.populate_epoch(
                epoch_key=epoch_keys[i],
                source_catalog=source_catalog,
            )
        
        result = benchmark(storage.rechunk_for_per_object_read, chunk_size=5000)
        assert result is None
    
    @pytest.mark.benchmark(group="scaling_rechunk")
    def test_benchmark_rechunk_100k_objects(
        self,
        benchmark,
        large_temp_dir,
        benchmark_catalog_100k,
    ):
        """Benchmark rechunking with 100K objects (10x scaling test)."""
        storage_path = large_temp_dir / "scaling_rechunk_100k"
        if storage_path.exists():
            shutil.rmtree(storage_path)
        
        source_catalog = self._create_source_catalog(benchmark_catalog_100k)
        epoch_keys = [f"r{58520832 + i * 256}" for i in range(100)]
        
        storage = LightcurveStorage(storage_path)
        # Use explicit zarr_chunks matching object count to avoid Dask chunking issues
        storage.create_for_per_epoch_write(
            source_catalog=source_catalog,
            epoch_keys=epoch_keys,
            zarr_chunks={"object": 100000},  # Explicit chunk size to match array
        )
        
        # Populate a few epochs to have realistic data
        for i in range(10):
            storage.populate_epoch(
                epoch_key=epoch_keys[i],
                source_catalog=source_catalog,
            )
        
        result = benchmark(storage.rechunk_for_per_object_read, chunk_size=10000)
        assert result is None
    
    @pytest.mark.benchmark(group="scaling_read_object")
    def test_benchmark_read_10k_objects(
        self,
        benchmark,
        large_temp_dir,
        benchmark_catalog_10k,
    ):
        """Benchmark object read performance with 10K objects."""
        storage_path = large_temp_dir / "scaling_read_10k"
        if storage_path.exists():
            shutil.rmtree(storage_path)
        
        source_catalog = self._create_source_catalog(benchmark_catalog_10k)
        epoch_keys = [f"r{58520832 + i * 256}" for i in range(100)]
        
        storage = LightcurveStorage(storage_path)
        storage.create_for_per_epoch_write(
            source_catalog=source_catalog,
            epoch_keys=epoch_keys,
        )
        
        # Populate epochs
        for i in range(10):
            storage.populate_epoch(
                epoch_key=epoch_keys[i],
                source_catalog=source_catalog,
            )
        
        # Rechunk for read
        storage.rechunk_for_per_object_read(chunk_size=1000)
        storage.load_for_per_object_read()
        
        # Benchmark reading an object in the middle
        result = benchmark(
            storage.get_object_lightcurve,
            object_key="5000",
            measurement_key="auto",
        )
        assert result is not None
    
    @pytest.mark.benchmark(group="scaling_read_object")
    def test_benchmark_read_50k_objects(
        self,
        benchmark,
        large_temp_dir,
        benchmark_catalog_50k,
    ):
        """Benchmark object read performance with 50K objects."""
        storage_path = large_temp_dir / "scaling_read_50k"
        if storage_path.exists():
            shutil.rmtree(storage_path)
        
        source_catalog = self._create_source_catalog(benchmark_catalog_50k)
        epoch_keys = [f"r{58520832 + i * 256}" for i in range(100)]
        
        storage = LightcurveStorage(storage_path)
        storage.create_for_per_epoch_write(
            source_catalog=source_catalog,
            epoch_keys=epoch_keys,
            zarr_chunks={"object": 50000},  # Explicit chunk size to avoid Dask chunking issues
        )
        
        # Populate epochs
        for i in range(10):
            storage.populate_epoch(
                epoch_key=epoch_keys[i],
                source_catalog=source_catalog,
            )
        
        # Rechunk for read
        storage.rechunk_for_per_object_read(chunk_size=5000)
        storage.load_for_per_object_read()
        
        # Benchmark reading an object in the middle
        result = benchmark(
            storage.get_object_lightcurve,
            object_key="25000",
            measurement_key="auto",
        )
        assert result is not None
    
    @pytest.mark.benchmark(group="scaling_read_object")
    def test_benchmark_read_100k_objects(
        self,
        benchmark,
        large_temp_dir,
        benchmark_catalog_100k,
    ):
        """Benchmark object read performance with 100K objects."""
        storage_path = large_temp_dir / "scaling_read_100k"
        if storage_path.exists():
            shutil.rmtree(storage_path)
        
        source_catalog = self._create_source_catalog(benchmark_catalog_100k)
        epoch_keys = [f"r{58520832 + i * 256}" for i in range(100)]
        
        storage = LightcurveStorage(storage_path)
        storage.create_for_per_epoch_write(
            source_catalog=source_catalog,
            epoch_keys=epoch_keys,
            zarr_chunks={"object": 100000},  # Explicit chunk size to avoid Dask chunking issues
        )
        
        # Populate epochs
        for i in range(10):
            storage.populate_epoch(
                epoch_key=epoch_keys[i],
                source_catalog=source_catalog,
            )
        
        # Rechunk for read
        storage.rechunk_for_per_object_read(chunk_size=10000)
        storage.load_for_per_object_read()
        
        # Benchmark reading an object in the middle
        result = benchmark(
            storage.get_object_lightcurve,
            object_key="50000",
            measurement_key="auto",
        )
        assert result is not None
