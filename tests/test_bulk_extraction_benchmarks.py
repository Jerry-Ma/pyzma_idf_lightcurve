"""Benchmark tests for bulk data extraction patterns.

These benchmarks test the actual data access patterns used in the dash app and notebooks:
1. Extracting all 1D variables along a dimension for DataFrame building
2. Comparing different extraction methods (.values vs .compute())
3. Testing chunking strategies for bulk extraction
4. Memory-efficient extraction patterns

Key Learnings from xarray docs:
- .values forces immediate computation (BAD for dask)
- .compute() respects lazy evaluation (GOOD)
- Chunking should align with access patterns
- Use coordinate-based indexing (.sel()) when possible

Run benchmarks:
    uv run pytest tests/test_bulk_extraction_benchmarks.py --benchmark-only -v
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from astropy.table import Table

from pyzma_idf_lightcurve.lightcurve.catalog import SourceCatalog
from pyzma_idf_lightcurve.lightcurve.datamodel import LightcurveStorage


# ============================================================================
# FIXTURES - Benchmark Catalogs
# ============================================================================


@pytest.fixture(scope="module")
def bulk_temp_dir():
    """Module-scoped temporary directory for bulk extraction benchmarks."""
    temp_path = Path(tempfile.mkdtemp(prefix="bulk_extract_"))
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


def _create_benchmark_catalog(n_objects, seed=42):
    """Helper function to create benchmark catalogs."""
    np.random.seed(seed)
    
    # IDF field coordinates
    ra = np.random.uniform(17.5, 17.7, n_objects)
    dec = np.random.uniform(-29.9, -29.7, n_objects)
    
    # SExtractor-like output
    x_image = np.random.uniform(1, 4000, n_objects)
    y_image = np.random.uniform(1, 4000, n_objects)
    flux_auto = np.random.lognormal(8, 1.5, n_objects)  # log-normal flux distribution
    
    # Generate unique sequential IDs
    source_ids = [f"I{i:06d}" for i in range(n_objects)]
    
    catalog_data = {
        "NUMBER": np.arange(1, n_objects + 1),
        "source_id": source_ids,
        "X_IMAGE": x_image,
        "Y_IMAGE": y_image,
        "ALPHA_J2000": ra,
        "DELTA_J2000": dec,
        "FLUX_AUTO": flux_auto,
        "FLUXERR_AUTO": flux_auto * 0.05,
        "MAG_AUTO": -2.5 * np.log10(flux_auto) + 25.0,
        "MAGERR_AUTO": 1.086 / (flux_auto / (flux_auto * 0.05)),
    }
    
    return Table(catalog_data)


@pytest.fixture(scope="module")
def bulk_catalog_1k():
    """Small catalog for quick tests."""
    return _create_benchmark_catalog(1000)


@pytest.fixture(scope="module")
def bulk_catalog_5k():
    """Medium catalog."""
    return _create_benchmark_catalog(5000)


@pytest.fixture(scope="module")
def bulk_catalog_10k():
    """Larger catalog."""
    return _create_benchmark_catalog(10000)


@pytest.fixture(scope="module")
def bulk_source_catalog_1k(bulk_catalog_1k):
    """SourceCatalog instance."""
    from pyzma_idf_lightcurve.lightcurve.catalog import SExtractorTableTransform
    return SourceCatalog(bulk_catalog_1k, table_transform=SExtractorTableTransform())


@pytest.fixture(scope="module")
def bulk_source_catalog_5k(bulk_catalog_5k):
    """SourceCatalog instance."""
    from pyzma_idf_lightcurve.lightcurve.catalog import SExtractorTableTransform
    return SourceCatalog(bulk_catalog_5k, table_transform=SExtractorTableTransform())


@pytest.fixture(scope="module")
def bulk_source_catalog_10k(bulk_catalog_10k):
    """SourceCatalog instance."""
    from pyzma_idf_lightcurve.lightcurve.catalog import SExtractorTableTransform
    return SourceCatalog(bulk_catalog_10k, table_transform=SExtractorTableTransform())


# ============================================================================
# BENCHMARK TESTS - Bulk Extraction Patterns
# ============================================================================


class TestBulkExtractionPatterns:
    """Test different patterns for bulk extraction of coordinates/variables."""
    
    @pytest.fixture
    def small_populated_storage(self, bulk_temp_dir, bulk_source_catalog_1k):
        """Create a small storage with 1K objects, 10 epochs, populated."""
        storage_path = bulk_temp_dir / "bulk_extract_small.zarr"
        epoch_keys = [f"r{58520832 + i * 256}" for i in range(10)]  # 10 epochs
        
        storage = LightcurveStorage(storage_path)
        storage.create_for_per_epoch_write(
            source_catalog=bulk_source_catalog_1k,
            epoch_keys=epoch_keys,
        )
        
        # Populate with some data
        for epoch_key in epoch_keys[:5]:  # Populate 5 epochs
            n_objs = bulk_source_catalog_1k.n_sources
            data = {
                "mag": np.random.normal(18.0, 0.5, n_objs),
                "mag_err": np.random.normal(0.05, 0.01, n_objs),
            }
            
            storage.add_epoch(
                epoch_key=epoch_key,
                source_ids=bulk_source_catalog_1k.source_ids.tolist(),
                lightcurve_data=[data],
                measurement_keys=["auto"],
            )
        
        storage.save()
        return storage
    
    @pytest.mark.benchmark(group="bulk_extraction")
    def test_extract_all_coords_along_object_dim_values(self, benchmark, small_storage):
        """Current pattern: Extract all 1D coords along object dim using .values.
        
        This is what the dash app does - extract ALL 1D variables at once.
        Uses .values which forces immediate computation.
        """
        def extract_with_values():
            ds = small_storage.dataset
            dim_name = 'object'
            
            # Extract coordinate values
            coord_values = ds.coords[dim_name].values
            data = {dim_name: coord_values}
            
            # Extract all 1D coordinates along this dimension
            for coord_name in ds.coords:
                coord = ds.coords[coord_name]
                if dim_name in coord.dims and len(coord.dims) == 1:
                    data[coord_name] = coord.values  # Forces computation!
            
            # Build DataFrame
            return pd.DataFrame(data)
        
        df = benchmark(extract_with_values)
        assert len(df) == 1000
        assert 'object' in df.columns
        assert 'ra' in df.columns  # Should have coordinate data
    
    @pytest.mark.benchmark(group="bulk_extraction")
    def test_extract_all_coords_along_object_dim_compute(self, benchmark, small_storage):
        """Optimized pattern: Extract using .compute() instead of .values.
        
        .compute() is the recommended way for dask arrays - respects lazy evaluation.
        """
        def extract_with_compute():
            ds = small_storage.dataset
            dim_name = 'object'
            
            # Extract coordinate values using .compute()
            coord_values = ds.coords[dim_name].compute()
            data = {dim_name: coord_values.values}  # Get numpy array from result
            
            # Extract all 1D coordinates along this dimension
            for coord_name in ds.coords:
                coord = ds.coords[coord_name]
                if dim_name in coord.dims and len(coord.dims) == 1:
                    computed = coord.compute()  # Lazy evaluation
                    data[coord_name] = computed.values
            
            # Build DataFrame
            return pd.DataFrame(data)
        
        df = benchmark(extract_with_compute)
        assert len(df) == 1000
        assert 'object' in df.columns
        assert 'ra' in df.columns
    
    @pytest.mark.benchmark(group="bulk_extraction")
    def test_extract_selected_coords_only(self, benchmark, small_storage):
        """Memory-efficient pattern: Extract only needed coords.
        
        Instead of extracting ALL 1D variables, only extract what we need.
        """
        def extract_selected_coords():
            ds = small_storage.dataset
            dim_name = 'object'
            
            # Define which coords we actually need
            needed_coords = ['object', 'ra', 'dec', 'x_image', 'y_image']
            
            # Extract only needed coords
            data = {}
            for coord_name in needed_coords:
                if coord_name in ds.coords:
                    coord = ds.coords[coord_name]
                    computed = coord.compute()
                    data[coord_name] = computed.values
                elif coord_name == dim_name:
                    coord_values = ds.coords[dim_name].compute()
                    data[coord_name] = coord_values.values
            
            # Build DataFrame
            return pd.DataFrame(data)
        
        df = benchmark(extract_selected_coords)
        assert len(df) == 1000
        assert 'object' in df.columns
        assert 'ra' in df.columns
    
    @pytest.mark.benchmark(group="bulk_extraction")
    def test_extract_with_to_dataframe_method(self, benchmark, small_storage):
        """Use xarray's built-in to_dataframe() method.
        
        xarray has a built-in method to convert to DataFrame - might be optimized.
        """
        def extract_with_to_dataframe():
            ds = small_storage.dataset
            
            # Select only 1D coordinates along object dimension
            coords_to_extract = []
            for coord_name in ds.coords:
                coord = ds.coords[coord_name]
                if 'object' in coord.dims and len(coord.dims) == 1:
                    coords_to_extract.append(coord_name)
            
            # Create a dataset with just these coords
            coord_ds = ds[coords_to_extract] if coords_to_extract else xr.Dataset()
            
            # Use built-in to_dataframe()
            df = coord_ds.to_dataframe()
            return df.reset_index()
        
        df = benchmark(extract_with_to_dataframe)
        assert len(df) == 1000
    
    @pytest.mark.benchmark(group="bulk_extraction")
    def test_extract_coords_batched(self, benchmark, small_storage):
        """Batched extraction: Process coordinates in smaller batches.
        
        Instead of extracting everything at once, process in batches.
        """
        def extract_batched():
            ds = small_storage.dataset
            dim_name = 'object'
            batch_size = 250  # Process 250 objects at a time
            
            # Get total number of objects
            n_objects = ds.sizes[dim_name]
            
            # Extract in batches
            dfs = []
            for start in range(0, n_objects, batch_size):
                end = min(start + batch_size, n_objects)
                
                # Slice the dataset for this batch
                batch_ds = ds.isel({dim_name: slice(start, end)})
                
                # Extract coords for this batch
                batch_data = {}
                for coord_name in batch_ds.coords:
                    coord = batch_ds.coords[coord_name]
                    if dim_name in coord.dims and len(coord.dims) == 1:
                        computed = coord.compute()
                        batch_data[coord_name] = computed.values
                
                dfs.append(pd.DataFrame(batch_data))
            
            # Concatenate all batches
            return pd.concat(dfs, ignore_index=True)
        
        df = benchmark(extract_batched)
        assert len(df) == 1000


class TestBulkExtractionWithRechunking:
    """Test how rechunking affects bulk extraction performance."""
    
    @pytest.fixture
    def medium_storage_write_optimized(self, test_catalog_10K, tmp_path):
        """Storage with write-optimized chunking (default)."""
        storage_path = tmp_path / "bulk_rechunk_write.zarr"
        catalog = test_catalog_10K.head(5000)  # 5K objects
        epoch_keys = [f"r{58520832 + i * 256}" for i in range(20)]  # 20 epochs
        
        storage = LightcurveStorage.create(
            storage_path=storage_path,
            catalog=catalog,
            epoch_keys=epoch_keys,
            measurement_keys=["auto"],
            value_keys=["mag", "mag_err"],
            force_overwrite=True,
        )
        
        # Populate some data
        for epoch_key in epoch_keys[:10]:
            n_objs = len(catalog)
            data = {
                "mag": np.random.normal(18.0, 0.5, n_objs),
                "mag_err": np.random.normal(0.05, 0.01, n_objs),
            }
            storage.add_epoch_data(
                epoch_key=epoch_key,
                measurement_key="auto",
                object_ids=catalog.index.tolist(),
                data=data,
            )
        
        storage.save()
        return storage
    
    @pytest.fixture
    def medium_storage_read_optimized(self, test_catalog_10K, tmp_path):
        """Storage with read-optimized chunking."""
        storage_path = tmp_path / "bulk_rechunk_read.zarr"
        catalog = test_catalog_10K.head(5000)  # 5K objects
        epoch_keys = [f"r{58520832 + i * 256}" for i in range(20)]  # 20 epochs
        
        storage = LightcurveStorage.create(
            storage_path=storage_path,
            catalog=catalog,
            epoch_keys=epoch_keys,
            measurement_keys=["auto"],
            value_keys=["mag", "mag_err"],
            force_overwrite=True,
        )
        
        # Populate some data
        for epoch_key in epoch_keys[:10]:
            n_objs = len(catalog)
            data = {
                "mag": np.random.normal(18.0, 0.5, n_objs),
                "mag_err": np.random.normal(0.05, 0.01, n_objs),
            }
            storage.add_epoch_data(
                epoch_key=epoch_key,
                measurement_key="auto",
                object_ids=catalog.index.tolist(),
                data=data,
            )
        
        # Rechunk for reading all objects
        storage.rechunk_for_per_object_read(chunk_size=5000)
        storage.save()
        return storage
    
    @pytest.mark.benchmark(group="bulk_rechunking")
    def test_bulk_extract_write_optimized(self, benchmark, medium_storage_write_optimized):
        """Extract all coords from write-optimized storage."""
        def extract():
            ds = medium_storage_write_optimized.dataset
            dim_name = 'object'
            
            data = {}
            for coord_name in ds.coords:
                coord = ds.coords[coord_name]
                if dim_name in coord.dims and len(coord.dims) == 1:
                    computed = coord.compute()
                    data[coord_name] = computed.values
            
            return pd.DataFrame(data)
        
        df = benchmark(extract)
        assert len(df) == 5000
    
    @pytest.mark.benchmark(group="bulk_rechunking")
    def test_bulk_extract_read_optimized(self, benchmark, medium_storage_read_optimized):
        """Extract all coords from read-optimized storage."""
        def extract():
            ds = medium_storage_read_optimized.dataset
            dim_name = 'object'
            
            data = {}
            for coord_name in ds.coords:
                coord = ds.coords[coord_name]
                if dim_name in coord.dims and len(coord.dims) == 1:
                    computed = coord.compute()
                    data[coord_name] = computed.values
            
            return pd.DataFrame(data)
        
        df = benchmark(extract)
        assert len(df) == 5000


class TestEpochDimensionExtraction:
    """Test extraction along epoch dimension (for epoch table)."""
    
    @pytest.fixture
    def storage_with_many_epochs(self, test_catalog_10K, tmp_path):
        """Storage with fewer objects but many epochs."""
        storage_path = tmp_path / "bulk_epochs.zarr"
        catalog = test_catalog_10K.head(500)  # Only 500 objects
        epoch_keys = [f"r{58520832 + i * 256}" for i in range(50)]  # 50 epochs
        
        storage = LightcurveStorage.create(
            storage_path=storage_path,
            catalog=catalog,
            epoch_keys=epoch_keys,
            measurement_keys=["auto"],
            value_keys=["mag", "mag_err"],
            force_overwrite=True,
        )
        
        # Populate all epochs
        for epoch_key in epoch_keys:
            n_objs = len(catalog)
            data = {
                "mag": np.random.normal(18.0, 0.5, n_objs),
                "mag_err": np.random.normal(0.05, 0.01, n_objs),
            }
            storage.add_epoch_data(
                epoch_key=epoch_key,
                measurement_key="auto",
                object_ids=catalog.index.tolist(),
                data=data,
            )
        
        storage.save()
        return storage
    
    @pytest.mark.benchmark(group="epoch_extraction")
    def test_extract_all_epoch_coords(self, benchmark, storage_with_many_epochs):
        """Extract all 1D variables along epoch dimension."""
        def extract():
            ds = storage_with_many_epochs.dataset
            dim_name = 'epoch'
            
            # Extract coordinate values
            coord_values = ds.coords[dim_name].compute()
            data = {dim_name: coord_values.values}
            
            # Extract all 1D variables along epoch dimension
            for var_name in ds.coords:
                var = ds.coords[var_name]
                if dim_name in var.dims and len(var.dims) == 1:
                    computed = var.compute()
                    data[var_name] = computed.values
            
            # Also check data_vars
            for var_name in ds.data_vars:
                var = ds[var_name]
                if dim_name in var.dims and len(var.dims) == 1:
                    computed = var.compute()
                    data[var_name] = computed.values
            
            return pd.DataFrame(data)
        
        df = benchmark(extract)
        assert len(df) == 50


class TestMemoryEfficientPatterns:
    """Test memory-efficient extraction patterns."""
    
    @pytest.fixture
    def large_storage(self, test_catalog_50K, tmp_path):
        """Larger storage for memory testing."""
        storage_path = tmp_path / "bulk_memory.zarr"
        catalog = test_catalog_50K.head(10000)  # 10K objects
        epoch_keys = [f"r{58520832 + i * 256}" for i in range(30)]  # 30 epochs
        
        storage = LightcurveStorage.create(
            storage_path=storage_path,
            catalog=catalog,
            epoch_keys=epoch_keys,
            measurement_keys=["auto"],
            value_keys=["mag", "mag_err"],
            force_overwrite=True,
        )
        
        # Populate some data
        for epoch_key in epoch_keys[:15]:
            n_objs = len(catalog)
            data = {
                "mag": np.random.normal(18.0, 0.5, n_objs),
                "mag_err": np.random.normal(0.05, 0.01, n_objs),
            }
            storage.add_epoch_data(
                epoch_key=epoch_key,
                measurement_key="auto",
                object_ids=catalog.index.tolist(),
                data=data,
            )
        
        storage.save()
        return storage
    
    @pytest.mark.benchmark(group="memory_efficient")
    def test_lazy_dict_build_then_compute(self, benchmark, large_storage):
        """Build dict of lazy operations, then compute all at once.
        
        This might be more efficient - build computation graph first,
        then compute everything in one go.
        """
        def extract():
            ds = large_storage.dataset
            dim_name = 'object'
            
            # Build dict of lazy operations (don't call .compute() yet)
            lazy_data = {}
            for coord_name in ds.coords:
                coord = ds.coords[coord_name]
                if dim_name in coord.dims and len(coord.dims) == 1:
                    lazy_data[coord_name] = coord  # Keep as lazy
            
            # Compute all at once using xarray's compute
            computed_data = xr.Dataset(lazy_data).compute()
            
            # Extract values to dict
            data = {k: v.values for k, v in computed_data.items()}
            
            return pd.DataFrame(data)
        
        df = benchmark(extract)
        assert len(df) == 10000
    
    @pytest.mark.benchmark(group="memory_efficient")
    def test_extract_coord_keys_only_first(self, benchmark, large_storage):
        """Extract only coordinate values first (cheapest operation).
        
        Sometimes we only need to know which objects/epochs exist,
        not all their metadata.
        """
        def extract():
            ds = large_storage.dataset
            dim_name = 'object'
            
            # Only extract the coordinate itself (object keys)
            coord_values = ds.coords[dim_name].compute()
            
            # Build minimal DataFrame
            return pd.DataFrame({dim_name: coord_values.values})
        
        df = benchmark(extract)
        assert len(df) == 10000
        assert list(df.columns) == ['object']
