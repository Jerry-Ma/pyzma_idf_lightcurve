"""Simplified benchmarks for xarray data extraction patterns.

Focus: Compare .values vs .compute() vs other patterns for extracting data from xarray.

This doesn't use the full LightcurveStorage - instead creates minimal xarray datasets
to directly test extraction patterns.

Run:
    uv run pytest tests/test_xarray_extraction_benchmarks.py --benchmark-only -v
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import dask.array as da


class TestExtractionMethods:
    """Compare different extraction methods on simple xarray datasets."""
    
    @pytest.fixture
    def simple_xr_dataset_1k(self):
        """Create a simple xarray dataset with 1K objects, mimicking our structure."""
        n_objects = 1000
        n_epochs = 10
        
        # Object keys and coordinates (similar to our storage)
        object_keys = [f"I{i:06d}" for i in range(n_objects)]
        ra = np.random.uniform(17.5, 17.7, n_objects)
        dec = np.random.uniform(-29.9, -29.7, n_objects)
        x_image = np.random.uniform(1, 4000, n_objects)
        y_image = np.random.uniform(1, 4000, n_objects)
        
        # Epoch keys
        epoch_keys = [f"r{58520832 + i * 256}" for i in range(n_epochs)]
        
        # Create dataset with dask arrays (lazy)
        ds = xr.Dataset(
            coords={
                'object': object_keys,
                'epoch': epoch_keys,
                'ra': ('object', da.from_array(ra, chunks=250)),
                'dec': ('object', da.from_array(dec, chunks=250)),
                'x_image': ('object', da.from_array(x_image, chunks=250)),
                'y_image': ('object', da.from_array(y_image, chunks=250)),
            }
        )
        
        return ds
    
    @pytest.fixture
    def simple_xr_dataset_10k(self):
        """Larger dataset with 10K objects."""
        n_objects = 10000
        n_epochs = 30
        
        object_keys = [f"I{i:06d}" for i in range(n_objects)]
        ra = np.random.uniform(17.5, 17.7, n_objects)
        dec = np.random.uniform(-29.9, -29.7, n_objects)
        x_image = np.random.uniform(1, 4000, n_objects)
        y_image = np.random.uniform(1, 4000, n_objects)
        
        epoch_keys = [f"r{58520832 + i * 256}" for i in range(n_epochs)]
        
        # Use smaller chunks for 10K objects
        ds = xr.Dataset(
            coords={
                'object': object_keys,
                'epoch': epoch_keys,
                'ra': ('object', da.from_array(ra, chunks=1000)),
                'dec': ('object', da.from_array(dec, chunks=1000)),
                'x_image': ('object', da.from_array(x_image, chunks=1000)),
                'y_image': ('object', da.from_array(y_image, chunks=1000)),
            }
        )
        
        return ds
    
    @pytest.mark.benchmark(group="extraction_1k")
    def test_extract_using_values_1k(self, benchmark, simple_xr_dataset_1k):
        """Current pattern: Use .values to extract - FORCES COMPUTATION immediately."""
        def extract():
            ds = simple_xr_dataset_1k
            dim_name = 'object'
            
            # Extract all 1D coords using .values
            data = {}
            for coord_name in ds.coords:
                coord = ds.coords[coord_name]
                if dim_name in coord.dims and len(coord.dims) == 1:
                    data[coord_name] = coord.values  # Forces immediate computation!
            
            return pd.DataFrame(data)
        
        df = benchmark(extract)
        assert len(df) == 1000
    
    @pytest.mark.benchmark(group="extraction_1k")
    def test_extract_using_compute_1k(self, benchmark, simple_xr_dataset_1k):
        """Recommended pattern: Use .compute() - respects lazy evaluation."""
        def extract():
            ds = simple_xr_dataset_1k
            dim_name = 'object'
            
            # Extract using .compute()
            data = {}
            for coord_name in ds.coords:
                coord = ds.coords[coord_name]
                if dim_name in coord.dims and len(coord.dims) == 1:
                    computed = coord.compute()  # Lazy evaluation
                    data[coord_name] = computed.values
            
            return pd.DataFrame(data)
        
        df = benchmark(extract)
        assert len(df) == 1000
    
    @pytest.mark.benchmark(group="extraction_1k")
    def test_extract_lazy_then_compute_all_1k(self, benchmark, simple_xr_dataset_1k):
        """Build lazy dict, then compute all at once (might be more efficient)."""
        def extract():
            ds = simple_xr_dataset_1k
            dim_name = 'object'
            
            # Collect lazy operations
            lazy_data = {}
            for coord_name in ds.coords:
                coord = ds.coords[coord_name]
                if dim_name in coord.dims and len(coord.dims) == 1:
                    lazy_data[coord_name] = coord
            
            # Compute all at once
            computed_ds = xr.Dataset(lazy_data).compute()
            
            # Extract to pandas
            data = {k: v.values for k, v in computed_ds.items()}
            return pd.DataFrame(data)
        
        df = benchmark(extract)
        assert len(df) == 1000
    
    @pytest.mark.benchmark(group="extraction_1k")
    def test_extract_using_to_dataframe_1k(self, benchmark, simple_xr_dataset_1k):
        """Use xarray's built-in to_dataframe() method."""
        def extract():
            ds = simple_xr_dataset_1k
            dim_name = 'object'
            
            # Select only 1D coords along object dimension
            coords_list = []
            for coord_name in ds.coords:
                coord = ds.coords[coord_name]
                if dim_name in coord.dims and len(coord.dims) == 1:
                    coords_list.append(coord_name)
            
            # Create dataset with just these coords and use to_dataframe()
            if coords_list:
                coord_ds = ds[coords_list]
                df = coord_ds.to_dataframe()
                return df.reset_index()
            else:
                return pd.DataFrame()
        
        df = benchmark(extract)
        assert len(df) == 1000
    
    @pytest.mark.benchmark(group="extraction_10k")
    def test_extract_using_values_10k(self, benchmark, simple_xr_dataset_10k):
        """Pattern: .values on larger dataset."""
        def extract():
            ds = simple_xr_dataset_10k
            dim_name = 'object'
            
            data = {}
            for coord_name in ds.coords:
                coord = ds.coords[coord_name]
                if dim_name in coord.dims and len(coord.dims) == 1:
                    data[coord_name] = coord.values
            
            return pd.DataFrame(data)
        
        df = benchmark(extract)
        assert len(df) == 10000
    
    @pytest.mark.benchmark(group="extraction_10k")
    def test_extract_using_compute_10k(self, benchmark, simple_xr_dataset_10k):
        """Pattern: .compute() on larger dataset."""
        def extract():
            ds = simple_xr_dataset_10k
            dim_name = 'object'
            
            data = {}
            for coord_name in ds.coords:
                coord = ds.coords[coord_name]
                if dim_name in coord.dims and len(coord.dims) == 1:
                    computed = coord.compute()
                    data[coord_name] = computed.values
            
            return pd.DataFrame(data)
        
        df = benchmark(extract)
        assert len(df) == 10000
    
    @pytest.mark.benchmark(group="extraction_10k")
    def test_extract_lazy_then_compute_all_10k(self, benchmark, simple_xr_dataset_10k):
        """Pattern: Build lazy dict, compute all at once (larger dataset)."""
        def extract():
            ds = simple_xr_dataset_10k
            dim_name = 'object'
            
            lazy_data = {}
            for coord_name in ds.coords:
                coord = ds.coords[coord_name]
                if dim_name in coord.dims and len(coord.dims) == 1:
                    lazy_data[coord_name] = coord
            
            computed_ds = xr.Dataset(lazy_data).compute()
            data = {k: v.values for k, v in computed_ds.items()}
            return pd.DataFrame(data)
        
        df = benchmark(extract)
        assert len(df) == 10000


class TestChunkingEffects:
    """Test how chunking affects extraction performance."""
    
    @pytest.mark.benchmark(group="chunking")
    def test_small_chunks_250(self, benchmark):
        """Extract with small chunks (250 objects/chunk)."""
        n_objects = 5000
        object_keys = [f"I{i:06d}" for i in range(n_objects)]
        ra = np.random.uniform(17.5, 17.7, n_objects)
        
        ds = xr.Dataset(coords={
            'object': object_keys,
            'ra': ('object', da.from_array(ra, chunks=250)),  # Small chunks
        })
        
        def extract():
            data = {'ra': ds.coords['ra'].compute().values}
            return pd.DataFrame(data)
        
        df = benchmark(extract)
        assert len(df) == 5000
    
    @pytest.mark.benchmark(group="chunking")
    def test_medium_chunks_1000(self, benchmark):
        """Extract with medium chunks (1000 objects/chunk)."""
        n_objects = 5000
        object_keys = [f"I{i:06d}" for i in range(n_objects)]
        ra = np.random.uniform(17.5, 17.7, n_objects)
        
        ds = xr.Dataset(coords={
            'object': object_keys,
            'ra': ('object', da.from_array(ra, chunks=1000)),  # Medium chunks
        })
        
        def extract():
            data = {'ra': ds.coords['ra'].compute().values}
            return pd.DataFrame(data)
        
        df = benchmark(extract)
        assert len(df) == 5000
    
    @pytest.mark.benchmark(group="chunking")
    def test_single_chunk(self, benchmark):
        """Extract with single chunk (all data)."""
        n_objects = 5000
        object_keys = [f"I{i:06d}" for i in range(n_objects)]
        ra = np.random.uniform(17.5, 17.7, n_objects)
        
        ds = xr.Dataset(coords={
            'object': object_keys,
            'ra': ('object', da.from_array(ra, chunks=-1)),  # Single chunk
        })
        
        def extract():
            data = {'ra': ds.coords['ra'].compute().values}
            return pd.DataFrame(data)
        
        df = benchmark(extract)
        assert len(df) == 5000


class TestBatchedExtraction:
    """Test batched extraction for memory efficiency."""
    
    @pytest.fixture
    def large_xr_dataset(self):
        """Large dataset to test batching."""
        n_objects = 10000
        object_keys = [f"I{i:06d}" for i in range(n_objects)]
        ra = np.random.uniform(17.5, 17.7, n_objects)
        dec = np.random.uniform(-29.9, -29.7, n_objects)
        x_image = np.random.uniform(1, 4000, n_objects)
        y_image = np.random.uniform(1, 4000, n_objects)
        
        ds = xr.Dataset(coords={
            'object': object_keys,
            'ra': ('object', da.from_array(ra, chunks=1000)),
            'dec': ('object', da.from_array(dec, chunks=1000)),
            'x_image': ('object', da.from_array(x_image, chunks=1000)),
            'y_image': ('object', da.from_array(y_image, chunks=1000)),
        })
        
        return ds
    
    @pytest.mark.benchmark(group="batching")
    def test_extract_all_at_once(self, benchmark, large_xr_dataset):
        """Extract all 10K objects at once."""
        def extract():
            ds = large_xr_dataset
            dim_name = 'object'
            
            data = {}
            for coord_name in ds.coords:
                coord = ds.coords[coord_name]
                if dim_name in coord.dims and len(coord.dims) == 1:
                    data[coord_name] = coord.compute().values
            
            return pd.DataFrame(data)
        
        df = benchmark(extract)
        assert len(df) == 10000
    
    @pytest.mark.benchmark(group="batching")
    def test_extract_in_batches_2k(self, benchmark, large_xr_dataset):
        """Extract in batches of 2K objects."""
        def extract():
            ds = large_xr_dataset
            dim_name = 'object'
            batch_size = 2000
            n_objects = ds.sizes[dim_name]
            
            dfs = []
            for start in range(0, n_objects, batch_size):
                end = min(start + batch_size, n_objects)
                batch_ds = ds.isel({dim_name: slice(start, end)})
                
                batch_data = {}
                for coord_name in batch_ds.coords:
                    coord = batch_ds.coords[coord_name]
                    if dim_name in coord.dims and len(coord.dims) == 1:
                        batch_data[coord_name] = coord.compute().values
                
                dfs.append(pd.DataFrame(batch_data))
            
            return pd.concat(dfs, ignore_index=True)
        
        df = benchmark(extract)
        assert len(df) == 10000
    
    @pytest.mark.benchmark(group="batching")
    def test_extract_selective_coords(self, benchmark, large_xr_dataset):
        """Extract only needed coordinates (not all)."""
        def extract():
            ds = large_xr_dataset
            
            # Only extract what we need
            needed = ['object', 'ra', 'dec']
            data = {}
            for coord_name in needed:
                if coord_name in ds.coords:
                    coord = ds.coords[coord_name]
                    data[coord_name] = coord.compute().values
            
            return pd.DataFrame(data)
        
        df = benchmark(extract)
        assert len(df) == 10000
