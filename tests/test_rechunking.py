"""Test rechunking workflow from epoch-chunked to object-chunked storage.

This module tests the concept of:
1. Creating a LightcurveStorage with per-epoch chunking (efficient for writing)
2. Populating data epoch-by-epoch (fast, sequential writes)
3. Rechunking to object-chunked storage (efficient for reading lightcurves)

Using the rechunker library allows us to optimize for both write and read patterns.
"""

import pytest
import numpy as np
import zarr
import xarray as xr
from pathlib import Path
import shutil

# Import rechunker
try:
    from rechunker import rechunk
    RECHUNKER_AVAILABLE = True
except ImportError:
    RECHUNKER_AVAILABLE = False
    rechunk = None

from pyzma_idf_lightcurve.lightcurve.datamodel import LightcurveStorage
from pyzma_idf_lightcurve.lightcurve.catalog import SourceCatalog


@pytest.mark.skipif(not RECHUNKER_AVAILABLE, reason="rechunker not installed")
class TestRechunkingWorkflow:
    """Test rechunking from epoch-chunked to object-chunked storage."""
    
    @pytest.fixture
    def test_catalog(self, tmp_path):
        """Create a test catalog with synthetic data."""
        n_objects = 100
        n_epochs = 10
        
        # Create synthetic catalog data
        objects_data = {
            'object_id': [f'obj_{i:04d}' for i in range(n_objects)],
            'ra': np.random.uniform(0, 360, n_objects),
            'dec': np.random.uniform(-90, 90, n_objects),
        }
        
        epochs_data = {
            'epoch_id': [f'epoch_{i:04d}' for i in range(n_epochs)],
            'mjd': 50000.0 + np.arange(n_epochs) * 10.0,
        }
        
        # Measurements: random magnitudes for each object-epoch pair
        measurements = []
        for epoch_idx in range(n_epochs):
            for obj_idx in range(n_objects):
                # 80% have valid measurements
                if np.random.random() < 0.8:
                    mag = 15.0 + np.random.normal(0, 0.5)
                    magerr = 0.01 + np.random.exponential(0.05)
                    measurements.append({
                        'object_id': objects_data['object_id'][obj_idx],
                        'epoch_id': epochs_data['epoch_id'][epoch_idx],
                        'MAG_AUTO': mag,
                        'MAGERR_AUTO': magerr,
                        'MAG_ISO': mag + np.random.normal(0, 0.1),
                        'MAGERR_ISO': magerr * 1.2,
                    })
        
        # Create catalog
        catalog = SourceCatalog.create_catalog(
            objects=objects_data,
            epochs=epochs_data,
            measurements=measurements,
            measurement_columns=[
                ('MAG_AUTO', 'MAGERR_AUTO'),
                ('MAG_ISO', 'MAGERR_ISO'),
            ],
        )
        
        return catalog
    
    @pytest.fixture
    def epoch_chunked_storage(self, tmp_path, test_catalog):
        """Create epoch-chunked storage (optimized for writing)."""
        storage_path = tmp_path / "epoch_chunked.zarr"
        
        # Create storage with EPOCH chunking
        # This is optimal for populating one epoch at a time
        storage = LightcurveStorage.create_storage(
            storage_path=storage_path,
            object_keys=test_catalog.object_keys,
            epoch_keys=test_catalog.epoch_keys,
            measurement_keys=test_catalog.measurement_keys,
            chunks={
                'object': 1000,     # Large chunk - entire object dimension
                'epoch': 1,          # Small chunk - one epoch at a time
                'measurement': 2,
                'value': 2,
            },
        )
        
        return storage
    
    def test_rechunking_concept(self, tmp_path, test_catalog, epoch_chunked_storage):
        """Test the complete workflow: create epoch-chunked, populate, rechunk to object-chunked."""
        
        # Step 1: Populate epoch-chunked storage (fast sequential writes)
        print("\n=== Step 1: Populate epoch-chunked storage ===")
        for epoch_key in test_catalog.epoch_keys[:5]:  # Use 5 epochs for faster test
            n_updated = epoch_chunked_storage.populate_epoch_from_catalog_v2(
                epoch_key=epoch_key,
                source_catalog=test_catalog,
                measurement_keys=test_catalog.measurement_keys,
            )
            print(f"  Populated {epoch_key}: {n_updated} measurements")
        
        # Verify data was written
        epoch_chunked_storage.load_storage(consolidated=False)
        sample_lc = epoch_chunked_storage.get_lightcurve(test_catalog.object_keys[0])
        n_valid = int(np.sum(~np.isnan(sample_lc['default-auto'].sel(value='value').values)))
        print(f"  Sample lightcurve has {n_valid} valid measurements")
        assert n_valid > 0, "No data written to epoch-chunked storage"
        
        # Check chunking
        print(f"  Epoch-chunked storage chunks: {epoch_chunked_storage.lightcurves.chunks}")
        
        # Step 2: Rechunk to object-chunked storage
        print("\n=== Step 2: Rechunk to object-chunked storage ===")
        
        source_zarr = zarr.open(str(epoch_chunked_storage.storage_path), mode='r')
        target_path = tmp_path / "object_chunked.zarr"
        temp_path = tmp_path / "temp_rechunk.zarr"
        
        # Define target chunks (optimized for reading individual lightcurves)
        target_chunks = {
            'object': 1,         # One object per chunk - optimal for lightcurve access
            'epoch': 100,        # All epochs together
            'measurement': 2,
            'value': 2,
        }
        
        print(f"  Source chunks: {source_zarr['lightcurves'].chunks}")
        print(f"  Target chunks: {target_chunks}")
        
        # Execute rechunking
        rechunked = rechunk(
            source_zarr['lightcurves'],
            target_chunks=target_chunks,
            target_store=str(target_path / 'lightcurves'),
            temp_store=str(temp_path / 'lightcurves'),
            max_mem='256MB',
        )
        
        print(f"  Rechunking plan created:")
        print(f"    Source: {rechunked}")
        
        result = rechunked.execute()
        print(f"  Rechunking executed: {result.chunks}")
        
        # Step 3: Load object-chunked storage and verify
        print("\n=== Step 3: Verify object-chunked storage ===")
        
        # Copy metadata to target
        shutil.copy(
            epoch_chunked_storage.storage_path / '.zattrs',
            target_path / '.zattrs'
        )
        shutil.copy(
            epoch_chunked_storage.storage_path / '.zgroup',
            target_path / '.zgroup'
        )
        shutil.copy(
            epoch_chunked_storage.storage_path / 'lightcurves' / '.zattrs',
            target_path / 'lightcurves' / '.zattrs'
        )
        
        # Load as LightcurveStorage
        object_chunked = LightcurveStorage(storage_path=target_path)
        object_chunked.load_storage(consolidated=False)
        
        print(f"  Object-chunked storage chunks: {object_chunked.lightcurves.chunks}")
        
        # Verify data integrity
        sample_lc_rechunked = object_chunked.get_lightcurve(test_catalog.object_keys[0])
        n_valid_rechunked = int(np.sum(~np.isnan(
            sample_lc_rechunked['default-auto'].sel(value='value').values
        )))
        
        print(f"  Sample lightcurve has {n_valid_rechunked} valid measurements (after rechunk)")
        
        # Compare data
        np.testing.assert_array_equal(
            sample_lc['default-auto'].values,
            sample_lc_rechunked['default-auto'].values,
            err_msg="Data changed after rechunking!"
        )
        
        print("  ✅ Data integrity verified - rechunking successful!")
        
        # Step 4: Compare performance characteristics
        print("\n=== Step 4: Performance comparison ===")
        
        # Epoch-chunked: Good for writing entire epochs
        print(f"  Epoch-chunked chunks: {epoch_chunked_storage.lightcurves.chunks}")
        print(f"    - Optimal for: Writing all objects in one epoch")
        print(f"    - Poor for: Reading one object's full lightcurve (many chunks)")
        
        # Object-chunked: Good for reading individual lightcurves
        print(f"  Object-chunked chunks: {object_chunked.lightcurves.chunks}")
        print(f"    - Optimal for: Reading one object's full lightcurve (1 chunk)")
        print(f"    - Poor for: Writing all objects in one epoch (many chunks)")
    
    def test_rechunk_with_different_strategies(self, tmp_path, test_catalog):
        """Test different rechunking strategies."""
        
        # Create a small test dataset
        storage_path = tmp_path / "source.zarr"
        storage = LightcurveStorage.create_storage(
            storage_path=storage_path,
            object_keys=test_catalog.object_keys[:20],  # Small dataset
            epoch_keys=test_catalog.epoch_keys[:5],
            measurement_keys=test_catalog.measurement_keys,
            chunks={'object': 20, 'epoch': 1, 'measurement': 2, 'value': 2},
        )
        
        # Populate
        for epoch_key in test_catalog.epoch_keys[:5]:
            storage.populate_epoch_from_catalog_v2(
                epoch_key=epoch_key,
                source_catalog=test_catalog,
                measurement_keys=test_catalog.measurement_keys,
            )
        
        storage.load_storage(consolidated=False)
        print(f"\nSource chunks: {storage.lightcurves.chunks}")
        
        # Test different target chunking strategies
        strategies = {
            'object_optimized': {'object': 1, 'epoch': 5, 'measurement': 2, 'value': 2},
            'epoch_optimized': {'object': 20, 'epoch': 1, 'measurement': 2, 'value': 2},
            'balanced': {'object': 5, 'epoch': 5, 'measurement': 2, 'value': 2},
        }
        
        for strategy_name, target_chunks in strategies.items():
            print(f"\n=== Testing {strategy_name} strategy ===")
            print(f"  Target chunks: {target_chunks}")
            
            target_path = tmp_path / f"{strategy_name}.zarr"
            temp_path = tmp_path / f"temp_{strategy_name}.zarr"
            
            source_zarr = zarr.open(str(storage.storage_path), mode='r')
            
            rechunked = rechunk(
                source_zarr['lightcurves'],
                target_chunks=target_chunks,
                target_store=str(target_path / 'lightcurves'),
                temp_store=str(temp_path / 'lightcurves'),
                max_mem='128MB',
            )
            
            result = rechunked.execute()
            print(f"  ✅ Rechunking completed: {result.chunks}")
            
            # Verify chunk size
            assert result.chunks == tuple(
                (target_chunks[dim],) if dim in target_chunks else (size,)
                for dim, size in zip(['object', 'epoch', 'measurement', 'value'], result.shape)
            ), f"Chunks don't match for {strategy_name}"


@pytest.mark.skipif(RECHUNKER_AVAILABLE, reason="Test rechunker import failure handling")
def test_rechunker_not_available():
    """Test graceful handling when rechunker is not available."""
    with pytest.raises(ImportError):
        from rechunker import rechunk
