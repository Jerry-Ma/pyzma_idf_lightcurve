"""
Test concurrent writing to zarr arrays with per-epoch chunking.

ARCHITECTURE:
1. Storage is created with epoch_chunk_size=1 (each epoch is a separate zarr chunk/file)
2. populate_epoch writes the ENTIRE (n_obj, n_mea, n_val, 1) array for a given epoch
3. Different epochs map to different zarr chunks/files

CONCURRENT WRITING STRATEGY:
- Safe pattern: Each worker handles DIFFERENT epoch_keys
- Workers write to different zarr chunks (files) - no conflicts
- All objects are written for each epoch by a single worker

This architecture enables true parallel population across epochs:
- Worker 1 populates epoch_0000, epoch_0004, epoch_0008, ...
- Worker 2 populates epoch_0001, epoch_0005, epoch_0009, ...
- Worker 3 populates epoch_0002, epoch_0006, epoch_0010, ...
- Worker 4 populates epoch_0003, epoch_0007, epoch_0011, ...

Each worker writes complete epoch slices to separate zarr chunks, ensuring
thread-safe concurrent population.
"""

import concurrent.futures

import numpy as np
import pytest
from astropy.table import Table

from pyzma_idf_lightcurve.lightcurve.catalog import SourceCatalog
from pyzma_idf_lightcurve.lightcurve.datamodel import LightcurveStorage


class TestConcurrentWriting:
    """
    Test concurrent writes to zarr lightcurve storage with per-epoch chunking.
    
    Storage is configured with epoch_chunk_size=1, so each epoch is a separate
    zarr chunk/file. This enables safe concurrent population across epochs.
    """
    
    @pytest.fixture
    def base_catalog(self):
        """Create a catalog with multiple objects and measurements."""
        n_objects = 500
        
        table = Table()
        table['id'] = np.arange(1, n_objects + 1)
        table['ra'] = np.linspace(150.0, 151.0, n_objects)
        table['dec'] = np.linspace(2.0, 3.0, n_objects)
        table['x'] = np.linspace(100.0, 1000.0, n_objects)
        table['y'] = np.linspace(100.0, 1000.0, n_objects)
        
        # Add measurements with correct value keys (mag, mag_err)
        for meas_name in ['AUTO', 'ISO']:
            table[f'mag_{meas_name.lower()}'] = 20.0 + np.random.uniform(0, 5, n_objects)
            table[f'magerr_{meas_name.lower()}'] = 0.05 + np.random.uniform(0, 0.1, n_objects)
        
        return SourceCatalog(table)
    
    @pytest.fixture
    def test_storage(self, tmp_path, base_catalog):
        """Create test storage with epoch_chunk_size=1 for per-epoch writing."""
        storage = LightcurveStorage(tmp_path)
        
        # Create 20 epochs for testing
        epoch_keys = [f"epoch_{i:04d}" for i in range(20)]
        
        storage.create_for_per_epoch_write(
            source_catalog=base_catalog,
            epoch_keys=epoch_keys,
        )
        
        return storage
    
    @pytest.mark.parametrize("n_workers", [1, 4])
    def test_per_epoch_writing(self, test_storage, base_catalog, n_workers):
        """
        Test per-epoch population in sequential and concurrent modes.
        
        With epoch_chunk_size=1, each epoch is a separate zarr chunk/file.
        Workers handle different epochs, writing complete (n_obj, n_mea, n_val, 1) arrays.
        
        EXPECTED BEHAVIOR:
        - n_workers=1 (sequential): Should PASS - baseline for comparison
        - n_workers=4 (concurrent): Should PASS - each worker writes different epochs
        """
        is_sequential = (n_workers == 1)
        
        # Use 12 epochs to test - enough to distribute across workers
        n_epochs = 12
        epoch_keys = [f"epoch_{i:04d}" for i in range(n_epochs)]
        
        print(f"\n{'='*60}")
        print(f"Testing per-epoch writing with n_workers={n_workers}")
        print(f"Objects: {len(base_catalog.object_keys)}, Epochs: {n_epochs}")
        print(f"Mode: {'SEQUENTIAL' if is_sequential else 'CONCURRENT (safe - different epochs per worker)'}")
        print(f"{'='*60}")
        
        def populate_worker(worker_id, assigned_epochs):
            """
            Worker function that populates assigned epochs.
            Each worker writes ENTIRE (n_obj, n_mea, n_val, 1) array for each epoch.
            """
            # Each worker gets its own storage instance
            worker_storage = LightcurveStorage(test_storage.storage_path)
            worker_storage.load_for_per_epoch_write()
            
            results = []
            for epoch_key in assigned_epochs:
                # Populate the entire epoch (all objects) from the catalog
                n_updated = worker_storage.populate_epoch(
                    source_catalog=base_catalog,
                    epoch_key=epoch_key
                )
                results.append((worker_id, epoch_key, n_updated))
                print(f"  Worker {worker_id}: populated {epoch_key} with {n_updated} measurements")
            
            return results
        
        # Distribute epochs across workers (round-robin)
        # Worker 0: epochs 0, 4, 8, ...
        # Worker 1: epochs 1, 5, 9, ...
        # Worker 2: epochs 2, 6, 10, ...
        # Worker 3: epochs 3, 7, 11, ...
        epoch_assignments = [[] for _ in range(n_workers)]
        for i, epoch_key in enumerate(epoch_keys):
            worker_id = i % n_workers
            epoch_assignments[worker_id].append(epoch_key)
        
        # Print assignment
        for worker_id, assigned in enumerate(epoch_assignments):
            print(f"Worker {worker_id} assigned: {assigned}")
        
        # Execute population
        if is_sequential:
            # Sequential execution (n_workers=1, so all epochs go to worker 0)
            all_results = populate_worker(0, epoch_assignments[0])
        else:
            # Concurrent execution - each worker handles different epochs
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = [
                    executor.submit(populate_worker, worker_id, assigned_epochs)
                    for worker_id, assigned_epochs in enumerate(epoch_assignments)
                ]
                all_results = []
                for future in concurrent.futures.as_completed(futures):
                    all_results.extend(future.result())
        
        print(f"\nCompleted: {len(all_results)} epoch populations")
        
        # Reload storage to verify all epochs were populated
        test_storage.load_for_per_epoch_write()
        
        # Verify data for each populated epoch
        sample_obj = base_catalog.object_keys[0]
        successful_epochs = 0
        
        for epoch_key in epoch_keys:
            data = test_storage.lightcurves.sel(object=sample_obj, epoch=epoch_key)
            has_data = np.any(~np.isnan(data.values))
            if has_data:
                successful_epochs += 1
        
        print(f"\nVerification: {successful_epochs}/{n_epochs} epochs have data")
        
        # All epochs should have data
        assert successful_epochs == n_epochs, \
            f"Expected all {n_epochs} epochs to have data, but only {successful_epochs} do"
        
        # Verify data correctness: compare stored values with catalog values
        print(f"\nValidating data values match catalog...")
        
        # Get catalog's measurement and value keys for indexing
        measurement_keys = sorted(base_catalog.measurement_keys)
        value_keys = sorted(base_catalog.value_keys)
        
        # Sample a few objects and epochs for detailed validation
        sample_objects = base_catalog.object_keys[:5]  # First 5 objects
        sample_epochs = epoch_keys[:3]  # First 3 epochs
        
        n_validated = 0
        for obj_key in sample_objects:
            # Find this object in the catalog
            obj_idx = np.where(base_catalog.object_keys == obj_key)[0][0]
            
            for epoch_key in sample_epochs:
                # Get data from storage
                stored_data = test_storage.lightcurves.sel(
                    object=obj_key, 
                    epoch=epoch_key
                )
                
                # Get expected values from catalog
                for mea_key in measurement_keys:
                    for val_key in value_keys:
                        stored_value = stored_data.sel(
                            measurement=mea_key,
                            value=val_key
                        ).values.item()
                        
                        # Get column name from catalog's data key mapping
                        from pyzma_idf_lightcurve.lightcurve.catalog import (
                            SourceCatalogDataKey,
                        )
                        data_key = SourceCatalogDataKey(
                            measurement=mea_key,
                            value=val_key,
                            epoch=None
                        )
                        catalog_col = base_catalog.data_key_info.data_keys_colname.get(data_key)
                        
                        if catalog_col is None:
                            continue  # This combination doesn't exist in catalog
                        
                        expected_value = base_catalog.table[catalog_col][obj_idx]
                        
                        # Compare values (allowing for float precision)
                        if not np.isnan(stored_value):
                            np.testing.assert_allclose(
                                stored_value, 
                                expected_value,
                                rtol=1e-6,
                                err_msg=f"Mismatch for object={obj_key}, epoch={epoch_key}, "
                                        f"measurement={mea_key}, value={val_key}"
                            )
                            n_validated += 1
        
        print(f"✓ Data values validated: {n_validated} measurements match catalog (from {len(sample_objects)} objects × {len(sample_epochs)} epochs)")
        print(f"\n✅ {'SEQUENTIAL' if is_sequential else 'CONCURRENT'} per-epoch writing works correctly!")


class TestZarrChunkingInfo:
    """Document zarr chunk configuration implications for concurrency."""
    
    def test_document_chunk_safety(self, tmp_path):
        """
        Document chunk-based concurrency implications with epoch_chunk_size=1.
        
        Since zarr v3 removed explicit region/synchronizer parameters,
        thread safety depends on chunk-level atomicity.
        
        For our lightcurve array with shape (object, measurement, value, epoch):
        - With epoch_chunk_size=1: each EPOCH is a separate zarr chunk/file
        - Writing to different epochs = different chunks = SAFE
        - Writing to same epoch, different objects = same chunk = UNSAFE
        
        RECOMMENDED PATTERN for concurrent population:
        1. Partition epochs into non-overlapping groups
        2. Each worker handles different epochs across all objects
        3. Workers write to different chunks (different epoch files), avoiding race conditions
        
        UNSAFE PATTERN:
        1. Multiple workers handling different objects, same epochs
        2. Workers compete for same chunks (same epoch files), potential data corruption
        """
        # Create small test storage to verify structure
        n_objects = 100
        table = Table()
        table['id'] = np.arange(1, n_objects + 1)
        table['ra'] = np.linspace(150.0, 150.5, n_objects)
        table['dec'] = np.linspace(2.0, 2.5, n_objects)
        table['x'] = np.random.uniform(100, 1000, n_objects)
        table['y'] = np.random.uniform(100, 1000, n_objects)
        table['mag_auto'] = np.random.uniform(18, 22, n_objects)
        table['magerr_auto'] = np.random.uniform(0.01, 0.1, n_objects)
        
        catalog = SourceCatalog(table)
        storage = LightcurveStorage(tmp_path)
        storage.create_for_per_epoch_write(
            source_catalog=catalog,
            epoch_keys=[f"epoch_{i:02d}" for i in range(10)]
        )
        
        # Verify storage exists
        assert (tmp_path / "lightcurves_write.zarr").exists()
        
        # Load and verify array structure  
        storage.load_for_per_epoch_write()
        assert storage.lightcurves is not None
        assert len(storage.lightcurves.coords['object']) == n_objects
        
        # Print recommendation
        print("\n" + "="*60)
        print("ZARR v3 CONCURRENT WRITING SAFETY (epoch_chunk_size=1)")
        print("="*60)
        print("Array dimensions: (object, measurement, value, epoch)")
        print("\nSAFE: Partition by EPOCHS (epoch_chunk_size=1)")
        print("  - Worker 1: all objects, epochs 0, 4, 8...")
        print("  - Worker 2: all objects, epochs 1, 5, 9...")
        print("  - Worker 3: all objects, epochs 2, 6, 10...")
        print("  - Worker 4: all objects, epochs 3, 7, 11...")
        print("  → Each worker writes to different chunks (different epoch files)")
        print("\nUNSAFE: Partition by OBJECTS")
        print("  - Worker 1: objects 0-249, all epochs")
        print("  - Worker 2: objects 250-499, all epochs")
        print("  → Workers compete for same chunks (same epoch files)")
        print("="*60)

