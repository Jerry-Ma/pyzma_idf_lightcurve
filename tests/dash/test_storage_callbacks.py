"""Integration tests for storage callbacks using real LightcurveStorage instances.

This test suite uses real storage instances created with sample catalog data
instead of mocks. This ensures we test the actual API contracts and catch
real integration issues.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock
from dash import no_update
from dash.exceptions import PreventUpdate

from pyzma_idf_lightcurve.lightcurve.dash.callbacks.storage_callbacks import register_storage_callbacks


class TestStorageLoadCallback:
    """Integration tests for the storage loading callback with real storage."""
    
    @pytest.fixture
    def callback_func(self):
        """Register callbacks and extract the load_storage function."""
        callbacks = {}
        
        def mock_callback(*outputs, **kwargs):
            def decorator(func):
                callbacks['load_storage'] = func
                return func
            return decorator
        
        mock_app = Mock()
        mock_app.callback = mock_callback
        register_storage_callbacks(mock_app)
        
        return callbacks['load_storage']
    
    def test_prevent_update_on_no_clicks(self, callback_func):
        """Test that callback raises PreventUpdate when n_clicks is None or 0."""
        with pytest.raises(PreventUpdate):
            callback_func(None, "some/path", "read")
        
        with pytest.raises(PreventUpdate):
            callback_func(0, "some/path", "read")
    
    def test_prevent_update_on_no_path(self, callback_func):
        """Test that callback raises PreventUpdate when storage_path is empty."""
        with pytest.raises(PreventUpdate):
            callback_func(1, "", "read")
        
        with pytest.raises(PreventUpdate):
            callback_func(1, None, "read")
    
    def test_path_does_not_exist(self, callback_func, temp_dir):
        """Test error handling when path doesn't exist."""
        nonexistent_path = str(temp_dir / "nonexistent" / "lightcurves_read.zarr")
        
        storage_data, notification = callback_func(
            1, nonexistent_path, "read"
        )
        
        # Should return error notification
        assert storage_data is no_update
        assert notification.title == "Error"
        assert notification.color == "red"
        assert "does not exist" in notification.message
    
    def test_successful_load_read_mode(self, callback_func, populated_storage_read):
        """Test successful storage loading in read mode using real storage."""
        from pyzma_idf_lightcurve.lightcurve.dash.storage_cache import StorageCache
        
        # Use the storage_path (parent directory), not the zarr path
        storage_path = str(populated_storage_read.storage_path)
        
        # Call callback with storage path
        storage_data, notification = callback_func(
            1, storage_path, "read"
        )
        
        # Verify success notification
        assert notification.title == "Success"
        assert notification.color == "green"
        
        # Verify storage data (cache key)
        assert storage_data is not None
        assert storage_data['storage_path'] == storage_path
        assert storage_data['mode'] == "read"
        
        # Verify storage was cached - retrieve it from cache
        cache = StorageCache.get_instance()
        cached_storage = cache.get(Path(storage_path), "read")
        assert cached_storage is not None
        assert cached_storage.storage_path == Path(storage_path)
    
    def test_successful_load_write_mode(self, callback_func, populated_storage_write):
        """Test successful storage loading in write mode using real storage."""
        from pyzma_idf_lightcurve.lightcurve.dash.storage_cache import StorageCache
        
        # Use the storage_path (parent directory), not the zarr path
        storage_path = str(populated_storage_write.storage_path)
        
        # Call callback with storage path
        storage_data, notification = callback_func(
            1, storage_path, "write"
        )
        
        # Verify success - storage_data is the cache key
        assert storage_data['storage_path'] == storage_path
        assert storage_data['mode'] == "write"
        assert notification.title == "Success"
        
        # Verify storage was cached - retrieve it from cache
        cache = StorageCache.get_instance()
        cached_storage = cache.get(Path(storage_path), "write")
        assert cached_storage is not None
        assert cached_storage.storage_path == Path(storage_path)
    
    def test_exception_handling(self, callback_func, temp_dir):
        """Test that callback handles exceptions gracefully with invalid zarr."""
        # Create an invalid zarr path (file instead of directory)
        invalid_path = temp_dir / "invalid.zarr"
        invalid_path.touch()  # Create a file, not a directory
        
        storage_data, notification = callback_func(
            1, str(invalid_path), "read"
        )
        
        # Should return error notification
        assert storage_data is no_update
        assert notification.title == "Error"
        assert notification.color == "red"
        # Error message should contain some diagnostic info
        assert len(notification.message) > 0
    
    def test_both_modes_with_same_storage(self, callback_func, temp_dir, source_catalog, epoch_keys):
        """Test that both read and write modes work with the same storage base."""
        from pyzma_idf_lightcurve.lightcurve.datamodel import LightcurveStorage
        
        # Create and populate storage
        storage_path = temp_dir / "test_both_modes"
        storage = LightcurveStorage(storage_path)
        storage.create_for_per_epoch_write(
            source_catalog=source_catalog,
            epoch_keys=epoch_keys,
        )
        storage.populate_epoch(source_catalog=source_catalog, epoch_key=epoch_keys[0])
        
        # Create read zarr by rechunking
        storage.rechunk_for_per_object_read(chunk_size=1000)
        storage.close()
        
        # Test read mode - use storage_path, not zarr path
        from pyzma_idf_lightcurve.lightcurve.dash.storage_cache import StorageCache
        
        base_path = str(storage_path)
        storage_data_read, notif_read = callback_func(
            1, base_path, "read"
        )
        assert notif_read.title == "Success"
        assert storage_data_read['mode'] == "read"
        
        # Test write mode - use storage_path, not zarr path
        storage_data_write, notif_write = callback_func(
            1, base_path, "write"
        )
        assert notif_write.title == "Success"
        assert storage_data_write['mode'] == "write"
        
        # Both should be cached
        cache = StorageCache.get_instance()
        cached_read = cache.get(storage_path, "read")
        cached_write = cache.get(storage_path, "write")
        assert cached_read is not None
        assert cached_write is not None
        
        # Both should have same dimensions and shape
        assert list(cached_read.dataset.dims) == list(cached_write.dataset.dims)
        assert dict(cached_read.dataset.sizes) == dict(cached_write.dataset.sizes)
