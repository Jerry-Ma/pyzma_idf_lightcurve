"""Tests for storage loading callback.

Based on Dash testing best practices from official documentation.
Key insight: dash.callback decorator returns the original function,
allowing direct invocation for testing (though without callback context).
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from dash import no_update
from dash.exceptions import PreventUpdate

from ..callbacks.storage_callbacks import register_storage_callbacks


class TestStorageLoadCallback:
    """Test suite for the storage loading callback."""
    
    @pytest.fixture
    def mock_app(self):
        """Create a mock Dash app for testing."""
        app = Mock()
        app.callback = Mock(return_value=lambda f: f)  # Return original function
        return app
    
    @pytest.fixture
    def callback_func(self, mock_app):
        """Register callbacks and extract the load_storage function."""
        # Register callbacks with mock app
        register_storage_callbacks(mock_app)
        
        # The callback decorator should have been called
        assert mock_app.callback.called
        
        # Get the decorated function (it's the function passed to callback)
        # The callback call_args are: ([outputs], inputs, [states], prevent_initial_call=True)
        callback_func = mock_app.callback.call_args[0][0]  # First positional arg is the function
        
        # Actually, the way callbacks work, the function is passed after all the decorator args
        # Let's extract it properly - it's wrapped, so we need the actual function
        for call_args in mock_app.callback.call_args_list:
            # The function is what's passed to the decorator result
            # We'll extract it from the wrapped function
            pass
        
        # Simpler approach: just call register and get the function from the closure
        # Let's re-register with a simpler mock
        callbacks = {}
        
        def mock_callback(*outputs, **kwargs):
            def decorator(func):
                callbacks['load_storage'] = func
                return func
            return decorator
        
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
    
    @patch('pyzma_idf_lightcurve.lightcurve.dash.callbacks.storage_callbacks.Path')
    def test_path_does_not_exist(self, mock_path_class, callback_func):
        """Test error handling when storage path doesn't exist."""
        # Setup mock
        mock_path = Mock()
        mock_path.exists.return_value = False
        mock_path.__truediv__ = lambda self, other: mock_path  # Support / operator
        mock_path.__str__ = lambda self: "fake/path/lightcurves_read.zarr"
        mock_path_class.return_value = mock_path
        
        # Call callback
        storage_data, storage_info, notification = callback_func(
            1, "fake/path", "read"
        )
        
        # Assertions
        assert storage_data is no_update
        assert storage_info is no_update
        assert notification['props']['title'] == "Error"
        assert "does not exist" in notification['props']['message']
        assert notification['props']['color'] == "red"
    
    @patch('pyzma_idf_lightcurve.lightcurve.dash.callbacks.storage_callbacks.LightcurveStorage')
    @patch('pyzma_idf_lightcurve.lightcurve.dash.callbacks.storage_callbacks.Path')
    def test_successful_load_read_mode(self, mock_path_class, mock_storage_class, callback_func):
        """Test successful storage loading in read mode."""
        # Setup path mock
        mock_path = Mock()
        mock_path.exists.return_value = True
        mock_path.is_dir.return_value = True
        mock_path.__truediv__ = lambda self, other: mock_path
        mock_path.__str__ = lambda self: "test/path/lightcurves_read.zarr"
        mock_path_class.return_value = mock_path
        
        # Setup storage mock
        mock_storage = Mock()
        mock_storage.get_storage_info.return_value = {
            'shape': (100, 200),
            'dimensions': ['time', 'object'],
            'coordinates': {'time': [], 'object': []},
            'variables': {'flux': {}, 'flux_err': {}},
            'chunks': {'time': 10, 'object': 20},
            'dtype': 'float32',
        }
        mock_storage_class.load_for_per_object_read.return_value = mock_storage
        
        # Call callback
        storage_data, storage_info, notification = callback_func(
            1, "test/path", "read"
        )
        
        # Assertions
        assert storage_data['path'] == "test/path/lightcurves_read.zarr"
        assert storage_data['mode'] == "read"
        assert storage_info['shape'] == (100, 200)
        assert 'flux' in storage_info['variables']
        assert notification['props']['title'] == "Success"
        assert notification['props']['color'] == "green"
        
        # Verify correct method was called
        mock_storage_class.load_for_per_object_read.assert_called_once()
    
    @patch('pyzma_idf_lightcurve.lightcurve.dash.callbacks.storage_callbacks.LightcurveStorage')
    @patch('pyzma_idf_lightcurve.lightcurve.dash.callbacks.storage_callbacks.Path')
    def test_successful_load_write_mode(self, mock_path_class, mock_storage_class, callback_func):
        """Test successful storage loading in write mode."""
        # Setup path mock
        mock_path = Mock()
        mock_path.exists.return_value = True
        mock_path.is_dir.return_value = True
        mock_path.__truediv__ = lambda self, other: mock_path
        mock_path.__str__ = lambda self: "test/path/lightcurves_write.zarr"
        mock_path_class.return_value = mock_path
        
        # Setup storage mock
        mock_storage = Mock()
        mock_storage.get_storage_info.return_value = {
            'shape': (100, 200),
            'dimensions': ['time', 'object'],
            'coordinates': {'time': [], 'object': []},
            'variables': {'flux': {}, 'flux_err': {}},
            'chunks': {'time': 10, 'object': 20},
            'dtype': 'float32',
        }
        mock_storage_class.load_for_per_epoch_write.return_value = mock_storage
        
        # Call callback
        storage_data, storage_info, notification = callback_func(
            1, "test/path", "write"
        )
        
        # Assertions
        assert storage_data['path'] == "test/path/lightcurves_write.zarr"
        assert storage_data['mode'] == "write"
        assert notification['props']['title'] == "Success"
        
        # Verify correct method was called
        mock_storage_class.load_for_per_epoch_write.assert_called_once()
    
    @patch('pyzma_idf_lightcurve.lightcurve.dash.callbacks.storage_callbacks.LightcurveStorage')
    @patch('pyzma_idf_lightcurve.lightcurve.dash.callbacks.storage_callbacks.Path')
    def test_exception_handling(self, mock_path_class, mock_storage_class, callback_func):
        """Test that exceptions are caught and returned as error notifications."""
        # Setup path mock
        mock_path = Mock()
        mock_path.exists.return_value = True
        mock_path.is_dir.return_value = True
        mock_path.__truediv__ = lambda self, other: mock_path
        mock_path.__str__ = lambda self: "test/path/lightcurves_read.zarr"
        mock_path_class.return_value = mock_path
        
        # Make storage loading raise an exception
        mock_storage_class.load_for_per_object_read.side_effect = ValueError("Invalid zarr format")
        
        # Call callback
        storage_data, storage_info, notification = callback_func(
            1, "test/path", "read"
        )
        
        # Assertions
        assert storage_data is no_update
        assert storage_info is no_update
        assert notification['props']['title'] == "Error"
        assert "Invalid zarr format" in notification['props']['message']
        assert notification['props']['color'] == "red"


# Integration test using actual Dash app (optional, requires dash.testing)
def test_storage_callback_integration():
    """
    Integration test using dash.testing framework.
    
    This test would require:
    - dash[testing] to be installed
    - An actual test zarr storage to exist
    - The full Dash app to be running
    
    Example structure:
    
    from dash.testing.application_runners import import_app
    
    def test_load_storage_integration(dash_duo):
        app = import_app("pyzma_idf_lightcurve.lightcurve.dash.app")
        dash_duo.start_server(app)
        
        # Enter storage path
        storage_input = dash_duo.find_element("#storage-path-input")
        storage_input.send_keys("test/path/storage.zarr")
        
        # Click load button
        load_button = dash_duo.find_element("#load-storage-button")
        load_button.click()
        
        # Wait for callback to complete
        dash_duo.wait_for_text_to_equal("#notifications-container", "Success", timeout=10)
        
        # Verify storage was loaded
        assert dash_duo.get_logs() == [], "No browser console errors"
    """
    pytest.skip("Integration test requires dash[testing] and test data")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
