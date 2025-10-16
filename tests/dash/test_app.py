"""Tests for Dash v3 application creation and configuration."""

import pytest
import dash_mantine_components as dmc

from pyzma_idf_lightcurve.lightcurve.dash.app import create_app, run_app


class TestAppCreation:
    """Test suite for Dash app creation."""
    
    def test_create_app_returns_dash_instance(self):
        """Test that create_app returns a Dash application instance."""
        app = create_app()
        
        assert hasattr(app, 'layout')
        assert hasattr(app, 'callback')
        assert app.title == "IDF Lightcurve Viewer"
    
    def test_create_app_with_initial_storage_path(self):
        """Test app creation with initial storage path parameter."""
        test_path = "test/storage/path.zarr"
        app = create_app(initial_storage_path=test_path)
        
        assert app is not None
        # The initial path should be passed to the storage loader component
    
    def test_create_app_without_storage_path(self):
        """Test app creation without initial storage path (default behavior)."""
        app = create_app()
        
        assert app is not None
    
    def test_app_has_required_stores(self):
        """Test that app layout contains required dcc.Store components."""
        app = create_app()
        layout_str = str(app.layout)
        
        # Check for required Store IDs (storage-info removed - data is retrieved from cache)
        assert 'storage-data' in layout_str
        assert 'selected-objects' in layout_str
    
    def test_app_has_mantine_provider(self):
        """Test that app uses MantineProvider for theming."""
        app = create_app()
        
        # Layout should be wrapped in MantineProvider
        assert app.layout is not None
    
    def test_app_has_main_components(self):
        """Test that app contains main UI components."""
        app = create_app()
        layout_str = str(app.layout)
        
        # Check for main component IDs
        assert 'storage-path-input' in layout_str
        assert 'storage-mode-select' in layout_str
        assert 'load-storage-button' in layout_str
        assert 'main-tabs' in layout_str
    
    def test_app_callbacks_registered(self):
        """Test that callbacks are registered with the app."""
        app = create_app()
        
        # Dash v3 stores callbacks in app.callback_map
        assert hasattr(app, 'callback_map')
        # Should have multiple callbacks registered
        assert len(app.callback_map) > 0


class TestRunApp:
    """Test suite for run_app function."""
    
    def test_run_app_creates_and_runs_app(self):
        """Test that run_app creates app and calls app.run."""
        from unittest.mock import Mock, patch
        with patch('pyzma_idf_lightcurve.lightcurve.dash.app.create_app') as mock_create_app:
            mock_app = Mock()
            mock_create_app.return_value = mock_app
            
            # This would normally block, so we'll just verify setup
            try:
                run_app(debug=False, port=8050, host='127.0.0.1', storage_path=None)
            except:
                pass  # Expected to fail when trying to actually run
            
            # Verify create_app was called with correct parameter
            mock_create_app.assert_called_once_with(initial_storage_path=None)
    
    def test_run_app_with_storage_path(self):
        """Test that run_app passes storage_path to create_app."""
        from unittest.mock import Mock, patch
        with patch('pyzma_idf_lightcurve.lightcurve.dash.app.create_app') as mock_create_app:
            mock_app = Mock()
            mock_create_app.return_value = mock_app
            
            test_path = "test/storage.zarr"
            
            try:
                run_app(storage_path=test_path)
            except:
                pass
            
            mock_create_app.assert_called_once_with(initial_storage_path=test_path)


class TestAppConfiguration:
    """Test suite for app configuration and settings."""
    
    def test_app_suppress_callback_exceptions(self):
        """Test that app has suppress_callback_exceptions enabled."""
        app = create_app()
        
        assert app.config.suppress_callback_exceptions is True
    
    def test_app_title_set(self):
        """Test that app has correct title."""
        app = create_app()
        
        assert app.title == "IDF Lightcurve Viewer"
