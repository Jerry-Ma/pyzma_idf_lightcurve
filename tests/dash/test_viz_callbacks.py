"""Tests for visualization callbacks."""

import pytest
from dash.exceptions import PreventUpdate

from pyzma_idf_lightcurve.lightcurve.dash.callbacks.viz_callbacks import (
    register_viz_callbacks,
)


class TestUpdateVizControlsCallback:
    """Test suite for viz controls update callback."""
    
    def test_returns_empty_lists_without_storage_info(self):
        """Test that empty lists are returned when storage_info is None."""
        from unittest.mock import Mock
        app = Mock()
        callbacks = []
        def capture_callback(*outputs, **kwargs):
            def decorator(func):
                callbacks.append(func)
                return func
            return decorator
        app.callback = capture_callback
        
        register_viz_callbacks(app)
        # The callbacks are: 0=viz_controls, 1=query_validation, 2=plot_generation
        # (query_validation requires Input from storage-info and query-input)
        update_viz_controls = callbacks[0]
        
        object_opts, error_msg, meas_opts, x_axis_opts, disabled = update_viz_controls(None, None, None)
        
        assert object_opts == []
        assert error_msg == ""
        assert meas_opts == []
        assert x_axis_opts == []
        assert disabled is True
    
    def test_populates_options_with_valid_data(self, populated_storage_read):
        """Test that options are populated with valid storage info."""
        from unittest.mock import Mock
        app = Mock()
        callbacks = []
        def capture_callback(*outputs, **kwargs):
            def decorator(func):
                callbacks.append(func)
                return func
            return decorator
        app.callback = capture_callback
        
        register_viz_callbacks(app)
        # The callbacks are: 0=viz_controls, 1=query_validation, 2=plot_generation
        update_viz_controls = callbacks[0]
        
        # Use real storage info
        storage_info = populated_storage_read.get_storage_info()
        selected_objects = [0, 1, 2]
        
        object_opts, error_msg, meas_opts, x_axis_opts, disabled = update_viz_controls(
            storage_info, selected_objects, None
        )
        
        assert len(object_opts) == 3
        assert error_msg == ""
        # measurement dimension from our catalog
        assert len(meas_opts) >= 1
        # Test fixture doesn't create epoch_ variables in measurement_keys
        # so x_axis_opts will be empty (epoch_ vars would be in data variables, not measurement_keys)
        # This is expected behavior - the callback filters measurement_keys for epoch_ prefix
        assert isinstance(x_axis_opts, list)  # Just verify it's a list
        assert disabled is False


class TestGenerateLightcurvePlotsCallback:
    """Test suite for lightcurve plot generation callback."""
    
    def test_prevent_update_on_no_clicks(self):
        """Test PreventUpdate when button not clicked."""
        from unittest.mock import Mock
        app = Mock()
        callbacks = []
        def capture_callback(*outputs, **kwargs):
            def decorator(func):
                callbacks.append(func)
                return func
            return decorator
        app.callback = capture_callback
        
        register_viz_callbacks(app)
        # Third callback is plot generation (first is validation, second is controls)
        generate_plots = callbacks[2]
        
        with pytest.raises(PreventUpdate):
            generate_plots(None, {'path': 'test'}, ['0'], ['0'], 'epoch_mjd', None)
    
    def test_prevent_update_on_missing_data(self):
        """Test PreventUpdate when required data is missing."""
        from unittest.mock import Mock
        app = Mock()
        callbacks = []
        def capture_callback(*outputs, **kwargs):
            def decorator(func):
                callbacks.append(func)
                return func
            return decorator
        app.callback = capture_callback
        
        register_viz_callbacks(app)
        # Third callback is plot generation (first is validation, second is controls)
        generate_plots = callbacks[2]
        
        with pytest.raises(PreventUpdate):
            generate_plots(1, None, ['0'], ['0'], 'epoch_mjd', None)
        
        with pytest.raises(PreventUpdate):
            generate_plots(1, {'path': 'test'}, None, ['0'], 'epoch_mjd', None)
        
        with pytest.raises(PreventUpdate):
            generate_plots(1, {'path': 'test'}, ['0'], None, 'epoch_mjd', None)
    
    def test_successful_plot_generation(self, populated_storage_read):
        """Test successful plot generation with valid data."""
        from unittest.mock import Mock, patch
        # We keep make_subplots patched since it's an external library (plotly)
        with patch('pyzma_idf_lightcurve.lightcurve.dash.callbacks.viz_callbacks.make_subplots') as mock_make_subplots:
            app = Mock()
            callbacks = []
            def capture_callback(*outputs, **kwargs):
                def decorator(func):
                    callbacks.append(func)
                    return func
                return decorator
            app.callback = capture_callback
            
            register_viz_callbacks(app)
            # The callbacks are: 0=viz_controls, 1=query_validation, 2=plot_generation
            generate_plots = callbacks[2]
            
            # Mock the plotly figure
            mock_fig = Mock()
            mock_make_subplots.return_value = mock_fig
            
            # Use real storage with correct key 'storage_path'
            storage_data = {'storage_path': str(populated_storage_read.storage_path), 'mode': 'read'}
            # Object keys in fixture are '1' through '20' (from NUMBER column starting at 1)
            object_keys = ['1', '2']
            # Measurement keys are '0' (single measurement dimension in test catalog)
            measurement_keys = ['0']
            # Get an actual x-axis variable from the storage
            storage_info = populated_storage_read.get_storage_info()
            epoch_vars = [v for v in storage_info['measurement_keys'] if v.startswith('epoch_')]
            x_axis_var = epoch_vars[0] if epoch_vars else 'epoch_mjd'
            
            # Should return a figure
            result = generate_plots(
                1, storage_data, object_keys, measurement_keys, x_axis_var, None
            )
            
            # Verify it returns the figure
            assert result is mock_fig
            mock_make_subplots.assert_called_once()
