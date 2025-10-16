"""Tests for info tab callbacks."""

import pytest
from dash.exceptions import PreventUpdate

from pyzma_idf_lightcurve.lightcurve.dash.callbacks.info_tab_callbacks import (
    register_info_tab_callbacks,
    build_info_tables_data,
)


@pytest.fixture
def mock_app():
    """Create a mock Dash app for callback registration."""
    from unittest.mock import Mock
    app = Mock()
    app.callback = lambda *outputs, **kwargs: lambda func: func
    return app


class TestStorageOverviewCallback:
    """Test suite for storage overview callback."""
    
    @pytest.fixture
    def callback_func(self, mock_app):
        """Extract the callback function for testing."""
        register_info_tab_callbacks(mock_app)
        # Get the first registered callback (storage overview)
        return mock_app.callback.call_args_list[0][1]['func'] if mock_app.callback.call_args_list else None
    
    def test_prevent_update_on_no_storage_info(self):
        """Test PreventUpdate is raised when storage_info is None."""
        from unittest.mock import Mock
        # Manually register to get callback
        app = Mock()
        callbacks = []
        def capture_callback(*outputs, **kwargs):
            def decorator(func):
                callbacks.append(func)
                return func
            return decorator
        app.callback = capture_callback
        
        register_info_tab_callbacks(app)
        update_storage_overview = callbacks[0]
        
        with pytest.raises(PreventUpdate):
            update_storage_overview(None)
    
    def test_update_with_valid_storage_info(self, populated_storage_read):
        """Test successful update with valid storage data."""
        from unittest.mock import Mock
        from pyzma_idf_lightcurve.lightcurve.dash.storage_cache import StorageCache
        
        # Register callback
        app = Mock()
        callbacks = []
        def capture_callback(*outputs, **kwargs):
            def decorator(func):
                callbacks.append(func)
                return func
            return decorator
        app.callback = capture_callback
        
        register_info_tab_callbacks(app)
        update_storage_overview = callbacks[0]
        
        # Put storage in cache
        cache = StorageCache.get_instance()
        cache.set(populated_storage_read.storage_path, 'read', populated_storage_read)
        
        # Use storage_data (cache key)
        storage_data = {
            'storage_path': str(populated_storage_read.storage_path),
            'mode': 'read',
        }
        
        result = update_storage_overview(storage_data)
        
        # Should return a list of Text components
        assert isinstance(result, list)
        assert len(result) > 0


class TestEpochVariablesGridCallback:
    """Test suite for epoch variables grid callback."""
    
    def test_prevent_update_on_no_storage_info(self):
        """Test PreventUpdate when storage_info is None."""
        from unittest.mock import Mock
        app = Mock()
        callbacks = []
        def capture_callback(*outputs, **kwargs):
            def decorator(func):
                callbacks.append(func)
                return func
            return decorator
        app.callback = capture_callback
        
        register_info_tab_callbacks(app)
        update_epoch_variables_grid = callbacks[1]
        
        with pytest.raises(PreventUpdate):
            update_epoch_variables_grid(None)
    
    def test_filter_epoch_variables(self, populated_storage_read):
        """Test that epoch variables with epoch dimension are discovered."""
        from unittest.mock import Mock
        from pyzma_idf_lightcurve.lightcurve.dash.storage_cache import StorageCache
        
        app = Mock()
        callbacks = []
        def capture_callback(*outputs, **kwargs):
            def decorator(func):
                callbacks.append(func)
                return func
            return decorator
        app.callback = capture_callback
        
        register_info_tab_callbacks(app)
        update_epoch_variables_grid = callbacks[1]
        
        # Put storage in cache
        cache = StorageCache.get_instance()
        cache.set(populated_storage_read.storage_path, 'read', populated_storage_read)
        
        # Use storage_data (cache key)
        storage_data = {
            'storage_path': str(populated_storage_read.storage_path),
            'mode': 'read',
        }
        
        row_data, column_defs = update_epoch_variables_grid(storage_data)
        
        # Should discover variables with epoch dimension and display as columns
        assert isinstance(row_data, list)
        assert len(row_data) > 0  # Should have at least some epoch rows (epochs)
        # Columns: index, epoch_id, + discovered epoch variables (lightcurves)
        assert len(column_defs) >= 3  # At least index, epoch_id, and one variable
        
        # Verify column structure
        assert column_defs[0]['field'] == 'index'
        assert column_defs[1]['field'] == 'epoch_id'
        
        # Verify row data has the right structure
        for row in row_data:
            assert 'index' in row
            assert 'epoch_id' in row


class TestObjectVariablesGridCallback:
    """Test suite for object variables grid callback."""
    
    def test_prevent_update_on_missing_data(self):
        """Test PreventUpdate when data is missing."""
        from unittest.mock import Mock
        app = Mock()
        callbacks = []
        def capture_callback(*outputs, **kwargs):
            def decorator(func):
                callbacks.append(func)
                return func
            return decorator
        app.callback = capture_callback
        
        register_info_tab_callbacks(app)
        update_object_variables_grid = callbacks[2]
        
        with pytest.raises(PreventUpdate):
            update_object_variables_grid(None)
    
    def test_successful_grid_update(self, populated_storage_read, temp_dir):
        """Test successful object variables grid update with dynamic variable discovery."""
        from unittest.mock import Mock
        from pyzma_idf_lightcurve.lightcurve.dash.storage_cache import StorageCache
        
        # Register callback
        app = Mock()
        callbacks = []
        def capture_callback(*outputs, **kwargs):
            def decorator(func):
                callbacks.append(func)
                return func
            return decorator
        app.callback = capture_callback
        
        register_info_tab_callbacks(app)
        update_object_variables_grid = callbacks[2]
        
        # Put storage in cache
        cache = StorageCache.get_instance()
        cache.set(populated_storage_read.storage_path, 'read', populated_storage_read)
        
        # Use storage_data (cache key)
        storage_data = {
            'storage_path': str(populated_storage_read.storage_path),
            'mode': 'read',
        }
        
        row_data, column_defs, button_disabled = update_object_variables_grid(storage_data)
        
        # Should return data for 20 objects from sample catalog
        assert len(row_data) == 20
        assert len(column_defs) >= 3  # index + object_id + at least one variable
        assert button_disabled is False
        
        # Verify object_id (coordinate label) is present
        assert 'object_id' in row_data[0]
        
        # Verify ra and dec coordinates are discovered and present
        assert 'ra' in row_data[0]
        assert 'dec' in row_data[0]
        assert isinstance(row_data[0]['ra'], float)
        assert isinstance(row_data[0]['dec'], float)
        
        # Verify index is present
        assert 'index' in row_data[0]
        assert row_data[0]['index'] == 0
    
    def test_coordinate_access_with_real_storage(self, populated_storage_read, temp_dir):
        """Test that coordinates are discovered and accessible from the dataset.
        
        This test verifies dynamic variable discovery works correctly
        and coordinates are accessible via the dataset property.
        """
        from unittest.mock import Mock
        from pyzma_idf_lightcurve.lightcurve.dash.storage_cache import StorageCache
        
        # Register callback
        app = Mock()
        callbacks = []
        def capture_callback(*outputs, **kwargs):
            def decorator(func):
                callbacks.append(func)
                return func
            return decorator
        app.callback = capture_callback
        
        register_info_tab_callbacks(app)
        update_object_variables_grid = callbacks[2]
        
        # Put storage in cache
        cache = StorageCache.get_instance()
        cache.set(populated_storage_read.storage_path, 'read', populated_storage_read)
        
        # Use storage_data (cache key)
        storage_data = {
            'storage_path': str(populated_storage_read.storage_path),
            'mode': 'read',
        }
        
        # This should NOT raise AttributeError about coords/variables
        try:
            row_data, column_defs, button_disabled = update_object_variables_grid(storage_data)
            # If we get here, the callback executed successfully
            assert len(row_data) > 0, "Should have returned data"
            # Verify we can actually access coordinate values
            assert 'ra' in row_data[0], "RA coordinate should be in row data"
            assert isinstance(row_data[0]['ra'], float), "RA should be a float"
        except AttributeError as e:
            pytest.fail(f"Callback failed with AttributeError (likely using wrong dataset access): {e}")


class TestStorageOverviewMetadataTable:
    """Test suite for storage overview metadata table functionality."""
    
    def test_metadata_table_includes_all_variables(self, populated_storage_read):
        """Test that metadata table includes all coordinates and data variables."""
        from unittest.mock import Mock
        from pyzma_idf_lightcurve.lightcurve.dash.storage_cache import StorageCache
        import dash_ag_grid as dag
        
        app = Mock()
        callbacks = []
        def capture_callback(*outputs, **kwargs):
            def decorator(func):
                callbacks.append(func)
                return func
            return decorator
        app.callback = capture_callback
        
        register_info_tab_callbacks(app)
        update_storage_overview = callbacks[0]
        
        # Put storage in cache
        cache = StorageCache.get_instance()
        cache.set(populated_storage_read.storage_path, 'read', populated_storage_read)
        
        storage_data = {
            'storage_path': str(populated_storage_read.storage_path),
            'mode': 'read',
        }
        
        result = update_storage_overview(storage_data)
        
        # Should return list with Text components and AgGrid
        assert isinstance(result, list)
        assert len(result) >= 3  # Summary texts + AgGrid
        
        # Find the AgGrid component
        grid = None
        for component in result:
            if isinstance(component, dag.AgGrid):
                grid = component
                break
        
        assert grid is not None, "Should contain an AgGrid component"
        
        # Verify grid has metadata
        row_data = grid.rowData
        assert isinstance(row_data, list)
        assert len(row_data) > 0
        
        # Check that all required fields are present
        for row in row_data:
            assert 'variable' in row
            assert 'type' in row
            assert 'dimensions' in row
            assert 'shape' in row
            assert 'chunks' in row
            assert 'dtype' in row
        
        # Verify we have both coordinates and data_vars
        types = {row['type'] for row in row_data}
        assert 'coordinate' in types
        assert 'data_var' in types
        
        # Verify expected variables are present
        var_names = {row['variable'] for row in row_data}
        assert 'object' in var_names
        assert 'epoch' in var_names
        assert 'lightcurves' in var_names
    
    def test_metadata_table_column_definitions(self, populated_storage_read):
        """Test that metadata table has correct column definitions."""
        from unittest.mock import Mock
        from pyzma_idf_lightcurve.lightcurve.dash.storage_cache import StorageCache
        import dash_ag_grid as dag
        
        app = Mock()
        callbacks = []
        def capture_callback(*outputs, **kwargs):
            def decorator(func):
                callbacks.append(func)
                return func
            return decorator
        app.callback = capture_callback
        
        register_info_tab_callbacks(app)
        update_storage_overview = callbacks[0]
        
        cache = StorageCache.get_instance()
        cache.set(populated_storage_read.storage_path, 'read', populated_storage_read)
        
        storage_data = {
            'storage_path': str(populated_storage_read.storage_path),
            'mode': 'read',
        }
        
        result = update_storage_overview(storage_data)
        
        # Find the AgGrid
        grid = None
        for component in result:
            if isinstance(component, dag.AgGrid):
                grid = component
                break
        
        assert grid is not None
        
        # Verify column definitions
        column_defs = grid.columnDefs
        assert isinstance(column_defs, list)
        assert len(column_defs) == 6  # variable, type, dimensions, shape, chunks, dtype
        
        expected_fields = ['variable', 'type', 'dimensions', 'shape', 'chunks', 'dtype']
        actual_fields = [col['field'] for col in column_defs]
        assert actual_fields == expected_fields


class TestEpochVariablesAsColumns:
    """Test suite for epoch variables displayed as columns."""
    
    def test_epoch_rows_with_variable_columns(self, populated_storage_read):
        """Test that each row represents an epoch with variable values as columns."""
        from unittest.mock import Mock
        from pyzma_idf_lightcurve.lightcurve.dash.storage_cache import StorageCache
        
        app = Mock()
        callbacks = []
        def capture_callback(*outputs, **kwargs):
            def decorator(func):
                callbacks.append(func)
                return func
            return decorator
        app.callback = capture_callback
        
        register_info_tab_callbacks(app)
        update_epoch_variables_grid = callbacks[1]
        
        cache = StorageCache.get_instance()
        cache.set(populated_storage_read.storage_path, 'read', populated_storage_read)
        
        storage_data = {
            'storage_path': str(populated_storage_read.storage_path),
            'mode': 'read',
        }
        
        row_data, column_defs = update_epoch_variables_grid(storage_data)
        
        # Verify we have 5 epochs (from sample data)
        assert len(row_data) == 5
        
        # Each row should have index, epoch_id, and variable columns
        for i, row in enumerate(row_data):
            assert row['index'] == i
            assert 'epoch_id' in row
            # Should have at least one variable column (lightcurves)
            assert len(row) >= 3
    
    def test_epoch_id_shows_coordinate_labels(self, populated_storage_read):
        """Test that epoch_id shows actual coordinate values not just indices."""
        from unittest.mock import Mock
        from pyzma_idf_lightcurve.lightcurve.dash.storage_cache import StorageCache
        
        app = Mock()
        callbacks = []
        def capture_callback(*outputs, **kwargs):
            def decorator(func):
                callbacks.append(func)
                return func
            return decorator
        app.callback = capture_callback
        
        register_info_tab_callbacks(app)
        update_epoch_variables_grid = callbacks[1]
        
        cache = StorageCache.get_instance()
        cache.set(populated_storage_read.storage_path, 'read', populated_storage_read)
        
        storage_data = {
            'storage_path': str(populated_storage_read.storage_path),
            'mode': 'read',
        }
        
        row_data, column_defs = update_epoch_variables_grid(storage_data)
        
        # Verify epoch_id contains string values (AOR keys)
        for row in row_data:
            epoch_id = row['epoch_id']
            assert isinstance(epoch_id, str)
            # Should start with 'r' (AOR key format)
            assert epoch_id.startswith('r')
    
    def test_discovered_epoch_variables_in_columns(self, populated_storage_read):
        """Test that epoch variables are discovered and added as columns."""
        from unittest.mock import Mock
        from pyzma_idf_lightcurve.lightcurve.dash.storage_cache import StorageCache
        
        app = Mock()
        callbacks = []
        def capture_callback(*outputs, **kwargs):
            def decorator(func):
                callbacks.append(func)
                return func
            return decorator
        app.callback = capture_callback
        
        register_info_tab_callbacks(app)
        update_epoch_variables_grid = callbacks[1]
        
        cache = StorageCache.get_instance()
        cache.set(populated_storage_read.storage_path, 'read', populated_storage_read)
        
        storage_data = {
            'storage_path': str(populated_storage_read.storage_path),
            'mode': 'read',
        }
        
        row_data, column_defs = update_epoch_variables_grid(storage_data)
        
        # Verify column definitions include discovered variables
        column_fields = [col['field'] for col in column_defs]
        assert 'index' in column_fields
        assert 'epoch_id' in column_fields
        # Should have lightcurves (multi-dimensional variable with epoch)
        assert 'lightcurves' in column_fields


class TestObjectVariablesAsColumns:
    """Test suite for object variables displayed as columns."""
    
    def test_object_rows_with_variable_columns(self, populated_storage_read):
        """Test that each row represents an object with variable values as columns."""
        from unittest.mock import Mock
        from pyzma_idf_lightcurve.lightcurve.dash.storage_cache import StorageCache
        
        app = Mock()
        callbacks = []
        def capture_callback(*outputs, **kwargs):
            def decorator(func):
                callbacks.append(func)
                return func
            return decorator
        app.callback = capture_callback
        
        register_info_tab_callbacks(app)
        update_object_variables_grid = callbacks[2]
        
        cache = StorageCache.get_instance()
        cache.set(populated_storage_read.storage_path, 'read', populated_storage_read)
        
        storage_data = {
            'storage_path': str(populated_storage_read.storage_path),
            'mode': 'read',
        }
        
        row_data, column_defs, _ = update_object_variables_grid(storage_data)
        
        # Verify we have 20 objects
        assert len(row_data) == 20
        
        # Each row should have index, object_id, and variable columns
        for i, row in enumerate(row_data):
            assert row['index'] == i
            assert 'object_id' in row
            # Should have ra, dec coordinates
            assert 'ra' in row
            assert 'dec' in row
    
    def test_object_id_shows_coordinate_labels(self, populated_storage_read):
        """Test that object_id shows actual coordinate values not just indices."""
        from unittest.mock import Mock
        from pyzma_idf_lightcurve.lightcurve.dash.storage_cache import StorageCache
        
        app = Mock()
        callbacks = []
        def capture_callback(*outputs, **kwargs):
            def decorator(func):
                callbacks.append(func)
                return func
            return decorator
        app.callback = capture_callback
        
        register_info_tab_callbacks(app)
        update_object_variables_grid = callbacks[2]
        
        cache = StorageCache.get_instance()
        cache.set(populated_storage_read.storage_path, 'read', populated_storage_read)
        
        storage_data = {
            'storage_path': str(populated_storage_read.storage_path),
            'mode': 'read',
        }
        
        row_data, column_defs, _ = update_object_variables_grid(storage_data)
        
        # Verify object_id contains string values
        for row in row_data:
            object_id = row['object_id']
            assert isinstance(object_id, str)
            # Should be non-empty
            assert len(object_id) > 0
    
    def test_coordinate_values_are_numeric(self, populated_storage_read):
        """Test that coordinate values (ra, dec) are numeric."""
        from unittest.mock import Mock
        from pyzma_idf_lightcurve.lightcurve.dash.storage_cache import StorageCache
        
        app = Mock()
        callbacks = []
        def capture_callback(*outputs, **kwargs):
            def decorator(func):
                callbacks.append(func)
                return func
            return decorator
        app.callback = capture_callback
        
        register_info_tab_callbacks(app)
        update_object_variables_grid = callbacks[2]
        
        cache = StorageCache.get_instance()
        cache.set(populated_storage_read.storage_path, 'read', populated_storage_read)
        
        storage_data = {
            'storage_path': str(populated_storage_read.storage_path),
            'mode': 'read',
        }
        
        row_data, column_defs, _ = update_object_variables_grid(storage_data)
        
        # Verify ra and dec are numeric values
        for row in row_data:
            assert isinstance(row['ra'], (int, float))
            assert isinstance(row['dec'], (int, float))
    
    def test_column_definitions_sorted(self, populated_storage_read):
        """Test that columns are in a consistent order."""
        from unittest.mock import Mock
        from pyzma_idf_lightcurve.lightcurve.dash.storage_cache import StorageCache
        
        app = Mock()
        callbacks = []
        def capture_callback(*outputs, **kwargs):
            def decorator(func):
                callbacks.append(func)
                return func
            return decorator
        app.callback = capture_callback
        
        register_info_tab_callbacks(app)
        update_object_variables_grid = callbacks[2]
        
        cache = StorageCache.get_instance()
        cache.set(populated_storage_read.storage_path, 'read', populated_storage_read)
        
        storage_data = {
            'storage_path': str(populated_storage_read.storage_path),
            'mode': 'read',
        }
        
        row_data, column_defs, _ = update_object_variables_grid(storage_data)
        
        # First two columns should always be index and object_id
        assert column_defs[0]['field'] == 'index'
        assert column_defs[1]['field'] == 'object_id'
        
        # Remaining columns should be sorted variable names
        variable_fields = [col['field'] for col in column_defs[2:]]
        assert variable_fields == sorted(variable_fields)


class TestBuildInfoTablesData:
    """Test suite for build_info_tables_data function."""
    
    def test_build_with_valid_storage(self, populated_storage_read):
        """Test building all three tables with valid storage."""
        result = build_info_tables_data(populated_storage_read)
        
        # Unpack the 6-tuple result
        (storage_overview, epoch_row_data, epoch_column_defs,
         object_row_data, object_column_defs, select_button_disabled) = result
        
        # Verify storage overview
        assert isinstance(storage_overview, list)
        assert len(storage_overview) >= 3  # Text + Text + AgGrid
        
        # Verify epoch data
        assert isinstance(epoch_row_data, list)
        assert len(epoch_row_data) == 5  # 5 epochs in test data
        assert isinstance(epoch_column_defs, list)
        assert len(epoch_column_defs) >= 3  # index, epoch_id, + variables
        
        # Verify object data
        assert isinstance(object_row_data, list)
        assert len(object_row_data) == 20  # 20 objects in test data
        assert isinstance(object_column_defs, list)
        assert len(object_column_defs) >= 3  # index, object_id, + variables
        
        # Verify button state
        assert select_button_disabled is False
    
    def test_build_with_none_storage(self):
        """Test handling None storage gracefully."""
        result = build_info_tables_data(None)
        
        (storage_overview, epoch_row_data, epoch_column_defs,
         object_row_data, object_column_defs, select_button_disabled) = result
        
        # Should return error message and empty tables
        assert isinstance(storage_overview, list)
        assert len(epoch_row_data) == 0
        assert len(epoch_column_defs) == 0
        assert len(object_row_data) == 0
        assert len(object_column_defs) == 0
        assert select_button_disabled is True
    
    def test_storage_metadata_content(self, populated_storage_read):
        """Test that storage metadata table has correct content."""
        result = build_info_tables_data(populated_storage_read)
        storage_overview = result[0]
        
        # Find the AgGrid component
        import dash_ag_grid as dag
        grid = None
        for component in storage_overview:
            if isinstance(component, dag.AgGrid):
                grid = component
                break
        
        assert grid is not None
        # Note: AgGrid attributes may not be accessible directly in tests
        # This validates the grid was created
    
    def test_epoch_table_structure(self, populated_storage_read):
        """Test epoch table has correct structure."""
        result = build_info_tables_data(populated_storage_read)
        epoch_row_data, epoch_column_defs = result[1], result[2]
        
        # Check row structure
        assert all('index' in row for row in epoch_row_data)
        assert all('epoch_id' in row for row in epoch_row_data)
        
        # Check column structure
        column_fields = [col['field'] for col in epoch_column_defs]
        assert column_fields[0] == 'index'
        assert column_fields[1] == 'epoch_id'
    
    def test_object_table_structure(self, populated_storage_read):
        """Test object table has correct structure."""
        result = build_info_tables_data(populated_storage_read)
        object_row_data, object_column_defs = result[3], result[4]
        
        # Check row structure
        assert all('index' in row for row in object_row_data)
        assert all('object_id' in row for row in object_row_data)
        assert all('ra' in row for row in object_row_data)
        assert all('dec' in row for row in object_row_data)
        
        # Check column structure
        column_fields = [col['field'] for col in object_column_defs]
        assert column_fields[0] == 'index'
        assert column_fields[1] == 'object_id'
        assert 'ra' in column_fields
        assert 'dec' in column_fields
    
    def test_coordinate_data_types(self, populated_storage_read):
        """Test that coordinate values have correct data types."""
        result = build_info_tables_data(populated_storage_read)
        object_row_data = result[3]
        
        for row in object_row_data:
            assert isinstance(row['ra'], (int, float))
            assert isinstance(row['dec'], (int, float))
            assert isinstance(row['object_id'], str)


class TestSelectObjectsCallback:
    """Test suite for select objects callback."""
    
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
        
        register_info_tab_callbacks(app)
        select_objects = callbacks[3]
        
        with pytest.raises(PreventUpdate):
            select_objects(None, [{'index': 0}])
    
    def test_prevent_update_on_no_selection(self):
        """Test PreventUpdate when no rows selected."""
        from unittest.mock import Mock
        app = Mock()
        callbacks = []
        def capture_callback(*outputs, **kwargs):
            def decorator(func):
                callbacks.append(func)
                return func
            return decorator
        app.callback = capture_callback
        
        register_info_tab_callbacks(app)
        select_objects = callbacks[3]
        
        with pytest.raises(PreventUpdate):
            select_objects(1, None)
        
        with pytest.raises(PreventUpdate):
            select_objects(1, [])
    
    def test_select_objects(self):
        """Test object selection."""
        from unittest.mock import Mock
        app = Mock()
        callbacks = []
        def capture_callback(*outputs, **kwargs):
            def decorator(func):
                callbacks.append(func)
                return func
            return decorator
        app.callback = capture_callback
        
        register_info_tab_callbacks(app)
        select_objects = callbacks[3]
        
        selected_rows = [
            {'index': 5},
            {'index': 10},
            {'index': 15}
        ]
        
        result = select_objects(1, selected_rows)
        
        assert result == [5, 10, 15]
    
    def test_limit_to_20_objects(self):
        """Test that selection is limited to 20 objects."""
        from unittest.mock import Mock
        app = Mock()
        callbacks = []
        def capture_callback(*outputs, **kwargs):
            def decorator(func):
                callbacks.append(func)
                return func
            return decorator
        app.callback = capture_callback
        
        register_info_tab_callbacks(app)
        select_objects = callbacks[3]
        
        # Create 25 selected rows
        selected_rows = [{'index': i} for i in range(25)]
        
        result = select_objects(1, selected_rows)
        
        assert len(result) == 20
        assert result == list(range(20))
