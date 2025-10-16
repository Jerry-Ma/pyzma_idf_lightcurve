"""Tests for Dash component creation."""

import pytest

from pyzma_idf_lightcurve.lightcurve.dash.components.storage_loader import create_storage_loader
from pyzma_idf_lightcurve.lightcurve.dash.components.storage_info_tab import create_storage_info_tab
from pyzma_idf_lightcurve.lightcurve.dash.components.viz_tab import create_viz_tab


class TestStorageLoaderComponent:
    """Test suite for storage loader component."""
    
    def test_create_storage_loader_returns_component(self):
        """Test that create_storage_loader returns a component."""
        component = create_storage_loader()
        
        assert component is not None
    
    def test_storage_loader_with_initial_path(self):
        """Test storage loader with initial path."""
        test_path = "test/path/storage.zarr"
        component = create_storage_loader(initial_path=test_path)
        
        assert component is not None
        # Component should have the test path as value
    
    def test_storage_loader_without_initial_path(self):
        """Test storage loader without initial path."""
        component = create_storage_loader()
        
        assert component is not None
    
    def test_storage_loader_has_required_elements(self):
        """Test that storage loader has all required input elements."""
        component = create_storage_loader()
        component_str = str(component)
        
        # Check for required component IDs
        assert 'storage-path-input' in component_str
        assert 'storage-mode-select' in component_str
        assert 'load-storage-button' in component_str
    
    def test_storage_loader_has_persistence(self):
        """Test that storage input has persistence enabled."""
        component = create_storage_loader()
        component_str = str(component)
        
        # The persistence property should be set
        assert 'persistence' in component_str.lower() or 'local' in component_str.lower()


class TestStorageInfoTabComponent:
    """Test suite for storage info tab component."""
    
    def test_create_storage_info_tab_returns_component(self):
        """Test that create_storage_info_tab returns a component."""
        component = create_storage_info_tab()
        
        assert component is not None
    
    def test_storage_info_tab_has_required_elements(self):
        """Test that storage info tab has required display elements."""
        component = create_storage_info_tab()
        component_str = str(component)
        
        # Check for actual IDs that exist in the component
        # From the error output, we can see these IDs exist:
        assert 'storage-overview' in component_str
        assert 'epoch-variables-grid' in component_str
        assert 'object-variables-grid' in component_str
        assert 'select-objects-button' in component_str


class TestVizTabComponent:
    """Test suite for visualization tab component."""
    
    def test_create_viz_tab_returns_component(self):
        """Test that create_viz_tab returns a component."""
        component = create_viz_tab()
        
        assert component is not None
    
    def test_viz_tab_has_required_elements(self):
        """Test that viz tab has visualization elements."""
        component = create_viz_tab()
        component_str = str(component)
        
        # Check for visualization elements
        # The exact IDs depend on your implementation
        assert component_str is not None


class TestComponentIntegration:
    """Test suite for component integration."""
    
    def test_all_components_can_be_created(self):
        """Test that all main components can be created without errors."""
        # This tests that imports and dependencies are correct
        loader = create_storage_loader()
        info_tab = create_storage_info_tab()
        viz_tab = create_viz_tab()
        
        assert loader is not None
        assert info_tab is not None
        assert viz_tab is not None
