"""Tests for image viewer callbacks."""

import pytest
import numpy as np
from pathlib import Path
from dash.exceptions import PreventUpdate
from astropy.io import fits

from pyzma_idf_lightcurve.lightcurve.dash.callbacks.image_callbacks import (
    register_image_callbacks,
)


@pytest.fixture
def sample_fits_file(temp_dir):
    """Create a sample FITS file for testing."""
    # Create a simple 100x100 image
    data = np.random.rand(100, 100).astype(np.float32)
    
    # Create FITS file with basic WCS header
    hdu = fits.PrimaryHDU(data)
    hdu.header['NAXIS'] = 2
    hdu.header['NAXIS1'] = 100
    hdu.header['NAXIS2'] = 100
    hdu.header['CRPIX1'] = 50.0
    hdu.header['CRPIX2'] = 50.0
    hdu.header['CRVAL1'] = 262.0  # IDF field RA
    hdu.header['CRVAL2'] = 58.0   # IDF field Dec
    hdu.header['CD1_1'] = -0.0003  # ~1 arcsec/pixel
    hdu.header['CD1_2'] = 0.0
    hdu.header['CD2_1'] = 0.0
    hdu.header['CD2_2'] = 0.0003
    hdu.header['CTYPE1'] = 'RA---TAN'
    hdu.header['CTYPE2'] = 'DEC--TAN'
    
    fits_path = temp_dir / "test_image.fits"
    hdu.writeto(fits_path, overwrite=True)
    
    return fits_path


class TestImageDisplayCallback:
    """Test suite for image display callback."""
    
    def test_prevent_update_on_missing_path(self):
        """Test PreventUpdate when image path is missing."""
        from unittest.mock import Mock
        app = Mock()
        callbacks = []
        def capture_callback(*outputs, **kwargs):
            def decorator(func):
                callbacks.append(func)
                return func
            return decorator
        app.callback = capture_callback
        
        register_image_callbacks(app)
        # Second callback is the display callback (first is validation)
        update_image_display = callbacks[1]
        
        with pytest.raises(PreventUpdate):
            update_image_display(None, 'linear', [1, 99], 'gray', None, None, None)
    
    def test_prevent_update_on_nonexistent_path(self, temp_dir):
        """Test PreventUpdate when image path doesn't exist."""
        from unittest.mock import Mock
        app = Mock()
        callbacks = []
        def capture_callback(*outputs, **kwargs):
            def decorator(func):
                callbacks.append(func)
                return func
            return decorator
        app.callback = capture_callback
        
        register_image_callbacks(app)
        # Second callback is the display callback (first is validation)
        update_image_display = callbacks[1]
        
        # Use a real path that doesn't exist
        nonexistent_path = str(temp_dir / 'fake' / 'path.fits')
        
        with pytest.raises(PreventUpdate):
            update_image_display(
                1, 'linear', [1, 99], 'gray', None, nonexistent_path, None
            )
    
    def test_successful_image_load_without_sources(self, sample_fits_file):
        """Test successful image loading without source overlay."""
        from unittest.mock import Mock
        # Register callback
        app = Mock()
        callbacks = []
        def capture_callback(*outputs, **kwargs):
            def decorator(func):
                callbacks.append(func)
                return func
            return decorator
        app.callback = capture_callback
        
        register_image_callbacks(app)
        # Second callback is the display callback (first is validation)
        update_image_display = callbacks[1]
        
        # Use real FITS file
        result = update_image_display(
            1, 'linear', [1, 99], 'gray', None, str(sample_fits_file), None
        )
        
        # Should return a figure (plotly figure object)
        assert result is not None
        # The result should have data attribute (plotly figure)
        assert hasattr(result, 'data') or hasattr(result, 'to_dict')
    
    def test_successful_image_load_with_sources(self, sample_fits_file, populated_storage_read):
        """Test successful image loading with source overlay."""
        from unittest.mock import Mock
        
        # Register callback
        app = Mock()
        callbacks = []
        def capture_callback(*outputs, **kwargs):
            def decorator(func):
                callbacks.append(func)
                return func
            return decorator
        app.callback = capture_callback
        
        register_image_callbacks(app)
        # Second callback is the display callback (first is validation)
        update_image_display = callbacks[1]
        
        # Use real storage and FITS file
        storage_data = {'path': str(populated_storage_read.storage_path / 'lightcurves_read.zarr')}
        selected_objects = [0, 1, 2]
        
        result = update_image_display(
            1, 'linear', [1, 99], 'gray', selected_objects, str(sample_fits_file), storage_data
        )
        
        # Should return figure with markers (plotly figure object)
        assert result is not None
        # The result should be a plotly figure
        assert hasattr(result, 'data') or hasattr(result, 'to_dict')
