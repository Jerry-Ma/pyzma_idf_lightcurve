"""Utility functions for image processing and normalization."""

import numpy as np
from astropy.io import fits
from astropy.visualization import (
    LinearStretch, LogStretch, SqrtStretch, AsinhStretch,
    PercentileInterval, ManualInterval, ImageNormalize
)


def load_fits_image(filepath, hdu_index=0):
    """Load FITS image data.
    
    Args:
        filepath: Path to FITS file
        hdu_index: HDU index to load (default: 0)
        
    Returns:
        tuple: (image_data, header)
    """
    with fits.open(filepath) as hdul:
        data = hdul[hdu_index].data
        header = hdul[hdu_index].header
    return data, header


def normalize_image(
    image_data,
    stretch='linear',
    vmin_percentile=1,
    vmax_percentile=99,
    vmin=None,
    vmax=None
):
    """Normalize image data for display.
    
    Args:
        image_data: 2D numpy array
        stretch: Stretch type ('linear', 'log', 'sqrt', 'asinh')
        vmin_percentile: Lower percentile for automatic scaling
        vmax_percentile: Upper percentile for automatic scaling  
        vmin: Manual minimum value (overrides percentile)
        vmax: Manual maximum value (overrides percentile)
        
    Returns:
        Normalized image array
    """
    # Select stretch function
    stretch_map = {
        'linear': LinearStretch(),
        'log': LogStretch(),
        'sqrt': SqrtStretch(),
        'asinh': AsinhStretch(),
    }
    stretch_func = stretch_map.get(stretch, LinearStretch())
    
    # Determine interval
    if vmin is not None and vmax is not None:
        interval = ManualInterval(vmin, vmax)
    else:
        interval = PercentileInterval(vmin_percentile, vmax_percentile)
    
    # Apply normalization
    norm = ImageNormalize(image_data, interval=interval, stretch=stretch_func)
    normalized = norm(image_data)
    
    return normalized


def get_image_extent(header):
    """Extract image extent from FITS header.
    
    Args:
        header: FITS header
        
    Returns:
        list: [xmin, xmax, ymin, ymax]
    """
    naxis1 = header.get('NAXIS1', 0)
    naxis2 = header.get('NAXIS2', 0)
    
    return [0, naxis1, 0, naxis2]


def pixel_to_world(x_pix, y_pix, header):
    """Convert pixel coordinates to world coordinates.
    
    Args:
        x_pix: X pixel coordinates
        y_pix: Y pixel coordinates
        header: FITS header with WCS info
        
    Returns:
        tuple: (ra, dec) in degrees
    """
    try:
        from astropy.wcs import WCS
        wcs = WCS(header)
        world = wcs.pixel_to_world(x_pix, y_pix)
        return world.ra.deg, world.dec.deg
    except Exception as e:
        print(f"Warning: Could not convert to world coordinates: {e}")
        return x_pix, y_pix


def world_to_pixel(ra, dec, header):
    """Convert world coordinates to pixel coordinates.
    
    Args:
        ra: Right ascension in degrees
        dec: Declination in degrees
        header: FITS header with WCS info
        
    Returns:
        tuple: (x_pix, y_pix)
    """
    try:
        from astropy.wcs import WCS
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        
        wcs = WCS(header)
        coord = SkyCoord(ra * u.deg, dec * u.deg)
        x_pix, y_pix = wcs.world_to_pixel(coord)
        return x_pix, y_pix
    except Exception as e:
        print(f"Warning: Could not convert to pixel coordinates: {e}")
        return ra, dec
