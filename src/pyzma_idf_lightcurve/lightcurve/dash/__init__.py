"""
Interactive visualization tools using Plotly Dash for lightcurve analysis.

This module provides a modern Dash application for exploring IDF lightcurve data
stored in xarray/zarr format.
"""

from .app import create_app, run_app

__all__ = [
    "create_app",
    "run_app",
]