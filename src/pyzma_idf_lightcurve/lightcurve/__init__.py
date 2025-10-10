"""
Core lightcurve functionality for IDF data analysis.

This package contains:
- datamodel: Main data structures for xarray-based lightcurve storage
- catalog: Source catalog management and operations
- dash: Visualization and interactive analysis tools (optional)
"""

from .datamodel import LightcurveStorage, SourceCatalog

# Optional dash imports (only if dependencies are available)
try:
    from .dash import LightcurveVisualizationApp
    _DASH_AVAILABLE = True
except ImportError:
    _DASH_AVAILABLE = False
    LightcurveVisualizationApp = None

__all__ = [
    "LightcurveStorage", 
    "SourceCatalog",
]

if _DASH_AVAILABLE:
    __all__.append("LightcurveVisualizationApp")