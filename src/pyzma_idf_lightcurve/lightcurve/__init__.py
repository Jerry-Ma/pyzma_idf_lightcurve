"""
Core lightcurve functionality for IDF data analysis.

This package contains:
- binary: Binary blob storage optimization for lightcurves
- data: Data structures and database schemas
- dash: Visualization and interactive analysis tools
"""

from .binary import BinaryLightcurveFormat, BinaryLightcurveDatabase
from .data import LightcurveDatabase, LightcurveQueryEngine

# Optional dash imports (only if dependencies are available)
try:
    from .dash import LightcurveVisualizationApp
    _DASH_AVAILABLE = True
except ImportError:
    _DASH_AVAILABLE = False
    LightcurveVisualizationApp = None

__all__ = [
    "BinaryLightcurveFormat",
    "BinaryLightcurveDatabase", 
    "LightcurveDatabase",
    "LightcurveQueryEngine",
]

if _DASH_AVAILABLE:
    __all__.append("LightcurveVisualizationApp")