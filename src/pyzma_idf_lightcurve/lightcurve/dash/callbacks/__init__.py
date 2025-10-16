"""Callback registration for the Dash app."""

from .storage_callbacks import register_storage_callbacks
from .info_tab_callbacks import register_info_tab_callbacks
from .viz_callbacks import register_viz_callbacks
from .image_callbacks import register_image_callbacks


def register_callbacks(app):
    """Register all callbacks with the Dash app.
    
    Args:
        app: Dash application instance
    """
    register_storage_callbacks(app)
    register_info_tab_callbacks(app)
    register_viz_callbacks(app)
    register_image_callbacks(app)
