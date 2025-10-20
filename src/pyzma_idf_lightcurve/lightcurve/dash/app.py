"""Modern Dash v3 application for IDF Lightcurve visualization."""

import time
import logging
from pathlib import Path
import diskcache
import dash
from dash import html, dcc, DiskcacheManager
import dash_mantine_components as dmc

from .components.storage_loader import create_storage_loader
from .components.storage_info_tab import create_storage_info_tab
from .components.viz_tab import create_viz_tab
from .components.object_stats_tab import create_object_stats_tab
from .components.image_tab import create_image_tab

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# Module-level cache for storage objects (shared across processes via disk)
_storage_cache = None

def get_storage_cache():
    """Get the shared storage cache instance."""
    global _storage_cache
    if _storage_cache is None:
        cache_dir = Path.cwd() / ".dash_cache"
        cache_dir.mkdir(exist_ok=True)
        _storage_cache = diskcache.Cache(str(cache_dir))
    return _storage_cache

def create_app(initial_storage_path=None):
    """Create and configure the Dash v3 application.
    
    Dash v3 automatically handles React 18.2.0+ configuration.
    No need to manually set React version.
    
    Args:
        initial_storage_path: Optional initial storage path to pre-populate the input field
    """
    start_time = time.time()
    logger.info("Creating Dash application...")
    
    # Setup background callback manager for long-running tasks
    cache = get_storage_cache()
    background_callback_manager = DiskcacheManager(cache)
    logger.info(f"Background callback cache: {cache.directory}")
    
    app = dash.Dash(
        __name__,
        external_stylesheets=dmc.styles.ALL,
        suppress_callback_exceptions=True,
        title="IDF Lightcurve Viewer",
        background_callback_manager=background_callback_manager,
    )
    logger.info(f"Dash app initialized ({time.time() - start_time:.2f}s)")
    
    layout_start = time.time()
    logger.info("Building layout...")
    app.layout = dmc.MantineProvider([
        dcc.Store(id='storage-data', storage_type='memory'),
        dcc.Store(id='selected-objects', storage_type='memory', data=[]),
        
        dmc.Container([
            dmc.Group([
                html.Div([
                    dmc.Title("IDF Lightcurve Viewer", order=1, mb=0),
                    dmc.Text("Visualize lightcurves from the Infrared Deep Field",
                            c="gray", size="sm"),
                ]),
                dmc.Button(
                    "📂 Load Storage",
                    id="open-storage-modal-button",
                    variant="filled",
                    color="blue",
                    size="lg",
                ),
            ], justify="space-between", align="center", mb="xl"),
        ], fluid=True, p="md"),
        
        # Storage loader modal dialog
        dmc.Modal(
            id="storage-modal",
            title="Load Storage",
            children=create_storage_loader(initial_storage_path),
            size="xl",
            zIndex=10000,
        ),
        
        dmc.Container([
            dmc.Tabs([
                dmc.TabsList([
                    dmc.TabsTab("Visualization", value="viz"),
                    dmc.TabsTab("Object Stats", value="stats"),
                    dmc.TabsTab("Image View", value="image"),
                    dmc.TabsTab("Storage Info", value="info"),
                ]),
                dmc.TabsPanel(create_viz_tab(), value="viz"),
                dmc.TabsPanel(create_object_stats_tab(), value="stats"),
                dmc.TabsPanel(create_image_tab(), value="image"),
                dmc.TabsPanel(create_storage_info_tab(), value="info"),
            ], id="main-tabs", value="viz")
        ], fluid=True, px="md"),
        
        html.Div(id='notifications-container'),
    ])
    logger.info(f"Layout built ({time.time() - layout_start:.2f}s)")
    
    callbacks_start = time.time()
    logger.info("Registering callbacks...")
    from . import callbacks
    callbacks.register_callbacks(app)
    logger.info(f"Callbacks registered ({time.time() - callbacks_start:.2f}s)")
    
    logger.info(f"App creation complete ({time.time() - start_time:.2f}s total)")
    return app


def run_app(debug=True, port=8050, host='0.0.0.0', storage_path=None):
    """Run the Dash application.
    
    Args:
        debug: Enable debug mode with hot reload
        port: Port to run the server on
        host: Host to bind to (default: 0.0.0.0 for all interfaces)
        storage_path: Optional initial storage path to pre-populate the input field
    
    Note: In Dash v3, app.run_server() was replaced by app.run()
    """
    logger.info(f"Starting app on {host}:{port} (debug={debug})")
    app = create_app(initial_storage_path=storage_path)
    
    # Configure Flask's logger to show at INFO level
    if debug:
        app.logger.setLevel(logging.INFO)
        logging.getLogger('werkzeug').setLevel(logging.INFO)
    
    logger.info("Starting server...")
    app.run(debug=debug, port=port, host=host)


if __name__ == "__main__":
    run_app()
