"""Callbacks for image viewer functionality."""

from pathlib import Path
from dash import Input, Output, State, callback, no_update, html
from dash.exceptions import PreventUpdate
import dash_mantine_components as dmc

from ..utils.image_utils import load_fits_image, normalize_image, get_image_extent
from ..utils.plotting import create_image_figure, add_source_markers
from ...datamodel import LightcurveStorage
from ..storage_cache import StorageCache


def register_image_callbacks(app):
    """Register all image-related callbacks."""
    _register_image_path_validation(app)
    _register_image_display_callback(app)


def _register_image_path_validation(app):
    """Register callback for real-time image path validation."""
    
    @app.callback(
        [
            Output('image-path-validation', 'children'),
            Output('load-image-button', 'disabled'),
        ],
        Input('image-path-input', 'value'),
        prevent_initial_call=False
    )
    def validate_image_path(image_path):
        """Validate image path and provide feedback."""
        
        # Empty path
        if not image_path or not image_path.strip():
            return (
                dmc.Text("Please enter a FITS image path", c="gray", size="sm", mt="xs"),
                True
            )
        
        path = Path(image_path.strip())
        
        # Check file existence
        if not path.exists():
            return (
                dmc.Alert(
                    "❌ Image file does not exist",
                    color="red",
                    variant="light",
                    mt="xs"
                ),
                True
            )
        
        # Check if it's a file
        if not path.is_file():
            return (
                dmc.Alert(
                    "❌ Path is not a file",
                    color="red",
                    variant="light",
                    mt="xs"
                ),
                True
            )
        
        # Check file extension
        if not path.suffix.lower() in ['.fits', '.fit']:
            return (
                dmc.Alert(
                    "⚠️ File does not have .fits extension. May not be a valid FITS file.",
                    color="yellow",
                    variant="light",
                    mt="xs"
                ),
                False  # Allow loading anyway
            )
        
        # Try to get file size for feedback
        try:
            file_size = path.stat().st_size
            size_mb = file_size / (1024 * 1024)
            return (
                dmc.Alert(
                    f"✓ Valid FITS file ({size_mb:.1f} MB)",
                    color="green",
                    variant="light",
                    mt="xs"
                ),
                False
            )
        except Exception as e:
            return (
                dmc.Alert(
                    f"⚠️ File exists but cannot read metadata: {str(e)}",
                    color="yellow",
                    variant="light",
                    mt="xs"
                ),
                False
            )


def _register_image_display_callback(app):
    """Register callback for displaying images."""
    
    @app.callback(
        Output('image-display', 'figure'),
        [
            Input('load-image-button', 'n_clicks'),
            Input('image-norm-select', 'value'),
            Input('image-stretch-slider', 'value'),
            Input('image-cmap-select', 'value'),
            Input('selected-objects', 'data'),
        ],
        [
            State('image-path-input', 'value'),
            State('storage-data', 'data'),
        ],
        prevent_initial_call=True
    )
    def update_image_display(
        n_clicks,
        norm_type,
        stretch_percentiles,
        colormap,
        selected_objects,
        image_path,
        storage_data
    ):
        """Load and display FITS image with source overlay."""
        if not image_path or not Path(image_path).exists():
            raise PreventUpdate
        
        try:
            # Load FITS image
            image_data, header = load_fits_image(image_path)
            
            # Normalize image
            vmin_pct, vmax_pct = stretch_percentiles
            normalized_data = normalize_image(
                image_data,
                stretch=norm_type,
                vmin_percentile=vmin_pct,
                vmax_percentile=vmax_pct
            )
            
            # Create figure
            extent = get_image_extent(header)
            fig = create_image_figure(
                normalized_data,
                extent=extent,
                title=f"Image: {Path(image_path).name}",
                colorscale=colormap
            )
            
            # Add source markers if storage is loaded
            if storage_data and selected_objects:
                try:
                    # Retrieve storage from cache
                    storage_path = Path(storage_data.get('storage_path'))
                    mode = storage_data.get('mode', 'read')
                    
                    cache = StorageCache.get_instance()
                    storage = cache.get(storage_path, mode)
                    
                    if storage is None:
                        # Fallback: load if not in cache
                        print(f"[WARNING] Storage not in cache for image, loading: {storage_path}")
                        storage = LightcurveStorage(storage_path=storage_path)
                        storage.load_for_per_object_read()
                        cache.set(storage_path, mode, storage)
                    else:
                        print(f"[INFO] Retrieved storage from cache for image (mode={mode})")
                    
                    ds = storage.ds
                    
                    # Get x_image and y_image coordinates for all objects
                    if 'x_image' in ds.coords and 'y_image' in ds.coords:
                        x_coords = ds.coords['x_image'].values
                        y_coords = ds.coords['y_image'].values
                        
                        # Add markers
                        fig = add_source_markers(
                            fig,
                            x_coords,
                            y_coords,
                            selected_indices=selected_objects,
                            marker_size=10
                        )
                except Exception as e:
                    print(f"Warning: Could not add source markers: {e}")
            
            return fig
            
        except Exception as e:
            print(f"Error loading image: {e}")
            raise PreventUpdate
