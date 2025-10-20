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
            Input('show-all-objects-toggle', 'checked'),
            Input('show-selected-objects-toggle', 'checked'),
            Input('marker-size-input', 'value'),
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
        show_all_objects,
        show_selected_objects,
        marker_size,
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
            
            # Add source markers if storage is loaded and toggles are on
            if storage_data and (show_all_objects or show_selected_objects):
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
                    
                    ds = storage.dataset
                    
                    # Debug: Print what we have
                    print(f"[DEBUG] Storage loaded for image overlay:")
                    print(f"  Dataset type: {type(ds)}")
                    print(f"  Available data_vars: {list(ds.data_vars.keys())}")
                    print(f"  show_all_objects: {show_all_objects}")
                    print(f"  show_selected_objects: {show_selected_objects}")
                    print(f"  selected_objects: {selected_objects}")
                    
                    # Get x_image and y_image coordinates for all objects
                    if 'object_x_image' in ds.data_vars and 'object_y_image' in ds.data_vars:
                        x_coords = ds['object_x_image'].values
                        y_coords = ds['object_y_image'].values
                        
                        print(f"[DEBUG] Object coordinates loaded:")
                        print(f"  X shape: {x_coords.shape}, range: [{x_coords.min():.1f}, {x_coords.max():.1f}]")
                        print(f"  Y shape: {y_coords.shape}, range: [{y_coords.min():.1f}, {y_coords.max():.1f}]")
                        
                        # Determine which objects to show
                        if show_all_objects and show_selected_objects:
                            # Show all objects, highlight selected
                            highlighted_indices = selected_objects if selected_objects else []
                            print(f"[DEBUG] Adding markers: ALL objects ({len(x_coords)}), highlighting {len(highlighted_indices)}")
                            fig = add_source_markers(
                                fig,
                                x_coords,
                                y_coords,
                                selected_indices=highlighted_indices,
                                marker_size=marker_size if marker_size else 8
                            )
                        elif show_all_objects:
                            # Show all objects without highlighting
                            print(f"[DEBUG] Adding markers: ALL objects ({len(x_coords)}), no highlighting")
                            fig = add_source_markers(
                                fig,
                                x_coords,
                                y_coords,
                                selected_indices=[],
                                marker_size=marker_size if marker_size else 8
                            )
                        elif show_selected_objects and selected_objects:
                            # Show only selected objects
                            selected_x = x_coords[selected_objects]
                            selected_y = y_coords[selected_objects]
                            print(f"[DEBUG] Adding markers: SELECTED objects only ({len(selected_objects)})")
                            # Mark all shown as selected
                            fig = add_source_markers(
                                fig,
                                selected_x,
                                selected_y,
                                selected_indices=list(range(len(selected_objects))),
                                marker_size=marker_size if marker_size else 8
                            )
                        else:
                            print(f"[DEBUG] No markers added (both toggles off or no selection)")
                    else:
                        print(f"[WARNING] Object position data not found in dataset!")
                        print(f"  Available data_vars: {list(ds.data_vars.keys())}")
                except Exception as e:
                    print(f"Warning: Could not add source markers: {e}")
                    import traceback
                    traceback.print_exc()
            
            return fig
            
        except Exception as e:
            print(f"Error loading image: {e}")
            import traceback
            traceback.print_exc()
            raise PreventUpdate
