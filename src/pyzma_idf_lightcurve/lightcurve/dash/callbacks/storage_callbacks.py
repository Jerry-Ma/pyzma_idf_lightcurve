"""Callbacks for storage loading and management."""

from pathlib import Path
from dash import Input, Output, State, callback, no_update
from dash.exceptions import PreventUpdate
import dash_mantine_components as dmc

from ...datamodel import LightcurveStorage
from ..storage_cache import StorageCache


def register_storage_callbacks(app):
    """Register all storage-related callbacks."""
    
    @app.callback(
        [
            Output('storage-path-validation', 'children'),
            Output('storage-mode-select', 'data'),
            Output('storage-mode-select', 'disabled'),
            Output('load-storage-button', 'disabled'),
        ],
        Input('storage-path-input', 'value'),
        prevent_initial_call=False
    )
    def validate_storage_path(storage_path):
        """Validate storage path and check which storage files exist."""
        if not storage_path or not storage_path.strip():
            return (
                dmc.Text("Please enter a storage path", c="gray", size="sm", mt="xs"),
                [
                    {"value": "read", "label": "Read-Optimized (lightcurves_read.zarr)"},
                    {"value": "write", "label": "Write-Optimized (lightcurves_write.zarr)"},
                ],
                True,  # Disable mode selector
                True,  # Disable load button
            )
        
        try:
            path = Path(storage_path.strip())
            
            # Check if path exists
            if not path.exists():
                return (
                    dmc.Alert(
                        "❌ Storage path does not exist",
                        color="red",
                        variant="light",
                        withCloseButton=False,
                    ),
                    [
                        {"value": "read", "label": "Read-Optimized (not available)"},
                        {"value": "write", "label": "Write-Optimized (not available)"},
                    ],
                    True,
                    True,
                )
            
            # Check for read and write zarr files
            read_zarr = path / "lightcurves_read.zarr"
            write_zarr = path / "lightcurves_write.zarr"
            
            read_exists = read_zarr.exists() and read_zarr.is_dir()
            write_exists = write_zarr.exists() and write_zarr.is_dir()
            
            # Build mode options based on what exists
            # Note: dmc.Select doesn't support 'disabled' in data items
            # We'll use visual indicators (✓/✗) and disable the entire Select if needed
            mode_options = []
            if read_exists:
                mode_options.append({
                    "value": "read", 
                    "label": "✓ Read-Optimized (lightcurves_read.zarr)",
                })
            else:
                mode_options.append({
                    "value": "read", 
                    "label": "✗ Read-Optimized (not found)",
                })
            
            if write_exists:
                mode_options.append({
                    "value": "write", 
                    "label": "✓ Write-Optimized (lightcurves_write.zarr)",
                })
            else:
                mode_options.append({
                    "value": "write", 
                    "label": "✗ Write-Optimized (not found)",
                })
            
            # Determine feedback message and button state
            if read_exists and write_exists:
                feedback = dmc.Alert(
                    "✓ Both storage files found. Select a mode to load.",
                    color="green",
                    variant="light",
                    withCloseButton=False,
                )
                button_disabled = False
                mode_disabled = False
            elif read_exists or write_exists:
                which = "read" if read_exists else "write"
                feedback = dmc.Alert(
                    f"⚠️ Only {which}-optimized storage found. Select it to load.",
                    color="yellow",
                    variant="light",
                    withCloseButton=False,
                )
                button_disabled = False
                mode_disabled = False
            else:
                feedback = dmc.Alert(
                    "❌ No storage files found at this path. Create storage first or check the path.",
                    color="red",
                    variant="light",
                    withCloseButton=False,
                )
                button_disabled = True
                mode_disabled = True
            
            return feedback, mode_options, mode_disabled, button_disabled
            
        except Exception as e:
            return (
                dmc.Alert(
                    f"Error validating path: {str(e)}",
                    color="red",
                    variant="light",
                    withCloseButton=False,
                    mt="xs",
                ),
                [
                    {"value": "read", "label": "Read-Optimized (error)", "disabled": True},
                    {"value": "write", "label": "Write-Optimized (error)", "disabled": True},
                ],
                True,
                True,
            )
    
    _register_load_storage_callback(app)


def _register_load_storage_callback(app):
    """Register the load storage callback (internal function)."""
    
    @app.callback(
        [
            Output('storage-data', 'data'),
            Output('notifications-container', 'children'),
        ],
        Input('load-storage-button', 'n_clicks'),
        [
            State('storage-path-input', 'value'),
            State('storage-mode-select', 'value'),
        ],
        prevent_initial_call=True
    )
    def load_storage(n_clicks, storage_path, storage_mode):
        """Load zarr storage and extract metadata."""
        # Debug output
        print("\n" + "-"*80)
        print(f"load_storage callback triggered:")
        print(f"  n_clicks: {n_clicks}")
        print(f"  storage_path: {storage_path!r}")
        print(f"  storage_mode: {storage_mode!r}")
        print("-"*80 + "\n")
        
        if not n_clicks or not storage_path:
            print("PreventUpdate: n_clicks or storage_path is empty")
            raise PreventUpdate
        
        try:
            # Use storage_path as-is - let LightcurveStorage handle zarr path details
            storage_path_obj = Path(storage_path)
            
            print(f"Storage path: {storage_path_obj}")
            
            if not storage_path_obj.exists():
                error_msg = f"Storage path does not exist: {storage_path_obj}"
                print(f"ERROR: {error_msg}")
                return (
                    no_update,
                    dmc.Notification(
                        title="Error",
                        message=error_msg,
                        color="red",
                        action="show",
                    )
                )
            
            # Load storage - LightcurveStorage will handle zarr path construction
            print(f"Loading storage in {storage_mode} mode...")
            storage = LightcurveStorage(storage_path=storage_path_obj)
            
            if storage_mode == "read":
                storage.load_for_per_object_read()
                zarr_path = storage.zarr_path_for_read
            else:
                storage.load_for_per_epoch_write()
                zarr_path = storage.zarr_path_for_write
            
            # Cache the loaded storage object for reuse in other callbacks
            cache = StorageCache.get_instance()
            cache.set(storage_path_obj, storage_mode, storage)
            print(f"Storage loaded and cached: {type(storage)}")
            print(f"Zarr path: {zarr_path}")
            
            # Get dataset info for display
            ds = storage.dataset
            data_shape = dict(ds.sizes)
            print(f"Dataset info: shape={data_shape}")
            
            # Prepare data for dcc.Store (must be JSON serializable)
            # This is the cache key for retrieving storage in other callbacks
            storage_data = {
                'storage_path': str(storage_path_obj),
                'mode': storage_mode,
            }
            
            print(f"✓ Successfully loaded storage from {zarr_path}")
            print(f"  Data shape: {data_shape}")
            print(f"  Cache key: {storage_path_obj}:{storage_mode}")
            
            return (
                storage_data,
                dmc.Notification(
                    title="Success",
                    message=f"Loaded storage from {zarr_path}",
                    color="green",
                    action="show",
                )
            )
            
        except Exception as e:
            # Print full traceback for debugging
            import traceback
            print("\n" + "="*80)
            print("ERROR in load_storage callback:")
            print("="*80)
            traceback.print_exc()
            print("="*80 + "\n")
            
            return (
                no_update,
                dmc.Notification(
                    title="Error",
                    message=f"Failed to load storage: {str(e)}",
                    color="red",
                    action="show",
                )
            )
