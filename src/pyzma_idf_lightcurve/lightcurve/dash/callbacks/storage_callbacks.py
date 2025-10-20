"""Callbacks for storage loading and management."""

import time
from pathlib import Path
from dash import Input, Output, State, no_update
from dash.exceptions import PreventUpdate
import dash_mantine_components as dmc
from dash_iconify import DashIconify
import pandas as pd

from ...datamodel import LightcurveStorage
from ..app import get_storage_cache


def register_storage_callbacks(app):
    """Register all storage-related callbacks."""
    
    @app.callback(
        Output('storage-modal', 'opened'),
        [
            Input('open-storage-modal-button', 'n_clicks'),
            Input('storage-data', 'data'),
        ],
        State('storage-modal', 'opened'),
        prevent_initial_call=True
    )
    def toggle_storage_modal(open_clicks, storage_data, is_open):
        """Open/close storage modal. Close when storage is loaded."""
        from dash import ctx
        
        if not ctx.triggered:
            raise PreventUpdate
        
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if trigger_id == 'open-storage-modal-button':
            return True  # Open modal
        elif trigger_id == 'storage-data':
            return False  # Close modal after data loads
        
        return is_open
    
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
    
    _register_background_load_callback(app)


def _register_background_load_callback(app):
    """Register background callback for storage loading with native progress tracking."""
    
    @app.callback(
        output=[
            Output('storage-data', 'data'),
            Output('notifications-container', 'children'),
        ],
        inputs=Input('load-storage-button', 'n_clicks'),
        state=[
            State('storage-path-input', 'value'),
            State('storage-mode-select', 'value'),
        ],
        background=True,
        running=[
            # Disable button and show loading state while callback is running
            (Output('load-storage-button', 'disabled'), True, False),
            (Output('load-storage-button', 'loading'), True, False),
        ],
        progress=[
            Output('loading-progress', 'value'),
            Output('loading-status-text', 'children'),
            Output('loading-detail-text', 'children'),
        ],
        progress_default=[0, "", ""],
        prevent_initial_call=True,
    )
    def load_storage_background(set_progress, n_clicks, storage_path, storage_mode):
        """Load zarr storage in background with real-time progress updates.
        
        This uses Dash's native background callbacks for long-running tasks.
        Progress is tracked via set_progress() function provided by Dash.
        
        Args:
            set_progress: Function to update progress (value, status, detail)
            n_clicks: Number of times load button was clicked
            storage_path: Path to storage directory
            storage_mode: Either 'read' or 'write'
        
        Returns:
            Tuple of (storage_data dict, notification component)
        """
        if not n_clicks:
            raise PreventUpdate
        
        if not storage_path or not storage_path.strip():
            set_progress((0, "Error", "Storage path is required"))
            return (
                no_update,
                dmc.Notification(
                    title="Error",
                    message="Storage path is required",
                    color="red",
                    icon=DashIconify(icon="mdi:alert-circle"),
                    action="show",
                )
            )
        
        try:
            storage_path_obj = Path(storage_path.strip())
            
            # Stage 1: Validate path (10%)
            set_progress((10, "Validating path...", str(storage_path_obj)))
            time.sleep(0.1)  # Small delay for UI update
            
            if not storage_path_obj.exists():
                error_msg = f"Storage path does not exist: {storage_path_obj}"
                set_progress((0, "Error", error_msg))
                return (
                    no_update,
                    dmc.Notification(
                        title="Error",
                        message=error_msg,
                        color="red",
                        icon=DashIconify(icon="mdi:alert-circle"),
                        action="show",
                    )
                )
            
            # Stage 2: Initialize storage object (30%)
            set_progress((30, "Initializing storage object...", f"Mode: {storage_mode}"))
            storage = LightcurveStorage(storage_path=storage_path_obj)
            time.sleep(0.1)
            
            # Stage 3: Load dataset from zarr (60%)
            if storage_mode == "read":
                set_progress((60, "Loading dataset for reading...", "Per-object optimization"))
                storage.load_for_per_object_read()
                zarr_path = storage.zarr_path_for_read
            else:
                set_progress((60, "Loading dataset for writing...", "Per-epoch optimization"))
                storage.load_for_per_epoch_write()
                zarr_path = storage.zarr_path_for_write
            
            # Stage 4: Extract metadata (70%)
            set_progress((70, "Extracting metadata...", ""))
            ds = storage.dataset
            data_shape = dict(ds.sizes)
            n_objects = data_shape.get('object', 0)
            n_epochs = data_shape.get('epoch', 0)
            time.sleep(0.1)
            
            # Stage 4.5: Pre-compute DataFrames for info tables (80%)
            set_progress((80, "Pre-computing info tables...", "Loading epoch and object metadata"))
            
            # Try to load from Parquet files first (fast path), fall back to Zarr (slow path)
            epoch_df = None
            object_df = None
            
            try:
                # Fast path: Load from Parquet files (0.2s vs 235s for Zarr)
                print(f"[DEBUG] Attempting to load metadata from Parquet files...")
                epoch_df = storage.load_metadata_table('epoch')
                object_df = storage.load_metadata_table('object')
                print(f"[DEBUG] ✓ Loaded from Parquet (fast path)")
                print(f"[DEBUG]   Epoch DF: {len(epoch_df)} rows, {len(epoch_df.columns)} columns")
                print(f"[DEBUG]   Object DF: {len(object_df)} rows, {len(object_df.columns)} columns")
                
                # Try to load object_stats table and merge with object_df
                try:
                    # Try both naming conventions: object_stats.parquet and object_stats_table.parquet
                    stats_path = storage.storage_path / 'object_stats.parquet'
                    if not stats_path.exists():
                        # Fall back to _table suffix naming
                        object_stats_df = storage.load_metadata_table('object_stats')
                    else:
                        object_stats_df = pd.read_parquet(stats_path)
                        
                    print(f"[DEBUG]   Object Stats DF: {len(object_stats_df)} rows, {len(object_stats_df.columns)} columns")
                    
                    # Merge object_df with object_stats_df on 'object' column
                    # Use left join to keep all objects even if some don't have stats
                    object_df = object_df.merge(object_stats_df, on='object', how='left')
                    print(f"[DEBUG] ✓ Merged object_stats into object table")
                    print(f"[DEBUG]   Merged Object DF: {len(object_df)} rows, {len(object_df.columns)} columns")
                except FileNotFoundError:
                    print(f"[DEBUG] ⚠️  object_stats table not found - proceeding without stats columns")
                    print(f"[DEBUG]   Run object stats computation to enable filtering by n_epochs")
            except FileNotFoundError:
                # Slow path: Build from Zarr data variables (backward compatibility)
                print(f"[DEBUG] Parquet files not found, building from Zarr (slow path)...")
                set_progress((80, "Pre-computing info tables...", "Building from Zarr (this may take a few minutes)"))
                
                def _make_df_from_dim_vars(dim_name):
                    """Extract 1D variables along a dimension into a DataFrame."""
                    coord_values = ds.coords[dim_name].values
                    data = {dim_name: coord_values}
                    vars_dict = {}
                    for var_name in ds.coords:
                        vars_dict[var_name] = ds.coords[var_name]
                    for var_name in ds.data_vars:
                        vars_dict[var_name] = ds[var_name]
                    for var_name, var in vars_dict.items():
                        if dim_name in var.dims and len(var.dims) == 1:
                            data[var_name] = var.values
                    return pd.DataFrame(data)
                
                print(f"[DEBUG] Building epoch DataFrame from Zarr...")
                epoch_df = _make_df_from_dim_vars('epoch')
                print(f"[DEBUG]   Epoch DF: {len(epoch_df)} rows, {len(epoch_df.columns)} columns")
                
                print(f"[DEBUG] Building object DataFrame from Zarr...")
                object_df = _make_df_from_dim_vars('object')
                print(f"[DEBUG]   Object DF: {len(object_df)} rows, {len(object_df.columns)} columns")
                print(f"[DEBUG] ⚠️  Consider exporting metadata to Parquet for 1000x faster loading")
            
            time.sleep(0.1)
            
            # Stage 5: Verify Dask array optimization (85%)
            set_progress((85, "Verifying data access optimization...", "Checking Dask array configuration"))
            
            # Get the lightcurves DataArray
            lc_data_array = storage.lightcurves
            
            print(f"\n[DEBUG] Analyzing lightcurves data access:")
            print(f"  Array shape: {lc_data_array.shape}")
            print(f"  Array dtype: {lc_data_array.dtype}")
            print(f"  Array size: {lc_data_array.nbytes / (1024**3):.2f} GB")
            
            # Check if it's already a Dask array (which it should be from zarr loading)
            is_dask = hasattr(lc_data_array.data, 'dask')
            print(f"  Is Dask array: {is_dask}")
            
            if is_dask:
                # Already optimized with Dask! Just verify chunking is good
                chunks = lc_data_array.chunks
                print(f"  Current chunks: {chunks}")
                
                # Check if chunking looks good
                # For .sel() operations on objects, we want reasonable object chunks
                if chunks:
                    object_chunk_size = chunks[0][0] if isinstance(chunks[0], tuple) else chunks[0]
                    total_objects = lc_data_array.shape[0]
                    print(f"  Object dimension: {object_chunk_size} per chunk (total: {total_objects})")
                    
                    if object_chunk_size == total_objects:
                        print(f"  ⚠️  Warning: Single chunk for all objects - may cause memory issues")
                        print(f"  Consider rechunking the zarr store for better performance")
                    else:
                        print(f"  ✓ Good chunking detected - {total_objects // object_chunk_size} chunks for objects")
                
                print(f"\n  Storage is already Dask-optimized:")
                print(f"    ✓ Data stays on disk (zarr-backed)")
                print(f"    ✓ Automatic chunk caching by Dask")
                print(f"    ✓ Efficient .sel() operations")
                print(f"    ✓ Multiprocessing-safe access")
            else:
                print(f"  ⚠️  Not a Dask array - using direct zarr access")
                print(f"  Consider loading with dask chunks for better performance")
            
            time.sleep(0.1)
            
            # Stage 6: Cache storage and DataFrames in diskcache (90%)
            set_progress((90, "Caching storage and tables...", "Saving to disk cache"))
            cache = get_storage_cache()
            cache_key_storage = f"storage:{storage_path_obj}:{storage_mode}"
            cache_key_epoch_df = f"epoch_df:{storage_path_obj}:{storage_mode}"
            cache_key_object_df = f"object_df:{storage_path_obj}:{storage_mode}"
            
            print(f"\n[DEBUG] Background callback caching storage and DataFrames:")
            print(f"  storage_path_obj: {storage_path_obj} (type: {type(storage_path_obj)})")
            print(f"  storage_mode: {storage_mode}")
            print(f"  cache_key_storage: {cache_key_storage}")
            print(f"  cache_key_epoch_df: {cache_key_epoch_df}")
            print(f"  cache_key_object_df: {cache_key_object_df}")
            print(f"  cache directory: {cache.directory}")
            print(f"  storage type: {type(storage)}")
            print(f"  epoch_df shape: {epoch_df.shape}")
            print(f"  object_df shape: {object_df.shape}")
            
            cache.set(cache_key_storage, storage)
            cache.set(cache_key_epoch_df, epoch_df)
            cache.set(cache_key_object_df, object_df)
            
            # Verify caching
            test_storage = cache.get(cache_key_storage)
            test_epoch = cache.get(cache_key_epoch_df)
            test_object = cache.get(cache_key_object_df)
            print(f"  Cache verification:")
            print(f"    storage: {test_storage is not None}")
            print(f"    epoch_df: {test_epoch is not None} (shape: {test_epoch.shape if test_epoch is not None else 'N/A'})")
            print(f"    object_df: {test_object is not None} (shape: {test_object.shape if test_object is not None else 'N/A'})")
            print()
            
            time.sleep(0.1)
            
            # Stage 7: Complete (100%)
            set_progress((
                100,
                "✓ Loading complete!",
                f"Loaded {n_objects:,} objects × {n_epochs:,} epochs (Dask-optimized lazy access)"
            ))
            
            # Prepare storage data for dcc.Store (must be JSON serializable)
            storage_data = {
                'storage_path': str(storage_path_obj),
                'mode': storage_mode,
                'n_objects': n_objects,
                'n_epochs': n_epochs,
                'timestamp': time.time(),
            }
            
            return (
                storage_data,
                dmc.Notification(
                    title="✓ Storage Loaded Successfully",
                    message=f"Loaded {n_objects:,} objects × {n_epochs:,} epochs from {zarr_path.name}",
                    color="green",
                    icon=DashIconify(icon="mdi:check-circle"),
                    action="show",
                )
            )
            
        except Exception as e:
            # Print full traceback for debugging
            import traceback
            print("\n" + "="*80)
            print("ERROR in load_storage_background:")
            print("="*80)
            traceback.print_exc()
            print("="*80 + "\n")
            
            set_progress((0, "✗ Error occurred", str(e)))
            
            return (
                no_update,
                dmc.Notification(
                    title="✗ Loading Failed",
                    message=f"Failed to load storage: {str(e)}",
                    color="red",
                    icon=DashIconify(icon="mdi:alert-circle"),
                    action="show",
                )
            )
