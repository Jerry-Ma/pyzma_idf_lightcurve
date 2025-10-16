"""Callbacks for visualization tab."""

from dash import Input, Output, State, callback, no_update, html
from dash.exceptions import PreventUpdate
import dash_mantine_components as dmc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from ...datamodel import LightcurveStorage
from ..storage_cache import StorageCache


def register_viz_callbacks(app):
    """Register all visualization-related callbacks."""
    _register_viz_controls_callback(app)
    _register_query_validation_callback(app)
    _register_plot_generation_callback(app)


def _register_query_validation_callback(app):
    """Register callback for real-time query validation."""
    
    @app.callback(
        Output('query-validation', 'children'),
        [
            Input('query-input', 'value'),
            Input('storage-data', 'data'),
        ],
        prevent_initial_call=False
    )
    def validate_query(query, storage_data):
        """Validate pandas query syntax."""
        
        # Empty query is valid (no filtering)
        if not query or not query.strip():
            return dmc.Text("No filter applied", c="gray", size="sm", mt="xs")
        
        # No storage loaded - can't validate yet
        if not storage_data:
            return dmc.Alert(
                "⚠️ Load storage first to validate query",
                color="yellow",
                variant="light",
                mt="xs"
            )
        
        # Try to validate query syntax with a mock DataFrame
        try:
            # Create a simple mock DataFrame to test syntax
            # We can't test actual columns without loading full storage
            mock_df = pd.DataFrame({'x': [1, 2, 3]})
            mock_df.query(query.strip())
            
            return dmc.Alert(
                "✓ Query syntax is valid",
                color="green",
                variant="light",
                mt="xs"
            )
        except SyntaxError as e:
            return dmc.Alert(
                f"❌ Syntax error: {str(e)}",
                color="red",
                variant="light",
                mt="xs"
            )
        except Exception as e:
            return dmc.Alert(
                f"⚠️ Cannot validate: {str(e)}. Column names will be checked when data loads.",
                color="yellow",
                variant="light",
                mt="xs"
            )


def _register_viz_controls_callback(app):
    """Register callback for updating visualization controls."""
    
    @app.callback(
        [
            Output('object-keys-select', 'data'),
            Output('object-keys-select', 'error'),
            Output('measurement-keys-select', 'data'),
            Output('x-axis-select', 'data'),
            Output('generate-plots-button', 'disabled'),
        ],
        [
            Input('storage-data', 'data'),
        ],
        prevent_initial_call=True
    )
    def update_viz_controls(storage_data):
        """Update visualization control options from xarray coordinates."""
        if not storage_data:
            return [], "", [], [], True
        
        # Get storage from cache to access metadata
        from pathlib import Path
        from ...datamodel import LightcurveStorage
        from ..storage_cache import StorageCache
        
        storage_path = Path(storage_data['storage_path'])
        mode = storage_data.get('mode', 'read')
        cache = StorageCache.get_instance()
        storage = cache.get(storage_path, mode)
        
        if not storage or not storage.lightcurves:
            return [], "", [], [], True
        
        lc_var = storage.lightcurves
        
        # Object keys - extract from 'object' coordinate (vectorized access)
        print("[DEBUG] Extracting object coordinate values...")
        if 'object' in lc_var.coords:
            object_values = lc_var.coords['object'].values
            object_options = [
                {'value': str(obj_key), 'label': str(obj_key)}
                for obj_key in object_values
            ]
            print(f"[DEBUG] Found {len(object_options)} objects")
        else:
            object_options = []
            print("[DEBUG] No 'object' coordinate found")
        
        # Measurement keys - extract from 'measurement' coordinate (vectorized access)
        print("[DEBUG] Extracting measurement coordinate values...")
        if 'measurement' in lc_var.coords:
            measurement_values = lc_var.coords['measurement'].values
            measurement_options = [
                {'value': str(meas_key), 'label': str(meas_key)}
                for meas_key in measurement_values
            ]
            print(f"[DEBUG] Found {len(measurement_options)} measurements")
        else:
            measurement_options = []
            print("[DEBUG] No 'measurement' coordinate found")
        
        # X-axis options - look for 1D epoch-related coordinates
        # Since lc_var is a DataArray, we only look at its coords
        print("[DEBUG] Scanning for epoch coordinates...")
        x_axis_options = []
        for coord in lc_var.coords:
            if coord == 'epoch':
                # Add epoch itself as option
                x_axis_options.append({'value': coord, 'label': coord})
            elif 'epoch' in lc_var.coords[coord].dims and len(lc_var.coords[coord].dims) == 1:
                x_axis_options.append({'value': coord, 'label': coord})
        print(f"[DEBUG] Found {len(x_axis_options)} epoch coordinates for x-axis")
        
        # Enable button if we have data
        disabled = len(object_options) == 0
        
        return object_options, "", measurement_options, x_axis_options, disabled


def _register_plot_generation_callback(app):
    """Register callback for generating lightcurve plots."""
    
    @app.callback(
        Output('lightcurve-plot', 'figure'),
        Input('generate-plots-button', 'n_clicks'),
        [
            State('storage-data', 'data'),
            State('object-keys-select', 'value'),
            State('measurement-keys-select', 'value'),
            State('x-axis-select', 'value'),
            State('query-input', 'value'),
        ],
        prevent_initial_call=True
    )
    def generate_lightcurve_plots(n_clicks, storage_data, object_keys, measurement_keys, x_axis_var, query):
        """Generate lightcurve plots for selected objects."""
        if not n_clicks or not storage_data or not object_keys or not measurement_keys:
            raise PreventUpdate
        
        try:
            # Retrieve storage from cache
            from pathlib import Path
            storage_path = Path(storage_data.get('storage_path'))
            mode = storage_data.get('mode', 'read')
            
            cache = StorageCache.get_instance()
            storage = cache.get(storage_path, mode)
            
            if storage is None:
                # Fallback: load if not in cache
                print(f"[WARNING] Storage not in cache for viz, loading: {storage_path}")
                storage = LightcurveStorage(storage_path=storage_path)
                storage.load_for_per_object_read()
                cache.set(storage_path, mode, storage)
            
            # Limit objects to 20
            # object_keys and measurement_keys are now actual coordinate values (strings), not indices
            object_keys_list = object_keys[:20]
            measurement_keys_list = measurement_keys
            
            # Get lightcurves DataArray for coordinate-based indexing
            lc_var = storage.lightcurves
            if lc_var is None:
                raise PreventUpdate
            
            # Create subplots
            n_objects = len(object_keys_list)
            n_cols = min(2, n_objects)
            n_rows = (n_objects + n_cols - 1) // n_cols
            
            fig = make_subplots(
                rows=n_rows,
                cols=n_cols,
                subplot_titles=[f'Object {obj_key}' for obj_key in object_keys_list],
                vertical_spacing=0.15 / n_rows if n_rows > 1 else 0.1,
                horizontal_spacing=0.1
            )
            
            # Color palette for measurements
            colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
            
            # Plot each object
            for obj_idx, object_key in enumerate(object_keys_list):
                row = (obj_idx // n_cols) + 1
                col = (obj_idx % n_cols) + 1
                
                # Select data for this object using xarray label-based indexing
                # This follows the pattern from get_object_lightcurve in datamodel.py
                lc_data = lc_var.sel(object=object_key)
                
                # Get x-axis data from dataset coordinates
                # Access epoch coordinate from the lightcurves DataArray
                if x_axis_var == 'epoch':
                    x_data = lc_var.coords['epoch'].values
                elif x_axis_var in lc_var.coords:
                    x_data = lc_var.coords[x_axis_var].values
                else:
                    # Fallback: use epoch indices
                    import numpy as np
                    x_data = np.arange(len(lc_var.coords['epoch']))
                
                # Plot each measurement
                for meas_idx, meas_key in enumerate(measurement_keys_list):
                    color = colors[meas_idx % len(colors)]
                    
                    # Extract magnitude and error using coordinate-based selection
                    # Assuming value dimension has coordinates like ['mag', 'mag_err']
                    # Use .sel() for coordinate-based indexing
                    import numpy as np
                    
                    try:
                        # Select this measurement and extract mag and mag_err
                        # lc_data is already the object's lightcurve data
                        meas_data = lc_data.sel(measurement=meas_key)
                        
                        # Try coordinate-based selection for value dimension
                        if 'mag' in lc_var.coords.get('value', []):
                            mag_data = meas_data.sel(value='mag').values
                            mag_err_data = meas_data.sel(value='mag_err').values
                        else:
                            # Fallback to index-based if coordinates not set properly
                            mag_data = meas_data.isel(value=0).values
                            mag_err_data = meas_data.isel(value=1).values
                        
                        # Filter out NaN values
                        mask = ~np.isnan(mag_data)
                        
                        if mask.sum() == 0:
                            continue
                        
                        fig.add_trace(
                            go.Scatter(
                                x=x_data[mask],
                                y=mag_data[mask],
                                error_y=dict(
                                    type='data',
                                    array=mag_err_data[mask],
                                    visible=True
                                ),
                                mode='markers',
                                marker=dict(size=4, color=color),
                                name=f'{meas_key}',
                                showlegend=(obj_idx == 0),
                                legendgroup=f'meas{meas_idx}',
                            ),
                            row=row,
                            col=col
                        )
                    except Exception as e:
                        print(f"[WARNING] Failed to plot measurement {meas_key} for object {object_key}: {e}")
                
                # Invert y-axis (magnitude scale)
                fig.update_yaxes(autorange="reversed", row=row, col=col)
            
            # Update layout
            fig.update_layout(
                height=400 * n_rows,
                showlegend=True,
                hovermode='closest',
                title_text="Lightcurves",
            )
            
            # Update axis labels
            fig.update_xaxes(title_text=x_axis_var.replace('epoch_', ''))
            fig.update_yaxes(title_text="Magnitude")
            
            return fig
            
        except Exception as e:
            print(f"Error generating plots: {e}")
            import traceback
            traceback.print_exc()
            raise PreventUpdate
