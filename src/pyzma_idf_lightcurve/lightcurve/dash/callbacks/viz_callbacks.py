"""Callbacks for visualization tab."""

from dash import Input, Output, State, callback, no_update, html, clientside_callback
from dash.exceptions import PreventUpdate
import dash_mantine_components as dmc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import logging
import time
import random

from ...datamodel import LightcurveStorage
from ..app import get_storage_cache

# Configure logging
logger = logging.getLogger(__name__)


def register_viz_callbacks(app):
    """Register all visualization-related callbacks."""
    cache = get_storage_cache()
    _register_viz_controls_callback(app)
    _register_query_validation_callback(app)
    _register_query_history_callback(app)
    _register_random_objects_callback(app, cache)
    _register_plot_generation_callback(app, cache)


def _register_query_history_callback(app):
    """Register callback to manage query history with localStorage."""
    
    # Clientside callback to load query history from localStorage
    app.clientside_callback(
        """
        function(n_intervals) {
            // Load query history from localStorage
            const historyJson = localStorage.getItem('queryHistory');
            if (historyJson) {
                const history = JSON.parse(historyJson);
                // Return as options for Select component
                return history.map(q => ({value: q, label: q}));
            }
            return [];
        }
        """,
        Output('query-history-select', 'data'),
        Input('query-history-select', 'id'),  # Trigger on mount
    )
    
    # Callback to update query input from history selection
    @app.callback(
        Output('object-query-input', 'value', allow_duplicate=True),
        Input('query-history-select', 'value'),
        prevent_initial_call=True,
    )
    def load_query_from_history(selected_query):
        """Load selected query from history into input field."""
        if selected_query:
            logger.info(f"Loading query from history: {selected_query}")
            return selected_query
        return no_update
    
    # Clientside callback to save query to localStorage when plots are generated
    app.clientside_callback(
        """
        function(n_clicks, query) {
            if (n_clicks && query && query.trim()) {
                // Load existing history
                let history = [];
                const historyJson = localStorage.getItem('queryHistory');
                if (historyJson) {
                    history = JSON.parse(historyJson);
                }
                
                // Add new query if not already in history
                if (!history.includes(query)) {
                    history.unshift(query);  // Add to front
                    // Keep only last 20 queries
                    if (history.length > 20) {
                        history = history.slice(0, 20);
                    }
                    localStorage.setItem('queryHistory', JSON.stringify(history));
                }
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output('object-query-input', 'id'),  # Dummy output
        Input('generate-plots-button', 'n_clicks'),
        State('object-query-input', 'value'),
        prevent_initial_call=True,
    )


def _register_random_objects_callback(app, cache):
    """Register callback to generate random object selection."""
    
    @app.callback(
        Output('object-query-input', 'value', allow_duplicate=True),
        Input('random-objects-button', 'n_clicks'),
        State('storage-data', 'data'),
        prevent_initial_call=True,
    )
    def generate_random_objects(n_clicks, storage_data):
        """Generate query string with 5 random object IDs."""
        if not n_clicks or not storage_data:
            raise PreventUpdate
        
        # Build cache key from storage_data
        from pathlib import Path
        storage_path = Path(storage_data['storage_path'])
        mode = storage_data.get('mode', 'read')
        path_mode = f"{storage_path}:{mode}"
        
        # Get object_df from cache
        object_df = cache.get(f"object_df:{path_mode}")
        if object_df is None:
            logger.warning("No object data in cache for random selection")
            raise PreventUpdate
        
        # Get all object IDs
        all_objects = object_df['object'].tolist()
        
        # Select 5 random objects (or fewer if less than 5 available)
        n_random = min(5, len(all_objects))
        random_objects = random.sample(all_objects, n_random)
        
        # Determine if numeric or string IDs
        try:
            float(random_objects[0])
            is_numeric = True
        except (ValueError, TypeError):
            is_numeric = False
        
        # Generate query string using 'in' operator
        if is_numeric:
            query_string = f"object in {random_objects}"
        else:
            # Use double quotes for consistency
            quoted_ids = [f'"{obj}"' for obj in random_objects]
            query_string = f"object in [{', '.join(quoted_ids)}]"
        
        logger.info(f"Generated random query with {n_random} objects: {query_string}")
        return query_string


def _register_query_validation_callback(app):
    """Register callback for real-time object query validation."""
    
    @app.callback(
        Output('object-query-validation', 'children'),
        [
            Input('object-query-input', 'value'),
            Input('storage-data', 'data'),
        ],
        prevent_initial_call=False
    )
    def validate_object_query(query, storage_data):
        """Validate pandas query syntax for object selection."""
        
        # Empty query is valid (plot all objects)
        if not query or not query.strip():
            return dmc.Text("No filter - will plot first N objects", c="gray", size="sm", mt="xs")
        
        # No storage loaded - can't validate column names yet
        if not storage_data:
            return dmc.Alert(
                "⚠️ Load storage first to validate query",
                color="yellow",
                variant="light",
                mt="xs"
            )
        
        # Try to validate query syntax
        try:
            # Get object DataFrame from cache to validate against real columns
            from pathlib import Path
            storage_path = Path(storage_data['storage_path'])
            mode = storage_data.get('mode', 'read')
            
            cache = get_storage_cache()
            path_mode = f"{storage_path}:{mode}"
            df_cache_key = f"object_df:{path_mode}"
            object_df = cache.get(df_cache_key)
            
            if object_df is not None:
                # Try to apply query to actual DataFrame
                result = object_df.query(query.strip())
                n_matched = len(result)
                return dmc.Alert(
                    f"✓ Query valid: matches {n_matched} objects",
                    color="green",
                    variant="light",
                    mt="xs"
                )
            else:
                # Fallback: basic syntax validation
                mock_df = pd.DataFrame({'object': [1, 2, 3], 'ra': [180, 181, 182], 'dec': [30, 31, 32]})
                mock_df.query(query.strip())
                return dmc.Alert(
                    "✓ Query syntax is valid (columns will be checked at plot time)",
                    color="blue",
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
                f"⚠️ Error: {str(e)}",
                color="yellow",
                variant="light",
                mt="xs"
            )


def _register_viz_controls_callback(app):
    """Register callback for updating visualization controls."""
    
    @app.callback(
        [
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
            return [], [], True
        
        # Get storage from cache to access metadata
        from pathlib import Path
        
        storage_path = Path(storage_data['storage_path'])
        mode = storage_data.get('mode', 'read')
        
        # Get cached storage from diskcache (shared across processes)
        cache = get_storage_cache()
        cache_key = f"storage:{storage_path}:{mode}"
        storage = cache.get(cache_key)
        
        if storage is None or storage.lightcurves is None:
            return [], [], True
        
        lc_var = storage.lightcurves
        
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
        disabled = False
        
        return measurement_options, x_axis_options, disabled


def _register_plot_generation_callback(app, cache):
    """Register callback for generating lightcurve plots."""
    
    @app.callback(
        Output('lightcurve-plot', 'figure'),
        Output('lightcurve-plot', 'style'),
        Input('generate-plots-button', 'n_clicks'),
        State('storage-data', 'data'),
        State('object-query-input', 'value'),
        State('max-objects-input', 'value'),
        State('measurement-keys-select', 'value'),
        State('x-axis-select', 'value'),
        prevent_initial_call=True,
    )
    def generate_lightcurve_plots(
        n_clicks,
        storage_data,
        object_query,
        max_objects,
        measurement_keys,
        x_axis_key,
    ):
        """Generate lightcurve plots based on current selections.
        
        Args:
            n_clicks: Number of times generate button clicked
            storage_data: Storage metadata dict with path and mode
            object_query: Query string for filtering objects
            max_objects: Maximum number of objects to plot
            measurement_keys: List of measurement columns to plot
            x_axis_key: Column to use for x-axis
            
        Returns:
            tuple: (plotly figure, style dict for container)
        """
        start_time = time.time()
        logger.info("="*80)
        logger.info("🚀 PLOT GENERATION STARTED")
        logger.info(f"   Query: {object_query!r}")
        logger.info(f"   Max objects: {max_objects}")
        logger.info(f"   Measurements: {measurement_keys}")
        logger.info(f"   X-axis: {x_axis_key}")
        
        if not n_clicks or not storage_data:
            raise PreventUpdate
        
        # Build cache key from storage_data
        from pathlib import Path
        storage_path = Path(storage_data['storage_path'])
        mode = storage_data.get('mode', 'read')
        path_mode = f"{storage_path}:{mode}"
        logger.info(f"   Cache key: {path_mode}")

        # Get data from cache with timing
        t0 = time.time()
        object_df = cache.get(f"object_df:{path_mode}")
        t1 = time.time()
        logger.info(f"⏱️  Loaded object_df from cache: {t1-t0:.4f}s (shape: {object_df.shape if object_df is not None else 'N/A'})")
        
        epoch_df = cache.get(f"epoch_df:{path_mode}")
        t2 = time.time()
        logger.info(f"⏱️  Loaded epoch_df from cache: {t2-t1:.4f}s (shape: {epoch_df.shape if epoch_df is not None else 'N/A'})")
        
        storage = cache.get(f"storage:{path_mode}")
        t3 = time.time()
        logger.info(f"⏱️  Loaded storage from cache: {t3-t2:.4f}s")
        logger.info(f"📊 Total data loading time: {t3-t0:.4f}s")
        
        # Validate data loaded
        if object_df is None or epoch_df is None or storage is None:
            logger.error("❌ Required data not found in cache")
            raise PreventUpdate
        
        # Filter objects by query
        t4 = time.time()
        if object_query and object_query.strip():
            try:
                filtered_df = object_df.query(object_query.strip())
                logger.info(f"⏱️  Applied query filter: {time.time()-t4:.4f}s (matched {len(filtered_df)} objects)")
            except Exception as e:
                logger.error(f"❌ Query filter failed: {e}")
                filtered_df = object_df
        else:
            filtered_df = object_df
            logger.info("   No query filter applied")
        
        # Limit to max_objects
        t5 = time.time()
        max_objects = max_objects or 20
        object_keys_list = filtered_df['object'].head(max_objects).tolist()
        logger.info(f"⏱️  Selected {len(object_keys_list)} objects (limit: {max_objects}): {time.time()-t5:.4f}s")
        
        if not object_keys_list:
            logger.warning("⚠️  No objects to plot")
            empty_fig = go.Figure()
            empty_fig.add_annotation(
                text="No objects match the query",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20, color="gray")
            )
            return empty_fig, {"height": "400px"}
        
        # Validate measurement keys
        if not measurement_keys:
            logger.warning("⚠️  No measurement keys selected")
            raise PreventUpdate
        
        # Create plots
        t6 = time.time()
        logger.info(f"🎨 Creating plots for {len(object_keys_list)} objects x {len(measurement_keys)} measurements...")
        
        try:
            # Get lightcurve data
            lc_var = storage.lightcurves
            
            # Create subplot figure
            n_rows = len(object_keys_list)
            n_cols = len(measurement_keys)
            
            fig = make_subplots(
                rows=n_rows,
                cols=n_cols,
                subplot_titles=[f"Object {obj} - {meas}" for obj in object_keys_list for meas in measurement_keys],
                vertical_spacing=0.05,
                horizontal_spacing=0.05
            )
            
            # Plot each object x measurement combination
            t_plot_start = time.time()
            traces_added = 0
            traces_failed = 0
            logger.info(f"🔄 Starting subplot iteration for {n_rows} × {n_cols} = {n_rows*n_cols} subplots...")
            for i, obj_key in enumerate(object_keys_list):
                for j, meas_key in enumerate(measurement_keys):
                    try:
                        t_trace_start = time.time()
                        logger.debug(f"   [{i},{j}] Loading {obj_key} - {meas_key}...")
                        # Load data for this object and measurement
                        obj_data = lc_var.sel(object=obj_key, measurement=meas_key)
                        
                        logger.debug(f"   [{i},{j}] Extracting values...")
                        # Get x and y data
                        if x_axis_key and x_axis_key in obj_data.coords:
                            x_data = obj_data.coords[x_axis_key].values
                        else:
                            x_data = obj_data.coords['epoch'].values
                        
                        # y_data has shape (2, n_epochs) where dim 0 is [value, uncertainty]
                        y_values_full = obj_data.values
                        logger.info(f"   [{i},{j}] Data shapes - x_data: {x_data.shape}, y_values_full: {y_values_full.shape}")
                        
                        if y_values_full.ndim == 2 and y_values_full.shape[0] == 2:
                            y_data = y_values_full[0, :]  # Values (first row)
                            y_error = y_values_full[1, :]  # Uncertainties (second row)
                            logger.info(f"   [{i},{j}] Split y_data - values: {y_data.shape}, errors: {y_error.shape}")
                            
                            # Filter data for magnitude measurements
                            if 'mag' in meas_key.lower():
                                import numpy as np
                                # Keep only valid magnitude points: mag < 90, not nan, and magerr < 1
                                valid_mask = (y_data < 90) & np.isfinite(y_data) & (y_error < 1)
                                x_data = x_data[valid_mask]
                                y_data = y_data[valid_mask]
                                y_error = y_error[valid_mask]
                                logger.info(f"   [{i},{j}] Magnitude filtering - kept {np.sum(valid_mask)}/{len(valid_mask)} points")
                        else:
                            # Fallback for unexpected shape
                            logger.warning(f"   [{i},{j}] Unexpected y_values shape: {y_values_full.shape}, using as-is")
                            y_data = y_values_full
                            y_error = None
                        
                        logger.debug(f"   [{i},{j}] Adding trace to subplot...")
                        # Add trace with error bars if available
                        fig.add_trace(
                            go.Scatter(
                                x=x_data,
                                y=y_data,
                                error_y=dict(type='data', array=y_error) if y_error is not None else None,
                                mode='markers+lines',
                                name=f"{obj_key} - {meas_key}",
                                showlegend=False
                            ),
                            row=i+1,
                            col=j+1
                        )
                        traces_added += 1
                        t_trace_end = time.time()
                        logger.debug(f"   [{i},{j}] Trace added in {t_trace_end-t_trace_start:.4f}s")
                    except Exception as e:
                        traces_failed += 1
                        logger.warning(f"   Failed to plot {obj_key} - {meas_key}: {e}")
                        import traceback
                        logger.warning(traceback.format_exc())
            
            t_plot_end = time.time()
            logger.info(f"⏱️  Subplot iteration: {t_plot_end-t_plot_start:.4f}s ({(t_plot_end-t_plot_start)/(n_rows*n_cols):.4f}s per subplot)")
            logger.info(f"📊 Traces added: {traces_added}, failed: {traces_failed}")
            
            # Update layout
            fig.update_layout(
                height=300 * n_rows,
                showlegend=False,
                title_text=f"Lightcurves: {len(object_keys_list)} objects × {len(measurement_keys)} measurements"
            )
            
            t7 = time.time()
            logger.info(f"⏱️  Plot generation: {t7-t6:.4f}s")
            logger.info(f"✅ PLOT GENERATION COMPLETE - Total time: {t7-start_time:.4f}s")
            logger.info("="*80)
            
            return fig, {"height": f"{300 * n_rows}px"}
            
        except Exception as e:
            logger.error(f"❌ Plot generation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise PreventUpdate