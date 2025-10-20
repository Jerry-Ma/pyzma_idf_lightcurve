"""Callbacks for object statistics tab with histogram visualization."""

import logging
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State
from dash.exceptions import PreventUpdate

from ..app import get_storage_cache

logger = logging.getLogger(__name__)


def register_object_stats_callbacks(app):
    """Register all object statistics tab callbacks."""
    cache = get_storage_cache()
    
    _register_column_selector_callback(app, cache)
    _register_histogram_plot_callback(app, cache)


def _register_column_selector_callback(app, cache):
    """Populate column selector when storage is loaded."""
    
    @app.callback(
        [
            Output('stats-column-select', 'data'),
            Output('stats-column-select', 'value'),
        ],
        Input('storage-data', 'data'),
        prevent_initial_call=True,
    )
    def populate_column_selector(storage_data):
        """Populate dropdown options with available columns from object table."""
        if not storage_data:
            return [], None
        
        from pathlib import Path
        storage_path = Path(storage_data['storage_path'])
        mode = storage_data.get('mode', 'read')
        path_mode = f"{storage_path}:{mode}"
        
        # Get object_df from cache
        object_df = cache.get(f"object_df:{path_mode}")
        if object_df is None:
            return [], None
        
        # Get numeric columns only
        numeric_cols = object_df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Create dropdown options
        options = [{"label": col, "value": col} for col in numeric_cols]
        
        # Set default value
        default_value = 'n_epochs_valid_all' if 'n_epochs_valid_all' in numeric_cols else (numeric_cols[0] if numeric_cols else None)
        
        return options, default_value


def _register_histogram_plot_callback(app, cache):
    """Generate histogram plot for selected column."""
    
    @app.callback(
        Output('object-stats-plot', 'figure'),
        [
            Input('stats-column-select', 'value'),
            Input('storage-data', 'data'),
        ],
        prevent_initial_call=True,
    )
    def generate_histogram(column, storage_data):
        """Generate histogram for selected column."""
        
        if not storage_data or not column:
            return {}
        
        from pathlib import Path
        storage_path = Path(storage_data['storage_path'])
        mode = storage_data.get('mode', 'read')
        path_mode = f"{storage_path}:{mode}"
        
        # Get object_df from cache
        object_df = cache.get(f"object_df:{path_mode}")
        if object_df is None:
            return {}
        
        # Extract data
        try:
            data = object_df[column].values
            
            # Filter out NaN/inf values
            data = data[np.isfinite(data)]
            
            if len(data) == 0:
                return {}
            
        except Exception as e:
            logger.error(f"Error extracting data: {e}")
            return {}
        
        # Create histogram
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=data,
            name=column,
            marker_color='steelblue',
            opacity=0.75,
            nbinsx=50,
        ))
        
        # Update layout
        fig.update_layout(
            title=f"Distribution of {column}",
            xaxis_title=column,
            yaxis_title="Count",
            height=600,
            showlegend=False,
            hovermode='closest',
            plot_bgcolor='white',
            paper_bgcolor='white',
            bargap=0.1,
        )
        
        # Grid lines
        fig.update_xaxes(showgrid=True, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridcolor='lightgray')
        
        # Add statistics annotation
        mean_val = np.mean(data)
        median_val = np.median(data)
        std_val = np.std(data)
        
        stats_text = f"N = {len(data)}<br>Mean = {mean_val:.2f}<br>Median = {median_val:.2f}<br>Std = {std_val:.2f}"
        
        fig.add_annotation(
            text=stats_text,
            xref="paper", yref="paper",
            x=0.98, y=0.98,
            xanchor='right', yanchor='top',
            showarrow=False,
            bgcolor="white",
            bordercolor="gray",
            borderwidth=1,
            borderpad=5,
        )
        
        logger.info(f"Generated histogram for {column}: {len(data)} objects")
        
        return fig
