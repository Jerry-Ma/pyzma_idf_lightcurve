"""Utility functions for plotting lightcurves."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_lightcurve_subplot(
    storage,
    object_indices,
    measurement_indices,
    x_axis_var='epoch_mjd',
    max_objects=20
):
    """Create a subplot figure with lightcurves for multiple objects.
    
    Args:
        storage: LightcurveStorage instance
        object_indices: List of object indices to plot
        measurement_indices: List of measurement indices to include
        x_axis_var: Variable name for x-axis (from epoch_ variables)
        max_objects: Maximum number of objects to plot
        
    Returns:
        plotly Figure object
    """
    # Limit objects
    object_indices = object_indices[:max_objects]
    n_objects = len(object_indices)
    
    # Calculate subplot layout
    n_cols = min(2, n_objects)
    n_rows = (n_objects + n_cols - 1) // n_cols
    
    # Create subplots
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[f'Object {idx}' for idx in object_indices],
        vertical_spacing=0.15 / n_rows if n_rows > 1 else 0.1,
        horizontal_spacing=0.1
    )
    
    # Color palette
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    # Plot each object
    for obj_idx, object_key in enumerate(object_indices):
        row = (obj_idx // n_cols) + 1
        col = (obj_idx % n_cols) + 1
        
        try:
            # Get lightcurve data
            lc_data = storage.get_object_lightcurve(object_key)
            
            # Get x-axis data
            if x_axis_var in lc_data:
                x_data = lc_data[x_axis_var].values
            else:
                x_data = np.arange(len(lc_data.coords['epoch']))
            
            # Plot each measurement
            for meas_idx, meas_key in enumerate(measurement_indices):
                color = colors[meas_idx % len(colors)]
                
                # Extract magnitude and error (assuming value dimension: [mag, mag_err, ...])
                mag_data = lc_data.isel(measurement=meas_key, value=0).values
                mag_err_data = lc_data.isel(measurement=meas_key, value=1).values
                
                # Filter NaN values
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
                        name=f'Meas {meas_key}',
                        showlegend=(obj_idx == 0),
                        legendgroup=f'meas{meas_key}',
                        hovertemplate=(
                            f'<b>Object {object_key}</b><br>'
                            f'Measurement {meas_key}<br>'
                            'X: %{x:.3f}<br>'
                            'Magnitude: %{y:.3f}<br>'
                            'Error: %{error_y.array:.4f}<br>'
                            '<extra></extra>'
                        ),
                    ),
                    row=row,
                    col=col
                )
            
            # Invert y-axis (magnitude scale - brighter is up)
            fig.update_yaxes(autorange="reversed", row=row, col=col)
            
        except Exception as e:
            print(f"Error plotting object {object_key}: {e}")
            continue
    
    # Update layout
    fig.update_layout(
        height=400 * n_rows,
        showlegend=True,
        hovermode='closest',
        title_text="Lightcurves",
    )
    
    # Update axis labels
    x_label = x_axis_var.replace('epoch_', '').replace('_', ' ').title()
    fig.update_xaxes(title_text=x_label)
    fig.update_yaxes(title_text="Magnitude")
    
    return fig


def create_image_figure(image_data, extent=None, title="Image", colorscale='gray'):
    """Create a plotly figure for displaying FITS image data.
    
    Args:
        image_data: 2D numpy array
        extent: [xmin, xmax, ymin, ymax] for axis limits
        title: Figure title
        colorscale: Plotly colorscale name
        
    Returns:
        plotly Figure object
    """
    fig = go.Figure()
    
    # Add image as heatmap
    fig.add_trace(go.Heatmap(
        z=image_data,
        colorscale=colorscale,
        hovertemplate='X: %{x}<br>Y: %{y}<br>Value: %{z}<extra></extra>',
    ))
    
    # Set aspect ratio and layout
    fig.update_layout(
        title=title,
        xaxis=dict(
            scaleanchor="y",
            scaleratio=1,
            constrain='domain',
        ),
        yaxis=dict(
            constrain='domain',
        ),
        height=600,
    )
    
    if extent:
        fig.update_xaxes(range=[extent[0], extent[1]])
        fig.update_yaxes(range=[extent[2], extent[3]])
    
    return fig


def add_source_markers(fig, x_coords, y_coords, selected_indices=None, marker_size=10):
    """Add source markers to an image figure.
    
    Args:
        fig: Plotly figure object
        x_coords: Array of x coordinates
        y_coords: Array of y coordinates
        selected_indices: List of indices for selected sources (highlighted differently)
        marker_size: Size of markers
        
    Returns:
        Updated figure
    """
    if selected_indices is None:
        selected_indices = []
    
    # Create mask for selected vs unselected
    n_sources = len(x_coords)
    selected_mask = np.array([i in selected_indices for i in range(n_sources)])
    
    # Add unselected sources
    if (~selected_mask).sum() > 0:
        fig.add_trace(go.Scatter(
            x=x_coords[~selected_mask],
            y=y_coords[~selected_mask],
            mode='markers',
            marker=dict(
                size=marker_size,
                color='cyan',
                symbol='circle-open',
                line=dict(width=2)
            ),
            name='Sources',
            hovertemplate='X: %{x:.1f}<br>Y: %{y:.1f}<extra></extra>',
        ))
    
    # Add selected sources
    if selected_mask.sum() > 0:
        fig.add_trace(go.Scatter(
            x=x_coords[selected_mask],
            y=y_coords[selected_mask],
            mode='markers',
            marker=dict(
                size=marker_size + 5,
                color='red',
                symbol='star',
                line=dict(width=2, color='yellow')
            ),
            name='Selected',
            hovertemplate='X: %{x:.1f}<br>Y: %{y:.1f}<extra></extra>',
        ))
    
    return fig
