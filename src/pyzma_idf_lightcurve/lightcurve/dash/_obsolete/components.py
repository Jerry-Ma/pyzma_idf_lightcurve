"""
Reusable components for the lightcurve visualization interface.
"""


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_lightcurve_plot(
    times: np.ndarray,
    magnitudes: np.ndarray,
    mag_errors: np.ndarray,
    title: str = "Lightcurve",
    flags: np.ndarray | None = None
) -> go.Figure:
    """
    Create a standard lightcurve plot.
    
    Args:
        times: Time values (MJD)
        magnitudes: Magnitude values
        mag_errors: Magnitude uncertainties
        title: Plot title
        flags: Optional flags for data quality
        
    Returns:
        Plotly Figure object
    """
    
    fig = go.Figure()
    
    # Color points by flags if provided
    if flags is not None:
        colors = ['blue' if f == 0 else 'orange' if f == 1 else 'red' for f in flags]
    else:
        colors = 'blue'
    
    # Add lightcurve points
    fig.add_trace(go.Scatter(
        x=times,
        y=magnitudes,
        error_y=dict(type='data', array=mag_errors, visible=True),
        mode='markers',
        marker=dict(size=4, color=colors),
        name='Lightcurve',
        hovertemplate='MJD: %{x:.3f}<br>Mag: %{y:.3f}<br>Error: %{error_y.array:.3f}<extra></extra>'
    ))
    
    # Customize layout
    fig.update_layout(
        title=title,
        xaxis_title="Time (MJD)",
        yaxis_title="Magnitude",
        yaxis=dict(autorange="reversed"),  # Magnitude scale (brighter is up)
        height=400,
        hovermode='closest'
    )
    
    return fig


def create_multi_panel_plot(
    lightcurves: dict[str, dict[str, np.ndarray]],
    title: str = "Multi-Panel Lightcurve"
) -> go.Figure:
    """
    Create multi-panel plot for comparing different measurement methods.
    
    Args:
        lightcurves: Dict with keys as panel names and values as lightcurve data
        title: Overall plot title
        
    Returns:
        Plotly Figure with subplots
    """
    
    n_panels = len(lightcurves)
    
    # Create subplots
    fig = make_subplots(
        rows=n_panels, cols=1,
        subplot_titles=list(lightcurves.keys()),
        shared_xaxes=True,
        vertical_spacing=0.05
    )
    
    for i, (panel_name, data) in enumerate(lightcurves.items(), 1):
        times = data['times']
        magnitudes = data['magnitudes']
        mag_errors = data.get('mag_errors', np.zeros_like(magnitudes))
        
        fig.add_trace(
            go.Scatter(
                x=times,
                y=magnitudes,
                error_y=dict(type='data', array=mag_errors, visible=True),
                mode='markers',
                marker=dict(size=3),
                name=panel_name,
                showlegend=False
            ),
            row=i, col=1
        )
        
        # Update y-axis to reverse (magnitude scale)
        fig.update_yaxes(autorange="reversed", row=i, col=1)
    
    # Update layout
    fig.update_layout(
        title=title,
        height=200 * n_panels,
        hovermode='closest'
    )
    
    # Update x-axis title only for bottom subplot
    fig.update_xaxes(title_text="Time (MJD)", row=n_panels, col=1)
    
    return fig


def create_object_search_interface():
    """Create object search and filtering interface components."""
    pass  # To be implemented as needed