"""
Main Plotly Dash application for interactive lightcurve visualization.
"""


import dash
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, dash_table, dcc, html

from ..binary import BinaryLightcurveDatabase
from ..data import LightcurveDatabase, LightcurveQueryEngine


class LightcurveVisualizationApp:
    """
    High-performance interactive lightcurve visualization using Plotly Dash.
    
    Features:
    - Interactive object selection and filtering
    - Multi-panel views (different methods, channels, flux types)
    - Real-time querying with database backend
    - Responsive design for large datasets
    - Export capabilities
    """
    
    def __init__(self, db_path: str, use_binary: bool = True):
        self.db_path = db_path
        self.use_binary = use_binary
        
        if use_binary:
            self.db = BinaryLightcurveDatabase(db_path)
        else:
            self.db = LightcurveDatabase(db_path)
            self.query_engine = LightcurveQueryEngine(self.db)
            
        self.app = dash.Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()
    
    def setup_layout(self):
        """Create the app layout with all components."""
        
        self.app.layout = html.Div([
            # Header
            html.H1("IDF Lightcurve Explorer", className="header"),
            
            # Control Panel
            html.Div([
                html.Div([
                    # Object Selection
                    html.Label("Object ID:"),
                    dcc.Input(
                        id='object-id-input',
                        type='number',
                        placeholder='Enter object ID',
                        value=1
                    ),
                ], className="control-group"),
                
                html.Div([
                    # Measurement Type
                    html.Label("Measurement Type:"),
                    dcc.Dropdown(
                        id='type-dropdown',
                        options=[
                            {'label': 'CH1 AUTO Original', 'value': 1},
                            {'label': 'CH1 AUTO LAC Cleaned', 'value': 2},
                            {'label': 'CH2 AUTO Original', 'value': 3},
                            {'label': 'CH2 AUTO LAC Cleaned', 'value': 4},
                        ],
                        value=1
                    ),
                ], className="control-group"),
                
                html.Div([
                    # Time Range
                    html.Label("Time Range (MJD):"),
                    dcc.RangeSlider(
                        id='time-range-slider',
                        min=55000,
                        max=58000,
                        step=1,
                        value=[55000, 58000],
                        marks={
                            55000: '2009',
                            56000: '2012', 
                            57000: '2015',
                            58000: '2018'
                        }
                    ),
                ], className="control-group"),
                
                html.Button("Load Lightcurve", id="load-button", n_clicks=0),
                
            ], className="control-panel"),
            
            # Main Plot Area
            dcc.Graph(id="lightcurve-plot"),
            
            # Statistics Panel
            html.Div(id="stats-panel"),
            
            # Data Table
            html.Div([
                html.H3("Lightcurve Data"),
                dash_table.DataTable(
                    id='lightcurve-table',
                    columns=[
                        {"name": "MJD", "id": "time"},
                        {"name": "Magnitude", "id": "magnitude"},
                        {"name": "Error", "id": "mag_error"},
                        {"name": "Flags", "id": "flags"},
                    ],
                    page_size=20,
                    sort_action="native",
                    filter_action="native"
                )
            ])
        ])
    
    def setup_callbacks(self):
        """Setup interactive callbacks."""
        
        @self.app.callback(
            [Output('lightcurve-plot', 'figure'),
             Output('stats-panel', 'children'),
             Output('lightcurve-table', 'data')],
            [Input('load-button', 'n_clicks')],
            [dash.dependencies.State('object-id-input', 'value'),
             dash.dependencies.State('type-dropdown', 'value'),
             dash.dependencies.State('time-range-slider', 'value')]
        )
        def update_lightcurve(n_clicks, object_id, type_id, time_range):
            if n_clicks == 0 or not object_id:
                return {}, "Select an object to view lightcurve", []
            
            # Get lightcurve data
            if self.use_binary:
                lightcurve = self.db.get_lightcurve(object_id, type_id)
                if not lightcurve:
                    return {}, f"No data found for object {object_id}", []
                
                # Apply time filtering
                times = lightcurve['times']
                magnitudes = lightcurve['magnitudes']
                mag_errors = lightcurve['mag_errors']
                flags = lightcurve['flags']
                
                # Filter by time range
                mask = (times >= time_range[0]) & (times <= time_range[1])
                times = times[mask]
                magnitudes = magnitudes[mask]
                mag_errors = mag_errors[mask]
                flags = flags[mask]
                
            else:
                # Traditional database query
                df = self.query_engine.get_lightcurve(
                    object_id=object_id,
                    time_range=tuple(time_range)
                )
                
                if df.empty:
                    return {}, f"No data found for object {object_id}", []
                
                times = df['obs_time'].values
                magnitudes = df['magnitude'].values
                mag_errors = df['mag_err'].values
                flags = df['flags'].values
            
            # Create plot
            fig = go.Figure()
            
            # Add lightcurve points
            fig.add_trace(go.Scatter(
                x=times,
                y=magnitudes,
                error_y=dict(type='data', array=mag_errors, visible=True),
                mode='markers',
                marker=dict(size=4, color='blue'),
                name='Lightcurve',
                hovertemplate='MJD: %{x:.3f}<br>Mag: %{y:.3f}<br>Error: %{error_y.array:.3f}<extra></extra>'
            ))
            
            # Customize layout
            fig.update_layout(
                title=f"Object {object_id} - Type {type_id}",
                xaxis_title="Time (MJD)",
                yaxis_title="Magnitude",
                yaxis=dict(autorange="reversed"),  # Magnitude scale (brighter is up)
                height=500,
                hovermode='closest'
            )
            
            # Calculate statistics
            if len(magnitudes) > 0:
                stats = html.Div([
                    html.H3("Statistics"),
                    html.P(f"Number of points: {len(magnitudes)}"),
                    html.P(f"Mean magnitude: {np.mean(magnitudes):.3f}"),
                    html.P(f"Magnitude std: {np.std(magnitudes):.4f}"),
                    html.P(f"Time span: {np.max(times) - np.min(times):.1f} days"),
                    html.P(f"Mean error: {np.mean(mag_errors):.4f}"),
                ])
            else:
                stats = html.P("No data points in selected range")
            
            # Prepare table data
            table_data = [
                {
                    "time": f"{t:.3f}",
                    "magnitude": f"{m:.3f}",
                    "mag_error": f"{e:.4f}",
                    "flags": int(f)
                }
                for t, m, e, f in zip(times, magnitudes, mag_errors, flags)
            ]
            
            return fig, stats, table_data
    
    def run_server(self, debug=True, port=8050):
        """Start the Dash server."""
        self.app.run_server(debug=debug, port=port)


def main():
    """CLI entry point for the visualization app."""
    try:
        import typer
    except ImportError:
        print("Error: typer is required for the CLI. Install with: pip install typer")
        return
    
    def run_viz(
        db_path: str = typer.Option("lightcurves.db", help="Path to lightcurve database"),
        port: int = typer.Option(8050, help="Port to run the server on"),
        debug: bool = typer.Option(False, help="Run in debug mode"),
        binary: bool = typer.Option(True, help="Use binary blob storage")
    ):
        app = LightcurveVisualizationApp(db_path, use_binary=binary)
        print(f"Starting lightcurve visualization server on port {port}")
        app.run_server(debug=debug, port=port)
    
    typer.run(run_viz)


if __name__ == "__main__":
    main()