"""Object statistics tab with histogram visualization."""

import dash_mantine_components as dmc
from dash import dcc


def create_object_stats_tab():
    """Create the object statistics tab with histogram plot."""
    
    return dmc.Stack([
        dmc.Paper([
            dmc.Title("Object Statistics", order=3, mb="md"),
            
            # Control panel for plot configuration
            dmc.Grid([
                # Column selection
                dmc.GridCol([
                    dmc.Select(
                        id='stats-column-select',
                        label="Select Column",
                        placeholder="Select column to plot",
                        data=[],
                        value=None,
                        searchable=True,
                        clearable=False,
                    ),
                ], span=12),
            ], gutter="md"),
            
        ], p="md", withBorder=True, mb="md"),
        
        # Plot container
        dmc.Paper([
            dcc.Graph(
                id='object-stats-plot',
                figure={},
                style={'height': '600px'},
                config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                }
            ),
        ], p="md", withBorder=True),
        
    ], gap="md")
