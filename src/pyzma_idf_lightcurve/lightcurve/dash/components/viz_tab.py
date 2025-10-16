"""Visualization tab component with controls, plots, and image viewer."""

import dash_mantine_components as dmc
from dash import html, dcc


def create_viz_tab():
    """Create the visualization tab with controls, plots, and image viewer.
    
    Returns:
        dmc component with lightcurve controls, plots, and optional image display.
    """
    return html.Div([
        # Control panel
        dmc.Paper(
            [
                dmc.Title("Lightcurve Controls", order=3, mb="md"),
                
                dmc.Grid([
                    # Object selection
                    dmc.GridCol(
                        [
                            dmc.MultiSelect(
                                id='object-keys-select',
                                label="Select Objects (max 20)",
                                placeholder="Choose objects to visualize",
                                data=[],
                                value=[],
                                maxValues=20,
                                searchable=True,
                                clearable=True,
                            )
                        ],
                        span=6
                    ),
                    
                    # Measurement selection
                    dmc.GridCol(
                        [
                            dmc.MultiSelect(
                                id='measurement-keys-select',
                                label="Select Measurements",
                                placeholder="Choose measurement types",
                                data=[],
                                value=[],
                                searchable=True,
                                clearable=True,
                            )
                        ],
                        span=6
                    ),
                    
                    # X-axis variable
                    dmc.GridCol(
                        [
                            dmc.Select(
                                id='x-axis-select',
                                label="X-Axis Variable",
                                placeholder="Choose time variable",
                                data=[],
                                value="epoch_mjd",
                                clearable=False,
                            )
                        ],
                        span=4
                    ),
                    
                    # Query filter
                    dmc.GridCol(
                        [
                            dmc.TextInput(
                                id='query-input',
                                label="Query Filter (optional)",
                                placeholder="e.g., ra > 180 & dec < 30",
                                description="Pandas query syntax for filtering objects",
                            ),
                            html.Div(id='query-validation'),
                        ],
                        span=5
                    ),
                    
                    # Plot button
                    dmc.GridCol(
                        [
                            html.Div(
                                dmc.Button(
                                    "Generate Plots",
                                    id='generate-plots-button',
                                    variant="filled",
                                    color="green",
                                    fullWidth=True,
                                    disabled=True,
                                ),
                                style={"paddingTop": "25px"}
                            )
                        ],
                        span=3
                    ),
                ]),
            ],
            p="md",
            mb="md",
            withBorder=True
        ),
        
        # Plot display area
        dmc.Paper(
            [
                dcc.Graph(
                    id='lightcurve-plot',
                    figure={},
                    config={
                        'displayModeBar': True,
                        'displaylogo': False,
                        'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                    },
                    style={"height": "800px"}
                )
            ],
            p="md",
            mb="md",
            withBorder=True
        ),
        
        # Image viewer section (optional, expandable)
        dmc.Accordion(
            [
                dmc.AccordionItem(
                    [
                        dmc.AccordionControl("Image Viewer with Source Overlay"),
                        dmc.AccordionPanel(
                            create_image_viewer()
                        ),
                    ],
                    value="image-viewer"
                )
            ],
            value=None,  # Collapsed by default
        )
    ])


def create_image_viewer():
    """Create the image viewer component with controls.
    
    Returns:
        Component with image display and overlay controls.
    """
    return html.Div([
        dmc.Grid([
            # Image path input
            dmc.GridCol(
                [
                    dmc.TextInput(
                        id='image-path-input',
                        label="FITS Image Path",
                        placeholder="Path to FITS image file",
                    ),
                    html.Div(id='image-path-validation'),
                ],
                span=6
            ),
            
            # Normalization
            dmc.GridCol(
                [
                    dmc.Select(
                        id='image-norm-select',
                        label="Normalization",
                        data=[
                            {"value": "linear", "label": "Linear"},
                            {"value": "log", "label": "Log"},
                            {"value": "sqrt", "label": "Sqrt"},
                            {"value": "asinh", "label": "Asinh"},
                        ],
                        value="linear",
                    )
                ],
                span=2
            ),
            
            # Colormap
            dmc.GridCol(
                [
                    dmc.Select(
                        id='image-cmap-select',
                        label="Colormap",
                        data=[
                            {"value": "gray", "label": "Gray"},
                            {"value": "viridis", "label": "Viridis"},
                            {"value": "hot", "label": "Hot"},
                            {"value": "cool", "label": "Cool"},
                        ],
                        value="gray",
                    )
                ],
                span=2
            ),
            
            # Load image button
            dmc.GridCol(
                [
                    html.Div(
                        dmc.Button(
                            "Load Image",
                            id='load-image-button',
                            variant="light",
                            fullWidth=True,
                        ),
                        style={"paddingTop": "25px"}
                    )
                ],
                span=2
            ),
        ]),
        
        # Stretch controls
        dmc.Grid([
            dmc.GridCol(
                [
                    dmc.Text("Stretch Percentiles:", size="sm", fw=500, mt="sm"),
                    dmc.RangeSlider(
                        id='image-stretch-slider',
                        min=0,
                        max=100,
                        value=[1, 99],
                        marks={0: "0", 25: "25", 50: "50", 75: "75", 100: "100"},
                        mb="md",
                    )
                ],
                span=12
            ),
        ]),
        
        # Image display
        dcc.Graph(
            id='image-display',
            figure={},
            config={
                'displayModeBar': True,
                'displaylogo': False,
            },
            style={"height": "600px"}
        ),
    ])
