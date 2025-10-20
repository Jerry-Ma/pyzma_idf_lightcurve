"""Visualization tab component with controls, plots, and image viewer."""

import dash_mantine_components as dmc
from dash import html, dcc
import dash_ag_grid as dag


def create_viz_tab():
    """Create the visualization tab with controls, plots, and image viewer.
    
    Returns:
        dmc component with lightcurve controls, plots, and optional image display.
    """
    return html.Div([
        # Store for current plot state (which object index is displayed)
        dcc.Store(id='current-plot-index', data=0),
        dcc.Store(id='filtered-object-list', data=[]),
        dcc.Store(id='query-is-valid', data=False),  # Store validation state
        dcc.Store(id='autocomplete-options', data=[]),  # Store autocomplete options
        
        # Main layout: Left column (controls) | Right column (plot)
        dmc.Grid([
            # Left column: All controls and navigator
            dmc.GridCol([
                dmc.Stack([
                    # Lightcurve Controls section
                    dmc.Paper(
                        [
                            dmc.Title("Lightcurve Controls", order=3, mb="md"),
                            
                            # Object query filter - full width
                            html.Div([
                                dmc.Textarea(
                                    id='object-query-input',
                                    label="Object Query Filter",
                                    placeholder='e.g., n_epochs_valid_all > 900 (Press Enter to update)',
                                    value='n_epochs_valid_all > 900',
                                    minRows=2,
                                    maxRows=2,
                                    description="Query filter (required). Press Enter to update table.",
                                    style={'width': '100%'},
                                ),
                            ], id='query-input-wrapper'),
                            
                            # Update button
                            dmc.Button(
                                "Update Table",
                                id='update-table-button',
                                variant="filled",
                                color="green",
                                disabled=True,
                                mt="xs",
                            ),
                            
                            html.Div(id='object-query-validation'),
                            html.Div(id='object-count-feedback'),  # Feedback for selection count
                            
                            # X-axis variable selection
                            dmc.Select(
                                id='x-axis-select',
                                label="X-Axis Variable",
                                placeholder="Choose time variable",
                                data=[],
                                value=None,
                                clearable=False,
                                mt="md",
                            ),
                            
                            # Measurement selection
                            dmc.MultiSelect(
                                id='measurement-keys-select',
                                label="Select Measurements",
                                placeholder="Choose measurement types",
                                data=[],
                                value=[],
                                searchable=True,
                                clearable=True,
                                mt="md",
                            ),
                        ],
                        p="md",
                        mb="md",
                        withBorder=True
                    ),
                    
                    # Object Navigator section
                    dmc.Paper(
                        [
                            dmc.Title("Object Navigator", order=4, mb="sm"),
                            
                            # Navigation buttons
                            dmc.Group([
                                dmc.Button("◄◄ -5", id="nav-prev5-button", size="xs", variant="light", color="blue"),
                                dmc.Button("◄◄ -2", id="nav-prev2-button", size="xs", variant="light", color="blue"),
                                dmc.Button("◄ -1", id="nav-prev1-button", size="xs", variant="light", color="blue"),
                                dmc.Button("+1 ►", id="nav-next1-button", size="xs", variant="light", color="green"),
                                dmc.Button("+2 ►►", id="nav-next2-button", size="xs", variant="light", color="green"),
                                dmc.Button("+5 ►►", id="nav-next5-button", size="xs", variant="light", color="green"),
                            ], gap="xs", mb="sm"),
                            
                            # Object table with AG Grid (sortable, filterable)
                            dag.AgGrid(
                                id="object-table",
                                columnDefs=[
                                    {"field": "#", "flex": 1, "minWidth": 50, "sortable": True, "checkboxSelection": True, "headerCheckboxSelection": True},
                                    {"field": "Object", "flex": 2, "minWidth": 100, "sortable": True},
                                    {"field": "Mag", "flex": 1, "minWidth": 60, "sortable": True},
                                    {"field": "MagErr", "flex": 1, "minWidth": 60, "sortable": True},
                                ],
                                rowData=[],
                                defaultColDef={"sortable": True, "filter": True, "resizable": True},
                                dashGridOptions={
                                    "rowSelection": "multiple",
                                    "animateRows": True,
                                    "pagination": False,
                                    "domLayout": "normal",
                                },
                                style={"height": "1200px", "width": "100%"},
                            ),
                        ],
                        p="md",
                        withBorder=True
                    ),
                ], gap="sm"),
            ], span=3),
            
            # Right column: Plot area
            dmc.GridCol([
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
                            style={"height": "2400px"}  # Large height for detailed multi-object plots
                        )
                    ],
                    p="md",
                    withBorder=True
                ),
            ], span=9),
        ]),
        
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
