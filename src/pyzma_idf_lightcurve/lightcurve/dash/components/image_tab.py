"""Image viewer tab component for displaying superstack with object overlays."""

import dash_mantine_components as dmc
from dash import dcc, html


def create_image_tab():
    """Create the image viewer tab layout.
    
    Features:
    - Load superstack FITS image
    - Display with adjustable stretch/colormap
    - Overlay selected objects from catalog
    """
    
    return dmc.Stack([
        # Image path input section
        dmc.Paper([
            dmc.Title("Superstack Image", order=3, mb="sm"),
            dmc.Text("Load and display the superstack FITS image with object overlays",
                    c="gray", size="sm", mb="md"),
            
            dmc.Grid([
                dmc.GridCol([
                    dmc.TextInput(
                        id='image-path-input',
                        label="FITS Image Path",
                        placeholder="e.g., ../superstack/coadd_ch1.fits",
                        value="../superstack/coadd_ch1.fits",
                        style={"width": "100%"}
                    ),
                    html.Div(id='image-path-validation'),
                ], span=9),
                
                dmc.GridCol([
                    dmc.Button(
                        "Load Image",
                        id='load-image-button',
                        fullWidth=True,
                        mt="xl",
                        variant="filled",
                    ),
                ], span=3),
            ]),
        ], shadow="xs", p="md", mb="md", withBorder=True),
        
        # Display controls
        dmc.Paper([
            dmc.Title("Display Settings", order=4, mb="md"),
            
            dmc.Grid([
                # Normalization
                dmc.GridCol([
                    dmc.Select(
                        id='image-norm-select',
                        label="Normalization",
                        description="Image scaling method",
                        value="linear",
                        data=[
                            {'value': 'linear', 'label': 'Linear'},
                            {'value': 'log', 'label': 'Log'},
                            {'value': 'sqrt', 'label': 'Square Root'},
                            {'value': 'asinh', 'label': 'Arcsinh'},
                        ],
                    ),
                ], span=3),
                
                # Colormap
                dmc.GridCol([
                    dmc.Select(
                        id='image-cmap-select',
                        label="Colormap",
                        description="Color scheme",
                        value="Greys_r",
                        data=[
                            {'value': 'Greys_r', 'label': 'Greys (inverted)'},
                            {'value': 'viridis', 'label': 'Viridis'},
                            {'value': 'plasma', 'label': 'Plasma'},
                            {'value': 'inferno', 'label': 'Inferno'},
                            {'value': 'RdYlBu_r', 'label': 'Red-Yellow-Blue'},
                            {'value': 'Spectral_r', 'label': 'Spectral'},
                        ],
                    ),
                ], span=3),
                
                # Stretch percentiles
                dmc.GridCol([
                    dmc.Text("Brightness Range", size="sm", fw="bold", mb="xs"),
                    dmc.Text("Adjust contrast using percentile clipping", 
                            size="xs", c="gray", mb="xs"),
                    dcc.RangeSlider(
                        id='image-stretch-slider',
                        min=0,
                        max=100,
                        step=0.5,
                        value=[1, 99],
                        marks={0: '0%', 25: '25%', 50: '50%', 75: '75%', 100: '100%'},
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                ], span=6),
            ], gutter="md"),
            
            # Object overlay controls
            dmc.Divider(label="Object Overlay", labelPosition="center", my="md"),
            
            dmc.Grid([
                dmc.GridCol([
                    dmc.Switch(
                        id='show-all-objects-toggle',
                        label="Show All Objects",
                        description="Display all catalog objects on image",
                        checked=False,
                        size="md",
                    ),
                ], span=4),
                
                dmc.GridCol([
                    dmc.Switch(
                        id='show-selected-objects-toggle',
                        label="Show Selected Objects",
                        description="Highlight selected objects from lightcurve queries",
                        checked=True,
                        size="md",
                    ),
                ], span=4),
                
                dmc.GridCol([
                    dmc.NumberInput(
                        id='marker-size-input',
                        label="Marker Size",
                        description="Size of overlay markers",
                        value=8,
                        min=2,
                        max=20,
                        step=1,
                    ),
                ], span=4),
            ], gutter="md"),
        ], shadow="xs", p="md", mb="md", withBorder=True),
        
        # Image display
        dmc.Paper([
            dcc.Loading(
                id="loading-image",
                type="circle",
                children=dcc.Graph(
                    id='image-display',
                    config={
                        'displayModeBar': True,
                        'displaylogo': False,
                        'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                        'toImageButtonOptions': {
                            'format': 'png',
                            'filename': 'idf_superstack',
                            'height': 1000,
                            'width': 1000,
                            'scale': 2
                        }
                    },
                    style={'height': '800px'}
                )
            )
        ], shadow="sm", p="md", withBorder=True),
    ], gap="md")
