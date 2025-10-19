"""Storage loader component for selecting and loading zarr storage."""

import dash_mantine_components as dmc
from dash import html, dcc
from dash_iconify import DashIconify


def create_storage_loader(initial_path=None):
    """Create the storage loader panel.
    
    Args:
        initial_path: Optional initial storage path to pre-populate the input field.
                     If provided, overrides any persisted value.
    
    Returns:
        dmc component with file path input, read/write selector, and load button.
    """
    return html.Div([
        dmc.Title("Load Storage", order=3, mb="sm"),
        
        dmc.Grid([
            # File path input
            dmc.GridCol(
                [
                    dmc.TextInput(
                        id='storage-path-input',
                        label="Storage Path",
                        placeholder="Enter path to zarr storage (e.g., scratch_dagster/idf_lightcurves.zarr)",
                        style={"width": "100%"},
                        value=initial_path or "",
                        persistence=True,
                        persistence_type='local',
                    )
                ],
                span=6
            ),
            
            # Read/Write selector
            dmc.GridCol(
                [
                    dmc.Select(
                        id='storage-mode-select',
                        label="Storage Mode",
                        data=[
                            {"value": "read", "label": "Read-Optimized (lightcurves_read.zarr)"},
                            {"value": "write", "label": "Write-Optimized (lightcurves_write.zarr)"},
                        ],
                        value="read",
                        clearable=False,
                        persistence=True,
                        persistence_type='local',
                    )
                ],
                span=3
            ),
            
            # Load button
            dmc.GridCol(
                [
                    html.Div(
                        dmc.Button(
                            "Load Storage",
                            id='load-storage-button',
                            variant="filled",
                            color="blue",
                            fullWidth=True,
                            leftSection=DashIconify(icon="mdi:database-import"),
                        ),
                        style={"paddingTop": "25px"}
                    )
                ],
                span=3
            ),
        ]),
        
        # Path validation feedback
        html.Div(id='storage-path-validation'),
        
        # Progress display (updated by background callback)
        dmc.Stack([
            dmc.Text(id='loading-status-text', size="sm", fw=500, c="dimmed", mt="md"),
            dmc.Progress(
                id='loading-progress',
                value=0,
                striped=True,
                animated=True,
                color="blue",
                size="lg",
            ),
            dmc.Text(id='loading-detail-text', size="xs", c="dimmed"),
        ], gap="xs"),
    ])
