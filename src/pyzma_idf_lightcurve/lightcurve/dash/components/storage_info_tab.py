"""Storage info tab component with overview and AG Grid tables."""

import dash_mantine_components as dmc
import dash_ag_grid as dag
from dash import html


def create_storage_info_tab():
    """Create the storage info tab with overview and data tables.
    
    Returns:
        dmc component with storage overview and AG Grid tables for epoch/object data.
    """
    return html.Div([
        # Storage overview
        dmc.Paper(
            [
                dmc.Title("Storage Overview", order=3, mb="sm"),
                html.Div(id='storage-overview', children=[
                    dmc.Text("No storage loaded", c="dimmed")
                ])
            ],
            p="md",
            mb="md",
            withBorder=True
        ),
        
        # Epoch variables table
        dmc.Paper(
            [
                dmc.Title("Epoch Variables", order=4, mb="sm"),
                dmc.Text(
                    "Time-indexed variables for each observation epoch",
                    size="sm",
                    c="dimmed",
                    mb="md"
                ),
                dag.AgGrid(
                    id='epoch-variables-grid',
                    columnDefs=[],
                    rowModelType="infinite",
                    defaultColDef={
                        "sortable": True,
                        "filter": True,
                        "resizable": True,
                    },
                    dashGridOptions={
                        "rowSelection": "single",
                        "rowBuffer": 0,
                        "cacheBlockSize": 100,
                        "maxBlocksInCache": 10,  # Increased from 2 to handle more blocks
                        "infiniteInitialRowCount": 1,  # Show loading state
                        "maxConcurrentDatasourceRequests": 2,
                        "cacheOverflowSize": 2,  # Keep extra blocks in memory
                    },
                    style={"height": "600px", "width": "100%"},
                )
            ],
            p="md",
            mb="md",
            withBorder=True
        ),
        
        # Object variables table
        dmc.Paper(
            [
                dmc.Group([
                    dmc.Title("Object Variables", order=4, mb=0),
                    dmc.Button(
                        "Select for Visualization",
                        id='select-objects-button',
                        variant="light",
                        color="blue",
                        size="sm",
                        disabled=True,
                    )
                ], justify="space-between", mb="sm"),
                dmc.Text(
                    "Object-specific variables (ra, dec, x_image, y_image, etc.)",
                    size="sm",
                    c="dimmed",
                    mb="md"
                ),
                dag.AgGrid(
                    id='object-variables-grid',
                    columnDefs=[],
                    rowModelType="infinite",
                    defaultColDef={
                        "sortable": True,
                        "filter": True,
                        "resizable": True,
                        "suppressMenu": False,
                    },
                    dashGridOptions={
                        "rowSelection": "multiple",
                        "rowBuffer": 0,
                        "cacheBlockSize": 100,
                        "maxBlocksInCache": 2,
                        "infiniteInitialRowCount": 1,
                        "maxConcurrentDatasourceRequests": 2,
                        "suppressColumnVirtualisation": False,
                        "suppressRowClickSelection": False,
                        "enableCellTextSelection": True,
                        "ensureDomOrder": False,
                        "animateRows": False,
                    },
                    style={"height": "600px", "width": "100%"},
                )
            ],
            p="md",
            withBorder=True
        ),
    ])
