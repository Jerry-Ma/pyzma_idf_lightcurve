"""Callbacks for storage info tab."""

from dash import Input, Output, State, callback, no_update
from dash.exceptions import PreventUpdate
import dash_mantine_components as dmc
import dash_ag_grid as dag
from tqdm import tqdm
import pandas as pd

from ...datamodel import LightcurveStorage
from ..storage_cache import StorageCache


def build_info_tables_data(storage):
    """Build data for all three info tables from storage.
    
    This function is separated for testing purposes.
    
    Args:
        storage: LightcurveStorage instance with loaded dataset
        
    Returns:
        tuple: (storage_overview_components, epoch_row_data, epoch_column_defs,
                object_row_data, object_column_defs, select_button_disabled)
    """
    print("[DEBUG] build_info_tables_data called")
    
    if storage is None:
        print("[ERROR] Storage is None")
        empty_overview = [dmc.Text("Storage not loaded", c="red", size="sm")]
        return empty_overview, [], [], [], [], True
    
    # Get dataset
    ds = storage.dataset
    print(f"[DEBUG] Dataset dimensions: {dict(ds.sizes)}")
    
    try:
        # ========== 1. BUILD STORAGE OVERVIEW ==========
        print("[DEBUG] Step 1: Building storage overview...")
        print(f"[DEBUG]   - Dataset has {len(ds.coords)} coordinates and {len(ds.data_vars)} data variables")
        var_metadata = []
        
        # Add coordinates
        print("[DEBUG]   - Processing coordinates...")
        for name in tqdm(list(ds.coords), desc="Processing coordinates", leave=False):
            coord = ds.coords[name]
            print(f"[DEBUG]     * Coordinate: {name}, dims: {coord.dims}, shape: {coord.shape}")
            var_metadata.append({
                'variable': name,
                'type': 'coordinate',
                'dimensions': ', '.join(coord.dims),
                'n_dims': len(coord.dims),
                'shape': str(coord.shape),
                'size': int(coord.size),
                'chunks': str(coord.chunks) if coord.chunks else 'None',
                'dtype': str(coord.dtype),
            })
        
        # Add data variables
        print("[DEBUG]   - Processing data variables...")
        for name in tqdm(list(ds.data_vars), desc="Processing data variables", leave=False):
            var = ds[name]
            print(f"[DEBUG]     * Data var: {name}, dims: {var.dims}, shape: {var.shape}")
            var_metadata.append({
                'variable': name,
                'type': 'data_var',
                'dimensions': ', '.join(var.dims),
                'n_dims': len(var.dims),
                'shape': str(var.shape),
                'size': int(var.size),
                'chunks': str(var.chunks) if var.chunks else 'None',
                'dtype': str(var.dtype),
            })
        
        # Column definitions for metadata table
        column_defs_storage = [
            {'field': 'variable', 'headerName': 'Variable'},
            {'field': 'type', 'headerName': 'Type'},
            {'field': 'dimensions', 'headerName': 'Dimensions'},
            {'field': 'n_dims', 'headerName': 'N Dims'},
            {'field': 'shape', 'headerName': 'Shape'},
            {'field': 'size', 'headerName': 'Size'},
            {'field': 'chunks', 'headerName': 'Chunks'},
            {'field': 'dtype', 'headerName': 'Dtype'},
        ]
        
        # Create summary text
        data_shape = dict(ds.sizes)
        dimensions = list(ds.dims)
        
        storage_overview = [
            dmc.Text(f"Shape: {data_shape}", size="sm", mb="xs"),
            dmc.Text(f"Dimensions: {', '.join(dimensions)}", size="sm", mb="md"),
            dag.AgGrid(
                id='storage-metadata-grid',
                rowData=var_metadata,
                columnDefs=column_defs_storage,
                defaultColDef={'sortable': True, 'filter': True, 'resizable': True},
                dashGridOptions={},
                style={'height': '500px', 'width': '100%'},
            ),
        ]
        print(f"[DEBUG] Step 1 complete: Storage overview created with {len(var_metadata)} variables")
        
        # ========== 2. BUILD EPOCH VARIABLES GRID ==========
        print("[DEBUG] Step 2: Building epoch variables grid...")
        
        def _make_df_from_dim_vars(dim_name):
            coord_values = ds.coords[dim_name].values
            data = {dim_name: coord_values}
            vars = {}
            for var_name in ds.coords:
                vars[var_name] = ds.coords[var_name]
            for var_name in ds.data_vars:
                vars[var_name] = ds[var_name]
            for var_name, var in tqdm(vars.items()):
                if dim_name in var.dims and len(var.dims) == 1:
                    data[var_name] = var.values
            return pd.DataFrame(data)

        def _convert_to_row_format(df):
            return df.to_dict(orient='records'), [{"field": i, "headerName": i, "width": 120} for i in df.columns]

        epoch_df = _make_df_from_dim_vars('epoch')
        epoch_row_data, epoch_column_defs = _convert_to_row_format(epoch_df)

        print(f"[DEBUG] Step 2 complete: Epoch grid - {len(epoch_row_data)} rows, {len(epoch_column_defs)} columns")

        # ========== 3. BUILD OBJECT VARIABLES GRID ==========
        print("[DEBUG] Step 3: Building object variables grid...")

        object_df = _make_df_from_dim_vars('object')
        object_row_data, object_column_defs = _convert_to_row_format(object_df)

        print(f"[DEBUG] Step 3 complete: Object grid - {len(object_row_data)} rows, {len(object_column_defs)} columns")
        
        # ========== RETURN ALL THREE OUTPUTS ==========
        print("[DEBUG] Returning all three table outputs:")
        print(f"[DEBUG]   - Storage overview: {len(var_metadata)} items")
        print(f"[DEBUG]   - Epoch grid: {len(epoch_row_data)} rows x {len(epoch_column_defs)} columns")
        print(f"[DEBUG]   - Object grid: {len(object_row_data)} rows x {len(object_column_defs)} columns")
        print("[DEBUG] Callback complete!")
        return (
            storage_overview,
            epoch_row_data,
            epoch_column_defs,
            object_row_data,
            object_column_defs,
            False  # enable select objects button
        )
        
    except Exception as e:
        import traceback
        print(f"[ERROR] build_info_tables_data failed:")
        print(f"  Error: {e}")
        print(f"  Traceback:\n{traceback.format_exc()}")
        
        empty_overview = [dmc.Text(f"Error: {str(e)}", c="red", size="sm")]
        return empty_overview, [], [], [], [], True


def register_info_tab_callbacks(app):
    """Register callbacks for the storage info tab."""
    
    @app.callback(
        [
            Output('storage-overview', 'children'),
            Output('epoch-variables-grid', 'rowData'),
            Output('epoch-variables-grid', 'columnDefs'),
            Output('object-variables-grid', 'rowData'),
            Output('object-variables-grid', 'columnDefs'),
            Output('select-objects-button', 'disabled'),
        ],
        Input('load-storage-button', 'n_clicks'),
        State('storage-data', 'data'),
        prevent_initial_call=True
    )
    def update_all_info_tables(n_clicks, storage_data):
        """Update all three info panels together to avoid race conditions with cache."""
        print("\n" + "="*80)
        print("[DEBUG] update_all_info_tables called")
        print(f"  n_clicks: {n_clicks}")
        print(f"  storage_data: {storage_data}")
        print("="*80 + "\n")
        
        if not n_clicks or not storage_data:
            print("[DEBUG] PreventUpdate: n_clicks or storage_data is None")
            raise PreventUpdate
        
        from pathlib import Path
        
        print(f"[DEBUG] Processing storage_data: {storage_data}")
        
        # Retrieve storage from cache - storage_data has 'storage_path' and 'mode' keys
        storage_path = Path(storage_data['storage_path'])
        mode = storage_data.get('mode', 'read')
        
        cache = StorageCache.get_instance()
        storage = cache.get(storage_path, mode)
        
        if storage is None:
            print("[ERROR] Storage not found in cache!")
            import dash_mantine_components as dmc
            empty_overview = [dmc.Text("Storage not loaded", c="red", size="sm")]
            return empty_overview, [], [], [], [], True
        
        print(f"[DEBUG] Retrieved storage from cache: {type(storage)}")
        
        # Call the extracted function to build all table data
        return build_info_tables_data(storage)
