"""Callbacks for storage info tab."""

from typing import Any
from dash import Input, Output, State, callback, no_update
from dash.exceptions import PreventUpdate
import dash_mantine_components as dmc
import dash_ag_grid as dag
from tqdm import tqdm
import pandas as pd
import logging

from ...datamodel import LightcurveStorage
from ..app import get_storage_cache

# Configure logging
logger = logging.getLogger(__name__)

# Cache key format constants
STORAGE_CACHE_KEY_FMT = "storage:{path}:{mode}"
DATAFRAME_CACHE_KEY_FMT = "{df_type}_df:{path}:{mode}"

# Module-level in-memory cache for DataFrames (faster access than diskcache)
# Type: dict with 'epoch' and 'object' keys containing DataFrames, 'cache_key' containing storage key
_df_cache: dict[str, pd.DataFrame | str | None] = {
    'epoch': None,
    'object': None,
    'cache_key': None,  # Track which data is cached
}


def _get_cached_dataframe(cache_key_storage: str, df_type: str):
    """Get DataFrame from two-tier cache (in-memory first, then diskcache).
    
    This function implements a two-tier caching strategy:
    1. First checks module-level in-memory cache for fastest access
    2. Falls back to diskcache if not in memory
    3. Updates in-memory cache on diskcache hit for future speed
    
    Args:
        cache_key_storage: Cache key for storage in format "storage:{path}:{mode}"
        df_type: Type of DataFrame ('epoch' or 'object')
        
    Returns:
        pandas.DataFrame or None if not found in either cache
    """
    logger.debug(f"_get_cached_dataframe(cache_key={cache_key_storage}, df_type={df_type})")
    
    # Check in-memory cache first (fastest path)
    if _df_cache['cache_key'] == cache_key_storage and _df_cache[df_type] is not None:
        df_cached = _df_cache[df_type]
        assert isinstance(df_cached, pd.DataFrame), "Cache should contain DataFrame"
        logger.debug(f"In-memory cache HIT for {df_type} (shape: {df_cached.shape})")
        return df_cached
    
    logger.debug(f"In-memory cache MISS for {df_type}, checking diskcache...")
    
    # Construct diskcache key from storage cache key
    # Format: "epoch_df:{path}:{mode}" or "object_df:{path}:{mode}"
    path_mode = cache_key_storage.replace('storage:', '')
    df_cache_key = f"{df_type}_df:{path_mode}"
    
    # Load from diskcache
    cache = get_storage_cache()
    df = cache.get(df_cache_key)  # type: ignore[assignment]
    
    if df is not None and isinstance(df, pd.DataFrame):
        logger.info(f"Diskcache HIT for {df_type} (shape: {df.shape}, key: {df_cache_key})")
        # Update in-memory cache for faster subsequent access
        _df_cache[df_type] = df  # type: ignore[assignment]
        _df_cache['cache_key'] = cache_key_storage
        return df
    
    logger.warning(f"Diskcache MISS for {df_type} (key: {df_cache_key})")
    return None


def build_info_tables_data(storage, storage_path, storage_mode):
    """Build data for all three info tables from storage.
    
    This function is separated for testing purposes.
    
    Args:
        storage: LightcurveStorage instance with loaded dataset
        storage_path: Path string used in cache keys (from storage_data)
        storage_mode: Mode string 'read' or 'write' (from storage_data)
        
    Returns:
        tuple: (storage_overview_components, epoch_row_data, epoch_column_defs,
                object_row_data, object_column_defs, select_button_disabled)
    """
    print("[DEBUG] build_info_tables_data called")
    print(f"[DEBUG]   storage_path: {storage_path}")
    print(f"[DEBUG]   storage_mode: {storage_mode}")
    
    if storage is None:
        print("[ERROR] Storage is None")
        empty_overview = [dmc.Text("Storage not loaded", c="red", size="sm")]
        return empty_overview, [], [], True
    
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
        
        # ========== 2. GET PRE-COMPUTED DATAFRAMES FROM CACHE ==========
        print("[DEBUG] Step 2: Getting pre-computed DataFrames from cache...")
        
        # Use the storage_path and storage_mode passed in (from storage_data)
        # These match exactly what was used in the background callback
        cache_key_storage = f"storage:{storage_path}:{storage_mode}"
        
        # Load DataFrames from cache (they're stored with keys like "epoch_df:{path}:{mode}")
        cache = get_storage_cache()
        cache_key_epoch = f"epoch_df:{storage_path}:{storage_mode}"
        cache_key_object = f"object_df:{storage_path}:{storage_mode}"
        
        print(f"[DEBUG]   Cache key base: {cache_key_storage}")
        print(f"[DEBUG]   Looking for epoch_df with key: {cache_key_epoch}")
        print(f"[DEBUG]   Looking for object_df with key: {cache_key_object}")
        
        epoch_df = cache.get(cache_key_epoch)
        object_df = cache.get(cache_key_object)
        
        # Also update in-memory cache for fast access by getRows callbacks
        global _df_cache
        if epoch_df is not None and object_df is not None:
            _df_cache['epoch'] = epoch_df
            _df_cache['object'] = object_df
            _df_cache['cache_key'] = cache_key_storage
            print(f"[DEBUG]   Updated in-memory cache for getRows callbacks")
            print(f"[DEBUG]     In-memory cache now contains:")
            print(f"[DEBUG]       - epoch: {epoch_df.shape} rows")
            print(f"[DEBUG]       - object: {object_df.shape} rows")
            print(f"[DEBUG]       - cache_key: {cache_key_storage}")
        
        if epoch_df is None or object_df is None:
            error_msg = "Pre-computed DataFrames not found in cache. Reload storage."
            print(f"[ERROR] {error_msg}")
            empty_overview = [dmc.Text(error_msg, c="red", size="sm")]
            return empty_overview, [], [], True
        
        print(f"[DEBUG] Step 2 complete: Retrieved DataFrames")
        print(f"[DEBUG]   Epoch DF: {epoch_df.shape}")
        print(f"[DEBUG]   Object DF: {object_df.shape}")
        
        # ========== 3. BUILD COLUMN DEFINITIONS (NO ROW DATA) ==========
        print("[DEBUG] Step 3: Building column definitions for infinite scroll...")
        
        def _build_column_defs(df, sort_field=None):
            """Build AG Grid column definitions for infinite row model."""
            column_defs = []
            for col in df.columns:
                col_def = {
                    "field": col,
                    "headerName": col,
                    "width": 150 if col in ['object', 'ra', 'dec'] else 120,
                    "filter": "agTextColumnFilter" if df[col].dtype == 'object' else "agNumberColumnFilter",
                }
                if sort_field and col == sort_field:
                    col_def["sort"] = "asc"
                    col_def["sortIndex"] = 0
                column_defs.append(col_def)
            return column_defs
        
        epoch_column_defs = _build_column_defs(epoch_df, sort_field='epoch' if 'epoch' in epoch_df.columns else None)
        object_column_defs = _build_column_defs(object_df, sort_field='object' if 'object' in object_df.columns else None)
        
        print(f"[DEBUG] Step 3 complete: Column definitions created")
        print(f"[DEBUG]   Epoch: {len(epoch_column_defs)} columns - {[c['field'] for c in epoch_column_defs[:5]]}...")
        print(f"[DEBUG]   Object: {len(object_column_defs)} columns - {[c['field'] for c in object_column_defs[:5]]}...")
        
        # ========== 4. RETURN OUTPUTS (NO ROW DATA - INFINITE SCROLL USES getRows CALLBACKS) ==========
        print("[DEBUG] Step 4: Returning outputs for infinite scroll mode...")
        print(f"[DEBUG]   - Storage overview: {len(var_metadata)} items")
        print(f"[DEBUG]   - Epoch grid: column defs only (getRows will provide data)")
        print(f"[DEBUG]   - Object grid: column defs only (getRows will provide data)")
        print(f"[DEBUG]   - DataFrames cached for getRows callbacks:")
        print(f"[DEBUG]     * Epoch: {epoch_df.shape[0]} rows")
        print(f"[DEBUG]     * Object: {object_df.shape[0]} rows")
        print("[DEBUG] Callback complete! getRows callbacks will now handle data loading.")
        
        # For infinite scroll: DO NOT set rowData at all!
        # AG Grid will call getRows callbacks to lazy-load data
        # Setting rowData (even to []) interferes with infinite scroll
        return (
            storage_overview,
            epoch_column_defs,
            object_column_defs,
            False  # enable select objects button
        )
        
    except Exception as e:
        import traceback
        print(f"[ERROR] build_info_tables_data failed:")
        print(f"  Error: {e}")
        print(f"  Traceback:\n{traceback.format_exc()}")
        
        empty_overview = [dmc.Text(f"Error: {str(e)}", c="red", size="sm")]
        return empty_overview, [], [], True


def register_info_tab_callbacks(app):
    """Register callbacks for the storage info tab."""
    
    print("\n" + "="*80)
    print("[DEBUG] Registering info tab callbacks...")
    print("="*80 + "\n")
    
    @app.callback(
        [
            Output('storage-overview', 'children'),
            Output('epoch-variables-grid', 'columnDefs'),
            Output('object-variables-grid', 'columnDefs'),
            Output('select-objects-button', 'disabled'),
        ],
        Input('storage-data', 'data'),
        prevent_initial_call=True
    )
    def update_all_info_tables(storage_data):
        """Update info panels when storage data changes.
        
        This callback is triggered when new storage data is loaded via the background
        callback. It:
        1. Loads the LightcurveStorage from the zarr path
        2. Fetches pre-computed DataFrames from cache (or computes if not cached)
        3. Builds column definitions for AG Grid infinite scroll
        4. Returns metadata overview and column definitions (NOT rowData)
        
        For AG Grid Infinite Scroll:
        - This callback only returns columnDefs (with initial sort)
        - The getRows callbacks (get_epoch_rows, get_object_rows) handle data loading
        - Never return rowData output - it breaks infinite scroll
        - Initial sort in columnDefs triggers AG Grid to call getRows
        
        Args:
            storage_data: Dict with 'storage_path' and 'mode' keys
            
        Returns:
            tuple: (storage_overview, epoch_columnDefs, object_columnDefs, select_button_disabled)
        
        Note: Background callbacks run in separate processes, so in-memory cache
        is not shared. This callback reloads from zarr/diskcache each time.
        """
        print("\n" + "="*80)
        print("[DEBUG] update_all_info_tables called")
        print(f"  storage_data: {storage_data}")
        print("="*80 + "\n")
        
        if not storage_data:
            print("[DEBUG] PreventUpdate: storage_data is None")
            raise PreventUpdate
        
        from pathlib import Path
        
        print(f"[DEBUG] Processing storage_data: {storage_data}")
        
        # Get cached storage from diskcache (shared across processes)
        storage_path = Path(storage_data['storage_path'])
        mode = storage_data.get('mode', 'read')
        
        cache = get_storage_cache()
        cache_key = f"storage:{storage_path}:{mode}"
        storage = cache.get(cache_key)
        
        print(f"[DEBUG] Retrieved storage from cache: key={cache_key}, found={storage is not None}")
        
        if storage is None:
            print(f"[ERROR] Storage not found in diskcache - was it loaded correctly?")
            import dash_mantine_components as dmc
            empty_overview = [dmc.Text("Storage not found in cache. Please reload.", c="red", size="sm")]
            return empty_overview, [], [], True
        
        print(f"[DEBUG] Storage retrieved successfully: {type(storage)}")
        
        # Call the extracted function to build all table data
        # Pass the storage_path and mode from storage_data to ensure cache keys match
        return build_info_tables_data(storage, str(storage_path), mode)
    
    @app.callback(
        [
            Output('object-query-input', 'value'),
            Output('main-tabs', 'value'),
        ],
        Input('select-objects-button', 'n_clicks'),
        State('object-variables-grid', 'selectedRows'),
        prevent_initial_call=True
    )
    def select_objects_for_viz(n_clicks, selected_rows):
        """Transfer selected objects from info tab to visualization tab.
        
        Gets selected rows from the object variables grid and generates a query
        string for the viz tab, then switches to viz tab.
        """
        if not n_clicks or not selected_rows:
            raise PreventUpdate
        
        logger.info(f"select_objects_for_viz: {len(selected_rows)} rows selected")
        
        # Extract object IDs from selected rows
        selected_object_ids = [row.get('object', '') for row in selected_rows if 'object' in row]
        
        if not selected_object_ids:
            raise PreventUpdate
        
        # Generate a pandas query string using 'in' operator
        # Format: object in ["I20000", "I24000"] or object in [123, 456]
        # Simpler and more readable than .isin() syntax
        
        # Determine if numeric or string IDs
        try:
            # Try to convert first ID to numeric
            float(selected_object_ids[0])
            is_numeric = True
        except (ValueError, TypeError):
            is_numeric = False
        
        # Generate query string with appropriate format
        if is_numeric:
            # Numeric IDs: object in [123, 456]
            query_string = f"object in {selected_object_ids}"
        else:
            # String IDs: object in ["I20000", "I24000"]
            # Use double quotes for consistency
            quoted_ids = [f'"{obj_id}"' for obj_id in selected_object_ids]
            query_string = f"object in [{', '.join(quoted_ids)}]"
        
        logger.info(f"Generated query: {query_string}")
        
        # Switch to viz tab and populate object query
        return query_string, "viz"
    
    # ========== INFINITE SCROLL: SHARED IMPLEMENTATION ==========
    
    def _handle_infinite_scroll_request(
        request: dict | None,
        storage_data: dict | None,
        df_type: str,
        debug_label: str
    ):
        """Shared implementation for infinite scroll getRows callbacks.
        
        This function handles AG Grid's infinite scroll data requests by:
        1. Validating the request and storage data
        2. Retrieving the cached DataFrame
        3. Applying sorting and filtering as requested by AG Grid
        4. Slicing the appropriate page of data
        5. Returning the data with total row count
        
        Args:
            request: AG Grid's getRowsRequest with startRow, endRow, sortModel, filterModel
            storage_data: Storage metadata dict with path and mode
            df_type: Type of DataFrame to retrieve ('epoch' or 'object')
            debug_label: Label for debug logging (e.g., 'Epoch', 'Object')
            
        Returns:
            dict with 'rowData' (list of row dicts) and 'rowCount' (total rows after filtering),
            or no_update if request/storage_data is None
        """
        if request is None or storage_data is None:
            print(f"[{debug_label.upper()}] Invalid request or storage_data - returning no_update")
            return no_update
        
        # Get DataFrame from cache
        cache_key_storage = f"storage:{storage_data['storage_path']}:{storage_data.get('mode', 'read')}"
        df = _get_cached_dataframe(cache_key_storage, df_type)
        
        if df is None:
            print(f"[ERROR] {debug_label} DataFrame not found in cache")
            return {'rowData': [], 'rowCount': 0}
        
        # Make defensive copy to avoid modifying cached DataFrame
        df = df.copy()
        
        # Apply sorting (can have multiple sort columns)
        sort_model = request.get('sortModel', [])
        if sort_model:
            sort_columns = [spec['colId'] for spec in sort_model]
            sort_ascending = [spec['sort'] == 'asc' for spec in sort_model]
            df = df.sort_values(by=sort_columns, ascending=sort_ascending)
        
        # Apply filtering
        filter_model = request.get('filterModel', {})
        if filter_model:
            for col, filter_spec in filter_model.items():
                if col not in df.columns:
                    continue
                    
                filter_type = filter_spec.get('filterType')
                
                if filter_type == 'text':
                    filter_val = filter_spec.get('filter', '')
                    if filter_val:
                        df = df[df[col].astype(str).str.contains(filter_val, case=False, na=False)]
                        
                elif filter_type == 'number':
                    filter_val = filter_spec.get('filter')
                    if filter_val is not None:
                        filter_op = filter_spec.get('type', 'equals')
                        if filter_op == 'equals':
                            df = df[df[col] == filter_val]
                        elif filter_op == 'notEqual':
                            df = df[df[col] != filter_val]
                        elif filter_op == 'greaterThan':
                            df = df[df[col] > filter_val]
                        elif filter_op == 'greaterThanOrEqual':
                            df = df[df[col] >= filter_val]
                        elif filter_op == 'lessThan':
                            df = df[df[col] < filter_val]
                        elif filter_op == 'lessThanOrEqual':
                            df = df[df[col] <= filter_val]
        
        # Get total row count after filtering/sorting
        total_rows = len(df)
        
        # Slice rows based on request
        start_row = request.get('startRow', 0)
        end_row = request.get('endRow', 100)
        rows_subset = df.iloc[start_row:end_row]
        
        # Convert to records
        rows_data = rows_subset.to_dict('records')
        
        print(f"[{debug_label.upper()}] Returning {len(rows_data)}/{total_rows} rows")
        
        return {
            'rowData': rows_data,
            'rowCount': total_rows
        }
    
    # ========== INFINITE SCROLL: getRows CALLBACKS ==========
    
    @app.callback(
        Output('epoch-variables-grid', 'getRowsResponse'),
        Input('epoch-variables-grid', 'getRowsRequest'),
        State('storage-data', 'data'),
        prevent_initial_call=True
    )
    def get_epoch_rows(request, storage_data):
        """Server-side data source for epoch grid infinite scroll."""
        return _handle_infinite_scroll_request(request, storage_data, 'epoch', 'Epoch')
    
    @app.callback(
        Output('object-variables-grid', 'getRowsResponse'),
        Input('object-variables-grid', 'getRowsRequest'),
        State('storage-data', 'data'),
        prevent_initial_call=True
    )
    def get_object_rows(request, storage_data):
        """Server-side data source for object grid infinite scroll."""
        return _handle_infinite_scroll_request(request, storage_data, 'object', 'Object')
    
    print("\n" + "="*80)
    print("[DEBUG] Info tab callbacks registered successfully!")
    print("[DEBUG] Registered callbacks:")
    print("  - update_all_info_tables (storage-data → grids)")
    print("  - select_objects_for_viz (button → viz tab)")
    print("  - get_epoch_rows (epoch-variables-grid.getRowsRequest)")
    print("  - get_object_rows (object-variables-grid.getRowsRequest)")
    print("="*80 + "\n")

