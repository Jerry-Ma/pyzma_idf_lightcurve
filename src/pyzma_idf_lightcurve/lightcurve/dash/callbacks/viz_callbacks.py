"""Callbacks for visualization tab."""

from dash import Input, Output, State, callback, no_update, html, clientside_callback, ALL, ctx
from dash.exceptions import PreventUpdate
import dash_mantine_components as dmc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import logging
import time
import random
from functools import lru_cache
from astropy.stats import SigmaClip
from typing import NamedTuple, Optional

from ...datamodel import LightcurveStorage
from ..app import get_storage_cache

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================

class SmoothingResult(NamedTuple):
    """Result from apply_sigma_clipped_smoothing function.
    
    All arrays are in time-sorted order for consistency.
    """
    x_smoothed: np.ndarray  # Smoothed x values (empty if smoothing disabled)
    y_smoothed: np.ndarray  # Smoothed y values (empty if smoothing disabled)
    y_err_smoothed: np.ndarray  # Smoothed y errors (empty if smoothing disabled)
    rejected_mask: Optional[np.ndarray]  # Boolean mask of rejected points (sorted order, None if no filtering)
    rates: Optional[np.ndarray]  # Rate of change (dmag/dt) between consecutive points (sorted order)
    mag_first_point: Optional[np.ndarray]  # Magnitude of first point in each consecutive pair (sorted order)
    x_sorted: np.ndarray  # Original x data in sorted order
    y_sorted: np.ndarray  # Original y data in sorted order
    y_err_sorted: np.ndarray  # Original y errors in sorted order


# =============================================================================
# Magnitude Conversion Utilities
# =============================================================================

# LRU cache for lightcurve data extraction (per object, measurement key)
@lru_cache(maxsize=1000)
def _get_cached_lightcurve(storage_id: str, obj_key: str, meas_key: str):
    """Extract lightcurve data from storage with caching.
    
    Args:
        storage_id: Unique identifier for storage (path:mode)
        obj_key: Object identifier
        meas_key: Measurement key
        
    Returns:
        tuple: (x_data, y_values, y_errors) as numpy arrays
    """
    # This will be called from the plot generation callback
    # The actual storage object is passed via closure
    return None  # Placeholder - will be populated by wrapper


def clear_lightcurve_cache():
    """Clear the lightcurve cache. Call on module reload or storage change."""
    _get_cached_lightcurve.cache_clear()
    logger.info("🗑️  Cleared lightcurve cache")


def _is_mag_key(meas_key: str) -> bool:
    """Check if measurement key represents magnitude data."""
    return '-mag_' in meas_key


def _is_div_key(meas_key: str) -> bool:
    """Check if measurement key represents ratio/division lightcurve."""
    return '_div-' in meas_key


def _parse_superstack_mag_key(meas_key: str) -> str:
    """Parse measurement key to get corresponding superstack magnitude column.
    
    Args:
        meas_key: Measurement key like 'ch1_sci-mag_auto' or 'ch1_sci_div-biweight_location_area3px_sigclip3'
        
    Returns:
        Superstack column name like 'superstack_mag_auto'
    """
    # For mag keys: extract the magnitude type (mag_auto, mag_aper_1, etc.)
    if '-mag_' in meas_key:
        mag_type = meas_key.split('-mag_')[1]  # e.g., 'auto', 'aper_1'
        return f'superstack_mag_{mag_type}'
    
    # For div keys: use mag_auto as default reference
    # (ratio lightcurves are typically compared to aperture photometry)
    return 'superstack_mag_auto'


def _convert_ratio_to_mag(ratio_values, ratio_errors, mag_ref, magerr_ref):
    """Convert ratio lightcurve to magnitude scale.
    
    Args:
        ratio_values: Array of flux ratios (F_epoch / F_superstack)
        ratio_errors: Array of ratio uncertainties
        mag_ref: Reference magnitude from superstack
        magerr_ref: Reference magnitude uncertainty from superstack
        
    Returns:
        tuple: (mag_values, magerr_values) scaled to magnitude system
    """
    # Convert ratio to delta magnitude: dmag = -2.5 * log10(ratio)
    with np.errstate(divide='ignore', invalid='ignore'):
        dmag = -2.5 * np.log10(ratio_values)
        # Uncertainty propagation: dmagerr = 2.5 * log10(e) * (ratio_error / ratio)
        # where ratio_error is absolute error on ratio
        dmagerr = 2.5 * np.log10(np.e) * np.abs(ratio_errors / ratio_values)
    
    # Scale to reference magnitude: mag = mag_ref + dmag
    mag_values = mag_ref + dmag
    
    # Combine uncertainties: total_err = sqrt(dmagerr^2 + magerr_ref^2)
    magerr_values = np.hypot(dmagerr, magerr_ref)
    
    return mag_values, magerr_values


def _get_reference_magnitude(object_df, obj_key, superstack_col):
    """Get reference magnitude from object table.
    
    Args:
        object_df: DataFrame with object metadata
        obj_key: Object identifier
        superstack_col: Column name for magnitude (e.g., 'superstack_mag_auto')
        
    Returns:
        tuple: (mag_ref, magerr_ref) or (None, None) if not found
    """
    try:
        obj_row = object_df[object_df['object'] == obj_key]
        if len(obj_row) == 0:
            logger.warning(f"Object {obj_key} not found in object_df")
            return None, None
        
        # Get magnitude and error
        mag_ref = obj_row[superstack_col].values[0]
        magerr_col = superstack_col.replace('_mag_', '_magerr_')
        magerr_ref = obj_row[magerr_col].values[0]
        
        # Check for invalid values
        if not np.isfinite(mag_ref) or not np.isfinite(magerr_ref):
            logger.warning(f"Invalid reference magnitude for {obj_key}: {mag_ref} ± {magerr_ref}")
            return None, None
        
        return float(mag_ref), float(magerr_ref)
    except Exception as e:
        logger.error(f"Error getting reference magnitude: {e}")
        return None, None


def prepare_magnitude_data(
    lc_var,
    object_df,
    obj_key: str,
    meas_key: str,
    measurement_keys: list,
):
    """Prepare data for magnitude plotting with proper scaling.
    
    This function handles two types of measurements:
    1. Direct magnitude measurements (-mag_ keys): no conversion needed
    2. Ratio measurements (_div- keys): convert to magnitude using superstack reference
    
    When mixing both types, all data is scaled to the first magnitude key's reference.
    
    Args:
        lc_var: xarray DataArray with lightcurve data
        object_df: DataFrame with object metadata including superstack magnitudes
        obj_key: Object identifier
        meas_key: Current measurement key
        measurement_keys: List of all measurement keys being plotted
        
    Returns:
        tuple: (y_values, y_errors) as magnitude and magnitude errors
               or (None, None) if conversion fails
    """
    # Extract raw data
    obj_data = lc_var.sel(object=obj_key, measurement=meas_key)
    y_values_full = obj_data.values
    
    if y_values_full.ndim != 2 or y_values_full.shape[0] != 2:
        logger.warning(f"Unexpected data shape for {obj_key}/{meas_key}: {y_values_full.shape}")
        return None, None
    
    y_values = y_values_full[0, :]  # Values
    y_errors = y_values_full[1, :]  # Uncertainties
    
    # Case 1: This is a magnitude key - no conversion needed
    if _is_mag_key(meas_key):
        return y_values, y_errors
    
    # Case 2: This is a div key - convert to magnitude
    if _is_div_key(meas_key):
        # Find first mag key to use as reference
        mag_keys = [k for k in measurement_keys if _is_mag_key(k)]
        
        if len(mag_keys) == 0:
            # No mag keys - cannot scale, log warning but return raw ratio
            logger.warning(f"No magnitude keys found for scaling {meas_key} - plotting raw ratio")
            return y_values, y_errors
        
        # Get reference magnitude from superstack
        first_mag_key = mag_keys[0]
        superstack_col = _parse_superstack_mag_key(first_mag_key)
        mag_ref, magerr_ref = _get_reference_magnitude(object_df, obj_key, superstack_col)
        
        if mag_ref is None:
            logger.warning(f"Cannot get reference magnitude for {obj_key}/{meas_key}")
            return None, None
        
        # Convert ratio to magnitude
        mag_values, magerr_values = _convert_ratio_to_mag(
            y_values, y_errors, mag_ref, magerr_ref
        )
        
        logger.debug(f"Converted {obj_key}/{meas_key} to magnitude (ref: {mag_ref:.2f})")
        return mag_values, magerr_values
    
    # Case 3: Unknown key type - return as is
    logger.warning(f"Unknown measurement key type: {meas_key}")
    return y_values, y_errors


def apply_sigma_clipped_smoothing(x_data, y_data, y_error, 
                                  prefilter_window_enabled=False,
                                  prefilter_window_samples=None, prefilter_window_sigma=None,
                                  prefilter_quantile_enabled=False,
                                  rate_quantile_range=None, prefilter_quantile_sigma=None,
                                  smoothing_enabled=False,
                                  window_size_days=None, smoothing_sigma=None):
    """Apply multi-stage filtering and smoothing to lightcurve data.
    
    This function can apply up to three stages (all optional and independent):
    1. Pre-filtering: Fixed window sigma clip (if enabled)
    2. Pre-filtering: dMag/dt quantile clipping (if enabled)
    3. Smoothing: Time-window sigma clipping for local averaging (if enabled)
    
    Args:
        x_data: Time values (e.g., MJD) - must be in days or convertible to float
        y_data: Magnitude values
        y_error: Magnitude errors
        prefilter_window_enabled: Whether to apply fixed window sigma clip pre-filtering
        prefilter_window_samples: Number of samples in window for pre-filtering
        prefilter_window_sigma: Sigma threshold for window pre-filtering
        prefilter_quantile_enabled: Whether to apply dMag/dt quantile pre-filtering
        rate_quantile_range: [q_low, q_high] percentiles for rate clipping (e.g., [2, 98])
        prefilter_quantile_sigma: Sigma threshold for magnitude outlier identification in rate filtering
        smoothing_enabled: Whether to apply time window smoothing
        window_size_days: Size of time window in days for smoothing
        smoothing_sigma: Sigma threshold for clipping in smoothing stage
        
    Returns:
        SmoothingResult: Named tuple containing all results in sorted order.
                        Returns empty arrays and Nones if processing fails.
    """
    if len(x_data) < 3:
        logger.warning("Not enough points for smoothing (need at least 3)")
        return SmoothingResult(
            x_smoothed=np.array([]),
            y_smoothed=np.array([]),
            y_err_smoothed=np.array([]),
            rejected_mask=None,
            rates=None,
            mag_first_point=None,
            x_sorted=np.array([]),
            y_sorted=np.array([]),
            y_err_sorted=np.array([])
        )
    
    # Store original dtype for conversion back
    original_dtype = x_data.dtype
    is_timedelta = hasattr(x_data, 'dtype') and np.issubdtype(x_data.dtype, np.timedelta64)
    is_datetime = hasattr(x_data, 'dtype') and np.issubdtype(x_data.dtype, np.datetime64)
    
    # Convert x_data to float if it's a timedelta or datetime type
    if is_timedelta:
        x_data_float = x_data / np.timedelta64(1, 'D')
    elif is_datetime:
        x_data_float = x_data.astype('datetime64[D]').astype(float)
    else:
        x_data_float = np.asarray(x_data, dtype=float)
    
    # Sort data by time
    sort_idx = np.argsort(x_data_float)
    x_sorted = x_data_float[sort_idx]
    y_sorted = y_data[sort_idx]
    y_err_sorted = y_error[sort_idx]
    
    # Convert x_sorted back to original dtype immediately after sorting
    # This ensures all returns have consistent dtype
    if is_timedelta:
        x_sorted_original = (x_sorted * np.timedelta64(1, 'D')).astype(original_dtype)
    elif is_datetime:
        x_sorted_original = x_sorted.astype('datetime64[D]').astype(original_dtype)
    else:
        x_sorted_original = x_sorted
    
    # Calculate rates for all consecutive points on SORTED data (for later use in plotting)
    dmag_sorted = np.diff(y_sorted)
    dt_sorted = np.diff(x_sorted)
    valid_dt_sorted = dt_sorted != 0
    rates_sorted = np.full_like(dmag_sorted, np.nan)
    if np.any(valid_dt_sorted):
        rates_sorted[valid_dt_sorted] = dmag_sorted[valid_dt_sorted] / dt_sorted[valid_dt_sorted]
    mag_first_point_sorted = y_sorted[:-1]  # Magnitude of first point in each pair
    
    # Stage 1: Pre-filtering (can apply both methods independently)
    mask = np.ones(len(x_sorted), dtype=bool)
    rejected_mask_sorted = np.zeros(len(x_sorted), dtype=bool)
    
    # Pre-filtering 1: Fixed window sigma clip
    if prefilter_window_enabled and prefilter_window_samples is not None and prefilter_window_sigma is not None:
        logger.info(f"Pre-filtering (fixed window): n_samples={prefilter_window_samples}, sigma={prefilter_window_sigma}")
        half_window = prefilter_window_samples // 2
        
        # Create SigmaClip object for window filtering
        sigclip_window = SigmaClip(sigma=prefilter_window_sigma, maxiters=2)
        
        for i in range(len(x_sorted)):
            # Find surrounding points by index (fixed number of samples)
            start_idx = max(0, i - half_window)
            end_idx = min(len(x_sorted), i + half_window + 1)
            
            if end_idx - start_idx < 3:
                continue
            
            window_y = y_sorted[start_idx:end_idx]
            
            try:
                clipped = sigclip_window(window_y, masked=True)
                # Mark ALL points in the window that are clipped as rejected
                for j, is_clipped in enumerate(clipped.mask):
                    if is_clipped:
                        global_idx = start_idx + j
                        mask[global_idx] = False
                        rejected_mask_sorted[global_idx] = True
            except Exception as e:
                logger.warning(f"Window pre-filter failed at index {i}: {e}")
                continue
        
        n_rejected = np.sum(rejected_mask_sorted)
        logger.info(f"Window pre-filtering removed {n_rejected} outliers ({n_rejected/len(mask)*100:.1f}%)")
    
    # Pre-filtering 2: Rate-based filtering with magnitude outlier identification
    if prefilter_quantile_enabled and rate_quantile_range is not None and len(rate_quantile_range) == 2 and prefilter_quantile_sigma is not None:
        # Quantile-based pre-filtering: rate clipping
        q_low, q_high = rate_quantile_range
        logger.info(f"Pre-filtering (rate-based): rate quantiles [{q_low}, {q_high}]%, sigma={prefilter_quantile_sigma}")
        
        # Use the already calculated rates from sorted data
        # Avoid division by zero - only use valid rates
        if np.any(valid_dt_sorted):
            # Compute quantile thresholds on valid rates only
            rate_low = np.percentile(rates_sorted[valid_dt_sorted], q_low)
            rate_high = np.percentile(rates_sorted[valid_dt_sorted], q_high)
            
            # Create separate SigmaClip object for rate filtering
            sigclip_rate = SigmaClip(sigma=prefilter_quantile_sigma, maxiters=2)
            
            # Calculate sigma-clipped median of all magnitudes to identify outliers
            clipped_mags = sigclip_rate(y_sorted[mask], masked=True)
            median_mag = np.ma.median(clipped_mags)

            logger.info(f"Rate thresholds: [{rate_low:.4f}, {rate_high:.4f}] mag/day median_mag={median_mag}")
            
            # For each extreme rate, reject only the point that is farther from median
            for i in range(len(rates_sorted)):
                if valid_dt_sorted[i]:
                    if rates_sorted[i] < rate_low or rates_sorted[i] > rate_high:
                        # This rate connects point i to point i+1
                        # Reject the one farther from the sigma-clipped median
                        dist_i = np.abs(y_sorted[i] - median_mag)
                        dist_i_plus_1 = np.abs(y_sorted[i + 1] - median_mag)
                        
                        if dist_i > dist_i_plus_1:
                            rejected_mask_sorted[i] = True
                            mask[i] = False
                        else:
                            rejected_mask_sorted[i + 1] = True
                            mask[i + 1] = False
            
            n_rejected = np.sum(rejected_mask_sorted)
            logger.info(f"Rate-based pre-filtering rejected {n_rejected} points ({n_rejected/len(mask)*100:.1f}%)")
        else:
            logger.warning("No valid time intervals for rate calculation")
    
    # Apply mask to data
    x_masked = x_sorted[mask]
    y_masked = y_sorted[mask]
    y_err_masked = y_err_sorted[mask]
    
    if len(x_masked) < 3:
        logger.warning("Too few points remaining after pre-filtering")
        # Return rejection mask in sorted order (all data is sorted)
        has_prefilter = prefilter_window_enabled or prefilter_quantile_enabled
        return SmoothingResult(
            x_smoothed=np.array([]),
            y_smoothed=np.array([]),
            y_err_smoothed=np.array([]),
            rejected_mask=rejected_mask_sorted if has_prefilter else None,
            rates=rates_sorted,
            mag_first_point=mag_first_point_sorted,
            x_sorted=x_sorted_original,
            y_sorted=y_sorted,
            y_err_sorted=y_err_sorted
        )
    
    # Stage 3: Time window smoothing (optional)
    if not smoothing_enabled or window_size_days is None or smoothing_sigma is None:
        # No smoothing requested, return original data (after pre-filtering)
        has_prefilter = prefilter_window_enabled or prefilter_quantile_enabled
        return SmoothingResult(
            x_smoothed=x_masked,
            y_smoothed=y_masked,
            y_err_smoothed=y_err_masked,
            rejected_mask=rejected_mask_sorted if has_prefilter else None,
            rates=rates_sorted,
            mag_first_point=mag_first_point_sorted,
            x_sorted=x_sorted_original,
            y_sorted=y_sorted,
            y_err_sorted=y_err_sorted
        )
    
    x_smoothed = []
    y_smoothed = []
    y_err_smoothed = []
    
    half_window = window_size_days / 2.0
    
    # Create SigmaClip object for smoothing stage
    sigclip = SigmaClip(sigma=smoothing_sigma, maxiters=2)
    
    for i, t in enumerate(x_masked):
        # Find points within time window
        window_mask = np.abs(x_masked - t) <= half_window
        
        if np.sum(window_mask) < 2:
            continue
        
        window_y = y_masked[window_mask]
        window_err = y_err_masked[window_mask]
        
        # Apply sigma clipping
        try:
            clipped = sigclip(window_y, masked=True)
            
            if np.sum(~clipped.mask) == 0:
                continue
            
            # Calculate median and standard deviation from non-clipped points
            y_median = np.ma.median(clipped)
            y_std = np.ma.std(clipped)
            
            x_smoothed.append(t)
            y_smoothed.append(y_median)
            y_err_smoothed.append(y_std)
            
        except Exception as e:
            logger.warning(f"Sigma clipping failed at t={t}: {e}")
            continue
    
    x_smoothed = np.array(x_smoothed)
    
    # Convert x_smoothed and x_sorted back to original dtype
    if is_timedelta:
        x_smoothed = (x_smoothed * np.timedelta64(1, 'D')).astype(original_dtype)
    elif is_datetime:
        x_smoothed = x_smoothed.astype('datetime64[D]').astype(original_dtype)
    
    # Return rejection mask in sorted order (all data is sorted)
    has_prefilter = prefilter_window_enabled or prefilter_quantile_enabled
    
    return SmoothingResult(
        x_smoothed=x_smoothed,
        y_smoothed=np.array(y_smoothed),
        y_err_smoothed=np.array(y_err_smoothed),
        rejected_mask=rejected_mask_sorted if has_prefilter else None,
        rates=rates_sorted,
        mag_first_point=mag_first_point_sorted,
        x_sorted=x_sorted_original,
        y_sorted=y_sorted,
        y_err_sorted=y_err_sorted
    )


def _register_object_table_callback(app, cache):
    """Register callback to populate and update the object table."""
    
    @app.callback(
        [
            Output('object-table', 'rowData'),
            Output('filtered-object-list', 'data'),
            Output('autocomplete-options', 'data'),
        ],
        [
            Input('update-table-button', 'n_clicks'),
            Input('object-query-input', 'n_submit'),  # Trigger on Enter key
            Input('storage-data', 'data'),
        ],
        [
            State('object-query-input', 'value'),
            State('autocomplete-options', 'data'),
        ],
        prevent_initial_call=True,
    )
    def update_object_table(n_clicks, n_submit, storage_data, object_query, current_history):
        """Update the object table based on query filter."""
        if not storage_data:
            return [], [], current_history or []
        
        # Require a query - don't load all objects
        if not object_query or not object_query.strip():
            return [], [], current_history or []
        
        from pathlib import Path
        storage_path = Path(storage_data['storage_path'])
        mode = storage_data.get('mode', 'read')
        path_mode = f"{storage_path}:{mode}"
        
        object_df = cache.get(f"object_df:{path_mode}")
        if object_df is None:
            return [], [], current_history or []
        
        # Filter objects by query
        try:
            filtered_df = object_df.query(object_query.strip())
        except Exception as e:
            logger.error(f"Query filter failed: {e}")
            return [], [], current_history or []
        
        # Warn if no results
        if len(filtered_df) == 0:
            logger.warning("Query returned 0 objects")
            return [], [], current_history or []
        
        # Sort by superstack magnitude (use mag_auto as default)
        if 'superstack_mag_auto' in filtered_df.columns:
            filtered_df = filtered_df.sort_values('superstack_mag_auto')
        
        # Store filtered object list
        object_list = filtered_df['object'].tolist()
        
        # Get superstack magnitudes and errors for display
        if 'superstack_mag_auto' in filtered_df.columns:
            mags = filtered_df['superstack_mag_auto'].tolist()
        else:
            mags = [None] * len(object_list)
        
        if 'superstack_magerr_auto' in filtered_df.columns:
            magerrs = filtered_df['superstack_magerr_auto'].tolist()
        else:
            magerrs = [None] * len(object_list)
        
        # Create AG Grid row data
        row_data = []
        for idx, (obj, mag, magerr) in enumerate(zip(object_list, mags, magerrs)):
            mag_str = f"{mag:.2f}" if mag is not None and np.isfinite(mag) else "N/A"
            magerr_str = f"{magerr:.2f}" if magerr is not None and np.isfinite(magerr) else "N/A"
            row_data.append({
                "#": idx,
                "Object": str(obj),
                "Mag": mag_str,
                "MagErr": magerr_str,
            })
        
        # Update query history (keep last 20 queries)
        history_list = current_history or []
        query_str = object_query.strip()
        
        # Add query to history if not already there
        if query_str and query_str not in history_list:
            history_list.insert(0, query_str)
            # Keep last 20 queries, but preserve column names that may be in the list
            # Column names typically don't have operators like >, <, =, in, etc.
            queries_only = [q for q in history_list if any(op in q for op in ['>', '<', '=', 'in', '!=', '>=', '<='])]
            history_list = queries_only[:20]
        
        logger.info(f"Query returned {len(filtered_df)} objects")
        
        return row_data, object_list, history_list


def _register_selection_feedback_callback(app):
    """Register callback to provide feedback on object selection count."""
    
    @app.callback(
        Output('object-count-feedback', 'children'),
        Input('object-table', 'selectedRows'),
        prevent_initial_call=False,
    )
    def show_selection_feedback(selected_rows):
        """Show feedback if too many objects are selected."""
        MAX_PLOT_OBJECTS = 10
        
        if not selected_rows:
            return None
        
        n_selected = len(selected_rows)
        
        if n_selected > MAX_PLOT_OBJECTS:
            return dmc.Alert(
                f"⚠️ {n_selected} objects selected. Plotting first {MAX_PLOT_OBJECTS} only.",
                color="orange",
                variant="light",
                mt="xs"
            )
        else:
            return dmc.Alert(
                f"✓ {n_selected} object(s) selected for plotting.",
                color="blue",
                variant="light",
                mt="xs"
            )


def _register_navigation_callbacks(app, cache):
    """Register callbacks for navigation buttons and AG Grid row clicks."""
    
    @app.callback(
        Output('object-table', 'selectedRows', allow_duplicate=True),
        Input('object-table', 'rowData'),
        prevent_initial_call='initial_duplicate',
    )
    def auto_select_first_row(row_data):
        """Auto-select first row when table loads with data."""
        if row_data and len(row_data) > 0:
            return [row_data[0]]
        return []
    
    @app.callback(
        [
            Output('current-plot-index', 'data'),
            Output('object-table', 'selectedRows'),
        ],
        [
            Input('nav-prev5-button', 'n_clicks'),
            Input('nav-prev2-button', 'n_clicks'),
            Input('nav-prev1-button', 'n_clicks'),
            Input('nav-next1-button', 'n_clicks'),
            Input('nav-next2-button', 'n_clicks'),
            Input('nav-next5-button', 'n_clicks'),
            Input('object-table', 'selectedRows'),
        ],
        [
            State('current-plot-index', 'data'),
            State('filtered-object-list', 'data'),
            State('object-table', 'rowData'),
        ],
        prevent_initial_call=True,
    )
    def navigate_plots(prev5, prev2, prev1, next1, next2, next5, selected_rows, current_index, object_list, row_data):
        """Handle navigation button clicks and AG Grid row selection."""
        
        if not ctx.triggered or not object_list:
            raise PreventUpdate
        
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        # Get current selection range
        if selected_rows:
            selected_indices = [row['#'] for row in selected_rows]
            first_selected = min(selected_indices)
            last_selected = max(selected_indices)
        else:
            first_selected = current_index if current_index is not None else 0
            last_selected = first_selected
        
        # Handle navigation buttons (select next N objects)
        if trigger_id.startswith('nav-prev'):
            # Previous N buttons - select N objects ending at first_selected - 1
            if trigger_id == 'nav-prev5-button':
                n = 5
            elif trigger_id == 'nav-prev2-button':
                n = 2
            else:  # nav-prev1-button
                n = 1
            end_index = first_selected
            start_index = max(0, end_index - n)
            new_index = start_index
        elif trigger_id.startswith('nav-next'):
            # Next N buttons - select N objects starting from last_selected + 1
            if trigger_id == 'nav-next5-button':
                n = 5
            elif trigger_id == 'nav-next2-button':
                n = 2
            else:  # nav-next1-button
                n = 1
            start_index = last_selected + 1
            end_index = min(len(object_list), start_index + n)
            new_index = start_index
        elif trigger_id == 'object-table' and selected_rows:
            # Handle AG Grid row selection
            new_index = selected_rows[0]['#']  # Get row index from selected row data
            logger.info(f"Navigation: {current_index} → {new_index}")
            return new_index, no_update  # Don't update selectedRows, it's already set
        else:
            raise PreventUpdate
        
        # For navigation buttons, select the range of rows
        logger.info(f"Navigation: selecting rows {start_index} to {end_index-1}")
        new_selected = [row for row in row_data if start_index <= row['#'] < end_index]
        return new_index, new_selected


def _register_smoothing_control_callback(app):
    """Register callback to enable/disable smoothing parameter inputs."""
    
    @app.callback(
        [
            Output('smoothing-prefilter-window', 'disabled'),
            Output('smoothing-prefilter-sigma', 'disabled'),
            Output('smoothing-window-size', 'disabled'),
            Output('smoothing-sigma', 'disabled'),
        ],
        Input('smoothing-enabled', 'checked'),
    )
    def toggle_smoothing_controls(enabled):
        """Enable or disable smoothing parameter inputs based on switch state."""
        return not enabled, not enabled, not enabled, not enabled


def register_viz_callbacks(app):
    """Register all visualization-related callbacks."""
    cache = get_storage_cache()
    
    # Clear lightcurve cache on registration (module reload)
    clear_lightcurve_cache()
    
    _register_query_autocomplete_callback(app, cache)
    _register_viz_controls_callback(app)
    _register_query_validation_callback(app)
    _register_object_table_callback(app, cache)
    _register_selection_feedback_callback(app)
    _register_navigation_callbacks(app, cache)
    _register_smoothing_control_callback(app)
    _register_plot_generation_callback(app, cache)


def _register_query_validation_callback(app):
    """Register callback for real-time object query validation."""
    
    @app.callback(
        [
            Output('object-query-validation', 'children'),
            Output('query-is-valid', 'data'),
            Output('update-table-button', 'disabled'),
            Output('query-input-wrapper', 'style'),
        ],
        [
            Input('object-query-input', 'value'),
            Input('storage-data', 'data'),
        ],
        prevent_initial_call=False
    )
    def validate_object_query(query, storage_data):
        """Validate pandas query syntax for object selection."""
        
        default_style = {'flex': '1'}
        error_style = {'flex': '1', 'border': '2px solid red', 'borderRadius': '4px', 'padding': '2px'}
        
        # Empty query is invalid - require a query
        if not query or not query.strip():
            return (
                dmc.Text("Enter a query filter (required)", c="gray", size="sm", mt="xs"),
                False,
                True,
                default_style
            )
        
        # No storage loaded - can't validate column names yet
        if not storage_data:
            return (
                dmc.Alert(
                    "⚠️ Load storage first to validate query",
                    color="yellow",
                    variant="light",
                    mt="xs"
                ),
                False,
                True,
                error_style
            )
        
        # Try to validate query syntax
        try:
            # Get object DataFrame from cache to validate against real columns
            from pathlib import Path
            storage_path = Path(storage_data['storage_path'])
            mode = storage_data.get('mode', 'read')
            
            cache = get_storage_cache()
            path_mode = f"{storage_path}:{mode}"
            df_cache_key = f"object_df:{path_mode}"
            object_df = cache.get(df_cache_key)
            
            if object_df is not None:
                # Try to apply query to actual DataFrame
                result = object_df.query(query.strip())
                n_matched = len(result)
                return (
                    dmc.Alert(
                        f"✓ Query valid: matches {n_matched} objects",
                        color="green",
                        variant="light",
                        mt="xs"
                    ),
                    True,
                    False,
                    default_style
                )
            else:
                # Fallback: basic syntax validation
                mock_df = pd.DataFrame({'object': [1, 2, 3], 'ra': [180, 181, 182], 'dec': [30, 31, 32]})
                mock_df.query(query.strip())
                return (
                    dmc.Alert(
                        "✓ Query syntax is valid (columns will be checked at plot time)",
                        color="blue",
                        variant="light",
                        mt="xs"
                    ),
                    True,
                    False,
                    default_style
                )
        except SyntaxError as e:
            return (
                dmc.Alert(
                    f"❌ Syntax error: {str(e)}",
                    color="red",
                    variant="light",
                    mt="xs"
                ),
                False,
                True,
                error_style
            )
        except Exception as e:
            return (
                dmc.Alert(
                    f"⚠️ Error: {str(e)}",
                    color="yellow",
                    variant="light",
                    mt="xs"
                ),
                False,
                True,
                error_style
            )


def _register_query_autocomplete_callback(app, cache):
    """Register callback to populate query input with column names for autocomplete."""
    
    @app.callback(
        Output('autocomplete-options', 'data', allow_duplicate=True),
        Input('storage-data', 'data'),
        State('autocomplete-options', 'data'),
        prevent_initial_call=True
    )
    def update_query_autocomplete(storage_data, current_data):
        """Update query input with filtered column names (object table, object, stats) for autocomplete."""
        if not storage_data:
            return current_data or []
        
        # Build cache key from storage_data
        from pathlib import Path
        storage_path = Path(storage_data['storage_path'])
        mode = storage_data.get('mode', 'read')
        path_mode = f"{storage_path}:{mode}"
        
        # Get dataframes from cache
        object_df = cache.get(f"object_df:{path_mode}")
        epoch_df = cache.get(f"epoch_df:{path_mode}")
        
        if object_df is None:
            return current_data or []
        
        # Filter to queryable columns: object table columns, object, and stats columns
        column_names = []
        
        # Add object column explicitly
        if 'object' in object_df.columns:
            column_names.append('object')
        
        # Identify object table columns and stats columns from object_df
        if epoch_df is not None:
            epoch_cols = set(epoch_df.columns)
            for col in object_df.columns:
                if col == 'object':
                    continue  # Already added
                if col.startswith('_'):
                    continue  # Skip internal columns
                # Object table columns exist in both epoch_df and object_df
                if col in epoch_cols:
                    column_names.append(col)
                # Object stats columns only exist in object_df (not in epoch_df)
                else:
                    column_names.append(col)
        else:
            # No epoch_df, add all object_df columns
            for col in object_df.columns:
                if col != 'object' and not col.startswith('_'):
                    column_names.append(col)
        
        column_names = sorted(column_names)
        
        # Merge with existing history (if any) - history items first
        existing_history = current_data or []
        
        # Create combined list: history first, then column names (avoiding duplicates)
        combined = []
        seen = set()
        
        # Add history items first
        for item in existing_history:
            if item not in seen:
                combined.append(item)
                seen.add(item)
        
        # Add column names that aren't in history
        for col in column_names:
            if col not in seen:
                combined.append(col)
                seen.add(col)
        
        logger.info(f"Updated query autocomplete with {len(column_names)} filtered columns and {len(existing_history)} history items")
        return combined


def _register_smoothing_control_callback(app):
    """Register callbacks to enable/disable controls for each smoothing/filtering method."""
    
    # Pre-filtering: Fixed window sigma clip controls
    @app.callback(
        [
            Output('smoothing-prefilter-window', 'disabled'),
            Output('smoothing-prefilter-sigma', 'disabled'),
        ],
        Input('prefilter-window-enabled', 'checked'),
    )
    def toggle_window_prefilter_controls(enabled):
        """Enable/disable window pre-filter controls."""
        disabled = not enabled
        return disabled, disabled
    
    # Pre-filtering: dMag/dt quantile controls
    @app.callback(
        [
            Output('rate-quantile-range', 'disabled'),
            Output('rate-quantile-symmetric', 'disabled'),
            Output('prefilter-quantile-sigma', 'disabled'),
        ],
        Input('prefilter-quantile-enabled', 'checked'),
    )
    def toggle_quantile_prefilter_controls(enabled):
        """Enable/disable quantile pre-filter controls."""
        disabled = not enabled
        return disabled, disabled, disabled
    
    # Fixed time window smoothing controls
    @app.callback(
        [
            Output('smoothing-window-size', 'disabled'),
            Output('smoothing-sigma', 'disabled'),
        ],
        Input('smoothing-enabled', 'checked'),
    )
    def toggle_smoothing_controls(enabled):
        """Enable/disable smoothing controls."""
        disabled = not enabled
        return disabled, disabled


def _register_symmetric_quantile_callback(app):
    """Register callback to enforce symmetric quantiles when switch is enabled."""
    
    @app.callback(
        Output('rate-quantile-range', 'value'),
        Input('rate-quantile-range', 'value'),
        State('rate-quantile-symmetric', 'checked'),
        prevent_initial_call=True,
    )
    def enforce_symmetric_quantiles(value, symmetric):
        """Enforce symmetric quantiles when switch is enabled.
        
        Ensures q_low + q_high = 100 for balanced outlier removal.
        Example: [2, 98], [5, 95], [10, 90]
        """
        if not symmetric or value is None or len(value) != 2:
            return no_update
        
        q_low, q_high = value
        
        # Check if already symmetric (within tolerance)
        if abs((q_low + q_high) - 100) < 0.1:
            return no_update
        
        # Determine which endpoint changed by checking which is closer to 50
        # Adjust the other endpoint to maintain symmetry
        if abs(q_low - 50) < abs(q_high - 50):
            # q_low changed less, adjust q_high to match
            new_high = 100 - q_low
        else:
            # q_high changed less, adjust q_low to match
            new_low = 100 - q_high
            return [new_low, q_high]
        
        # Ensure within bounds [0, 100]
        new_high = max(0, min(100, new_high))
        new_low = 100 - new_high
        
        return [new_low, new_high]


def _register_viz_controls_callback(app):
    """Register callback for updating visualization controls."""
    
    @app.callback(
        [
            Output('measurement-keys-select', 'data'),
            Output('measurement-keys-select', 'value'),
            Output('x-axis-select', 'data'),
            Output('x-axis-select', 'value'),
        ],
        [
            Input('storage-data', 'data'),
        ],
        prevent_initial_call=True
    )
    def update_viz_controls(storage_data):
        """Update visualization control options from xarray coordinates."""
        if not storage_data:
            return [], [], [], None
        
        # Get storage from cache to access metadata
        from pathlib import Path
        
        storage_path = Path(storage_data['storage_path'])
        mode = storage_data.get('mode', 'read')
        
        # Get cached storage from diskcache (shared across processes)
        cache = get_storage_cache()
        cache_key = f"storage:{storage_path}:{mode}"
        storage = cache.get(cache_key)
        
        if storage is None or storage.lightcurves is None:
            return [], [], [], None
        
        lc_var = storage.lightcurves
        
        # Measurement keys - extract from 'measurement' coordinate (vectorized access)
        print("[DEBUG] Extracting measurement coordinate values...")
        if 'measurement' in lc_var.coords:
            measurement_values = lc_var.coords['measurement'].values
            measurement_options = [
                {'value': str(meas_key), 'label': str(meas_key)}
                for meas_key in measurement_values
            ]
            # Pre-select first measurement
            default_measurement = [str(measurement_values[0])] if len(measurement_values) > 0 else []
            print(f"[DEBUG] Found {len(measurement_options)} measurements, pre-selected: {default_measurement}")
        else:
            measurement_options = []
            default_measurement = []
            print("[DEBUG] No 'measurement' coordinate found")
        
        # X-axis options - get all epoch table columns from cached DataFrame
        print("[DEBUG] Loading epoch metadata from cache...")
        cache_key_epoch = f"epoch_df:{storage_path}:{mode}"
        epoch_df = cache.get(cache_key_epoch)
        
        x_axis_options = []
        default_x_axis = None
        if epoch_df is not None:
            # Add all columns from epoch DataFrame as x-axis options
            for col in epoch_df.columns:
                x_axis_options.append({'value': col, 'label': col})
            print(f"[DEBUG] Found {len(x_axis_options)} epoch columns for x-axis: {list(epoch_df.columns)}")
            
            # Set default x-axis value (prefer time-related columns)
            if 'epoch_mjd' in epoch_df.columns:
                default_x_axis = 'epoch_mjd'
            elif 'epoch_t0' in epoch_df.columns:
                default_x_axis = 'epoch_t0'
            elif 'epoch_reqbegintime' in epoch_df.columns:
                default_x_axis = 'epoch_reqbegintime'
            elif 'epoch' in epoch_df.columns:
                default_x_axis = 'epoch'
            elif len(x_axis_options) > 0:
                default_x_axis = x_axis_options[0]['value']
        else:
            # Fallback: scan lightcurve coordinates if epoch_df not in cache
            print("[DEBUG] Epoch DataFrame not found in cache, falling back to coordinate scan...")
            for coord in lc_var.coords:
                if coord == 'epoch':
                    x_axis_options.append({'value': coord, 'label': coord})
                elif 'epoch' in lc_var.coords[coord].dims and len(lc_var.coords[coord].dims) == 1:
                    x_axis_options.append({'value': coord, 'label': coord})
            print(f"[DEBUG] Found {len(x_axis_options)} epoch coordinates for x-axis")
            if len(x_axis_options) > 0:
                default_x_axis = x_axis_options[0]['value']
        
        print(f"[DEBUG] Returning {len(measurement_options)} measurement options (default: {default_measurement}), {len(x_axis_options)} x-axis options (default: {default_x_axis})")
        return measurement_options, default_measurement, x_axis_options, default_x_axis


def _register_plot_generation_callback(app, cache):
    """Register callback for generating lightcurve plots."""
    
    @app.callback(
        [
            Output('lightcurve-plot', 'figure'),
            Output('lightcurve-plot', 'style'),
        ],
        [
            Input('current-plot-index', 'data'),
            Input('object-table', 'selectedRows'),
            Input('measurement-keys-select', 'value'),
            Input('x-axis-select', 'value'),
            Input('prefilter-window-enabled', 'checked'),
            Input('smoothing-prefilter-window', 'value'),
            Input('smoothing-prefilter-sigma', 'value'),
            Input('prefilter-quantile-enabled', 'checked'),
            Input('rate-quantile-range', 'value'),
            Input('prefilter-quantile-sigma', 'value'),
            Input('smoothing-enabled', 'checked'),
            Input('smoothing-window-size', 'value'),
            Input('smoothing-sigma', 'value'),
        ],
        [
            State('storage-data', 'data'),
            State('filtered-object-list', 'data'),
        ],
        prevent_initial_call=True,
    )
    def generate_lightcurve_plots(
        plot_index,
        selected_rows,
        measurement_keys,
        x_axis_key,
        prefilter_window_enabled,
        smoothing_prefilter_window,
        smoothing_prefilter_sigma,
        prefilter_quantile_enabled,
        rate_quantile_range,
        prefilter_quantile_sigma,
        smoothing_enabled,
        smoothing_window_days,
        smoothing_sigma,
        storage_data,
        object_list,
    ):
        """Generate lightcurve plots for selected objects (multi-object support).
        
        Args:
            plot_index: Current object index (fallback if no selection)
            selected_rows: List of selected rows from AG Grid
            measurement_keys: List of measurement columns to plot
            x_axis_key: Column to use for x-axis
            prefilter_window_enabled: Whether to apply fixed window sigma clip pre-filtering
            smoothing_prefilter_window: Pre-filter window size (n_samples)
            smoothing_prefilter_sigma: Pre-filter sigma threshold for window filtering
            prefilter_quantile_enabled: Whether to apply dMag/dt quantile pre-filtering
            rate_quantile_range: [q_low, q_high] percentiles for rate clipping
            prefilter_quantile_sigma: Sigma threshold for magnitude outlier identification in rate filtering
            smoothing_enabled: Whether to apply time window smoothing
            smoothing_window_days: Window size in days for smoothing
            smoothing_sigma: Sigma threshold for outlier rejection in smoothing
            storage_data: Storage metadata dict with path and mode
            object_list: Filtered list of object IDs
            
        Returns:
            tuple: (plotly figure, style dict for container)
        """
        # Always combine measurements on same panel
        combine_measurements = True
        
        start_time = time.time()
        logger.info("="*80)
        logger.info("🚀 PLOT GENERATION STARTED")
        logger.info(f"   Plot index: {plot_index}")
        logger.info(f"   Selected rows: {len(selected_rows) if selected_rows else 0}")
        logger.info(f"   Object list length: {len(object_list) if object_list else 0}")
        logger.info(f"   Measurements: {measurement_keys}")
        logger.info(f"   X-axis: {x_axis_key}")
        logger.info(f"   Combine measurements: {combine_measurements}")
        
        if not storage_data or not object_list:
            raise PreventUpdate
        
        # Determine which objects to plot
        MAX_PLOT_OBJECTS = 10
        objects_to_plot = []
        if selected_rows and len(selected_rows) > 0:
            # Use selected rows from AG Grid
            for row in selected_rows:
                if 'Object' in row:
                    obj_key = row['Object']
                    if obj_key in object_list:
                        objects_to_plot.append(obj_key)
            
            # Enforce limit
            if len(objects_to_plot) > MAX_PLOT_OBJECTS:
                logger.warning(f"   {len(objects_to_plot)} objects selected, limiting to first {MAX_PLOT_OBJECTS}")
                objects_to_plot = objects_to_plot[:MAX_PLOT_OBJECTS]
            
            logger.info(f"   Plotting {len(objects_to_plot)} selected objects: {objects_to_plot}")
        elif plot_index is not None and 0 <= plot_index < len(object_list):
            # Fallback to single object from plot_index
            objects_to_plot = [object_list[plot_index]]
            logger.info(f"   Plotting single object from index: {objects_to_plot[0]}")
        else:
            logger.warning("   No valid objects to plot")
            raise PreventUpdate
        
        # Validate measurement keys
        if not measurement_keys:
            logger.warning("⚠️  No measurement keys selected")
            raise PreventUpdate
        
        # Build cache key from storage_data
        from pathlib import Path
        storage_path = Path(storage_data['storage_path'])
        mode = storage_data.get('mode', 'read')
        path_mode = f"{storage_path}:{mode}"

        # Get data from cache
        object_df = cache.get(f"object_df:{path_mode}")
        epoch_df = cache.get(f"epoch_df:{path_mode}")
        storage = cache.get(f"storage:{path_mode}")
        
        if object_df is None or epoch_df is None or storage is None:
            logger.error("❌ Required data not found in cache")
            raise PreventUpdate
        
        # Get lightcurve data
        lc_var = storage.lightcurves
        
        # Determine subplot layout based on multi-object and combine_measurements
        n_objects = len(objects_to_plot)
        
        if combine_measurements:
            # One row per object, all measurements combined
            n_rows = n_objects
            n_cols = 1
            subplot_titles = [f"Object {obj}" for obj in objects_to_plot]
        else:
            # One row per (object, measurement) pair
            n_rows = n_objects * len(measurement_keys)
            n_cols = 1
            subplot_titles = []
            for obj in objects_to_plot:
                for meas in measurement_keys:
                    subplot_titles.append(f"Object {obj} - {meas}")
        
        logger.info(f"🎨 Creating {n_rows} subplots (2 columns: lightcurve + rate diagnostic) for {n_objects} objects with {len(measurement_keys)} measurements...")
        
        # Create subplot figure with 2 columns: lightcurve on left, rate diagnostic on right
        # Share y-axis between columns since both plot magnitude
        subplot_titles_2col = []
        for title in subplot_titles:
            subplot_titles_2col.append(title)
            subplot_titles_2col.append(f"{title} (Rate Diagnostic)")
        
        fig = make_subplots(
            rows=n_rows,
            cols=2,
            subplot_titles=subplot_titles_2col,
            vertical_spacing=0.05 if n_rows > 1 else 0.08,
            horizontal_spacing=0.08,
            specs=[[{"secondary_y": False}, {"secondary_y": False}] for _ in range(n_rows)],
            column_widths=[0.6, 0.4],
            shared_yaxes=True
        )
        
        # Store reference magnitudes for legends
        ref_mags = {}
        
        # Plot each object
        current_row = 1
        traces_added = 0
        traces_failed = 0
        
        for obj_idx, obj_key in enumerate(objects_to_plot):
            logger.info(f"📊 Plotting object {obj_idx+1}/{n_objects}: {obj_key}")
            
            # Plot each measurement for this object
            for meas_idx, meas_key in enumerate(measurement_keys):
                try:
                    # Determine subplot row
                    if combine_measurements:
                        # All measurements for this object go to same row
                        subplot_row = obj_idx + 1
                    else:
                        # Each measurement gets its own row
                        subplot_row = current_row
                        current_row += 1
                    
                    # Prepare magnitude data with proper conversion
                    result = prepare_magnitude_data(
                        lc_var, object_df, obj_key, meas_key, measurement_keys
                    )
                    
                    if result[0] is None:
                        logger.warning(f"Failed to prepare data for {obj_key}/{meas_key}")
                        traces_failed += 1
                        continue
                    
                    y_data, y_error = result
                    
                    # Get reference magnitude for div keys (include in trace name)
                    trace_name = meas_key
                    if _is_div_key(meas_key):
                        # Find first mag key to get reference
                        mag_keys = [k for k in measurement_keys if _is_mag_key(k)]
                        if mag_keys:
                            first_mag_key = mag_keys[0]
                            superstack_col = _parse_superstack_mag_key(first_mag_key)
                            mag_ref, magerr_ref = _get_reference_magnitude(object_df, obj_key, superstack_col)
                            if mag_ref is not None:
                                ref_key = f"{obj_key}:{meas_key}"
                                ref_mags[ref_key] = (mag_ref, magerr_ref)
                                # Include ref mag in trace name for legend
                                trace_name = f"{obj_key}-{meas_key} (ref: {mag_ref:.2f}±{magerr_ref:.3f})"
                    else:
                        trace_name = f"{obj_key}-{meas_key}"
                    
                    # Get x-axis data
                    obj_data = lc_var.sel(object=obj_key, measurement=meas_key)
                    if x_axis_key and x_axis_key in obj_data.coords:
                        x_data = obj_data.coords[x_axis_key].values
                    elif x_axis_key and x_axis_key in epoch_df.columns:
                        x_data = epoch_df[x_axis_key].values
                    else:
                        x_data = obj_data.coords['epoch'].values
                    
                    # Filter data for valid magnitude measurements
                    valid_mask = (y_data < 90) & np.isfinite(y_data) & (y_error < 1) & np.isfinite(y_error)
                    if np.sum(valid_mask) > 0:
                        x_data = x_data[valid_mask]
                        y_data = y_data[valid_mask]
                        y_error = y_error[valid_mask]
                        logger.info(f"[{obj_idx}.{meas_idx}] Kept {np.sum(valid_mask)} valid points for {obj_key}/{meas_key}")
                    else:
                        logger.warning(f"[{obj_idx}.{meas_idx}] No valid points for {obj_key}/{meas_key}")
                        traces_failed += 1
                        continue
                    
                    # Use ScatterGL for better performance with large datasets
                    # Washed-out errorbars: semi-transparent color
                    error_color = 'rgba(100, 100, 100, 0.3)'
                    marker_size = 4  # Reduced marker size
                    
                    # Assign different colors for each measurement
                    color_palette = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']
                    marker_color = color_palette[meas_idx % len(color_palette)]
                    
                    # Always process data through apply_sigma_clipped_smoothing to get consistent sorted data
                    # This ensures x_data_sorted, y_data_sorted, rates, and rejected_mask are all aligned
                    logger.info(f"Processing data for consistent sorting:")
                    if prefilter_window_enabled or prefilter_quantile_enabled or smoothing_enabled:
                        logger.info(f"  Pre-filter/smoothing enabled:")
                        if prefilter_window_enabled:
                            logger.info(f"    Pre-filter (fixed window): n_samples={smoothing_prefilter_window}, sigma={smoothing_prefilter_sigma}")
                        if prefilter_quantile_enabled:
                            logger.info(f"    Pre-filter (quantile): rate quantiles {rate_quantile_range}%")
                        if smoothing_enabled:
                            logger.info(f"    Smoothing: window={smoothing_window_days} days, sigma={smoothing_sigma}")
                    
                    result = apply_sigma_clipped_smoothing(
                        x_data, y_data, y_error,
                        prefilter_window_enabled=prefilter_window_enabled,
                        prefilter_window_samples=smoothing_prefilter_window,
                        prefilter_window_sigma=smoothing_prefilter_sigma,
                        prefilter_quantile_enabled=prefilter_quantile_enabled,
                        rate_quantile_range=rate_quantile_range,
                        prefilter_quantile_sigma=prefilter_quantile_sigma,
                        smoothing_enabled=smoothing_enabled,
                        window_size_days=smoothing_window_days,
                        smoothing_sigma=smoothing_sigma
                    )
                    
                    # Plot raw data (sorted - returned from apply_sigma_clipped_smoothing)
                    fig.add_trace(
                        go.Scattergl(
                            x=result.x_sorted,
                            y=result.y_sorted,
                            error_y=dict(
                                type='data',
                                array=result.y_err_sorted,
                                visible=True,
                                color=error_color,
                                thickness=1,
                                width=0,
                            ),
                            mode='markers',
                            name=trace_name,  # Use enhanced name with ref mags
                            showlegend=True,
                            marker=dict(
                                size=marker_size,
                                color=marker_color,
                                line=dict(width=0.5, color='white'),
                            ),
                        ),
                        row=subplot_row,
                        col=1
                    )
                    traces_added += 1
                    
                    # Add rate diagnostic plot using precomputed rates (sorted and aligned)
                    if result.rates is not None and result.mag_first_point is not None and len(result.rates) > 0:
                        # Filter out NaN rates
                        valid_rates = ~np.isnan(result.rates)
                        if np.any(valid_rates):
                            # Plot rate diagnostic
                            fig.add_trace(
                                go.Scattergl(
                                    x=result.rates[valid_rates],
                                    y=result.mag_first_point[valid_rates],
                                    mode='markers',
                                    name=f"{trace_name} (rate)",
                                    showlegend=False,
                                    marker=dict(
                                        size=4,
                                        color=marker_color,
                                        line=dict(width=0.5, color='white'),
                                    ),
                                ),
                                row=subplot_row,
                                col=2
                            )
                            
                            # Add vertical line at rate=0
                            fig.add_shape(
                                type="line",
                                x0=0, x1=0,
                                y0=np.min(result.mag_first_point[valid_rates]), y1=np.max(result.mag_first_point[valid_rates]),
                                line=dict(color="gray", width=1, dash="dash"),
                                row=subplot_row,
                                col=2
                            )
                    
                    # Plot rejected points if any
                    if result.rejected_mask is not None and np.any(result.rejected_mask):
                        # Plot rejected points with cross symbols in lightcurve (use sorted data)
                        x_rejected = result.x_sorted[result.rejected_mask]
                        y_rejected = result.y_sorted[result.rejected_mask]
                        y_err_rejected = result.y_err_sorted[result.rejected_mask]
                        
                        fig.add_trace(
                            go.Scattergl(
                                x=x_rejected,
                                y=y_rejected,
                                error_y=dict(
                                    type='data',
                                    array=y_err_rejected,
                                    visible=True,
                                    color='rgba(255, 0, 0, 0.3)',
                                    thickness=1,
                                    width=0,
                                ),
                                mode='markers',
                                name=f"{trace_name} (rejected)",
                                showlegend=True,
                                marker=dict(
                                    size=8,
                                    color='red',
                                    symbol='x',
                                    line=dict(width=2),
                                ),
                            ),
                            row=subplot_row,
                            col=1
                        )
                        traces_added += 1
                        
                        # Also plot rejected points in rate diagnostic
                        # rejected_mask indices directly correspond to sorted data indices
                        if result.rates is not None and result.mag_first_point is not None:
                            # Rates are calculated between consecutive points, so mask needs to be trimmed
                            rejected_mask_rates = result.rejected_mask[:-1]
                            
                            # Apply mask to get rejected rates
                            rate_rej = result.rates[rejected_mask_rates]
                            mag_rej = result.mag_first_point[rejected_mask_rates]
                            
                            # Filter out NaN rates
                            valid_rej = ~np.isnan(rate_rej)
                            rate_rej_list = rate_rej[valid_rej].tolist()
                            mag_rej_list = mag_rej[valid_rej].tolist()
                            
                            if len(rate_rej_list) > 0:
                                fig.add_trace(
                                    go.Scattergl(
                                        x=rate_rej_list,
                                        y=mag_rej_list,
                                        mode='markers',
                                        name=f"{trace_name} (rejected rate)",
                                        showlegend=False,
                                        marker=dict(
                                            size=6,
                                            color='red',
                                            symbol='x',
                                            line=dict(width=1.5),
                                        ),
                                    ),
                                    row=subplot_row,
                                    col=2
                                )
                    
                    if smoothing_enabled and len(result.x_smoothed) > 0:
                        # Plot smoothed data as a line on top
                        logger.info(f"Adding smoothed trace: {len(result.x_smoothed)} points")
                        logger.info(f"  X range: [{result.x_smoothed.min()}, {result.x_smoothed.max()}]")
                        logger.info(f"  Y range: [{result.y_smoothed.min()}, {result.y_smoothed.max()}]")
                        logger.info(f"  Raw X range: [{result.x_sorted.min()}, {result.x_sorted.max()}]")
                        logger.info(f"  Raw Y range: [{result.y_sorted.min()}, {result.y_sorted.max()}]")
                        fig.add_trace(
                            go.Scattergl(
                                x=result.x_smoothed,
                                y=result.y_smoothed,
                                error_y=dict(
                                    type='data',
                                    array=result.y_err_smoothed,
                                    visible=True,
                                    color='rgba(255, 0, 255, 0.5)',  # Bright magenta, semi-transparent
                                    thickness=2,
                                    width=3,
                                ),
                                mode='lines+markers',
                                name=f"{trace_name} (smoothed)",
                                showlegend=True,
                                line=dict(
                                    color='magenta',  # Bright magenta line
                                    width=3,
                                ),
                                marker=dict(
                                    size=8,  # Larger markers
                                    color='magenta',  # Bright magenta
                                    symbol='diamond',
                                    line=dict(width=1, color='white'),
                                ),
                            ),
                            row=subplot_row,
                            col=1
                        )
                        logger.info(f"Added smoothed trace with {len(result.x_smoothed)} points")
                    elif smoothing_enabled:
                        logger.warning(f"Smoothing enabled but returned no points for {obj_key}/{meas_key} (input had {len(result.x_sorted)} points)")
                    else:
                        logger.info(f"Smoothing not enabled for {obj_key}/{meas_key}")
                    
                except Exception as e:
                    traces_failed += 1
                    logger.warning(f"Failed to plot {obj_key} - {meas_key}: {e}")
                    import traceback
                    logger.warning(traceback.format_exc())
            
            # Reset current_row if not combining (already incremented in loop)
            if not combine_measurements:
                current_row = current_row  # Already incremented
        
        logger.info(f"📊 Traces added: {traces_added}, failed: {traces_failed}")
        
        # Calculate flexible height: minimum 800px (doubled), or 300px per row
        plot_height = max(800, 300 * n_rows)
        
        fig.update_layout(
            height=plot_height,
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor="lightgray",
                borderwidth=1,
            ),
            title_text=f"{n_objects} Object(s) - {len(measurement_keys)} measurements",
            margin=dict(r=200),  # Add right margin for legend
        )
        
        # Invert y-axis for all subplots (magnitude convention)
        # Only set y-axis properties on left column since they're shared
        for row_idx in range(1, n_rows + 1):
            # Left column: lightcurve (y-axis properties apply to both columns)
            fig.update_yaxes(autorange='reversed', title_text="Magnitude", row=row_idx, col=1)
            fig.update_xaxes(title_text=x_axis_key or "Epoch", row=row_idx, col=1)
            
            # Right column: rate diagnostic (only update x-axis, y-axis is shared)
            fig.update_xaxes(title_text="dMag/dt (mag/day)", row=row_idx, col=2)
        
        logger.info(f"✅ PLOT GENERATION COMPLETE ({time.time() - start_time:.2f}s)")
        logger.info("="*80)
        
        # Return flexible height style (doubled minimum)
        return fig, {"height": f"{plot_height}px", "minHeight": "800px"}


def register_viz_callbacks(app):
    """Register all visualization tab callbacks.
    
    Args:
        app: Dash application instance
    """
    cache = get_storage_cache()
    
    # Register all callbacks
    _register_object_table_callback(app, cache)
    _register_selection_feedback_callback(app)
    _register_navigation_callbacks(app, cache)
    _register_query_autocomplete_callback(app, cache)
    _register_smoothing_control_callback(app)
    _register_symmetric_quantile_callback(app)
    _register_viz_controls_callback(app)
    _register_plot_generation_callback(app, cache)