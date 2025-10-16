# IDF Lightcurve Dash Visualization App

Modern interactive visualization application for IDF lightcurve data using **Plotly Dash v3**.

## Technology Stack

- **Dash v3.2.0** - Modern web framework with automatic React 18.2+ support
- **Dash Mantine Components v2.3.0** - Beautiful UI components
- **Dash AG Grid v32.3.2** - High-performance data tables
- **Plotly v5+** - Interactive plotting library

## Features

- **Storage Loader**: Load zarr storage files (read or write optimized)
- **Storage Info Tab**: Browse storage metadata, epoch variables, and object variables with AG Grid tables
- **Visualization Tab**: Create multi-object lightcurve plots with customizable controls
- **Image Viewer**: Display FITS images with source overlay and adjustable normalization

## Installation

The Dash app requires Dash v3 and companion libraries:

```bash
pip install "dash>=3.0.0,<4.0.0" "dash-mantine-components>=0.14.0" "dash-ag-grid>=31.0.0" plotly
```

## Usage

### Using CLI (Recommended)

```bash
# Start the visualization app
idflc viz --port 8050 --debug

# View help
idflc viz --help
```

### From Python

```python
from pyzma_idf_lightcurve.lightcurve.dash import run_app

# Run with default settings
run_app(debug=True, port=8050)
```

### From Command Line

```bash
# Navigate to the dash directory
cd src/pyzma_idf_lightcurve/lightcurve/dash

# Run the app
python test_app_run.py
```

Then open your browser to `http://localhost:8050`

## Workflow

1. **Load Storage**:
   - Enter path to zarr storage base directory (e.g., `scratch_dagster/idf_lightcurves.zarr`)
   - Select storage mode (read-optimized or write-optimized)
   - Click "Load Storage"

2. **Browse Storage Info**:
   - View storage overview (shape, dimensions, chunks)
   - Browse epoch variables table
   - Browse object variables table with coordinates (ra, dec, x_image, y_image)
   - Select objects for visualization using checkboxes
   - Click "Select for Visualization" to send to viz tab

3. **Generate Lightcurve Plots**:
   - Select objects (auto-populated from info tab selection)
   - Select measurement types
   - Choose x-axis variable (e.g., epoch_mjd)
   - Optionally add query filter
   - Click "Generate Plots"
   - View interactive subplots with error bars

4. **View Images (Optional)**:
   - Expand "Image Viewer" accordion
   - Enter path to FITS image file
   - Adjust normalization (linear, log, sqrt, asinh)
   - Adjust stretch percentiles
   - Choose colormap
   - Click "Load Image"
   - Selected objects will be overlaid as markers

## Architecture

### Components (`components/`)
- `storage_loader.py`: File path input, mode selector, load button
- `storage_info_tab.py`: Overview panel and AG Grid tables
- `viz_tab.py`: Control panel, plot area, image viewer accordion

### Callbacks (`callbacks/`)
- `storage_callbacks.py`: Load storage and extract metadata
- `info_tab_callbacks.py`: Update grids, handle object selection
- `viz_callbacks.py`: Update controls, generate plots
- `image_callbacks.py`: Load and display images with overlay

### Utilities (`utils/`)
- `plotting.py`: Plotly figure creation functions
- `image_utils.py`: FITS image loading and normalization

## API Reference

### LightcurveStorage Integration

The app uses the `LightcurveStorage` API from `datamodel.py`:

```python
# Load storage
storage = LightcurveStorage.load_for_per_object_read(path)

# Get storage info
info = storage.get_storage_info()

# Get lightcurve for specific object
lc = storage.get_object_lightcurve(object_index)

# Access xarray dataset directly
ds = storage.ds
```

### Expected Data Structure

The app expects xarray DataArray with:

**Dimensions:**
- `object`: Number of objects (sources)
- `measurement`: Number of measurement types (e.g., different apertures)
- `value`: Data values (mag, mag_err, flux, etc.)
- `epoch`: Number of time points

**Coordinates:**
- `ra`, `dec`: Sky coordinates
- `x_image`, `y_image`: Image pixel coordinates

**Variables:**
- `epoch_*`: Time-indexed data (e.g., `epoch_mjd`, `epoch_aor_key`)
- `object_*`: Object-specific data (e.g., `object_id`)

## Performance Notes

- Object table limited to 10,000 rows for browser performance
- Maximum 20 objects in subplot plots
- Storage info cached in `dcc.Store` (memory)
- AG Grid uses pagination and filtering for large datasets

## Troubleshooting

**Storage Not Loading:**
- Check file path is correct
- Verify storage mode matches actual storage type
- Check console for error messages

**Plots Not Generating:**
- Ensure objects and measurements are selected
- Check that storage has data for selected objects
- Look for NaN values in data

**Image Not Displaying:**
- Verify FITS file path exists
- Check FITS file format (should be 2D image)
- Try adjusting stretch percentiles

## Development

To modify the app:

1. Edit components in `components/` folder
2. Update callbacks in `callbacks/` folder
3. Add utility functions to `utils/` folder
4. Test changes by running app locally

## Dependencies

- `dash`: Web application framework
- `dash-mantine-components`: UI components
- `dash-ag-grid`: High-performance data tables
- `plotly`: Interactive plots
- `astropy`: FITS file handling
- `xarray`: Multi-dimensional array operations
- `zarr`: Storage backend
