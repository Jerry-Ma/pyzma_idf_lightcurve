#!/usr/bin/env python
"""
Realistic IDF Lightcurve Storage Benchmarks.

This test suite accurately mimics the real IDF storage structure:
- 50K objects (typical IDF field size)
- 1000 epochs (multi-year monitoring)
- 40 measurements (various apertures and methods)
- 2 values (mag, mag_err)
- ~50 variables per dimension (from tbl_aor and source catalog)

This is the critical benchmark that reveals the real performance bottleneck:
When xarray has many variables (50+ columns), extracting coordinates becomes
significantly slower because it needs to iterate through all variables.

Run benchmarks:
    # Run all realistic benchmarks
    uv run pytest tests/test_realistic_idf_benchmarks.py --benchmark-only -v
    
    # Run only coordinate extraction benchmarks
    uv run pytest tests/test_realistic_idf_benchmarks.py::TestRealisticCoordinateExtraction --benchmark-only -v
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import dask.array as da

# ============================================================================
# FIXTURES - Realistic IDF Dataset
# ============================================================================


@pytest.fixture(scope="module")
def realistic_temp_dir():
    """Module-scoped temporary directory for realistic benchmarks."""
    temp_path = Path(tempfile.mkdtemp(prefix="realistic_idf_benchmark_"))
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture(scope="module")
def realistic_idf_dataset():
    """Create realistic IDF storage dataset structure.
    
    Mimics the actual IDF lightcurve storage with:
    - 50K objects (typical IDF field)
    - 1000 epochs (multi-year monitoring)
    - 40 measurements (various apertures: AUTO, ISO, APER_1-38)
    - 2 values (mag, mag_err)
    - ~50 variables for object dimension (from source catalog)
    - ~50 variables for epoch dimension (from tbl_aor)
    - ~10 variables for measurement dimension
    """
    np.random.seed(42)
    
    # Dimensions matching real IDF storage
    n_objects = 50_000
    n_epochs = 1_000
    n_measurements = 40
    n_values = 2
    
    print(f"\n[SETUP] Creating realistic IDF dataset: {n_objects=} {n_epochs=} {n_measurements=} {n_values=}")
    
    # Create dimension coordinates
    object_keys = [f"I{i+1}" for i in range(n_objects)]
    epoch_keys = [f"r{58520832 + i * 256}" for i in range(n_epochs)]
    measurement_keys = ['auto', 'iso'] + [f'aper_{i}' for i in range(1, n_measurements-1)]
    value_keys = ['mag', 'mag_err']
    
    # Create the main data variable (lightcurves) with dask
    print("[SETUP] Creating lightcurve data array (dask-backed)...")
    lightcurve_data = da.random.uniform(
        15, 25, 
        size=(n_objects, n_epochs, n_measurements, n_values),
        chunks=(5000, -1, -1, -1)  # Chunk along object dimension
    )
    
    # Create ~50 variables for OBJECT dimension (from source catalog)
    print("[SETUP] Creating 50 object variables (source catalog columns)...")
    object_vars = {
        # Spatial coordinates
        'ra': (['object'], np.random.uniform(17.5, 17.7, n_objects)),
        'dec': (['object'], np.random.uniform(-29.9, -29.7, n_objects)),
        'x_image': (['object'], np.random.uniform(0, 750, n_objects)),
        'y_image': (['object'], np.random.uniform(0, 750, n_objects)),
        
        # SExtractor photometry outputs (~30 columns)
        'flux_auto': (['object'], np.random.uniform(100, 10000, n_objects)),
        'fluxerr_auto': (['object'], np.random.uniform(1, 100, n_objects)),
        'flux_iso': (['object'], np.random.uniform(100, 10000, n_objects)),
        'fluxerr_iso': (['object'], np.random.uniform(1, 100, n_objects)),
        'flux_aper': (['object'], np.random.uniform(100, 10000, n_objects)),
        'fluxerr_aper': (['object'], np.random.uniform(1, 100, n_objects)),
        'kron_radius': (['object'], np.random.uniform(1, 10, n_objects)),
        'fwhm_image': (['object'], np.random.uniform(2, 5, n_objects)),
        'ellipticity': (['object'], np.random.uniform(0, 0.8, n_objects)),
        'theta_image': (['object'], np.random.uniform(-90, 90, n_objects)),
        'a_image': (['object'], np.random.uniform(2, 10, n_objects)),
        'b_image': (['object'], np.random.uniform(1, 8, n_objects)),
        'isoarea_image': (['object'], np.random.uniform(10, 1000, n_objects).astype(int)),
        'flags': (['object'], np.random.randint(0, 8, n_objects)),
        'class_star': (['object'], np.random.uniform(0, 1, n_objects)),
        
        # Additional photometric columns
        'background': (['object'], np.random.uniform(100, 500, n_objects)),
        'threshold': (['object'], np.random.uniform(10, 50, n_objects)),
        'flux_max': (['object'], np.random.uniform(1000, 50000, n_objects)),
        'flux_min': (['object'], np.random.uniform(0, 100, n_objects)),
        'x_win': (['object'], np.random.uniform(0, 750, n_objects)),
        'y_win': (['object'], np.random.uniform(0, 750, n_objects)),
        'xpeak_image': (['object'], np.random.uniform(0, 750, n_objects).astype(int)),
        'ypeak_image': (['object'], np.random.uniform(0, 750, n_objects).astype(int)),
        
        # Morphology parameters
        'elongation': (['object'], np.random.uniform(1, 5, n_objects)),
        'flux_radius': (['object'], np.random.uniform(2, 8, n_objects)),
        'petro_radius': (['object'], np.random.uniform(3, 15, n_objects)),
        'x2_image': (['object'], np.random.uniform(1, 100, n_objects)),
        'y2_image': (['object'], np.random.uniform(1, 100, n_objects)),
        'xy_image': (['object'], np.random.uniform(-50, 50, n_objects)),
        
        # Astrometry
        'errx2_image': (['object'], np.random.uniform(0.01, 1, n_objects)),
        'erry2_image': (['object'], np.random.uniform(0.01, 1, n_objects)),
        'errxy_image': (['object'], np.random.uniform(-0.5, 0.5, n_objects)),
        'errcxx_image': (['object'], np.random.uniform(0.001, 0.1, n_objects)),
        'errcyy_image': (['object'], np.random.uniform(0.001, 0.1, n_objects)),
        'errcxy_image': (['object'], np.random.uniform(-0.05, 0.05, n_objects)),
        'erra_image': (['object'], np.random.uniform(0.1, 2, n_objects)),
        'errb_image': (['object'], np.random.uniform(0.1, 2, n_objects)),
        'errtheta_image': (['object'], np.random.uniform(0, 180, n_objects)),
        
        # Detection metadata
        'number': (['object'], np.arange(1, n_objects + 1)),
        'alpha_j2000': (['object'], np.random.uniform(17.5, 17.7, n_objects)),
        'delta_j2000': (['object'], np.random.uniform(-29.9, -29.7, n_objects)),
        'x2_world': (['object'], np.random.uniform(0, 0.001, n_objects)),
        'y2_world': (['object'], np.random.uniform(0, 0.001, n_objects)),
        'xy_world': (['object'], np.random.uniform(-0.0005, 0.0005, n_objects)),
        'cxx_image': (['object'], np.random.uniform(0.001, 0.1, n_objects)),
        'cyy_image': (['object'], np.random.uniform(0.001, 0.1, n_objects)),
        'cxy_image': (['object'], np.random.uniform(-0.05, 0.05, n_objects)),
    }
    print(f"[SETUP] Created {len(object_vars)} object variables")
    
    # Create ~50 variables for EPOCH dimension (from tbl_aor)
    print("[SETUP] Creating 50 epoch variables (tbl_aor columns)...")
    epoch_vars = {
        # Temporal information
        'mjd': (['epoch'], np.random.uniform(55000, 59000, n_epochs)),
        'date_obs': (['epoch'], np.array([f"2010-{i%12+1:02d}-{i%28+1:02d}" for i in range(n_epochs)])),
        'ut_time': (['epoch'], np.random.uniform(0, 24, n_epochs)),
        
        # AOR/observation metadata (~30 columns)
        'aor_key': (['epoch'], epoch_keys),
        'program_id': (['epoch'], np.random.randint(10000, 99999, n_epochs)),
        'observation_id': (['epoch'], np.arange(1, n_epochs + 1)),
        'exptimen': (['epoch'], np.random.choice([30, 60, 100], n_epochs)),
        'nframes': (['epoch'], np.random.choice([1, 2, 4], n_epochs)),
        'channel': (['epoch'], np.random.choice([1, 2, 3, 4], n_epochs)),
        
        # Telescope pointing
        'ra_targ': (['epoch'], np.random.uniform(17.5, 17.7, n_epochs)),
        'dec_targ': (['epoch'], np.random.uniform(-29.9, -29.7, n_epochs)),
        'pa': (['epoch'], np.random.uniform(0, 360, n_epochs)),
        'crval1': (['epoch'], np.random.uniform(17.5, 17.7, n_epochs)),
        'crval2': (['epoch'], np.random.uniform(-29.9, -29.7, n_epochs)),
        'crpix1': (['epoch'], np.random.uniform(100, 150, n_epochs)),
        'crpix2': (['epoch'], np.random.uniform(100, 150, n_epochs)),
        
        # Image quality metrics
        'seeing': (['epoch'], np.random.uniform(1.5, 3.5, n_epochs)),
        'airmass': (['epoch'], np.random.uniform(1.0, 2.0, n_epochs)),
        'sky_background': (['epoch'], np.random.uniform(100, 500, n_epochs)),
        'zeropoint': (['epoch'], np.random.uniform(19, 21, n_epochs)),
        'extinction': (['epoch'], np.random.uniform(0.1, 0.3, n_epochs)),
        
        # Calibration
        'gain': (['epoch'], np.random.uniform(3.5, 4.5, n_epochs)),
        'readnoise': (['epoch'], np.random.uniform(10, 20, n_epochs)),
        'saturate': (['epoch'], np.random.uniform(50000, 60000, n_epochs)),
        
        # Processing metadata
        'pipeline_version': (['epoch'], np.array([f"S{20 + i%3}.{i%10}.0" for i in range(n_epochs)])),
        'fluxconv': (['epoch'], np.random.uniform(0.1, 0.2, n_epochs)),
        'magzpt': (['epoch'], np.random.uniform(17, 19, n_epochs)),
        
        # WCS information
        'cd1_1': (['epoch'], np.random.uniform(-0.0005, -0.0004, n_epochs)),
        'cd1_2': (['epoch'], np.random.uniform(-0.0001, 0.0001, n_epochs)),
        'cd2_1': (['epoch'], np.random.uniform(-0.0001, 0.0001, n_epochs)),
        'cd2_2': (['epoch'], np.random.uniform(0.0004, 0.0005, n_epochs)),
        
        # Additional metadata
        'naxis1': (['epoch'], np.full(n_epochs, 256)),
        'naxis2': (['epoch'], np.full(n_epochs, 256)),
        'exptime': (['epoch'], np.random.uniform(25, 35, n_epochs)),
        'framtime': (['epoch'], np.random.uniform(0.01, 0.03, n_epochs)),
        
        # Observatory/instrument
        'telescop': (['epoch'], np.array(['Spitzer'] * n_epochs)),
        'instrume': (['epoch'], np.array(['IRAC'] * n_epochs)),
        'filter': (['epoch'], np.random.choice(['IRAC.1', 'IRAC.2', 'IRAC.3', 'IRAC.4'], n_epochs)),
        
        # Observation mode
        'readmode': (['epoch'], np.array(['full'] * n_epochs)),
        'dithpos': (['epoch'], np.random.randint(1, 5, n_epochs)),
        'cyclenum': (['epoch'], np.random.randint(1, 10, n_epochs)),
        
        # Quality flags
        'quality': (['epoch'], np.random.randint(0, 4, n_epochs)),
        'status': (['epoch'], np.array(['OK'] * n_epochs)),
        
        # Additional processing info
        'mopex_version': (['epoch'], np.array([f"18.{i%5}.0" for i in range(n_epochs)])),
        'combine_method': (['epoch'], np.array(['median'] * n_epochs)),
        'outlier_rejection': (['epoch'], np.random.choice(['sigma_clip', 'minmax'], n_epochs)),
        'n_input_frames': (['epoch'], np.random.randint(10, 50, n_epochs)),
        'n_rejected_frames': (['epoch'], np.random.randint(0, 5, n_epochs)),
    }
    print(f"[SETUP] Created {len(epoch_vars)} epoch variables")
    
    # Create ~10 variables for MEASUREMENT dimension
    print("[SETUP] Creating 10 measurement variables...")
    measurement_vars = {
        'aperture_radius': (['measurement'], np.concatenate([
            [5.0, 3.0],  # auto, iso
            np.arange(1, n_measurements-1) * 0.5  # aper_1 to aper_38
        ])),
        'aperture_correction': (['measurement'], np.random.uniform(0.9, 1.1, n_measurements)),
        'zeropoint_offset': (['measurement'], np.random.uniform(-0.1, 0.1, n_measurements)),
        'measurement_type': (['measurement'], np.array(
            ['AUTO', 'ISO'] + [f'APER_{i}' for i in range(1, n_measurements-1)]
        )),
        'is_primary': (['measurement'], np.array(
            [True, False] + [False] * (n_measurements - 2)
        )),
        'background_subtracted': (['measurement'], np.array([True] * n_measurements)),
        'sky_annulus_inner': (['measurement'], np.random.uniform(10, 15, n_measurements)),
        'sky_annulus_outer': (['measurement'], np.random.uniform(20, 30, n_measurements)),
        'contamination_flag': (['measurement'], np.random.randint(0, 2, n_measurements)),
        'measurement_flag': (['measurement'], np.random.randint(0, 4, n_measurements)),
    }
    print(f"[SETUP] Created {len(measurement_vars)} measurement variables")
    
    # Create ~2 variables for VALUE dimension
    value_vars = {
        'unit': (['value'], np.array(['mag', 'mag'])),
        'description': (['value'], np.array(['magnitude', 'magnitude error'])),
    }
    print(f"[SETUP] Created {len(value_vars)} value variables")
    
    # Build the complete dataset
    print("[SETUP] Building xarray Dataset...")
    coords = {
        'object': object_keys,
        'epoch': epoch_keys,
        'measurement': measurement_keys,
        'value': value_keys,
    }
    coords.update(object_vars)
    coords.update(epoch_vars)
    coords.update(measurement_vars)
    coords.update(value_vars)
    
    ds = xr.Dataset(
        data_vars={
            'lightcurves': (
                ['object', 'epoch', 'measurement', 'value'],
                lightcurve_data
            )
        },
        coords=coords,
        attrs={
            'description': 'Realistic IDF lightcurve storage benchmark',
            'n_objects': n_objects,
            'n_epochs': n_epochs,
            'n_measurements': n_measurements,
            'n_values': n_values,
            'n_object_vars': len(object_vars),
            'n_epoch_vars': len(epoch_vars),
            'n_measurement_vars': len(measurement_vars),
            'n_value_vars': len(value_vars),
        }
    )
    
    print(f"[SETUP] Dataset created successfully!")
    print(f"[SETUP] Total coordinates: {len(ds.coords)}")
    print(f"[SETUP] Dataset size: {ds.nbytes / 1e9:.2f} GB")
    
    return ds


# ============================================================================
# BENCHMARK TESTS - Realistic Coordinate Extraction
# ============================================================================


class TestRealisticCoordinateExtraction:
    """Benchmark coordinate extraction with realistic IDF storage structure.
    
    This is THE critical test that reveals the real performance issue.
    With 50+ variables per dimension, coordinate extraction becomes
    significantly slower because xarray needs to iterate through all variables.
    """
    
    @pytest.mark.benchmark(group="realistic_extraction")
    def test_realistic_extract_all_object_coords(self, benchmark, realistic_idf_dataset):
        """Benchmark extracting ALL object coordinates (current pattern).
        
        This mimics the current implementation that extracts all coordinates
        with the 'object' dimension. With 50 variables, this is SLOW!
        """
        ds = realistic_idf_dataset
        
        def extract_all_coords():
            dim_name = 'object'
            data = {dim_name: ds.coords[dim_name].values}
            
            # Extract ALL coordinates with 'object' dimension
            for coord_name in ds.coords:
                coord = ds.coords[coord_name]
                if dim_name in coord.dims and len(coord.dims) == 1:
                    data[coord_name] = coord.values
            
            return pd.DataFrame(data)
        
        result = benchmark(extract_all_coords)
        n_vars_extracted = len(result.columns)
        print(f"\n[RESULT] Extracted {n_vars_extracted} variables for {len(result)} objects")
        assert len(result) == 50_000
    
    @pytest.mark.benchmark(group="realistic_extraction")
    def test_realistic_extract_selective_object_coords(self, benchmark, realistic_idf_dataset):
        """Benchmark extracting SELECTIVE object coordinates (proposed optimization).
        
        This tests the proposed optimization: only extract the 5 coordinates
        that are actually displayed in the AgGrid table.
        
        Expected: MUCH faster than extracting all 50+ coordinates!
        """
        ds = realistic_idf_dataset
        
        def extract_selective_coords():
            # Only extract what we display in AgGrid
            needed_coords = ['object', 'ra', 'dec', 'x_image', 'y_image']
            data = {}
            
            for coord_name in needed_coords:
                if coord_name in ds.coords:
                    data[coord_name] = ds.coords[coord_name].values
            
            return pd.DataFrame(data)
        
        result = benchmark(extract_selective_coords)
        n_vars_extracted = len(result.columns)
        print(f"\n[RESULT] Extracted {n_vars_extracted} variables for {len(result)} objects")
        assert len(result) == 50_000
    
    @pytest.mark.benchmark(group="realistic_extraction")
    def test_realistic_extract_all_epoch_coords(self, benchmark, realistic_idf_dataset):
        """Benchmark extracting ALL epoch coordinates (current pattern).
        
        With 50 epoch variables from tbl_aor, this should also be slow.
        """
        ds = realistic_idf_dataset
        
        def extract_all_coords():
            dim_name = 'epoch'
            data = {dim_name: ds.coords[dim_name].values}
            
            # Extract ALL coordinates with 'epoch' dimension
            for coord_name in ds.coords:
                coord = ds.coords[coord_name]
                if dim_name in coord.dims and len(coord.dims) == 1:
                    data[coord_name] = coord.values
            
            return pd.DataFrame(data)
        
        result = benchmark(extract_all_coords)
        n_vars_extracted = len(result.columns)
        print(f"\n[RESULT] Extracted {n_vars_extracted} variables for {len(result)} epochs")
        assert len(result) == 1_000
    
    @pytest.mark.benchmark(group="realistic_extraction")
    def test_realistic_extract_selective_epoch_coords(self, benchmark, realistic_idf_dataset):
        """Benchmark extracting SELECTIVE epoch coordinates (proposed optimization).
        
        Only extract the 3 coordinates displayed in the AgGrid table.
        """
        ds = realistic_idf_dataset
        
        def extract_selective_coords():
            # Only extract what we display in AgGrid
            needed_coords = ['epoch', 'mjd', 'aor_key']
            data = {}
            
            for coord_name in needed_coords:
                if coord_name in ds.coords:
                    data[coord_name] = ds.coords[coord_name].values
            
            return pd.DataFrame(data)
        
        result = benchmark(extract_selective_coords)
        n_vars_extracted = len(result.columns)
        print(f"\n[RESULT] Extracted {n_vars_extracted} variables for {len(result)} epochs")
        assert len(result) == 1_000
    
    @pytest.mark.benchmark(group="realistic_extraction")
    def test_realistic_extract_measurement_table(self, benchmark, realistic_idf_dataset):
        """Benchmark extracting measurement table.
        
        This should be fast since there are only 40 measurements.
        """
        ds = realistic_idf_dataset
        
        def extract_measurement_table():
            dim_name = 'measurement'
            data = {dim_name: ds.coords[dim_name].values}
            
            for coord_name in ds.coords:
                coord = ds.coords[coord_name]
                if dim_name in coord.dims and len(coord.dims) == 1:
                    data[coord_name] = coord.values
            
            return pd.DataFrame(data)
        
        result = benchmark(extract_measurement_table)
        n_vars_extracted = len(result.columns)
        print(f"\n[RESULT] Extracted {n_vars_extracted} variables for {len(result)} measurements")
        assert len(result) == 40


# ============================================================================
# BENCHMARK TESTS - Direct Coordinate Access
# ============================================================================


class TestRealisticDirectAccess:
    """Benchmark direct coordinate access patterns.
    
    These tests show the overhead of iterating through all coordinates
    vs. directly accessing specific coordinates by name.
    """
    
    @pytest.mark.benchmark(group="realistic_direct_access")
    def test_realistic_direct_coordinate_access(self, benchmark, realistic_idf_dataset):
        """Benchmark directly accessing 5 specific coordinates by name.
        
        This is the optimal pattern - no iteration, just direct access.
        """
        ds = realistic_idf_dataset
        
        def direct_access():
            data = {
                'object': ds.coords['object'].values,
                'ra': ds.coords['ra'].values,
                'dec': ds.coords['dec'].values,
                'x_image': ds.coords['x_image'].values,
                'y_image': ds.coords['y_image'].values,
            }
            return pd.DataFrame(data)
        
        result = benchmark(direct_access)
        assert len(result) == 50_000
    
    @pytest.mark.benchmark(group="realistic_direct_access")
    def test_realistic_iteration_overhead(self, benchmark, realistic_idf_dataset):
        """Benchmark the overhead of iterating through all 50+ coordinates.
        
        This measures how much time is spent just iterating and checking
        coordinate dimensions, without extracting values.
        """
        ds = realistic_idf_dataset
        
        def iteration_only():
            dim_name = 'object'
            coord_names = []
            
            for coord_name in ds.coords:
                coord = ds.coords[coord_name]
                if dim_name in coord.dims and len(coord.dims) == 1:
                    coord_names.append(coord_name)
            
            return coord_names
        
        result = benchmark(iteration_only)
        print(f"\n[RESULT] Found {len(result)} coordinates for 'object' dimension")
        assert len(result) > 40  # Should find ~50 object coordinates


# ============================================================================
# BENCHMARK TESTS - Data Variables Pattern (Actual datamodel.py Pattern)
# ============================================================================


class TestRealisticDataVarsPattern:
    """Benchmark the actual pattern used in datamodel.py.
    
    In the real implementation, dimension metadata is stored as DATA VARIABLES
    (e.g., 'object_ra', 'object_dec') rather than as coordinates. Only the
    dimension keys and a few core spatial coords (ra, dec, x_image, y_image)
    are stored as coordinates.
    
    This is a CRITICAL test to see if this pattern performs differently!
    """
    
    @pytest.fixture(scope="class")
    def realistic_datavars_dataset(self):
        """Create dataset using the actual datamodel.py pattern.
        
        Key differences from coordinate-based pattern:
        - Only dimension keys are coordinates (object, epoch, measurement, value)
        - Only 4 spatial coords (ra, dec, x_image, y_image) are coordinates
        - All other metadata (50+ columns) stored as DATA VARIABLES with prefix
          (e.g., 'object_flux_auto', 'epoch_mjd', 'measurement_aperture_radius')
        """
        np.random.seed(42)
        
        n_objects = 50_000
        n_epochs = 1_000
        n_measurements = 40
        n_values = 2
        
        print(f"\n[SETUP] Creating realistic dataset with DATA VARS pattern...")
        print(f"[SETUP] {n_objects=} {n_epochs=} {n_measurements=} {n_values=}")
        
        # Dimension coordinates (ONLY the dimension keys!)
        object_keys = [f"I{i+1}" for i in range(n_objects)]
        epoch_keys = [f"r{58520832 + i * 256}" for i in range(n_epochs)]
        measurement_keys = ['auto', 'iso'] + [f'aper_{i}' for i in range(1, n_measurements-1)]
        value_keys = ['mag', 'mag_err']
        
        # Main data variable
        lightcurve_data = da.random.uniform(
            15, 25,
            size=(n_objects, n_epochs, n_measurements, n_values),
            chunks=(5000, -1, -1, -1)
        )
        
        # Create dataset with MINIMAL coordinates (only dimension keys)
        coords = {
            'object': object_keys,
            'epoch': epoch_keys,
            'measurement': measurement_keys,
            'value': value_keys,
        }
        
        ds = xr.Dataset(
            data_vars={
                'lightcurves': (
                    ['object', 'epoch', 'measurement', 'value'],
                    lightcurve_data
                )
            },
            coords=coords,
        )
        
        # Add core spatial coordinates (these ARE coordinates in real implementation)
        ds = ds.assign_coords({
            'ra': (['object'], np.random.uniform(17.5, 17.7, n_objects)),
            'dec': (['object'], np.random.uniform(-29.9, -29.7, n_objects)),
            'x_image': (['object'], np.random.uniform(0, 750, n_objects)),
            'y_image': (['object'], np.random.uniform(0, 750, n_objects)),
        })
        
        # Add ~45 object variables as DATA VARIABLES (with 'object_' prefix)
        print("[SETUP] Adding 45 object data variables...")
        object_data_vars = {
            'object_flux_auto': (['object'], np.random.uniform(100, 10000, n_objects)),
            'object_fluxerr_auto': (['object'], np.random.uniform(1, 100, n_objects)),
            'object_flux_iso': (['object'], np.random.uniform(100, 10000, n_objects)),
            'object_fluxerr_iso': (['object'], np.random.uniform(1, 100, n_objects)),
            'object_flux_aper': (['object'], np.random.uniform(100, 10000, n_objects)),
            'object_fluxerr_aper': (['object'], np.random.uniform(1, 100, n_objects)),
            'object_kron_radius': (['object'], np.random.uniform(1, 10, n_objects)),
            'object_fwhm_image': (['object'], np.random.uniform(2, 5, n_objects)),
            'object_ellipticity': (['object'], np.random.uniform(0, 0.8, n_objects)),
            'object_theta_image': (['object'], np.random.uniform(-90, 90, n_objects)),
            'object_a_image': (['object'], np.random.uniform(2, 10, n_objects)),
            'object_b_image': (['object'], np.random.uniform(1, 8, n_objects)),
            'object_isoarea_image': (['object'], np.random.uniform(10, 1000, n_objects).astype(int)),
            'object_flags': (['object'], np.random.randint(0, 8, n_objects)),
            'object_class_star': (['object'], np.random.uniform(0, 1, n_objects)),
            'object_background': (['object'], np.random.uniform(100, 500, n_objects)),
            'object_threshold': (['object'], np.random.uniform(10, 50, n_objects)),
            'object_flux_max': (['object'], np.random.uniform(1000, 50000, n_objects)),
            'object_flux_min': (['object'], np.random.uniform(0, 100, n_objects)),
            'object_x_win': (['object'], np.random.uniform(0, 750, n_objects)),
            'object_y_win': (['object'], np.random.uniform(0, 750, n_objects)),
            'object_xpeak_image': (['object'], np.random.uniform(0, 750, n_objects).astype(int)),
            'object_ypeak_image': (['object'], np.random.uniform(0, 750, n_objects).astype(int)),
            'object_elongation': (['object'], np.random.uniform(1, 5, n_objects)),
            'object_flux_radius': (['object'], np.random.uniform(2, 8, n_objects)),
            'object_petro_radius': (['object'], np.random.uniform(3, 15, n_objects)),
            'object_x2_image': (['object'], np.random.uniform(1, 100, n_objects)),
            'object_y2_image': (['object'], np.random.uniform(1, 100, n_objects)),
            'object_xy_image': (['object'], np.random.uniform(-50, 50, n_objects)),
            'object_errx2_image': (['object'], np.random.uniform(0.01, 1, n_objects)),
            'object_erry2_image': (['object'], np.random.uniform(0.01, 1, n_objects)),
            'object_errxy_image': (['object'], np.random.uniform(-0.5, 0.5, n_objects)),
            'object_errcxx_image': (['object'], np.random.uniform(0.001, 0.1, n_objects)),
            'object_errcyy_image': (['object'], np.random.uniform(0.001, 0.1, n_objects)),
            'object_errcxy_image': (['object'], np.random.uniform(-0.05, 0.05, n_objects)),
            'object_erra_image': (['object'], np.random.uniform(0.1, 2, n_objects)),
            'object_errb_image': (['object'], np.random.uniform(0.1, 2, n_objects)),
            'object_errtheta_image': (['object'], np.random.uniform(0, 180, n_objects)),
            'object_number': (['object'], np.arange(1, n_objects + 1)),
            'object_alpha_j2000': (['object'], np.random.uniform(17.5, 17.7, n_objects)),
            'object_delta_j2000': (['object'], np.random.uniform(-29.9, -29.7, n_objects)),
            'object_x2_world': (['object'], np.random.uniform(0, 0.001, n_objects)),
            'object_y2_world': (['object'], np.random.uniform(0, 0.001, n_objects)),
            'object_xy_world': (['object'], np.random.uniform(-0.0005, 0.0005, n_objects)),
            'object_cxx_image': (['object'], np.random.uniform(0.001, 0.1, n_objects)),
            'object_cyy_image': (['object'], np.random.uniform(0.001, 0.1, n_objects)),
        }
        
        # Add to dataset as data variables
        for name, (dims, array) in object_data_vars.items():
            ds[name] = (dims, array)
        
        # Add ~47 epoch variables as DATA VARIABLES (with 'epoch_' prefix)
        print("[SETUP] Adding 47 epoch data variables...")
        epoch_data_vars = {
            'epoch_mjd': (['epoch'], np.random.uniform(55000, 59000, n_epochs)),
            'epoch_date_obs': (['epoch'], np.array([f"2010-{i%12+1:02d}-{i%28+1:02d}" for i in range(n_epochs)])),
            'epoch_ut_time': (['epoch'], np.random.uniform(0, 24, n_epochs)),
            'epoch_aor_key': (['epoch'], epoch_keys),
            'epoch_program_id': (['epoch'], np.random.randint(10000, 99999, n_epochs)),
            'epoch_observation_id': (['epoch'], np.arange(1, n_epochs + 1)),
            'epoch_exptimen': (['epoch'], np.random.choice([30, 60, 100], n_epochs)),
            'epoch_nframes': (['epoch'], np.random.choice([1, 2, 4], n_epochs)),
            'epoch_channel': (['epoch'], np.random.choice([1, 2, 3, 4], n_epochs)),
            'epoch_ra_targ': (['epoch'], np.random.uniform(17.5, 17.7, n_epochs)),
            'epoch_dec_targ': (['epoch'], np.random.uniform(-29.9, -29.7, n_epochs)),
            'epoch_pa': (['epoch'], np.random.uniform(0, 360, n_epochs)),
            'epoch_crval1': (['epoch'], np.random.uniform(17.5, 17.7, n_epochs)),
            'epoch_crval2': (['epoch'], np.random.uniform(-29.9, -29.7, n_epochs)),
            'epoch_crpix1': (['epoch'], np.random.uniform(100, 150, n_epochs)),
            'epoch_crpix2': (['epoch'], np.random.uniform(100, 150, n_epochs)),
            'epoch_seeing': (['epoch'], np.random.uniform(1.5, 3.5, n_epochs)),
            'epoch_airmass': (['epoch'], np.random.uniform(1.0, 2.0, n_epochs)),
            'epoch_sky_background': (['epoch'], np.random.uniform(100, 500, n_epochs)),
            'epoch_zeropoint': (['epoch'], np.random.uniform(19, 21, n_epochs)),
            'epoch_extinction': (['epoch'], np.random.uniform(0.1, 0.3, n_epochs)),
            'epoch_gain': (['epoch'], np.random.uniform(3.5, 4.5, n_epochs)),
            'epoch_readnoise': (['epoch'], np.random.uniform(10, 20, n_epochs)),
            'epoch_saturate': (['epoch'], np.random.uniform(50000, 60000, n_epochs)),
            'epoch_pipeline_version': (['epoch'], np.array([f"S{20 + i%3}.{i%10}.0" for i in range(n_epochs)])),
            'epoch_fluxconv': (['epoch'], np.random.uniform(0.1, 0.2, n_epochs)),
            'epoch_magzpt': (['epoch'], np.random.uniform(17, 19, n_epochs)),
            'epoch_cd1_1': (['epoch'], np.random.uniform(-0.0005, -0.0004, n_epochs)),
            'epoch_cd1_2': (['epoch'], np.random.uniform(-0.0001, 0.0001, n_epochs)),
            'epoch_cd2_1': (['epoch'], np.random.uniform(-0.0001, 0.0001, n_epochs)),
            'epoch_cd2_2': (['epoch'], np.random.uniform(0.0004, 0.0005, n_epochs)),
            'epoch_naxis1': (['epoch'], np.full(n_epochs, 256)),
            'epoch_naxis2': (['epoch'], np.full(n_epochs, 256)),
            'epoch_exptime': (['epoch'], np.random.uniform(25, 35, n_epochs)),
            'epoch_framtime': (['epoch'], np.random.uniform(0.01, 0.03, n_epochs)),
            'epoch_telescop': (['epoch'], np.array(['Spitzer'] * n_epochs)),
            'epoch_instrume': (['epoch'], np.array(['IRAC'] * n_epochs)),
            'epoch_filter': (['epoch'], np.random.choice(['IRAC.1', 'IRAC.2', 'IRAC.3', 'IRAC.4'], n_epochs)),
            'epoch_readmode': (['epoch'], np.array(['full'] * n_epochs)),
            'epoch_dithpos': (['epoch'], np.random.randint(1, 5, n_epochs)),
            'epoch_cyclenum': (['epoch'], np.random.randint(1, 10, n_epochs)),
            'epoch_quality': (['epoch'], np.random.randint(0, 4, n_epochs)),
            'epoch_status': (['epoch'], np.array(['OK'] * n_epochs)),
            'epoch_mopex_version': (['epoch'], np.array([f"18.{i%5}.0" for i in range(n_epochs)])),
            'epoch_combine_method': (['epoch'], np.array(['median'] * n_epochs)),
            'epoch_outlier_rejection': (['epoch'], np.random.choice(['sigma_clip', 'minmax'], n_epochs)),
            'epoch_n_input_frames': (['epoch'], np.random.randint(10, 50, n_epochs)),
            'epoch_n_rejected_frames': (['epoch'], np.random.randint(0, 5, n_epochs)),
        }
        
        for name, (dims, array) in epoch_data_vars.items():
            ds[name] = (dims, array)
        
        print(f"[SETUP] Dataset created!")
        print(f"[SETUP] Coordinates: {len(ds.coords)} (only dims + spatial)")
        print(f"[SETUP] Data variables: {len(ds.data_vars)} (lightcurves + metadata)")
        print(f"[SETUP] Total variables: {len(ds.coords) + len(ds.data_vars)}")
        
        return ds
    
    @pytest.mark.benchmark(group="datavars_pattern")
    def test_datavars_extract_from_coords_only(self, benchmark, realistic_datavars_dataset):
        """Benchmark extracting ONLY from coordinates (4 spatial + 1 dim = 5 total).
        
        This mimics extracting the AgGrid table using ONLY coordinates.
        Should be very fast since there are only ~8 coordinates total!
        """
        ds = realistic_datavars_dataset
        
        def extract_from_coords():
            dim_name = 'object'
            data = {dim_name: ds.coords[dim_name].values}
            
            # Extract ALL coordinates with 'object' dimension
            for coord_name in ds.coords:
                coord = ds.coords[coord_name]
                if dim_name in coord.dims and len(coord.dims) == 1:
                    data[coord_name] = coord.values
            
            return pd.DataFrame(data)
        
        result = benchmark(extract_from_coords)
        n_vars = len(result.columns)
        print(f"\n[RESULT] Extracted {n_vars} variables from COORDINATES")
        assert len(result) == 50_000
        assert n_vars == 5  # object, ra, dec, x_image, y_image
    
    @pytest.mark.benchmark(group="datavars_pattern")
    def test_datavars_extract_from_data_vars(self, benchmark, realistic_datavars_dataset):
        """Benchmark extracting from DATA VARIABLES (50+ object_* vars).
        
        This mimics extracting additional table columns from data variables.
        Should be slower since there are 45+ data variables with 'object_' prefix.
        """
        ds = realistic_datavars_dataset
        
        def extract_from_data_vars():
            dim_name = 'object'
            data = {dim_name: ds.coords[dim_name].values}
            
            # Extract ALL data variables with 'object' dimension
            for var_name in ds.data_vars:
                if var_name.startswith(f'{dim_name}_'):
                    var = ds[var_name]
                    if dim_name in var.dims and len(var.dims) == 1:
                        data[var_name] = var.values
            
            return pd.DataFrame(data)
        
        result = benchmark(extract_from_data_vars)
        n_vars = len(result.columns)
        print(f"\n[RESULT] Extracted {n_vars} variables from DATA VARS")
        assert len(result) == 50_000
        assert n_vars > 40  # Should get 45+ object_* data variables
    
    @pytest.mark.benchmark(group="datavars_pattern")
    def test_datavars_extract_selective_mixed(self, benchmark, realistic_datavars_dataset):
        """Benchmark selective extraction from BOTH coords and data vars.
        
        This is the optimal pattern for real datamodel.py structure:
        - Get dimension key from coords
        - Get 4 spatial coords (ra, dec, x_image, y_image) from coords
        - Optionally get specific columns from data vars if needed
        """
        ds = realistic_datavars_dataset
        
        def extract_selective_mixed():
            # Direct access to coordinates (fast!)
            data = {
                'object': ds.coords['object'].values,
                'ra': ds.coords['ra'].values,
                'dec': ds.coords['dec'].values,
                'x_image': ds.coords['x_image'].values,
                'y_image': ds.coords['y_image'].values,
            }
            
            # Optionally add specific data vars if needed
            # (Commented out since AgGrid only shows coords)
            # data['flux_auto'] = ds['object_flux_auto'].values
            
            return pd.DataFrame(data)
        
        result = benchmark(extract_selective_mixed)
        n_vars = len(result.columns)
        print(f"\n[RESULT] Extracted {n_vars} variables (selective)")
        assert len(result) == 50_000
        assert n_vars == 5  # Only coords for AgGrid table
    
    @pytest.mark.benchmark(group="datavars_pattern_full")
    def test_datavars_extract_full_table_iterate(self, benchmark, realistic_datavars_dataset):
        """Benchmark extracting FULL table by iterating through coords + data_vars.
        
        This mimics the real dash app use case:
        - Extract ALL coordinates with 'object' dimension
        - Extract ALL data_vars with 'object_' prefix
        - Results in ~50 columns total (5 coords + 45 data_vars)
        """
        ds = realistic_datavars_dataset
        
        def extract_full_table_iterate():
            dim_name = 'object'
            data = {dim_name: ds.coords[dim_name].values}
            
            # Extract ALL coordinates with 'object' dimension
            for coord_name in ds.coords:
                coord = ds.coords[coord_name]
                if dim_name in coord.dims and len(coord.dims) == 1:
                    data[coord_name] = coord.values
            
            # Extract ALL data variables with 'object_' prefix
            for var_name in ds.data_vars:
                if var_name.startswith(f'{dim_name}_'):
                    var = ds[var_name]
                    if dim_name in var.dims and len(var.dims) == 1:
                        # Store without prefix for cleaner column names
                        clean_name = var_name.replace(f'{dim_name}_', '')
                        data[clean_name] = var.values
            
            return pd.DataFrame(data)
        
        result = benchmark(extract_full_table_iterate)
        n_vars = len(result.columns)
        print(f"\n[RESULT] Extracted {n_vars} variables (full table with iteration)")
        assert len(result) == 50_000
        assert n_vars > 45  # Should get ~50 total columns
    
    @pytest.mark.benchmark(group="datavars_pattern_full")
    def test_datavars_extract_full_table_direct(self, benchmark, realistic_datavars_dataset):
        """Benchmark extracting FULL table by directly accessing known variables.
        
        This is the optimized pattern:
        - Define list of coordinate names to extract
        - Define list of data_var names to extract  
        - Direct access to each (no iteration)
        """
        ds = realistic_datavars_dataset
        
        def extract_full_table_direct():
            dim_name = 'object'
            
            # Core coordinates (direct access)
            coord_names = ['object', 'ra', 'dec', 'x_image', 'y_image']
            data = {name: ds.coords[name].values for name in coord_names}
            
            # Data variables (direct access with prefix)
            # In real app, this list would come from configuration or metadata
            data_var_names = [
                'flux_auto', 'fluxerr_auto', 'flux_iso', 'fluxerr_iso',
                'flux_aper', 'fluxerr_aper', 'kron_radius', 'fwhm_image',
                'ellipticity', 'theta_image', 'a_image', 'b_image',
                'isoarea_image', 'flags', 'class_star', 'background',
                'threshold', 'flux_max', 'flux_min', 'x_win', 'y_win',
                'xpeak_image', 'ypeak_image', 'elongation', 'flux_radius',
                'petro_radius', 'x2_image', 'y2_image', 'xy_image',
                'errx2_image', 'erry2_image', 'errxy_image', 'errcxx_image',
                'errcyy_image', 'errcxy_image', 'erra_image', 'errb_image',
                'errtheta_image', 'number', 'alpha_j2000', 'delta_j2000',
                'x2_world', 'y2_world', 'xy_world', 'cxx_image', 'cyy_image',
            ]
            
            for var_name in data_var_names:
                full_var_name = f'{dim_name}_{var_name}'
                if full_var_name in ds.data_vars:
                    data[var_name] = ds[full_var_name].values
            
            return pd.DataFrame(data)
        
        result = benchmark(extract_full_table_direct)
        n_vars = len(result.columns)
        print(f"\n[RESULT] Extracted {n_vars} variables (full table with direct access)")
        assert len(result) == 50_000
        assert n_vars > 45  # Should get ~50 total columns
    
    @pytest.mark.benchmark(group="datavars_pattern_full")
    def test_datavars_extract_full_table_listcomp(self, benchmark, realistic_datavars_dataset):
        """Benchmark extracting FULL table using list comprehension.
        
        This is a hybrid approach:
        - Use list comprehension to find matching data_vars (faster than loop)
        - Direct extraction after finding names
        """
        ds = realistic_datavars_dataset
        
        def extract_full_table_listcomp():
            dim_name = 'object'
            
            # Core coordinates (direct access)
            coord_names = [name for name in ds.coords if name == dim_name or 
                          (dim_name in ds.coords[name].dims and len(ds.coords[name].dims) == 1)]
            data = {name: ds.coords[name].values for name in coord_names}
            
            # Data variables (list comprehension to find, then extract)
            data_var_names = [name for name in ds.data_vars 
                             if name.startswith(f'{dim_name}_') and 
                             dim_name in ds[name].dims and len(ds[name].dims) == 1]
            
            for var_name in data_var_names:
                clean_name = var_name.replace(f'{dim_name}_', '')
                data[clean_name] = ds[var_name].values
            
            return pd.DataFrame(data)
        
        result = benchmark(extract_full_table_listcomp)
        n_vars = len(result.columns)
        print(f"\n[RESULT] Extracted {n_vars} variables (full table with list comprehension)")
        assert len(result) == 50_000
        assert n_vars > 45  # Should get ~50 total columns
    
    @pytest.mark.benchmark(group="datavars_pattern_full_table")
    def test_datavars_extract_full_table_iteration(self, benchmark, realistic_datavars_dataset):
        """Benchmark extracting FULL table with ALL coords + ALL data_vars.
        
        This is the REAL DASH APP pattern - build complete AgGrid table with:
        - All coordinates for the dimension (5 items)
        - All data_vars for the dimension (45+ items with 'object_' prefix)
        - Total: ~50 columns for AgGrid table
        
        Uses iteration pattern to find all variables.
        """
        ds = realistic_datavars_dataset
        
        def extract_full_table_iteration():
            dim_name = 'object'
            data = {}
            
            # Extract dimension key
            data[dim_name] = ds.coords[dim_name].values
            
            # Extract ALL coordinates with this dimension (iteration)
            for coord_name in ds.coords:
                coord = ds.coords[coord_name]
                if coord_name != dim_name and dim_name in coord.dims and len(coord.dims) == 1:
                    data[coord_name] = coord.values
            
            # Extract ALL data_vars with this dimension (iteration)
            for var_name in ds.data_vars:
                if var_name.startswith(f'{dim_name}_'):
                    var = ds[var_name]
                    if dim_name in var.dims and len(var.dims) == 1:
                        # Remove prefix for cleaner column names
                        col_name = var_name.replace(f'{dim_name}_', '')
                        data[col_name] = var.values
            
            return pd.DataFrame(data)
        
        result = benchmark(extract_full_table_iteration)
        n_vars = len(result.columns)
        print(f"\n[RESULT] Extracted {n_vars} variables (full table with iteration)")
        assert len(result) == 50_000
        assert n_vars > 40  # Should get 50+ columns total
    
    @pytest.mark.benchmark(group="datavars_pattern_full_table")
    def test_datavars_extract_full_table_direct(self, benchmark, realistic_datavars_dataset):
        """Benchmark extracting FULL table with direct access (no iteration).
        
        This is the OPTIMIZED pattern - directly access known variable names
        instead of iterating through coords/data_vars dictionaries.
        
        In real dash app, you know which columns you want to display, so
        you can list them explicitly rather than discovering via iteration.
        """
        ds = realistic_datavars_dataset
        
        def extract_full_table_direct():
            dim_name = 'object'
            
            # Known coordinate names
            coord_names = ['object', 'ra', 'dec', 'x_image', 'y_image']
            
            # Known data_var names (without prefix) - in real app, you'd list the ones you want
            # Here we'll just list a representative subset for the benchmark
            datavar_names = [
                'flux_auto', 'fluxerr_auto', 'flux_iso', 'fluxerr_iso',
                'flux_aper', 'fluxerr_aper', 'kron_radius', 'fwhm_image',
                'ellipticity', 'theta_image', 'a_image', 'b_image',
                'isoarea_image', 'flags', 'class_star', 'background',
                'threshold', 'flux_max', 'flux_min', 'x_win', 'y_win',
                'xpeak_image', 'ypeak_image', 'elongation', 'flux_radius',
                'petro_radius', 'x2_image', 'y2_image', 'xy_image',
                'errx2_image', 'erry2_image', 'errxy_image', 'errcxx_image',
                'errcyy_image', 'errcxy_image', 'erra_image', 'errb_image',
                'errtheta_image', 'number', 'alpha_j2000', 'delta_j2000',
                'x2_world', 'y2_world', 'xy_world', 'cxx_image', 'cyy_image',
            ]
            
            # Direct access - build dictionary in one go
            data = {name: ds.coords[name].values for name in coord_names}
            
            # Add data_vars with direct access
            for col_name in datavar_names:
                var_name = f'{dim_name}_{col_name}'
                if var_name in ds.data_vars:
                    data[col_name] = ds[var_name].values
            
            return pd.DataFrame(data)
        
        result = benchmark(extract_full_table_direct)
        n_vars = len(result.columns)
        print(f"\n[RESULT] Extracted {n_vars} variables (full table with direct access)")
        assert len(result) == 50_000
        assert n_vars > 40  # Should get 50+ columns total
    
    @pytest.mark.benchmark(group="datavars_pattern_full_table")
    def test_datavars_extract_full_table_cached_discovery(self, benchmark, realistic_datavars_dataset):
        """Benchmark with one-time discovery + cached column list.
        
        This is the BEST PRACTICE pattern:
        1. On first call: Discover all coord/datavar names (one-time cost)
        2. Cache the column name list
        3. On subsequent calls: Use cached list for direct access (fast!)
        
        This benchmark shows the steady-state performance after caching.
        """
        ds = realistic_datavars_dataset
        dim_name = 'object'
        
        # Simulate the one-time discovery (not benchmarked)
        coord_names_cache = [dim_name]
        for coord_name in ds.coords:
            coord = ds.coords[coord_name]
            if coord_name != dim_name and dim_name in coord.dims and len(coord.dims) == 1:
                coord_names_cache.append(coord_name)
        
        datavar_names_cache = []
        for var_name in ds.data_vars:
            if var_name.startswith(f'{dim_name}_'):
                var = ds[var_name]
                if dim_name in var.dims and len(var.dims) == 1:
                    col_name = var_name.replace(f'{dim_name}_', '')
                    datavar_names_cache.append((col_name, var_name))
        
        print(f"\n[CACHE] Cached {len(coord_names_cache)} coords + {len(datavar_names_cache)} data_vars")
        
        # Benchmark the extraction using cached names (fast path)
        def extract_with_cached_names():
            # Direct access using cached coordinate names
            data = {name: ds.coords[name].values for name in coord_names_cache}
            
            # Direct access using cached data_var names
            for col_name, var_name in datavar_names_cache:
                data[col_name] = ds[var_name].values
            
            return pd.DataFrame(data)
        
        result = benchmark(extract_with_cached_names)
        n_vars = len(result.columns)
        print(f"[RESULT] Extracted {n_vars} variables (cached discovery)")
        assert len(result) == 50_000
        assert n_vars > 40
