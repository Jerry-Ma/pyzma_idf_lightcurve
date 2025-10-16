#!/usr/bin/env python3
"""Verify the actual storage pattern used in LightcurveStorage.

This script checks whether the stored datasets use Pattern A (all as coords) 
or Pattern B (data_vars with minimal coords).
"""

import sys
from pathlib import Path
from pyzma_idf_lightcurve.lightcurve.datamodel import LightcurveStorage


def verify_storage_pattern(storage_path: Path):
    """Verify the storage pattern of a LightcurveStorage instance."""
    print(f"\n{'='*80}")
    print(f"Verifying storage pattern: {storage_path}")
    print(f"{'='*80}\n")
    
    # Create storage instance
    storage = LightcurveStorage(storage_path)
    
    # Try to load read dataset
    try:
        ds = storage.load_for_read()
        print("✅ Successfully loaded dataset for reading\n")
    except Exception as e:
        print(f"❌ Could not load dataset: {e}")
        return
    
    # Analyze structure
    print(f"Dataset Structure:")
    print(f"  Dimensions: {dict(ds.dims)}")
    print(f"  Total coordinates: {len(ds.coords)}")
    print(f"  Total data variables: {len(ds.data_vars)}")
    print()
    
    # List all coordinates
    print(f"Coordinates ({len(ds.coords)} total):")
    for i, coord_name in enumerate(sorted(ds.coords.keys()), 1):
        coord = ds.coords[coord_name]
        dims_str = f"({', '.join(coord.dims)})"
        print(f"  {i:3d}. {coord_name:30s} {dims_str:30s} shape={coord.shape}")
    print()
    
    # List first 20 data variables
    data_var_names = list(ds.data_vars.keys())
    print(f"Data Variables ({len(data_var_names)} total, showing first 20):")
    for i, var_name in enumerate(sorted(data_var_names)[:20], 1):
        var = ds.data_vars[var_name]
        dims_str = f"({', '.join(var.dims)})"
        print(f"  {i:3d}. {var_name:30s} {dims_str:30s} shape={var.shape}")
    print()
    
    # Determine pattern
    print(f"{'='*80}")
    print("Pattern Analysis:")
    print(f"{'='*80}\n")
    
    # Count coordinates by dimension
    dim_names = ['object', 'epoch', 'measurement', 'value']
    coord_counts = {dim: 0 for dim in dim_names}
    
    for coord_name in ds.coords:
        coord = ds.coords[coord_name]
        for dim in dim_names:
            if dim in coord.dims and len(coord.dims) == 1:
                coord_counts[dim] += 1
    
    print("Coordinates per dimension:")
    for dim, count in coord_counts.items():
        print(f"  {dim:15s}: {count:3d} coordinates")
    print()
    
    # Count data_vars with prefix pattern
    datavar_counts = {dim: 0 for dim in dim_names}
    
    for var_name in ds.data_vars:
        for dim in dim_names:
            if var_name.startswith(f'{dim}_'):
                var = ds.data_vars[var_name]
                if dim in var.dims and len(var.dims) == 1:
                    datavar_counts[dim] += 1
    
    print("Data variables per dimension (with prefix pattern):")
    for dim, count in datavar_counts.items():
        print(f"  {dim:15s}: {count:3d} data variables")
    print()
    
    # Determine which pattern
    print(f"{'='*80}")
    print("Conclusion:")
    print(f"{'='*80}\n")
    
    total_coords = sum(coord_counts.values())
    total_datavars_with_prefix = sum(datavar_counts.values())
    
    if total_coords > 50:
        print("⚠️  PATTERN A detected: ALL VARIABLES AS COORDINATES")
        print(f"    - {total_coords} dimension-specific coordinates found")
        print(f"    - This is the SLOW pattern (81.8 ms per table)")
        print(f"    - RECOMMENDATION: Migrate to Pattern B (data_vars)")
        pattern = "A"
    elif total_datavars_with_prefix > 50:
        print("✅ PATTERN B detected: DATA VARS WITH MINIMAL COORDS")
        print(f"    - Only {total_coords} dimension-specific coordinates")
        print(f"    - {total_datavars_with_prefix} variables stored as data_vars")
        print(f"    - This is the FAST pattern (7.2 ms per table)")
        print(f"    - RECOMMENDATION: Optimize extraction to 4.0 ms (direct access)")
        pattern = "B"
    else:
        print("🤔 MIXED PATTERN or SMALL DATASET detected")
        print(f"    - {total_coords} coordinates")
        print(f"    - {total_datavars_with_prefix} prefixed data variables")
        print(f"    - May be a test dataset with limited variables")
        pattern = "MIXED"
    
    print()
    
    # Specific recommendations
    if pattern == "A":
        print("Action Items for Pattern A → Pattern B Migration:")
        print("  1. Modify LightcurveStorage._create() to use data_vars pattern")
        print("  2. Store only dimension keys in coords initially")
        print("  3. Add spatial coords (ra, dec, x_image, y_image) via assign_coords")
        print("  4. Store metadata as data_vars with prefix (e.g., 'object_flux_auto')")
        print("  5. Expected speedup: 11.4x (81.8 ms → 7.2 ms)")
        print()
    elif pattern == "B":
        print("Action Items for Pattern B Optimization:")
        print("  1. Find coordinate extraction code in dash app")
        print("  2. Search for: 'for coord_name in ds.coords'")
        print("  3. Replace with direct access:")
        print("     coord_names = ['object', 'ra', 'dec', 'x_image', 'y_image']")
        print("     data = {name: ds.coords[name].values for name in coord_names}")
        print("  4. Expected speedup: 1.8x (7.2 ms → 4.0 ms)")
        print()
    
    return ds


if __name__ == "__main__":
    # Look for test storage directories
    test_dir = Path(__file__).parent
    
    # Check for existing test storage
    test_storage_paths = [
        test_dir / "test_data" / "lightcurve_storage",
        test_dir / "test_benchmarks_storage" / "lightcurve_storage",
        test_dir.parent.parent / "scratch" / "lightcurve_storage",
    ]
    
    # Check command line args
    if len(sys.argv) > 1:
        custom_path = Path(sys.argv[1])
        if custom_path.exists():
            verify_storage_pattern(custom_path)
        else:
            print(f"❌ Path not found: {custom_path}")
            sys.exit(1)
    else:
        # Try to find any existing storage
        found = False
        for path in test_storage_paths:
            if path.exists():
                verify_storage_pattern(path)
                found = True
                break
        
        if not found:
            print("❌ No test storage found. Creating a small test dataset...\n")
            
            # Create a minimal test storage
            import tempfile
            import numpy as np
            
            with tempfile.TemporaryDirectory() as tmpdir:
                storage_path = Path(tmpdir) / "test_storage"
                
                # Create minimal test storage
                storage = LightcurveStorage.create(
                    storage_path=storage_path,
                    n_objects=100,
                    n_epochs=10,
                    n_measurements=5,
                    n_values=2,
                    object_vars={
                        'ra': np.random.uniform(17.5, 17.7, 100),
                        'dec': np.random.uniform(-29.9, -29.7, 100),
                        'flux_auto': np.random.uniform(100, 10000, 100),
                    },
                    epoch_vars={
                        'mjd': np.random.uniform(55000, 59000, 10),
                        'seeing': np.random.uniform(1.5, 3.5, 10),
                    },
                    measurement_vars={
                        'aperture': np.array([2.0, 4.0, 6.0, 8.0, 10.0]),
                    },
                    value_vars={},
                )
                
                print("Created temporary test storage\n")
                verify_storage_pattern(storage_path)
