"""
Installation and testing guide for the pyzma_idf_lightcurve package.
"""

def check_installation():
    """Check if the package can be imported and basic functionality works."""
    
    print("🔍 Testing pyzma_idf_lightcurve installation...")
    
    # Test core imports
    try:
        from pyzma_idf_lightcurve.pipeline import IDFPipelineConfig
        from pyzma_idf_lightcurve.pipeline.templates import PartitionKey, IDFFilename
        print("✅ Pipeline imports successful")
    except ImportError as e:
        print(f"❌ Pipeline import failed: {e}")
        return False
    
    try:
        from pyzma_idf_lightcurve.lightcurve import BinaryLightcurveDatabase
        print("✅ Lightcurve imports successful")  
    except ImportError as e:
        print(f"❌ Lightcurve import failed: {e}")
        return False
    
    # Test template functionality
    try:
        # Test filename parsing
        filename = "IDF_gr123_ch1_sci.fits"
        parsed = IDFFilename.parse(filename)
        reconstructed = IDFFilename.make(**parsed)
        
        if reconstructed == filename:
            print("✅ Template parsing works correctly")
        else:
            print(f"❌ Template parsing failed: {filename} != {reconstructed}")
            return False
            
        # Test partition key
        partition = PartitionKey.make(group_name="gr123", chan="ch1")
        if partition == "gr123_ch1":
            print("✅ Partition key generation works")
        else:
            print(f"❌ Partition key failed: expected 'gr123_ch1', got '{partition}'")
            return False
            
    except Exception as e:
        print(f"❌ Template testing failed: {e}")
        return False
    
    # Test binary format
    try:
        import numpy as np
        from pyzma_idf_lightcurve.lightcurve.binary import BinaryLightcurveFormat
        
        # Create test data
        times = np.linspace(55000, 55100, 100)
        mags = 15.0 + 0.1 * np.random.randn(100)
        errs = 0.05 * np.ones(100)
        flags = np.zeros(100, dtype=int)
        
        # Test binary round-trip
        binary_data = BinaryLightcurveFormat.pack_lightcurve(1, 1, times, mags, errs, flags)
        unpacked = BinaryLightcurveFormat.unpack_lightcurve(binary_data)
        
        if len(unpacked['times']) == 100:
            print("✅ Binary format works correctly")
        else:
            print("❌ Binary format failed")
            return False
            
    except Exception as e:
        print(f"❌ Binary format testing failed: {e}")
        return False
    
    # Test optional dash imports
    try:
        from pyzma_idf_lightcurve.lightcurve import LightcurveVisualizationApp
        if LightcurveVisualizationApp is not None:
            print("✅ Dash visualization available")
        else:
            print("⚠️  Dash visualization not available (missing dependencies)")
    except ImportError:
        print("⚠️  Dash visualization not available (missing dependencies)")
    
    print("\n🎉 Basic installation check passed!")
    return True


def show_usage_examples():
    """Show basic usage examples."""
    
    print("\n📖 Usage Examples:")
    print("="*50)
    
    print("\n1. Start Pipeline Development Server:")  
    print("   idf-pipeline-dev")
    print("   # Opens Dagster UI at http://localhost:3001")
    
    print("\n2. Launch Lightcurve Visualization:")
    print("   idf-lightcurve-viz --db-path lightcurves.db --port 8050")
    print("   # Opens visualization at http://localhost:8050")
    
    print("\n3. Use Binary Storage in Code:")
    print("""
    from pyzma_idf_lightcurve.lightcurve import BinaryLightcurveDatabase
    import numpy as np
    
    # Create database
    db = BinaryLightcurveDatabase("test.db")
    
    # Store lightcurve
    times = np.linspace(55000, 58000, 1200)
    mags = 15.0 + 0.1 * np.random.randn(1200)
    errs = 0.05 * np.ones(1200)
    flags = np.zeros(1200, dtype=int)
    
    db.store_lightcurve(12345, 1, times, mags, errs, flags)
    
    # Retrieve lightcurve (ultra-fast!)
    lightcurve = db.get_lightcurve(12345, 1)
    """)
    
    print("\n4. Configure Pipeline:")
    print("""
    from pyzma_idf_lightcurve.pipeline import IDFPipelineConfig
    
    config = IDFPipelineConfig(
        input_dir="per_aor_images",
        workdir="scratch",
        coadd_dir="superstack"
    )
    """)


if __name__ == "__main__":
    success = check_installation()
    
    if success:
        show_usage_examples()
        print("\n🚀 Ready to process IDF lightcurves with binary optimization!")
    else:
        print("\n❌ Installation issues found. Please check dependencies.")
        print("Try: uv pip install -e .")