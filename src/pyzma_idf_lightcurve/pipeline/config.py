"""
Configuration schemas for the IDF lightcurve processing pipeline.

Following Dagster best practices with native Config class and pydantic Field integration.
"""

from dagster import Config
from pydantic import Field


class IDFPipelineConfig(Config):
    """
    Dagster-native configuration for the IDF lightcurve processing pipeline.
    
    This follows Dagster framework best practices by using the native Config class
    with pydantic Field support for enhanced metadata and documentation. All 
    configuration values must be provided via YAML configuration files.
    """
    
    # Directory paths - all required, no defaults
    input_dir: str = Field(description="Directory containing per-AOR input IDF files for processing")
    workdir: str = Field(description="Working directory for intermediate processing files and temporary data") 
    coadd_dir: str = Field(description="Directory containing coadd/superstack reference files")
    
    # Script paths - all required, no defaults
    imarith_script: str = Field(description="Path to imarith.py script for image arithmetic operations")
    make_wht_script: str = Field(description="Path to make_wht.py script for weight map creation")
    lac_script: str = Field(description="Path to lac.py script for lightcurve analysis and extraction")
    cat2cat_script: str = Field(description="Path to cat2cat.py script for catalog cross-matching")
    
    # Configuration files - all required with validation
    sex_config_file: str = Field(
        description="SExtractor configuration file path (should end with .sexcfg)",
        pattern=r".*\.sexcfg$"
    )
    sextractor_config: str = Field(
        description="SExtractor main configuration parameter name",
        min_length=1
    )
    
    # Lightcurve storage configuration
    lightcurve_storage_filename: str = Field(
        description="Zarr storage filename for lightcurves (should end with .zarr)",
        pattern=r".*\.zarr$",
        default="lightcurves.zarr"
    )
    aor_info_table: str = Field(
        description="Path to ECSV file containing AOR information and metadata"
    )
