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
    with pydantic Field support for enhanced metadata and documentation. The 
    configuration uses pydantic Field with defaults and descriptions.
    """
    
    # Directory paths with pydantic Field defaults and descriptions
    input_dir: str = Field(default="per_aor_images", description="Directory containing per-AOR input IDF files for processing")
    workdir: str = Field(default="scratch", description="Working directory for intermediate processing files and temporary data") 
    coadd_dir: str = Field(default="superstack", description="Directory containing coadd/superstack reference files")
    
    # Script paths with pydantic Field metadata
    imarith_script: str = Field(default="imarith.py", description="Path to imarith.py script for image arithmetic operations")
    make_wht_script: str = Field(default="make_wht.py", description="Path to make_wht.py script for weight map creation")
    lac_script: str = Field(default="lac.py", description="Path to lac.py script for lightcurve analysis and extraction")
    cat2cat_script: str = Field(default="cat2cat.py", description="Path to cat2cat.py script for catalog cross-matching")
    
    # Configuration files with pydantic Field descriptions, defaults, and validation
    sex_config_file: str = Field(
        default="conf_sex.d/conf_per_aor.sexcfg", 
        description="SExtractor configuration file path (should end with .sexcfg)",
        pattern=r".*\.sexcfg$"
    )
    sextractor_config: str = Field(
        default="config", 
        description="SExtractor main configuration parameter name",
        min_length=1
    )
    final_table_name: str = Field(
        default="idf_lightcurves.ecsv", 
        description="Final merged lightcurve table filename (should end with .ecsv)",
        pattern=r".*\.ecsv$"
    )