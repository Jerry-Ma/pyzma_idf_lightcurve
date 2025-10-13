"""
Main Dagster definitions for the IDF lightcurve processing pipeline.
"""

import os

import yaml
from dagster import Config, ConfigurableResource, Definitions
from loguru import logger

from .assets import asset_defs
from .config import IDFPipelineConfig
from .io_managers import duckdb_json_io_manager


def _load_defs() -> Definitions:
    """Load Dagster definitions, applying configuration if IDFLC_RESOURCE_DEFS_PATH is set."""
    resource_defs = {
                "idf_pipeline_config": IDFPipelineConfig,
                "duckdb_io_manager": duckdb_json_io_manager,
       
    }
 
    # Check for configuration file path in environment
    resource_defs_path = os.environ.get('IDFLC_RESOURCE_DEFS_PATH', None)
    
    if resource_defs_path is not None:
        logger.info(f"load resource defs from {resource_defs_path}")
        with open(resource_defs_path, 'r') as fo:
            d = yaml.safe_load(fo)
            for k, v in d.items():
                if k in resource_defs:
                    # Handle Config/ConfigurableResource classes with direct instantiation
                    if (isinstance(resource_defs[k], type) and 
                        (issubclass(resource_defs[k], ConfigurableResource) or 
                         issubclass(resource_defs[k], Config))):
                        resource_defs[k] = resource_defs[k](**v)
                    # Handle traditional resources (like duckdb_io_manager) with .configured()
                    elif hasattr(resource_defs[k], "configured"):
                        resource_defs[k] = resource_defs[k].configured(v)
                    else:
                        raise ValueError(f"Unknown resource type for key: {k}")
    else:
        logger.info(f"no resource defs config found, skip.")

    logger.debug(f"loaded resource defs:\n{resource_defs}")
    
    return Definitions(
            assets=asset_defs,
            resources=resource_defs,
        )


# Load definitions (will use config if environment variable is set)
defs = _load_defs()