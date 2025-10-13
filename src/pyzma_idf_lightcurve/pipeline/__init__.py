"""
Dagster pipeline package for IDF lightcurve processing.

This package contains:
- assets: All Dagster assets for the IDF processing workflow
- io_managers: Custom I/O managers for data persistence
- configs: Configuration schemas and utilities
- dev_server: Development server utilities
"""

from .assets import partition_files, prepared_input_file_symlinks
from .config import IDFPipelineConfig
from .definitions import defs
from .dev_server import get_optimal_process_count, setup_multiprocess_instance
from .io_managers import DuckDBJSONIOManager, duckdb_json_io_manager

__all__ = [
    "IDFPipelineConfig",
    "prepared_input_file_symlinks",
    "partition_files", 
    "DuckDBJSONIOManager",
    "duckdb_json_io_manager",
    "get_optimal_process_count",
    "setup_multiprocess_instance",
    "defs",
]