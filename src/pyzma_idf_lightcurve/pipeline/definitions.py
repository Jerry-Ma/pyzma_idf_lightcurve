"""
Main Dagster definitions for the IDF lightcurve processing pipeline.
"""

from dagster import Definitions

from .assets import (
    prepared_input_file_symlinks,
    partition_files,
    group_chan_partitions,
)
from .io_managers import duckdb_json_io_manager
from .config import IDFPipelineConfig


# Define the Dagster definitions
defs = Definitions(
    assets=[
        prepared_input_file_symlinks,
        partition_files,
        # Add more assets as they are implemented
    ],
    resources={
        "config": IDFPipelineConfig(),
        "io_manager": duckdb_json_io_manager.configured({
            "database": "scratch_dagster/idf_lightcurve/io.duckdb"
        }),
    }
)