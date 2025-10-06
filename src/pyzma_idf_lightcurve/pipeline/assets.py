"""
Core asset definitions for the IDF lightcurve processing pipeline.
"""

import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Literal, TypeVar, Generic
from typing_extensions import TypedDict
import multiprocessing

from loguru import logger
import numpy as np
from astropy.table import Table, vstack

from dagster import (
    asset,
    AssetExecutionContext,
    AssetKey,
    Config,
    Definitions,
    MaterializeResult,
    MetadataValue,
    DynamicPartitionsDefinition,
    DefaultSensorStatus,
    sensor,
    SensorEvaluationContext,
    SensorResult,
    RunRequest,
    multiprocess_executor,
    define_asset_job,
)

from .config import IDFPipelineConfig
from .templates import PartitionKey, IDFFilename
from .utils import check_files_and_timestamps, run_subprocess_command


# Define dynamic partitions for group-channel combinations
group_chan_partitions = DynamicPartitionsDefinition(name="group_chan")


def discover_idf_files(input_dir: str) -> Dict[str, Dict[str, str]]:
    """Discover IDF files and group by partition key. Returns original file paths.
    
    Args:
        input_dir: Directory containing original IDF files
        
    Returns:
        Dict mapping partition keys to dict of kind -> original_file_path
    """
    files_by_partition = {}
    input_path = Path(input_dir)
    
    if not input_path.exists():
        logger.warning(f"Input directory does not exist: {input_dir}")
        return files_by_partition

    for file_path in input_path.rglob("IDF_*.fits"):
        try:
            parsed = IDFFilename.parse(file_path.name)
            group_name = parsed['group_name']
            chan = parsed['chan']
            kind = parsed['kind']
            
            # Create partition key
            partition_key = PartitionKey.make(group_name=group_name, chan=chan)
            
            # Add to files dictionary (original paths)
            if partition_key not in files_by_partition:
                files_by_partition[partition_key] = {}
            files_by_partition[partition_key][kind] = str(file_path)

        except ValueError:
            logger.debug(f"Filename {file_path.name} does not match IDF pattern")
            continue

    return files_by_partition


def get_or_create_symlinks_for_partition(files: Dict[str, str], workdir: str, recreate=False) -> Tuple[Dict[str, str], Dict[str, int]]:
    """Create symlinks in workdir for files of a specific partition. Skip existing symlinks unless recreate=True.
    
    Args:
        files: Dict of kind -> original_file_path for one partition
        workdir: Working directory where symlinks will be created
        recreate: If True, recreate symlinks even if they exist
        
    Returns:
        Tuple of (symlinked_files_dict, stats_dict)
        - symlinked_files_dict: Dict of kind -> symlink_path
        - stats_dict: Dict with "created", "skipped", "recreated" counts
    """
    workdir_path = Path(workdir)
    workdir_path.mkdir(parents=True, exist_ok=True)
    
    symlinked_files = {}
    stats = {"created": 0, "skipped": 0, "recreated": 0}
    
    for kind, source_file_str in files.items():
        source_path = Path(source_file_str)
        
        # Generate the target filename using proper IDF naming
        target_filename = IDFFilename.remake_filepath(
            source_path, 
            parent_path=workdir_path,
        )
        target_path = target_filename  # remake_filepath returns full path now

        # Check if symlink already exists and is valid
        if target_path.is_symlink() and target_path.exists() and not recreate:
            # Symlink exists and points to valid file
            stats["skipped"] += 1
            symlinked_files[kind] = str(target_path)
        else:
            # Handle existing files/symlinks before creating new symlink
            if target_path.exists():
                if target_path.is_symlink():
                    # Remove existing symlink
                    target_path.unlink()
                    if recreate:
                        stats["recreated"] += 1
                else:
                    # Regular file exists - don't remove it, raise error
                    raise FileExistsError(
                        f"Regular file exists at symlink target path: {target_path}. "
                        f"Cannot create symlink. Please remove or move the file manually."
                    )
            elif target_path.is_symlink():
                # Broken symlink (exists as symlink but target doesn't exist)
                target_path.unlink()
                logger.debug(f"Removed broken symlink: {target_path}")
            
            # Create the symlink
            try:
                target_path.symlink_to(source_path.resolve())
                stats["created"] += 1
                symlinked_files[kind] = str(target_path)
                logger.debug(f"Created symlink: {source_path} -> {target_path}")
            except OSError as e:
                logger.error(f"Failed to create symlink {target_path}: {e}")
                raise

    return symlinked_files, stats


@asset(
    description="Symlinked IDF files organized by partition in workdir"
)
def prepared_input_file_symlinks(
    context: AssetExecutionContext, 
    config: IDFPipelineConfig
) -> MaterializeResult:
    """Discover all IDF files and create symlinks for all partitions in workdir, skipping existing ones."""
    
    # Always discover all files
    context.log.info("🔍 Discovering all IDF files...")
    files_by_partition = discover_idf_files(config.input_dir)

    if not files_by_partition:
        context.log.warning("No IDF files found in input directory")
        return MaterializeResult(
            value={},
            metadata={"partitions_prepared": MetadataValue.int(0)}
        )

    context.log.info(f"📁 Found {len(files_by_partition)} partitions with IDF files")

    # Ensure workdir exists
    workdir_path = Path(config.workdir)
    workdir_path.mkdir(parents=True, exist_ok=True)
    
    # Track what happens during linking
    prepared_files = {}
    partitions_with_existing_links = []
    partitions_with_new_links = []
    total_existing_files = 0
    total_new_files = 0
    
    # Process ALL partitions, but skip existing symlinks
    for partition_key, files in files_by_partition.items():
        # Use the get_or_create_symlinks_for_partition function for consistent logic
        partition_files, stats = get_or_create_symlinks_for_partition(
            files,
            config.workdir,
            recreate=False
        )
        
        # Record results for this partition
        prepared_files[partition_key] = partition_files

        # Track statistics
        created_count = stats["created"]
        skipped_count = stats["skipped"]
        recreated_count = stats["recreated"]
        
        if skipped_count > 0 and created_count == 0 and recreated_count == 0:
            # All symlinks existed already
            partitions_with_existing_links.append(partition_key)
            total_existing_files += skipped_count
        else:
            # Some symlinks were created or recreated
            partitions_with_new_links.append(partition_key)
            total_new_files += created_count + recreated_count
            total_existing_files += skipped_count
    
    # Log summary
    context.log.info(f"✅ Processing complete:")
    context.log.info(f"   📂 {len(partitions_with_existing_links)} partitions had all symlinks already")
    context.log.info(f"   🆕 {len(partitions_with_new_links)} partitions needed new symlinks")
    context.log.info(f"   📄 {total_existing_files} files were already linked")
    context.log.info(f"   🔗 {total_new_files} new symlinks created")
    
    # Return results with comprehensive metadata
    return MaterializeResult(
        value=prepared_files,
        metadata={
            "total_partitions": MetadataValue.int(len(prepared_files)),
            "total_files_linked": MetadataValue.int(total_existing_files + total_new_files),
            "existing_files_count": MetadataValue.int(total_existing_files),
            "new_files_count": MetadataValue.int(total_new_files),
            "partitions_already_linked": MetadataValue.int(len(partitions_with_existing_links)),
            "partitions_with_new_links": MetadataValue.int(len(partitions_with_new_links)),
            "new_partitions_list": MetadataValue.text(", ".join(sorted(str(p) for p in partitions_with_new_links)) if partitions_with_new_links else "None"),
        }
    )


@asset(
    partitions_def=group_chan_partitions,
    deps=["prepared_input_file_symlinks"],
    description="Files for a specific partition extracted from symlinked IDF files"
)
def partition_files(
    context: AssetExecutionContext,
    prepared_input_file_symlinks: Dict[str, Dict[str, str]]
) -> MaterializeResult:
    """Extract files for the current partition from the prepared files."""
    partition_key = context.partition_key
    
    # Extract files for this partition
    files = prepared_input_file_symlinks.get(partition_key, {})
    n_files = len(files)
    
    if not files:
        context.log.warning(f"No files found for partition {partition_key}")
        return MaterializeResult(
            value={},
            metadata={"n_files": MetadataValue.int(0), "partition_key": MetadataValue.text(partition_key)}
        )

    context.log.info(f"Retrieved {n_files} files for partition {partition_key}")

    return MaterializeResult(
        value=files,
        metadata={
            "n_files": MetadataValue.int(n_files),
            "partition_key": MetadataValue.text(partition_key)
        }
    )