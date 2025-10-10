"""
Core asset definitions for the IDF lightcurve processing pipeline.
"""

import re
import subprocess
from pathlib import Path
from typing import Literal, TypeVar, Generic, TypedDict, get_args
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

from ..utils.naming import NameTemplate, make_regex_stub_from_literal
from ..types import ChanT, GroupNameT, IDFFilename
from ..lightcurve.catalog import SourceCatalog, MeasurementType
from .config import IDFPipelineConfig
from .utils import check_files_and_timestamps, run_subprocess_command



class PartitionKeyT(TypedDict):
    """Type for partition key components."""
    group_name: GroupNameT
    chan: ChanT


class PartitionKey(NameTemplate[PartitionKeyT]):
    """Template for IDF partition keys."""
    template = "{group_name}_{chan}"
    pattern = re.compile(rf"^(?P<group_name>gr\d+)_{make_regex_stub_from_literal('chan', ChanT)}$")
    
    @classmethod
    def get_components(cls, partition_key: str) -> tuple[GroupNameT, ChanT]:
        """Extract group_name and chan from partition key."""
        parsed = cls.parse(partition_key)
        return parsed['group_name'], parsed['chan']

# Define dynamic partitions for group-channel combinations
group_chan_partitions = DynamicPartitionsDefinition(name="group_chan")

def _make_measurement_types() -> list[str]:
    chan_names = get_args(ChanT)

    # TODO load the sextractor config to get aperture info
    col_suffixes_sex = ["auto", "iso"] + [f"aper_{i}" for i in range(8)]

    # TODO implement this later 
    col_suffixes_div = []

    measurement_types = []
    for chan in chan_names:
        for kind, suffix in (("sci", ""), ("sci", "_clean")):
            for col_suffix in col_suffixes_sex:
                measurement_types.append(
                    MeasurementType.make(
                        chan=chan, kind=kind, suffix=suffix, col_suffix=col_suffix
                    )
                )
        for col_suffix in col_suffixes_div:
            measurement_types.append(
                MeasurementType.make(
                    chan=chan, kind="sci", suffix="_div", col_suffix=col_suffix
                )
            )
    return measurement_types 


measurement_types_all = _make_measurement_types()


def discover_idf_files(input_dir: str) -> dict[str, dict[str, str]]:
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


def get_or_create_symlinks_for_partition(files: dict[str, str], workdir: str, recreate=False) -> tuple[dict[str, str], dict[str, int]]:
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
    description="Symlinked IDF files organized by partition in workdir",
    required_resource_keys={"idf_pipeline_config"}
)
def prepared_input_file_symlinks(
    context: AssetExecutionContext
) -> MaterializeResult:
    """Discover all IDF files and create symlinks for all partitions in workdir, skipping existing ones."""
    
    # Get static pipeline config from resources
    config: IDFPipelineConfig = context.resources.idf_pipeline_config
    
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
    prepared_input_file_symlinks: dict[str, dict[str, str]]
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

@asset(
    deps=["partition_files"],
    partitions_def=group_chan_partitions,
    description="Normalized images from dividing sci files by coadd images",
    required_resource_keys={"idf_pipeline_config"}
)
def sci_divided_by_coadd(
    context: AssetExecutionContext,
    partition_files: dict[str, str]
) -> MaterializeResult:
    """Run imarith.py to divide sci file by coadd file."""
    # Get static pipeline config from resources
    config: IDFPipelineConfig = context.resources.idf_pipeline_config
    
    partition_key = context.partition_key
    op_name = "divide sci by coadd {partition_key}"
    
    if not partition_files or "sci" not in partition_files:
        context.log.warning(f"No sci file found for partition {partition_key}")
        return MaterializeResult(
            value="",
            metadata={"files_divided": MetadataValue.int(0)}
        )
    
    # Extract group_name and chan from partition key
    group_name, chan = PartitionKey.get_components(partition_key)
    
    # Input files
    sci_file = Path(partition_files["sci"])
    coadd_file = Path(config.coadd_dir) / f"coadd_{chan}.fits"  # Use configurable coadd directory
    
    # Output file (add _div suffix to sci file)
    output_file = IDFFilename.remake_filepath(sci_file, suffix='_div')
    
    # Check files and timestamps
    inputs_exist, outputs_up_to_date = check_files_and_timestamps(
        context,
        input_files=[sci_file, coadd_file],
        output_files=[output_file],
        operation_name=op_name
    )
    
    if not inputs_exist:
        return MaterializeResult(
            value="",
            metadata={"files_divided": MetadataValue.int(0)}
        )
    
    if outputs_up_to_date:
        return MaterializeResult(
            value=str(output_file),
            metadata={
                "files_divided": MetadataValue.int(1),
                "output_file": MetadataValue.text(str(output_file))
            }
        )
    
    # Run imarith.py
    cmd = [
        "python", config.imarith_script,
        str(sci_file),
        str(coadd_file),
        "divide",
        "-o", str(output_file)
    ]
    
    result = run_subprocess_command(
        context=context,
        cmd=cmd,
        operation_name=op_name,
        partition_key=partition_key
    )
    
    return MaterializeResult(
        value=str(output_file),
        metadata={
            "files_divided": MetadataValue.int(1),
            "output_file": MetadataValue.text(str(output_file)),
            "command": MetadataValue.text(" ".join(cmd))
        }
    )

@asset(
    deps=["partition_files"],
    partitions_def=group_chan_partitions,
    description="Weight files created from uncertainty files",
    required_resource_keys={"idf_pipeline_config"}
)
def wht_from_unc(
    context: AssetExecutionContext,
    partition_files: dict[str, str]
) -> MaterializeResult:
    """Run make_wht.py to create weight files from uncertainty files."""
    # Get static pipeline config from resources
    config: IDFPipelineConfig = context.resources.idf_pipeline_config
    
    partition_key = context.partition_key
    op_name = f"make weight files {partition_key}"

    if not partition_files or "unc" not in partition_files:
        context.log.warning(f"No unc file found for partition {partition_key}")
        return MaterializeResult(
            value="",
            metadata={"weight_files_created": MetadataValue.int(0)}
        )
    
    # Input file
    unc_file = Path(partition_files["unc"])
    
    # Output file - infer from unc filename by replacing .fits with _wht.fits
    output_file = IDFFilename.remake_filepath(unc_file, suffix="_wht")
    
    # Check files and timestamps
    inputs_exist, outputs_up_to_date = check_files_and_timestamps(
        context,
        input_files=[unc_file],
        output_files=[output_file],
        operation_name=op_name
    )
    
    if not inputs_exist:
        return MaterializeResult(
            value="",
            metadata={"weight_files_created": MetadataValue.int(0)}
        )
    
    if outputs_up_to_date:
        return MaterializeResult(
            value=str(output_file),
            metadata={
                "weight_files_created": MetadataValue.int(1),
                "output_file": MetadataValue.text(str(output_file))
            }
        )
    
    # Run make_wht.py
    cmd = [
        "python", config.make_wht_script,
        str(unc_file)
    ]
    
    result = run_subprocess_command(
        context=context,
        cmd=cmd,
        operation_name=op_name,
        partition_key=partition_key
    )
    
    return MaterializeResult(
        value=str(output_file),
        metadata={
            "weight_files_created": MetadataValue.int(1),
            "output_file": MetadataValue.text(str(output_file)),
            "command": MetadataValue.text(" ".join(cmd))
        }
    )

@asset(
    deps=["partition_files"],
    partitions_def=group_chan_partitions,
    description="Cleaned images processed with lacosmic cleaning algorithm",
    required_resource_keys={"idf_pipeline_config"}
)
def sci_lac_cleaned(
    context: AssetExecutionContext,
    partition_files: dict[str, str],
) -> MaterializeResult:
    """Run lac.py to produce cleaned sci files."""
    # Get static pipeline config from resources
    config: IDFPipelineConfig = context.resources.idf_pipeline_config
    
    partition_key = context.partition_key
    op_name = f"lac clean files {partition_key}"

    if not partition_files or "sci" not in partition_files or "unc" not in partition_files:
        context.log.warning(f"Required sci or unc file not found for partition {partition_key}")
        return MaterializeResult(
            value="",
            metadata={"files_cleaned": MetadataValue.int(0)}
        )
    
    # Extract group_name and chan from partition key
    group_name, chan = PartitionKey.get_components(partition_key)
    
    # Find sci files and weight files
    sci_file = Path(partition_files["sci"])
    unc_file = Path(partition_files["unc"])
    output_file = IDFFilename.remake_filepath(sci_file, suffix='_clean')
    
    # Check files and timestamps
    inputs_exist, outputs_up_to_date = check_files_and_timestamps(
        context,
        input_files=[sci_file, unc_file],
        output_files=[output_file],
        operation_name=op_name,
    )
    
    if not inputs_exist:
        return MaterializeResult(
            value="",
            metadata={"files_cleaned": MetadataValue.int(0)}
        )
    
    if outputs_up_to_date:
        return MaterializeResult(
            value=str(output_file),
            metadata={
                "files_cleaned": MetadataValue.int(1),
                "output_file": MetadataValue.text(str(output_file))
            }
        )
    
    # Run lac.py
    cmd = ["python", config.lac_script, str(sci_file), str(unc_file)]
    
    result = run_subprocess_command(
        context=context,
        cmd=cmd,
        operation_name=op_name,
        partition_key=partition_key
    )
    
    return MaterializeResult(
        value=str(output_file),
        metadata={
            "files_cleaned": MetadataValue.int(1),
            "output_file": MetadataValue.text(str(output_file)),
            "command": MetadataValue.text(" ".join(cmd))
        }
    )

def _run_sextractor_on_file(
    context: AssetExecutionContext,
    config: IDFPipelineConfig,
    partition_key: str,
    sci_file: Path,
    unc_file: Path,
    sexcat_file: Path,
    ecsv_file: Path,
    label: str
) -> bool:
    """
    Shared logic to run SExtractor on a single FITS file.
    Returns True if successful, False otherwise.
    """
    coadd_path = Path(config.coadd_dir)
    op_name = f"SExtractor {label} {partition_key}"
    
    chan_detect = "ch1"  # always use ch1 for detection
    coadd_file = coadd_path / f"coadd_{chan_detect}.fits"
    coadd_wht_file = coadd_path / f"coadd_{chan_detect}_wht.fits"
    
    # Check files and timestamps
    inputs_exist, outputs_up_to_date = check_files_and_timestamps(
        context,
        input_files=[sci_file, unc_file, coadd_file, coadd_wht_file],
        output_files=[sexcat_file, ecsv_file],
        operation_name=op_name,
    )
    
    if not inputs_exist:
        return False
    
    if outputs_up_to_date:
        return True
    
    # Run SExtractor
    cmd_sex = [
        "sex", str(coadd_file), str(sci_file),
        "-c", config.sex_config_file,
        "-WEIGHT_TYPE", "MAP_WEIGHT,MAP_RMS",
        "-WEIGHT_IMAGE", ",".join([str(coadd_wht_file), str(unc_file)]),
        "-CATALOG_NAME", str(sexcat_file)
    ]
    
    result_sex = run_subprocess_command(
        context=context,
        cmd=cmd_sex,
        operation_name=op_name,
        partition_key=partition_key
    )
    # Convert sexcat -> ecsv using cat2cat.py
    cmd_ecsv = [
        "python", config.cat2cat_script,
        str(sexcat_file),
        "--fmt-in", "ascii.sextractor",
        "--fmt-out", "ascii.ecsv",
        "--output", str(ecsv_file)
    ]

    result_ecsv = run_subprocess_command(
        context=context,
        cmd=cmd_ecsv,
        operation_name=f"convert to ecsv {label} {partition_key}",
        partition_key=partition_key
    ) 
    return True

@asset(
    deps=["partition_files"],
    partitions_def=group_chan_partitions,
    description="Source catalogs extracted from sci files",
    required_resource_keys={"idf_pipeline_config"}
)
def cat_from_sci(
    context: AssetExecutionContext,
    partition_files: dict[str, str],
) -> MaterializeResult:
    """Run SExtractor on sci files to produce .sexcat files."""
    # Get static pipeline config from resources
    config: IDFPipelineConfig = context.resources.idf_pipeline_config
    
    partition_key = context.partition_key

    if not partition_files or "sci" not in partition_files or "unc" not in partition_files:
        context.log.warning(f"No sci or unc file found for partition {partition_key}")
        return MaterializeResult(
            value="",
            metadata={"catalog_created": MetadataValue.int(0)}
        )

    sci_file = Path(partition_files["sci"])
    unc_file = Path(partition_files["unc"])
    sexcat_file = IDFFilename.remake_filepath(sci_file, fileext='sexcat')
    ecsv_file = IDFFilename.remake_filepath(sci_file, fileext='ecsv')
    
    success = _run_sextractor_on_file(
        context, config, partition_key, sci_file, unc_file, sexcat_file, ecsv_file, "sci"
    )
    
    if success:
        return MaterializeResult(
            value=str(ecsv_file),
            metadata={
                "catalog_created": MetadataValue.int(1),
                "output_file": MetadataValue.text(str(ecsv_file)),
                "input_label": MetadataValue.text("sci")
            }
        )
    else:
        return MaterializeResult(
            value="",
            metadata={"catalog_created": MetadataValue.int(0)}
        )

@asset(
    deps=["partition_files", "sci_lac_cleaned"],
    partitions_def=group_chan_partitions,
    description="Source catalogs extracted from lac cleaned sci files",
    required_resource_keys={"idf_pipeline_config"}
)
def cat_from_sci_lac_cleaned(
    context: AssetExecutionContext,
    sci_lac_cleaned: str,
    partition_files: dict[str, str]
) -> MaterializeResult:
    """Run SExtractor on sci_clean files to produce .sexcat files."""
    # Get static pipeline config from resources
    config: IDFPipelineConfig = context.resources.idf_pipeline_config
    
    partition_key = context.partition_key

    if sci_lac_cleaned == "" or "unc" not in partition_files:
        context.log.warning(f"No lac cleaned sci file or unc found for partition {partition_key}")
        return MaterializeResult(
            value="",
            metadata={"catalog_created": MetadataValue.int(0)}
        )
    
    sci_clean_file = Path(sci_lac_cleaned)
    unc_file = Path(partition_files["unc"])
    sexcat_file = IDFFilename.remake_filepath(sci_clean_file, fileext='sexcat')
    ecsv_file = IDFFilename.remake_filepath(sci_clean_file, fileext='ecsv')

    success = _run_sextractor_on_file(
        context, config, partition_key, sci_clean_file, unc_file, sexcat_file, ecsv_file, "sci_clean"
    )
    
    if success:
        return MaterializeResult(
            value=str(ecsv_file),
            metadata={
                "catalog_created": MetadataValue.int(1),
                "output_file": MetadataValue.text(str(ecsv_file)),
                "file_type": MetadataValue.text("sci_clean")
            }
        )
    else:
        return MaterializeResult(
            value="",
            metadata={"catalog_created": MetadataValue.int(0)}
        )


asset_defs = [
    prepared_input_file_symlinks,
    partition_files,
    sci_divided_by_coadd,
    wht_from_unc,
    sci_lac_cleaned,
    cat_from_sci,
    cat_from_sci_lac_cleaned,
]