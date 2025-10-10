"""
Utility functions for the IDF lightcurve processing pipeline.
"""

import subprocess
from pathlib import Path


from dagster import AssetExecutionContext


def check_files_and_timestamps(
    context: AssetExecutionContext,
    input_files: list[Path],
    output_files: list[Path],
    operation_name: str = "operation"
) -> tuple[bool, bool]:
    """
    Helper function to check file existence and timestamps.
    
    Returns:
        (all_inputs_exist, all_outputs_up_to_date)
    """
    # Check if all input files exist
    missing_files = [str(f) for f in input_files if not f.exists()]
    if missing_files:
        context.log.error(f"Input files not found for {operation_name}: {missing_files}")
        return False, False
    
    # Check if all output files exist
    if not all(f.exists() for f in output_files):
        return True, False  # inputs exist, but not all outputs
    
    # Check timestamps - outputs should be newer than all inputs
    latest_input_time = max(f.stat().st_mtime for f in input_files)
    all_outputs_newer = all(f.stat().st_mtime > latest_input_time for f in output_files)
    
    if all_outputs_newer:
        context.log.info(f"All outputs are up to date for {operation_name}")
        return True, True
    
    return True, False  # inputs exist, outputs exist but not up to date


def run_subprocess_command(
    context: AssetExecutionContext,
    cmd: list[str],
    operation_name: str,
    partition_key: str,
    cwd: None | Path = None
) -> subprocess.CompletedProcess:
    """
    Helper function to run subprocess commands with consistent logging and error handling.
    
    Args:
        context: Dagster execution context for logging
        cmd: Command list to execute
        operation_name: Description of operation for logging
        partition_key: Partition key for context in logs
        cwd: Working directory for command (defaults to current directory)
        
    Returns:
        CompletedProcess result
        
    Raises:
        subprocess.CalledProcessError: If command fails
    """
    if cwd is None:
        cwd = Path.cwd()
        
    context.log.info(f"Running {operation_name} for {partition_key}: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd, 
            cwd=cwd, 
            capture_output=True, 
            text=True, 
            check=True
        )
        context.log.info(f"{operation_name} completed successfully for {partition_key}")
        context.log.debug(f"stdout: {result.stdout}")
        if result.stderr:
            context.log.debug(f"stderr: {result.stderr}")
        return result
    except subprocess.CalledProcessError as e:
        context.log.error(f"{operation_name} failed for {partition_key}: {e.stderr}")
        raise