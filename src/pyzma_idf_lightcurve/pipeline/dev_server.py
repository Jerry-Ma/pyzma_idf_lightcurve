#!/usr/bin/env python3
"""
Fast development server for Dagster with multiprocessing support
"""

import os
import sys
import tempfile
from dagster import DagsterInstance
import subprocess
from pathlib import Path
import yaml
import multiprocessing

def get_optimal_process_count():
    """Get optimal number of processes based on CPU cores."""
    cpu_count = multiprocessing.cpu_count()
    # Use 75% of available cores, but at least 2 and at most 16
    optimal = max(2, min(16, int(cpu_count * 0.75)))
    print(f"CPU cores detected: {multiprocessing.cpu_count()} -> using {optimal} processes")
    return optimal

def setup_multiprocess_instance(dev_home: Path):
    """Setup Dagster instance with multiprocessing executor."""
    
    max_processes = get_optimal_process_count()
    
    # Create dagster.yaml with multiprocess executor
    dagster_yaml_content = {
        'telemetry': {
            'enabled': False
        },
        "run_coordinator": {
            "module": "dagster._core.run_coordinator.queued_run_coordinator",
            "class": "QueuedRunCoordinator",
            "config": {
                "max_concurrent_runs": max_processes,
            }
        },
        'backfills': {
            'use_threads': True,
            'num_workers': max_processes,
        }
    }
    
    # Write the configuration
    dagster_yaml_path = dev_home / 'dagster.yaml'
    with open(dagster_yaml_path, 'w') as f:
        yaml.dump(dagster_yaml_content, f, default_flow_style=False)
    
    print(f"Created multiprocess-enabled dagster.yaml at: {dagster_yaml_path}")
    print(f"Configured for {max_processes} concurrent processes")
    return dagster_yaml_path, max_processes

def run(dev_home: str, host: str = "127.0.0.1", port: int = 3001, config_file: str | None = None):
    """Run Dagster dev server with specified configuration.
    
    Args:
        dev_home: Directory for Dagster instance (required)
        host: Host to bind the server to
        port: Port to bind the server to
        config_file: Path to YAML configuration file (optional)
    """
    # Convert dev_home to Path and resolve
    dev_home_path = Path(dev_home).resolve()
    if dev_home_path.exists():
        print(f"Using existing DAGSTER_HOME: {dev_home_path}")
    else:
        dev_home_path.mkdir(parents=True, exist_ok=True)
        print(f"Created new DAGSTER_HOME: {dev_home_path}")

    # Setup multiprocess instance configuration
    dagster_yaml_path, max_processes = setup_multiprocess_instance(dev_home_path)
    
    print(f"Using DAGSTER_HOME: {dev_home_path}")
    
    # Set environment variables for optimized dev mode with multiprocessing
    env = os.environ.copy()
    env.update({
        'DAGSTER_HOME': str(dev_home_path),
        'DAGSTER_AUTO_RELOAD': 'false',
    })
    
    # Pass configuration file path via environment variable
    if config_file:
        config_file_abs = Path(config_file).resolve()
        print(f"Loading configuration from: {config_file_abs}")
        env['IDFLC_RESOURCE_DEFS_PATH'] = str(config_file_abs)
    
    # Get current Python executable to ensure we use the same environment
    python_executable = sys.executable
    
    # Run dagster dev with multiprocessing optimizations using current Python executable
    cmd = [
        python_executable, '-m', 'dagster', 'dev', 
        '-m', 'pyzma_idf_lightcurve.pipeline',
        '-p', str(port),
        '--host', host,
        '--empty-workspace',
    ]
    
    print(f"Using Python executable: {python_executable}")
    print(f"Starting dev server: {' '.join(cmd)}")
    subprocess.run(cmd, env=env)