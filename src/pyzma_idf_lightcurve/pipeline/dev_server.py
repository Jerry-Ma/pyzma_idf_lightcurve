#!/usr/bin/env python3
"""
Fast development server for Dagster with multiprocessing support
"""

import os
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

def main():
    # Create temporary directory for dev
    dev_home = Path("scratch_dagster/idf_lightcurve").resolve()
    if dev_home.exists():
        print(f"Using existing DAGSTER_HOME: {dev_home}")
    else:
        dev_home.mkdir(parents=True, exist_ok=True)
        print(f"Created new DAGSTER_HOME: {dev_home}")

    # Setup multiprocess instance configuration
    dagster_yaml_path, max_processes = setup_multiprocess_instance(dev_home)
    
    print(f"Using temp DAGSTER_HOME: {dev_home}")
    
    # Set environment variables for optimized dev mode with multiprocessing
    env = os.environ.copy()
    env.update({
        'DAGSTER_HOME': str(dev_home),
        'DAGSTER_AUTO_RELOAD': 'false',
    })
    
    # Run dagster dev with multiprocessing optimizations
    cmd = [
        'dagster', 'dev', 
        '-m', 'pyzma_idf_lightcurve.pipeline',
        '-p', '3001',
        '--host', '127.0.0.1',
        '--empty-workspace',
    ]
    
    print(f"Starting dev server: {' '.join(cmd)}")
    subprocess.run(cmd, env=env)

if __name__ == "__main__":
    main()