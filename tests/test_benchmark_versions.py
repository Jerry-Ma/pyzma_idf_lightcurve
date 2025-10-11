"""
Benchmark comparison for different populate_epoch_from_catalog implementations.

This module tests v0, v1, and v2 implementations to identify the best strategy.
"""
import shutil
from pathlib import Path

import pytest

from pyzma_idf_lightcurve.lightcurve.datamodel import LightcurveStorage


class TestPopulateVersionComparison:
    """Compare different versions of populate_epoch_from_catalog."""
    
    @pytest.fixture
    def setup_storage(self, tmp_path, benchmark_source_catalog):
        """Create storage for each benchmark run."""
        def _setup(version_name):
            storage_path = tmp_path / f"storage_{version_name}"
            if storage_path.exists():
                shutil.rmtree(storage_path)
            
            storage = LightcurveStorage(storage_path)
            storage.create_storage(
                source_catalog=benchmark_source_catalog,
                epoch_keys=[f"r{58520832 + i * 256}" for i in range(100)],
            )
            return storage
        return _setup
    
    @pytest.mark.benchmark(group="populate_versions")
    def test_benchmark_populate_v0(
        self,
        benchmark,
        setup_storage,
        benchmark_source_catalog,
        benchmark_measurement_keys,
    ):
        """Benchmark v0: Original xarray .loc[] + zarr write."""
        storage = setup_storage("v0")
        epoch_key = "r58520832"
        
        def run_populate_v0():
            return storage.populate_epoch_from_catalog_v0(
                epoch_key=epoch_key,
                source_catalog=benchmark_source_catalog,
                measurement_keys=benchmark_measurement_keys,
            )
        
        result = benchmark(run_populate_v0)
        assert result > 0
    
    @pytest.mark.benchmark(group="populate_versions")
    def test_benchmark_populate_v1(
        self,
        benchmark,
        setup_storage,
        benchmark_source_catalog,
        benchmark_measurement_keys,
    ):
        """Benchmark v1: Read-modify-write with pre-computed indices."""
        storage = setup_storage("v1")
        epoch_key = "r58520832"
        
        def run_populate_v1():
            return storage.populate_epoch_from_catalog_v1(
                epoch_key=epoch_key,
                source_catalog=benchmark_source_catalog,
                measurement_keys=benchmark_measurement_keys,
            )
        
        result = benchmark(run_populate_v1)
        assert result > 0
    
    @pytest.mark.benchmark(group="populate_versions")
    def test_benchmark_populate_v2(
        self,
        benchmark,
        setup_storage,
        benchmark_source_catalog,
        benchmark_measurement_keys,
    ):
        """Benchmark v2: User's handwritten implementation."""
        storage = setup_storage("v2")
        epoch_key = "r58520832"
        
        def run_populate_v2():
            return storage.populate_epoch_from_catalog_v2(
                epoch_key=epoch_key,
                source_catalog=benchmark_source_catalog,
                measurement_keys=benchmark_measurement_keys,
            )
        
        result = benchmark(run_populate_v2)
        assert result > 0
    
    @pytest.mark.benchmark(group="populate_versions_multi")
    def test_benchmark_populate_v0_10_epochs(
        self,
        benchmark,
        tmp_path,
        benchmark_source_catalog,
        benchmark_measurement_keys,
    ):
        """Benchmark v0 with 10 epochs (realistic workflow)."""
        storage_path = tmp_path / "storage_v0_multi"
        if storage_path.exists():
            shutil.rmtree(storage_path)
        
        epoch_keys = [f"r{58520832 + i * 256}" for i in range(10)]
        
        def run_populate_all_v0():
            storage = LightcurveStorage(storage_path)
            storage.create_storage(
                source_catalog=benchmark_source_catalog,
                epoch_keys=epoch_keys,
            )
            
            for epoch_key in epoch_keys:
                storage.populate_epoch_from_catalog_v0(
                    epoch_key=epoch_key,
                    source_catalog=benchmark_source_catalog,
                    measurement_keys=benchmark_measurement_keys,
                )
            return storage
        
        result = benchmark(run_populate_all_v0)
        assert result.lightcurves is not None
    
    @pytest.mark.benchmark(group="populate_versions_multi")
    def test_benchmark_populate_v1_10_epochs(
        self,
        benchmark,
        tmp_path,
        benchmark_source_catalog,
        benchmark_measurement_keys,
    ):
        """Benchmark v1 with 10 epochs (realistic workflow)."""
        storage_path = tmp_path / "storage_v1_multi"
        if storage_path.exists():
            shutil.rmtree(storage_path)
        
        epoch_keys = [f"r{58520832 + i * 256}" for i in range(10)]
        
        def run_populate_all_v1():
            storage = LightcurveStorage(storage_path)
            storage.create_storage(
                source_catalog=benchmark_source_catalog,
                epoch_keys=epoch_keys,
            )
            
            for epoch_key in epoch_keys:
                storage.populate_epoch_from_catalog_v1(
                    epoch_key=epoch_key,
                    source_catalog=benchmark_source_catalog,
                    measurement_keys=benchmark_measurement_keys,
                )
            return storage
        
        result = benchmark(run_populate_all_v1)
        assert result.lightcurves is not None
    
    @pytest.mark.benchmark(group="populate_versions_multi")
    def test_benchmark_populate_v2_10_epochs(
        self,
        benchmark,
        tmp_path,
        benchmark_source_catalog,
        benchmark_measurement_keys,
    ):
        """Benchmark v2 with 10 epochs (realistic workflow)."""
        storage_path = tmp_path / "storage_v2_multi"
        if storage_path.exists():
            shutil.rmtree(storage_path)
        
        epoch_keys = [f"r{58520832 + i * 256}" for i in range(10)]
        
        def run_populate_all_v2():
            storage = LightcurveStorage(storage_path)
            storage.create_storage(
                source_catalog=benchmark_source_catalog,
                epoch_keys=epoch_keys,
            )
            
            for epoch_key in epoch_keys:
                storage.populate_epoch_from_catalog_v2(
                    epoch_key=epoch_key,
                    source_catalog=benchmark_source_catalog,
                    measurement_keys=benchmark_measurement_keys,
                )
            return storage
        
        result = benchmark(run_populate_all_v2)
        assert result.lightcurves is not None
