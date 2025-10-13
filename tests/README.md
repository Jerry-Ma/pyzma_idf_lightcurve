# Tests

Comprehensive test suite for the IDF Lightcurve Storage system.

## Test Files

### `test_catalog.py` (21 tests)
Tests for the `SourceCatalog` class:
- Measurement key extraction and format validation
- Object key generation (string-based)
- Spatial ordering with grid-based algorithm
- Coordinate handling and dictionary generation
- Valid measurement masking

### `test_datamodel.py` (39 tests)
Tests for the `LightcurveStorage` class:
- Storage initialization and creation
- Zarr backend configuration
- Epoch population with vectorized assignment
- Object lightcurve retrieval
- Epoch data retrieval
- Storage metadata and info
- Spatial region queries
- Data persistence across reloads
- Error handling for invalid operations
- Integration workflows and edge cases

### `test_benchmarks.py`
Performance benchmarks comparing different implementations.

### `test_concurrent_writing.py`
Tests for concurrent epoch writing capabilities.

### `test_naming.py`
Tests for catalog naming and identification.

## Running Tests

```bash
# All tests
uv run pytest tests/

# Specific module
uv run pytest tests/test_catalog.py -v
uv run pytest tests/test_datamodel.py -v

# With coverage report
uv run pytest tests/ --cov=src/pyzma_idf_lightcurve --cov-report=html

# Watch mode (requires pytest-watch)
uv run ptw tests/
```

## Test Coverage

Current test suite includes:
- **test_catalog.py**: SourceCatalog functionality
- **test_datamodel.py**: LightcurveStorage functionality (39 tests)
- **test_benchmarks.py**: Performance comparisons
- **test_concurrent_writing.py**: Concurrent operations
- **test_naming.py**: Naming conventions

## Removed Test Files

The following test files have been removed as they tested obsolete API:
- `test_populate_correctness.py`: Tested versioned populate methods (v0/v1/v2) that no longer exist
- `test_zarr_storage.py`: Tested old storage API that has been replaced

The functionality they tested is now covered by `test_datamodel.py` with the current API.

## Key Testing Patterns

### Fixtures
All tests use pytest fixtures for setup:
- `sample_catalog`: Test catalog with standard columns
- `measurement_keys`: Properly formatted measurement keys
- `epoch_keys`: Test epoch identifiers
- `storage`: Pre-configured LightcurveStorage instance
- `tmp_path`: Temporary directory (pytest built-in)

### Test Organization
- One test class per module: `TestSourceCatalog`, `TestLightcurveStorage`
- Integration tests in dedicated class: `TestIntegrationScenarios`
- Descriptive test names: `test_<feature>_<scenario>`

## Performance Validations

Tests verify xarray optimization patterns:
- ✅ Vectorized assignment (batched updates)
- ✅ View-based access (no unnecessary copies)
- ✅ Efficient spatial queries (boolean indexing)
- ✅ Region-based Zarr writes

## Documentation

For detailed information about:
- Test consolidation history: `/design/TEST_CONSOLIDATION.md`
- API changes and updates: `/design/XARRAY_MIGRATION.md`
- Performance optimizations: `/design/XARRAY_OPTIMIZATIONS.md`
- Architecture overview: `/design/ARCHITECTURE_OVERVIEW.md`
