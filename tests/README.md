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

### `test_datamodel.py` (11 tests)
Tests for the `LightcurveStorage` class:
- Storage initialization and creation
- Zarr backend configuration
- Epoch population with vectorized assignment
- Object lightcurve retrieval
- Epoch data retrieval
- Storage metadata and info
- Spatial region queries
- Integration workflows

## Running Tests

```bash
# All tests (32 tests, ~1.3s)
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

- **Catalog Module:** 100%
- **Datamodel Module:** 100%
- **Overall:** 32/32 tests passing

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
