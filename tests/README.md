# StreamHalo Tests

This folder contains pytest tests demonstrating and verifying different features and capabilities of the StreamHalo package.

## Quick Navigation

### Core Tests
- **`test_mock_halo.py`** - Mock stellar halo generation test
  - Complete pipeline with 100,000 particles (80% streams, 20% background)
  - Mass-proportional particle allocation
  - Power-law satellite mass function
  - Realistic Hernquist profiles
  - Escape velocity constraints
  - Visualization generation

- **`test_satellite_sampling.py`** - Satellite population sampling test
  - Power-law mass function sampling
  - Orbital parameter generation
  - Mass distribution visualization
  - Sampling verification

### Performance Tests
- **`test_parallelization.py`** - JAX parallelization verification
  - Monitor CPU usage during halo generation
  - Verify multi-threading performance
  - Generate performance metrics JSON
  - Thread pool scaling analysis

## Running Tests

### Run all tests
```bash
conda activate StreaMAX
pytest tests/ -v
```

### Run specific test
```bash
pytest tests/test_mock_halo.py -v
pytest tests/test_satellite_sampling.py -v
pytest tests/test_parallelization.py -v
```

### Run with output
```bash
pytest tests/ -v -s
```

### Run as standalone scripts
```bash
# Run mock halo generation
python tests/test_mock_halo.py

# Run satellite sampling
python tests/test_satellite_sampling.py

# Run parallelization test
python tests/test_parallelization.py
```

## Test Outputs

Generated files are saved to `tests/outputs/`:

- **`full_pipeline_example/`**
  - `mock_halo_visualization.png` - 3D halo structure
  - `mock_halo_detailed.png` - Statistical analysis plots

- **`satellite_sampling/`**
  - `mass_function_distribution.png` - Mass distribution plot
  - `satellite_mass_function.png` - Detailed mass function visualization

- **`parallelization_test/`**
  - `parallelization_results.json` - CPU/memory metrics

## Test Configuration

### conftest.py
Pytest fixtures for:
- Random seed management
- Mock halo configuration
- Shared test utilities

### __init__.py
Package initialization for proper pytest discovery.

## Test Assertions

Each test includes assertions to verify:

### test_mock_halo
- Total particles generated (100,000)
- Stream/background particle counts
- Satellite count > 0
- Position/velocity array sizes
- Mass-proportional allocation correlation > 0.5

### test_satellite_sampling
- Satellite count matches request
- All masses within bounds (1e7 - 1e10 M_sun)
- Log mass array correctness
- Satellite data structure completeness
- Large sample size (10,000)

### test_parallelization
- CPU measurements recorded
- Valid CPU percentage values
- Thread count > 0
- Status format validation

## System Requirements

- Python 3.8+
- NumPy, Matplotlib, SciPy
- pytest
- JAX environment (StreaMAX conda environment)
- Multi-core CPU recommended

## Performance Notes

- First run compiles JAX JIT kernels (~3-5 sec overhead)
- Subsequent runs faster (~20 sec total)
- Mock halo generation: ~6.87 sec (parallelized)
- Background generation: ~2.01 sec
- Visualization: ~1-2 sec

## Example: Running with pytest

```bash
conda activate StreaMAX
cd /path/to/StreamHalo
pytest tests/ -v --tb=short
```

Expected output:
```
tests/test_mock_halo.py::test_mock_halo_generation PASSED
tests/test_satellite_sampling.py::test_satellite_sampling PASSED
tests/test_parallelization.py::test_jax_parallelization PASSED
```

## Troubleshooting

**Q: Tests fail with ModuleNotFoundError**
```bash
# Ensure StreaMAX environment is activated
conda activate StreaMAX
```

**Q: Parallelization test is slow**
- First run includes JIT compilation
- Subsequent runs are faster
- Check CPU usage with: `pytest tests/test_parallelization.py -v -s`

**Q: Output files not created**
- Check `tests/outputs/` directory exists
- Verify write permissions
- Check disk space

## Further Reading

- See `../README.md` for StreamHalo overview
- See `conftest.py` for pytest fixtures
- See individual test files for implementation details
