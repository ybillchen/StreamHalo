"""Enforce test execution order and manage optional tests.

Execution order: fast unit tests → mock halo (generates cache) → comparison
tests that consume the cache → sweeps (longest, run last).

Optional tests (marked with @pytest.mark.optional) are skipped by default.
Run them with:
    pytest tests/ --run-optional
"""

import os
import pytest

# Use the platform allocator (raw cudaMalloc) instead of the BFC pooled allocator.
# BFC with allow_growth can fail to satisfy large one-shot allocations (e.g. 1.67 GiB
# for precompute_prog_trajectories with n_steps=8000) due to CUDA virtual-address
# fragmentation, even when total free GPU memory is sufficient.
os.environ.setdefault('XLA_PYTHON_CLIENT_ALLOCATOR', 'platform')


def pytest_addoption(parser):
    parser.addoption(
        '--run-optional', action='store_true', default=False,
        help='Also run tests marked @pytest.mark.optional (skipped by default).',
    )


def pytest_configure(config):
    config.addinivalue_line(
        'markers',
        'optional: mark test as optional; skipped unless --run-optional is passed.',
    )


def pytest_collection_modifyitems(config, items):
    run_optional = config.getoption('--run-optional')
    skip_optional = pytest.mark.skip(reason='optional test; use --run-optional to run')

    ORDER = [
        'test_satellite_sampling',
        'test_parallelization',
        'test_satellite_kinematics',
        'test_mock_halo',
        'test_bj05_comparison',
        'test_two_point_correlation',
        'test_sweep_subsample_size',
        'test_alpha_sweep',
    ]
    rank = {name: i for i, name in enumerate(ORDER)}

    for item in items:
        if not run_optional and item.get_closest_marker('optional'):
            item.add_marker(skip_optional)

    items.sort(key=lambda item: rank.get(
        item.module.__name__.split('.')[-1], len(ORDER)))
