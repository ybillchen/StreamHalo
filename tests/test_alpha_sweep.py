"""Alpha-sweep tests: generate mock halos for different mass-function slopes,
then compare their 2PCF.

Both tests are optional (skipped by default). Run with:
    pytest tests/ --run-optional

Typical workflow:
    pytest tests/test_alpha_sweep.py --run-optional   # runs both in order
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import matplotlib.pyplot as plt
import numpy as np
import pytest

from tests.plot_style import apply_style, style_ax, legend as _legend
apply_style()

from streamhalo import MockHalo
from streamhalo.potentials import tidal_radius
from streamhalo.sampling import (
    make_broken_powerlaw_position_sampler,
    uniform_velocity_sampler,
)
from tests.test_two_point_correlation import (
    compute_w_with_errors,
    xyz_to_lb,
    OBSERVER,
    SIGMA_ANGLE,
    N_RESAMPLE,
    N_RANDOM,
)

# Use a sub-sample size below the smallest alpha halo (~1,475 particles for α=-2.3)
# so that all three halos have genuine subsampling and non-zero error bands.
N_SUB_ALPHA = 1_000

ALPHA_VALUES = [-1.5, -1.9, -2.3]
OUTPUT_DIR   = 'tests/outputs/alpha_sweep'

_ORIGIN = {'x_origin': 0.0, 'y_origin': 0.0, 'z_origin': 0.0,
           'dirx': 0.0, 'diry': 0.0, 'dirz': 1.0}
HOST_POTENTIAL_TYPE = 'Composite:MiyamotoNagai,Hernquist,Hernquist,NFW'
HOST_PARAMS = [
    {'logM': float(np.log10(4.7717e10)), 'Rs': 2.6, 'Hs': 0.3, **_ORIGIN},
    {'logM': float(np.log10(5.00e9)),    'Rs': 1.0,             **_ORIGIN},
    {'logM': float(np.log10(1.8142e9)),  'Rs': 0.0688867,       **_ORIGIN},
    {'logM': float(np.log10(5.5427e11)), 'Rs': 15.626,
     'a': 1.0, 'b': 1.0, 'c': 1.0,     **_ORIGIN},
]
NFW = HOST_PARAMS[3]

THETA_BINS = np.logspace(np.log10(0.3), 2.0, 15)


def _cache_path(alpha):
    return f'{OUTPUT_DIR}/mock_halo_alpha_{alpha:.1f}.npz'


def _build_halo(alpha, rng):
    halo = MockHalo(HOST_POTENTIAL_TYPE, HOST_PARAMS, rng=rng,
                    mass_function='powerlaw', alpha=alpha, M_min=1e7, M_max=3e10)
    halo.sample_satellites(
        stellar_mass_target=1e9,
        position_sampler=make_broken_powerlaw_position_sampler(
            r_min=2.0, r_max=100.0, alpha_inner=-3.0, alpha_outer=-3.0),
        velocity_sampler=uniform_velocity_sampler,
    )
    halo.generate_streams(
        target_particles=50_000,
        min_particles=500,
        integration_time=4.0,
        n_steps=8_000,
        plummer_scale=lambda r_sat, M_sat: (
            0.1 * tidal_radius(r_sat, M_sat, NFW['logM'], NFW['Rs'])),
    )
    return halo


# ---------------------------------------------------------------------------
# Test 1: generate and save
# ---------------------------------------------------------------------------

@pytest.mark.optional
def test_alpha_sweep_generate():
    """Generate a mock halo per alpha and save to NPZ.

    Runs in-process sequentially. GPU memory between satellites is managed by
    converting each JAX result to numpy immediately inside generate_streams,
    so only one satellite's computation lives on GPU at a time.
    """
    rng = np.random.default_rng(42)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for alpha in ALPHA_VALUES:
        path = _cache_path(alpha)
        print(f'\nalpha = {alpha}: generating ...', flush=True)
        halo = _build_halo(alpha, rng)
        halo.save(path)
        n = len(halo.stream_positions)
        del halo
        print(f'  saved {n:,} particles → {path}', flush=True)
        assert os.path.exists(path), f'Cache not written: {path}'


# ---------------------------------------------------------------------------
# Test 2: compare 2PCF
# ---------------------------------------------------------------------------

@pytest.mark.optional
def test_alpha_sweep_compare():
    """Load saved alpha-sweep caches and compare w(θ) across slopes."""
    missing = [a for a in ALPHA_VALUES if not os.path.exists(_cache_path(a))]
    if missing:
        pytest.skip(
            f'Cache files missing for alpha={missing}. '
            'Run test_alpha_sweep_generate first (--run-optional).'
        )

    rng = np.random.default_rng(0)
    results = {}

    for alpha in ALPHA_VALUES:
        d   = np.load(_cache_path(alpha))
        pos = d['stream_positions']
        l, b = xyz_to_lb(pos, observer=OBSERVER)
        n    = len(pos)
        print(f'\nalpha = {alpha}: {n:,} particles loaded from cache')

        w, err, theta_c = compute_w_with_errors(
            l, b, THETA_BINS,
            n_random=N_RANDOM, n_resample=N_RESAMPLE, n_sub=N_SUB_ALPHA,
            rng=rng, sigma_angle=SIGMA_ANGLE)
        results[alpha] = (w, err, theta_c, n)

    # --- print table ---
    theta_c = results[ALPHA_VALUES[0]][2]
    header  = f"{'theta_c':>7}  " + "  ".join(f"{'a='+str(a):>10}" for a in ALPHA_VALUES)
    print(f'\n{header}')
    print('-' * len(header))
    for k, tc in enumerate(theta_c):
        row = f'{tc:7.2f}  ' + '  '.join(
            f"{results[a][0][k]:10.4f}" for a in ALPHA_VALUES)
        print(row)

    # --- plot ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    for i, alpha in enumerate(ALPHA_VALUES):
        w, err, theta_c, n = results[alpha]
        ax.fill_between(theta_c, w - err, w + err, alpha=0.35, color=f'C{i}')
        ax.semilogx(theta_c, w, 'o-', color=f'C{i}', linewidth=1.8, markersize=4,
                    label=rf'$\alpha={alpha}$ ($N={n:,}$)')
    ax.axhline(0, color='k', linewidth=0.7, linestyle=':')
    style_ax(ax, xlabel=r'$\theta$ (deg)', ylabel=r'$w(\theta)$')
    ax.set_title(r'2PCF vs mass-function slope $\alpha$')
    ax.set_xlim(THETA_BINS[0], THETA_BINS[-1])
    _legend(ax, loc='upper right')
    plt.tight_layout()
    out = f'{OUTPUT_DIR}/w_theta_alpha_sweep.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nPlot saved to {out}')

    # --- assertions ---
    small = theta_c < 10.0
    for alpha in ALPHA_VALUES:
        w, err, _, _ = results[alpha]
        assert np.any(w[small] > 0), \
            f'alpha={alpha}: w not positive at any scale below 10 deg'

    w_flat  = results[max(ALPHA_VALUES)][0]
    w_steep = results[min(ALPHA_VALUES)][0]
    assert w_flat[small].mean() != w_steep[small].mean(), \
        'w(theta) identical for flattest and steepest alpha — no sensitivity to slope'

    print('\nAll assertions passed.')
