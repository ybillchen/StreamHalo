"""Sample 10000 satellites and plot |V| vs galactocentric radius."""

import os

import matplotlib.pyplot as plt
import numpy as np

from tests.test_mock_halo import (
    _POT_FNS, v_esc,
    sample_r_broken_powerlaw,
    sample_v_hernquist,
)


def test_satellite_kinematics():
    output_dir = 'tests/outputs/satellite_kinematics'
    os.makedirs(output_dir, exist_ok=True)

    rng = np.random.default_rng(42)
    n_satellites = 10000
    r_min, r_max = 1.0, 100.0

    _origin = {'x_origin': 0.0, 'y_origin': 0.0, 'z_origin': 0.0,
               'dirx': 0.0, 'diry': 0.0, 'dirz': 1.0}
    host_potential_type = 'Composite:MiyamotoNagai,Hernquist,Hernquist,NFW'
    host_params = [
        {'logM': np.log10(4.7717e10), 'Rs': 2.6, 'Hs': 0.3, **_origin},
        {'logM': np.log10(5e9),       'Rs': 1.0,             **_origin},
        {'logM': np.log10(1.8142e9),  'Rs': 0.0688867,       **_origin},
        {'logM': np.log10(5.5427e11), 'Rs': 15.626,
         'a': 1.0, 'b': 1.0, 'c': 1.0, **_origin},
    ]

    phi_scale = 0.5 * v_esc(r_min, host_potential_type, host_params)**2

    radii = np.empty(n_satellites)
    speeds = np.empty(n_satellites)

    for i in range(n_satellites):
        r = sample_r_broken_powerlaw(rng, r_min=r_min, r_max=r_max,
                                     alpha_inner=-3, alpha_outer=-3)
        v_e = v_esc(r, host_potential_type, host_params)
        v = sample_v_hernquist(rng, v_e, phi_scale)
        radii[i] = r
        speeds[i] = v

    r_curve = np.logspace(np.log10(1e0), np.log10(1e3), 300)
    v_esc_curve = np.array([v_esc(r, host_potential_type, host_params) for r in r_curve])

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(radii, speeds, s=1, alpha=0.3, rasterized=True, label='Satellites')
    ax.plot(r_curve, v_esc_curve, 'k--', linewidth=1.5, label=r'$v_{\rm esc}$')
    ax.set_xlabel('Galactocentric distance (kpc)')
    ax.set_ylabel(r'$|V|$ (km/s)')
    ax.set_xscale('log')
    ax.set_xlim(1e0, 1e3)
    ax.set_ylim(0, 700)
    ax.legend(fontsize=8)
    ax.set_title(f'{n_satellites:,} satellites — Hernquist DF')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/speed_vs_radius.png', dpi=150, bbox_inches='tight')
    plt.close()

    v_esc_local = np.array([v_esc(r, host_potential_type, host_params) for r in radii])
    assert len(radii) == n_satellites
    assert np.all(speeds >= 0)
    assert np.all(speeds < v_esc_local * 1.05)
