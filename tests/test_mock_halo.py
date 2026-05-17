#!/usr/bin/env python3
"""Mock halo generation test."""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np

from streamhalo import MockHalo
from streamhalo.potentials import v_esc, tidal_radius
from streamhalo.sampling import (
    make_broken_powerlaw_position_sampler,
    uniform_velocity_sampler,
)


def dndlogm(data, bins):
    """Return (bin_centers, dN/d log M) for log-spaced bins."""
    counts, edges = np.histogram(data, bins=bins)
    return np.sqrt(edges[:-1] * edges[1:]), counts / np.log10(edges[1:] / edges[:-1])


def test_mock_halo_generation():
    """End-to-end test: sample satellites, generate streams + background, plot, cache."""

    rng = np.random.default_rng(42)

    fstream = 0.5
    satellite_stellar_mass_target = 1e9
    total_particles = 10000
    target_stream_particles = int(total_particles * fstream)
    stream_integration_time = 4.0
    dt = 0.5  # Myr
    stream_n_steps = int(round(stream_integration_time * 1000 / dt))
    min_stream_particles = 100
    r_min, r_max = 2.0, 200.0
    projection_range = 200.0

    _origin = {'x_origin': 0.0, 'y_origin': 0.0, 'z_origin': 0.0,
               'dirx': 0.0, 'diry': 0.0, 'dirz': 1.0}
    host_potential_type = 'Composite:MiyamotoNagai,Hernquist,Hernquist,NFW'
    host_params = [
        {'logM': np.log10(4.7717e10), 'Rs': 2.6, 'Hs': 0.3, **_origin},   # disk
        {'logM': np.log10(5e9),       'Rs': 1.0,             **_origin},   # bulge
        {'logM': np.log10(1.8142e9),  'Rs': 0.0688867,       **_origin},   # nucleus
        {'logM': np.log10(5.5427e11), 'Rs': 15.626,                        # halo
         'a': 1.0, 'b': 1.0, 'c': 1.0, **_origin},
    ]
    nfw = host_params[3]

    halo = MockHalo(
        host_potential_type, host_params, rng=rng,
        mass_function='powerlaw', alpha=-1.9, M_min=1e7, M_max=3e10,
    )

    halo.sample_satellites(
        stellar_mass_target=satellite_stellar_mass_target,
        position_sampler=make_broken_powerlaw_position_sampler(
            r_min=r_min, r_max=100.0, alpha_inner=-3, alpha_outer=-3),
        velocity_sampler=uniform_velocity_sampler,
    )

    halo.generate_streams(
        target_particles=target_stream_particles,
        min_particles=min_stream_particles,
        integration_time=stream_integration_time,
        n_steps=stream_n_steps,
        plummer_scale=lambda r_sat, M_sat: 0.1 * tidal_radius(r_sat, M_sat, nfw['logM'], nfw['Rs']),
    )

    os.makedirs('tests/outputs', exist_ok=True)
    halo.save('tests/outputs/mock_halo_cache.npz')

    n_stream = len(halo.stream_positions)
    n_bg = int(n_stream * (1 - fstream) / fstream)
    if n_stream + n_bg < total_particles * 0.9:
        warnings.warn(f"Total particles ({n_stream + n_bg}) below 90% of target ({total_particles})")

    halo.generate_background(
        n_particles=n_bg,
        position_sampler=make_broken_powerlaw_position_sampler(
            r_min=r_min, r_max=r_max, alpha_inner=-3, alpha_outer=-3),
    )

    combined_pos = halo.combined_positions
    combined_idx = halo.combined_index
    stream_mask = combined_idx >= 0
    n_stream_actual = stream_mask.sum()
    n_bg_actual = (~stream_mask).sum()
    fraction_actual = n_stream_actual / len(combined_pos)
    radii = np.linalg.norm(combined_pos, axis=1)

    halo_masses = halo.satellite_halo_masses
    stellar_masses = halo.satellite_stellar_masses
    n_satellites = halo.n_satellites

    output_dir = 'tests/outputs/full_pipeline_example'
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 3))
    xh, yh = dndlogm(halo_masses,    np.logspace(np.log10(halo_masses.min()),    np.log10(halo_masses.max()),    30))
    xs, ys = dndlogm(stellar_masses, np.logspace(np.log10(stellar_masses.min()), np.log10(stellar_masses.max()), 30))
    ax.step(xh, yh, where='mid', linewidth=2, color='C0', label='Halo mass')
    ax.step(xs, ys, where='mid', linewidth=2, color='C1', label='Stellar mass (Behroozi+13)')
    ax.set_xlabel(r'Mass ($M_\odot$)')
    ax.set_ylabel(r'$dN/d\log M$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(stellar_masses.min(), halo_masses.max())
    ax.legend(loc='upper right', fontsize=8)
    ax.text(0.05, 0.05, f'$N_{{\\rm sat}}={n_satellites}$\n$\\alpha={halo.satellites.alpha}$',
            transform=ax.transAxes, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=9)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/mock_halo_mass_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(combined_pos[~stream_mask, 0], combined_pos[~stream_mask, 1],
               c='gray', s=1, alpha=0.3, label=f'Background ({n_bg_actual:,})')
    ax.scatter(combined_pos[stream_mask, 0], combined_pos[stream_mask, 1],
               c=combined_idx[stream_mask], cmap='tab20', s=1, alpha=0.6,
               label=f'Streams ({n_stream_actual:,})')
    ax.set_xlabel('X (kpc)')
    ax.set_ylabel('Y (kpc)')
    ax.set_xlim(-projection_range, projection_range)
    ax.set_ylim(-projection_range, projection_range)
    ax.set_aspect('equal')
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/mock_halo_xy_projection.png', dpi=150, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(4, 3))
    r_bins = np.logspace(np.log10(10.0), np.log10(1000.0), 40)
    xr, yr_stream = dndlogm(radii[stream_mask],  r_bins)
    _,  yr_bg     = dndlogm(radii[~stream_mask], r_bins)
    ax.step(xr, yr_stream, where='mid', linewidth=2, label=f'Streams ({n_stream_actual:,})', color='C0')
    ax.step(xr, yr_bg,     where='mid', linewidth=2, label=f'Background ({n_bg_actual:,})', color='gray')
    ax.set_xlabel('Radius (kpc)')
    ax.set_ylabel(r'$dN/d\log r$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(10.0, 1000.0)
    ax.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/mock_halo_radial_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()

    allocated = sorted(halo.particles_per_satellite)
    correlation = np.corrcoef(
        np.log10([stellar_masses[i] for i in allocated]),
        [halo.particles_per_satellite[i] for i in allocated],
    )[0, 1]

    assert n_stream_actual > 0
    assert n_bg_actual > 0
    assert abs(fraction_actual - fstream) < 0.1
    assert correlation > 0.4


if __name__ == '__main__':
    test_mock_halo_generation()
