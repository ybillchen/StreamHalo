#!/usr/bin/env python3
"""Mock halo generation test."""

import os
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from StreaMAX.StreaMAX.potentials import (
    NFW_potential, Hernquist_potential, MiyamotoNagai_potential
)
from StreaMAX.StreaMAX.utils import prepare_params as stremax_prepare_params
from StreaMAX.StreaMAX.constants import KPCGYR_TO_KMS
from streamhalo import MockHalo

_POT_FNS = {
    'NFW': NFW_potential,
    'Hernquist': Hernquist_potential,
    'MiyamotoNagai': MiyamotoNagai_potential,
}


def v_esc(r, host_potential_type, host_params, r_vir=200.0):
    """Escape velocity (km/s) with virial radius as zero-point: v_esc²=2*(Phi(r_vir)-Phi(r))."""
    comp_types = host_potential_type.split(':')[1].split(',')
    def _phi(radius):
        return sum(
            float(_POT_FNS[t](float(radius), 0.0, 0.0, stremax_prepare_params(p)))
            for t, p in zip(comp_types, host_params)
        )
    return float(np.sqrt(2.0 * (_phi(r_vir) - _phi(r))) * KPCGYR_TO_KMS)


def sample_r_broken_powerlaw(rng, r_min=1.0, r_max=300.0, r_break=100.0,
                             alpha_inner=-3, alpha_outer=-4):
    """Sample galactocentric radius from a broken power-law 3D number density n(r)~r^alpha."""
    ei = 2 + alpha_inner   # exponent of dN/dr in inner region
    eo = 2 + alpha_outer   # exponent of dN/dr in outer region
    ratio = r_break ** (ei - eo)  # continuity normalisation B/A

    def _piece(r_lo, r_hi, exp, scale=1.0):
        if exp == -1:
            return scale * np.log(r_hi / r_lo)
        return scale * (r_hi**(exp+1) - r_lo**(exp+1)) / (exp + 1)

    I1 = _piece(r_min, r_break, ei)
    I2 = _piece(r_break, r_max, eo, scale=ratio)
    total = I1 + I2

    u = rng.uniform(0, 1)
    if u * total <= I1:
        if ei == -1:
            return r_min * np.exp(u * total)
        return (r_min**(ei+1) + u * total * (ei+1)) ** (1.0/(ei+1))
    else:
        u_adj = (u * total - I1) / ratio
        if eo == -1:
            return r_break * np.exp(u_adj)
        return (r_break**(eo+1) + u_adj * (eo+1)) ** (1.0/(eo+1))


def _df_hernquist(E):
    """Hernquist (1990) isotropic DF; E in [-1, 0] (dimensionless)."""
    q = np.sqrt(-E)
    return (1.0 / (8.0 * np.sqrt(2.0) * np.pi**3)
            * (1.0 - q*q)**(-2.5)
            * (3.0*np.arcsin(q) + q*np.sqrt(1.0 - q*q)*(1.0 - 2.0*q*q)*(8.0*q**4 - 8.0*q*q - 3.0)))


def sample_v_hernquist(rng, v_esc_val, phi_scale, n_grid=1000):
    """Sample speed from the Hernquist isotropic DF.

    Dimensionless energy: E_dim = (v^2/2 + Phi(r)) / phi_scale in [E_min, 0].
    phi_scale = |Phi(r_ref)| sets the global energy normalisation (use r_min as ref).
    Samples E_dim via numerical CDF inversion, returns v = sqrt(v_esc^2 - 2*q^2*phi_scale).
    """
    phi_r = -0.5 * v_esc_val**2          # Phi(r) < 0
    E_min = phi_r / phi_scale             # in (-1, 0]
    q_max = np.sqrt(-E_min)               # q at v=0; < 1 for r > r_min

    q_grid = np.linspace(1e-6, q_max * (1.0 - 1e-9), n_grid)
    pdf_q = np.maximum(np.array([_df_hernquist(-q**2) for q in q_grid]) * 2.0 * q_grid, 0.0)

    cdf = np.cumsum(pdf_q)
    cdf /= cdf[-1]

    q_s = np.interp(rng.uniform(0.0, 1.0), cdf, q_grid)
    return float(np.sqrt(v_esc_val**2 - 2.0 * q_s**2 * phi_scale))


def tidal_radius(r_sat, M_sat, logM_host, Rs_host, r_vir=200.0):
    """King tidal radius: r_t = r_sat * (M_sat / (3 * M_host(<r_sat)))^(1/3)."""
    M_host = 10.0 ** logM_host
    c = r_vir / Rs_host
    f_c = np.log(1.0 + c) - c / (1.0 + c)
    x = r_sat / Rs_host
    M_enc = M_host * (np.log(1.0 + x) - x / (1.0 + x)) / f_c
    return r_sat * (M_sat / (3.0 * M_enc)) ** (1.0 / 3.0)


def test_mock_halo_generation():
    """Test mock halo generation with mass-proportional allocation."""

    rng = np.random.default_rng(42)

    fstream = 0.5
    satellite_stellar_mass_target = 1e9  # target total stellar mass in streams (M_sun)

    mass_function_params = {
        'alpha': -1.9,
        'M_min': 1e7,
        'M_max': 3e10,
    }

    total_particles = 100000
    target_stream_particles = int(total_particles * fstream)
    stream_integration_time = 4.0
    dt = 0.5 # Myr
    stream_n_steps = int(round(stream_integration_time * 1000 / dt))
    min_stream_particles = 100
    r_min = 2.0
    r_max = 200.0
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

    # Initialize MockHalo
    halo = MockHalo(
        host_potential_type=host_potential_type,
        host_params=host_params,
        n_satellites=1,
        mass_function='powerlaw',
        n_stream_particles=1000,
        rng=rng
    )

    halo.satellites.alpha = mass_function_params['alpha']
    halo.satellites.M_min = mass_function_params['M_min']
    halo.satellites.M_max = mass_function_params['M_max']

    # Sample satellites

    _nfw = host_params[3]  # NFW halo component (used for tidal radius)
    r_vir = 200.0

    # Global energy normalisation: |Phi(r_min)| sets the most-bound reference energy
    phi_scale = 0.5 * v_esc(r_min, host_potential_type, host_params)**2

    satellite_masses = []
    satellite_positions = []
    satellite_velocities = []

    total_sampled_stellar_mass = 0

    while total_sampled_stellar_mass < satellite_stellar_mass_target:
        mass = halo.satellites.sample(1)[0]
        satellite_masses.append(mass)
        total_sampled_stellar_mass += halo.satellites.stellar_mass(mass)

        r = sample_r_broken_powerlaw(rng, r_min=r_min, r_max=100.0, alpha_inner=-3, alpha_outer=-3)

        theta = np.arccos(rng.uniform(-1, 1))
        phi = rng.uniform(0, 2 * np.pi)

        position = np.array([
            r * np.sin(theta) * np.cos(phi),
            r * np.sin(theta) * np.sin(phi),
            r * np.cos(theta)
        ])
        satellite_positions.append(position)

        escape_velocity = v_esc(r, host_potential_type, host_params)
        # v_mag = sample_v_hernquist(rng, escape_velocity, phi_scale)
        v_mag = rng.uniform(0.0 * escape_velocity, 1.0 * escape_velocity)

        v_theta = np.arccos(rng.uniform(-1, 1))
        v_phi = rng.uniform(0, 2 * np.pi)

        velocity = np.array([
            v_mag * np.sin(v_theta) * np.cos(v_phi),
            v_mag * np.sin(v_theta) * np.sin(v_phi),
            v_mag * np.cos(v_theta)
        ])
        satellite_velocities.append(velocity)

    satellite_masses = np.array(satellite_masses)
    satellite_positions = np.array(satellite_positions)
    satellite_velocities = np.array(satellite_velocities)
    n_satellites = len(satellite_masses)

    sat_data = {
        'halo_masses': satellite_masses,
        'stellar_masses': halo.satellites.stellar_mass(satellite_masses),
        'logM_halo': np.log10(satellite_masses),
        'positions': satellite_positions,
        'velocities': satellite_velocities,
        'initial_conditions': np.hstack([satellite_positions, satellite_velocities]),
    }

    halo.n_satellites = n_satellites

    # Generate streams
    t_start = time.time()

    all_positions = []
    all_velocities = []
    all_stream_index = []
    total_stellar = np.sum(sat_data['stellar_masses'])
    particles_allocated = {}

    for i in range(n_satellites):
        mass_fraction = sat_data['stellar_masses'][i] / total_stellar
        n_particles_target = int(np.round(mass_fraction * target_stream_particles))

        if n_particles_target < min_stream_particles:
            continue

        particles_allocated[i] = n_particles_target

        r_sat = np.linalg.norm(satellite_positions[i])
        r_t = tidal_radius(r_sat, satellite_masses[i],
                           _nfw['logM'], _nfw['Rs'])
        satellite_params = {
            'logM': float(sat_data['logM_halo'][i]),
            'Rs': float(0.1 * r_t)
        }

        n_particles_generate = stream_n_steps * max(1, n_particles_target // stream_n_steps)

        t_sat, xv_sat, xv_stream, xhi_stream = halo.add_stream(
            sat_data['initial_conditions'][i],
            'Plummer',
            satellite_params,
            stream_integration_time,
            n_steps=stream_n_steps,
            unroll=False,
            n_particles=n_particles_generate
        )

        positions = xv_stream[:, :3]
        velocities = xv_stream[:, 3:]

        if n_particles_target < len(positions):
            indices = rng.choice(len(positions), n_particles_target, replace=False)
            positions = positions[indices]
            velocities = velocities[indices]

        all_positions.append(positions)
        all_velocities.append(velocities)
        all_stream_index.append(np.full(len(positions), i, dtype=int))

    t_streams = time.time() - t_start

    halo_particles_streams = {
        'positions': np.vstack(all_positions),
        'velocities': np.vstack(all_velocities),
        'stream_index': np.concatenate(all_stream_index),
    }

    os.makedirs('tests/outputs', exist_ok=True)
    np.savez(
        'tests/outputs/mock_halo_cache.npz',
        stream_positions=halo_particles_streams['positions'],
        stream_velocities=halo_particles_streams['velocities'],
        stream_index=halo_particles_streams['stream_index'],
        satellite_halo_masses=sat_data['halo_masses'],
        satellite_stellar_masses=sat_data['stellar_masses'],
        satellite_positions=sat_data['positions'],
        satellite_velocities=sat_data['velocities'],
    )

    # Generate background
    t_start = time.time()

    total_stream_particles_generated = len(halo_particles_streams['positions'])
    n_bg_target = int(total_stream_particles_generated * (1 - fstream) / fstream)
    total_particles_target = total_stream_particles_generated + n_bg_target

    if total_particles_target < total_particles * 0.9:
        warnings.warn(f"Total particles target ({total_particles_target}) below 90% of target ({total_particles})")

    radii_bg = np.array([sample_r_broken_powerlaw(rng, r_min=r_min, r_max=r_max, alpha_inner=-3, alpha_outer=-3)
                         for _ in range(n_bg_target)])
    thetas_bg = np.arccos(rng.uniform(-1, 1, size=n_bg_target))
    phis_bg = rng.uniform(0, 2 * np.pi, size=n_bg_target)
    positions_bg = np.column_stack([
        radii_bg * np.sin(thetas_bg) * np.cos(phis_bg),
        radii_bg * np.sin(thetas_bg) * np.sin(phis_bg),
        radii_bg * np.cos(thetas_bg),
    ])
    velocities_bg = np.zeros((n_bg_target, 3))

    t_bg = time.time() - t_start

    # Combine particles

    combined_positions = np.vstack([
        halo_particles_streams['positions'],
        positions_bg
    ])

    combined_velocities = np.vstack([
        halo_particles_streams['velocities'],
        velocities_bg
    ])

    stream_index = np.concatenate([
        halo_particles_streams['stream_index'],
        np.full(n_bg_target, -1, dtype=int)
    ])

    total_particles_actual = len(combined_positions)
    total_stream_particles_actual = np.sum(stream_index >= 0)
    total_background_particles = np.sum(stream_index == -1)

    # Verify background particle fraction matches target
    fraction_actual = total_stream_particles_actual / total_particles_actual

    # Statistics

    radii = np.linalg.norm(combined_positions, axis=1)
    speeds = np.linalg.norm(combined_velocities, axis=1)

    # Visualization

    stream_mask = stream_index >= 0

    output_dir = 'tests/outputs/full_pipeline_example'
    os.makedirs(output_dir, exist_ok=True)

    # Figure 1: Satellite mass distribution histogram (log-spaced bins)
    fig, ax = plt.subplots(figsize=(5, 3))
    masses_list = sat_data['halo_masses']
    stellar_masses_list = sat_data['stellar_masses']

    log_bins_h = np.logspace(np.log10(masses_list.min()), np.log10(masses_list.max()), 30)
    counts_h, edges_h = np.histogram(masses_list, bins=log_bins_h)
    density_h = counts_h / np.log10(edges_h[1:] / edges_h[:-1])
    x_step_h = np.concatenate([edges_h[:-1], edges_h[-1:]])
    y_step_h = np.concatenate([density_h, [density_h[-1]]])
    ax.step(x_step_h, y_step_h, where='post', linewidth=2, color='C0', label='Halo mass')

    log_bins_s = np.logspace(np.log10(stellar_masses_list.min()),
                              np.log10(stellar_masses_list.max()), 30)
    counts_s, edges_s = np.histogram(stellar_masses_list, bins=log_bins_s)
    density_s = counts_s / np.log10(edges_s[1:] / edges_s[:-1])
    x_step_s = np.concatenate([edges_s[:-1], edges_s[-1:]])
    y_step_s = np.concatenate([density_s, [density_s[-1]]])
    ax.step(x_step_s, y_step_s, where='post', linewidth=2, color='C1',
            label='Stellar mass (Behroozi+13)')

    ax.set_xlabel(r'Mass ($M_\odot$)')
    ax.set_ylabel(r'$dN/d\log M$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(stellar_masses_list.min(), mass_function_params['M_max'])
    ax.legend(loc='upper right', fontsize=8)
    info_text = (f'$N_{{\\rm sat}}={n_satellites}$\n'
                 f'$\\alpha={mass_function_params["alpha"]}$')
    ax.text(0.05, 0.05, info_text, transform=ax.transAxes, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=9)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/mock_halo_mass_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 2: 2D projection (XY plane)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(combined_positions[~stream_mask, 0],
              combined_positions[~stream_mask, 1],
              c='gray', s=1, alpha=0.3, label=f'Background ({total_background_particles:,})')
    ax.scatter(combined_positions[stream_mask, 0],
              combined_positions[stream_mask, 1],
              c=stream_index[stream_mask], cmap='tab20', s=1, alpha=0.6, label=f'Streams ({total_stream_particles_actual:,})')
    ax.set_xlabel('X (kpc)')
    ax.set_ylabel('Y (kpc)')
    ax.set_xlim(-projection_range, projection_range)
    ax.set_ylim(-projection_range, projection_range)
    ax.set_aspect('equal')
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/mock_halo_xy_projection.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 3: Radial distribution histogram (log-spaced bins)
    fig, ax = plt.subplots(figsize=(4, 3))
    r_min = 10.0
    r_max = 1000.0
    log_bins = np.logspace(np.log10(r_min), np.log10(r_max), 40)

    counts_stream, edges_stream = np.histogram(radii[stream_mask], bins=log_bins)
    bin_widths = edges_stream[1:] - edges_stream[:-1]
    density_stream = counts_stream / np.log10(edges_stream[1:] / edges_stream[:-1])

    counts_bg, _ = np.histogram(radii[~stream_mask], bins=log_bins)
    density_bg = counts_bg / np.log10(edges_stream[1:] / edges_stream[:-1])

    x_step = np.concatenate([edges_stream[:-1], edges_stream[-1:]])
    y_stream_step = np.concatenate([density_stream, [density_stream[-1]]])
    y_bg_step = np.concatenate([density_bg, [density_bg[-1]]])

    ax.step(x_step, y_stream_step, where='post', linewidth=2, label=f'Streams ({total_stream_particles_actual:,})', color='C0')
    ax.step(x_step, y_bg_step, where='post', linewidth=2, label=f'Background ({total_background_particles:,})', color='gray')

    ax.set_xlabel('Radius (kpc)')
    ax.set_ylabel(r'$dN/d\log r$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(r_min, r_max)
    ax.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/mock_halo_radial_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Summary
    allocated_indices = sorted(particles_allocated.keys())
    particles_list = [particles_allocated[i] for i in allocated_indices]
    stellar_masses_list = sat_data['stellar_masses']
    masses_allocated = [stellar_masses_list[i] for i in allocated_indices]
    correlation = np.corrcoef(np.log10(masses_allocated), particles_list)[0, 1]

    # Assertions for pytest
    assert total_particles_actual > total_particles_target * 0.90, f"Generated {total_particles_actual} < 90% of target {total_particles_target}"
    assert n_satellites > 0, f"Expected satellites, got {n_satellites}"
    assert len(combined_positions) == total_particles_actual, "Position array size mismatch"
    assert len(combined_velocities) == total_particles_actual, "Velocity array size mismatch"
    assert total_stream_particles_actual > 0, "No stream particles generated"
    assert total_background_particles > 0, "No background particles generated"
    assert total_stream_particles_actual + total_background_particles == total_particles_actual, \
        "Particle count mismatch"
    assert abs(fraction_actual - fstream) < 0.1, f"Stream fraction {fraction_actual:.2%} deviates from expected {fstream:.0%}"
    assert correlation > 0.4, f"Mass-proportional allocation correlation too low: {correlation}"

if __name__ == '__main__':
    test_mock_halo_generation()
