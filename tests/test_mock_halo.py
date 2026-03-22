#!/usr/bin/env python3
"""Mock halo generation test."""

import os
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from streamhalo import MockHalo

def v_esc(r, M_h, Rs):
    """Calculate escape velocity at radius r for NFW halo."""
    GM_sun = 1.32712e11
    kpc_to_km = 3.0857e16
    x = r / Rs
    M_enclosed = M_h * (np.log(1.0 + x) - x / (1.0 + x))
    r_km = r * kpc_to_km
    v_esc = np.sqrt(2.0 * GM_sun * M_enclosed / r_km)
    return v_esc

def hernquist_density(r, a):
    """Hernquist density profile: rho(r) = 1/(r(r+a)^3)"""
    return 1.0 / (r * (r + a)**3)

def test_mock_halo_generation():
    """Test mock halo generation with mass-proportional allocation."""

    rng = np.random.default_rng(42)

    total_halo_mass = 1e10
    fstream = 0.5
    satellite_total_mass = total_halo_mass * fstream
    background_total_mass = total_halo_mass * (1 - fstream)

    mass_function_params = {
        'alpha': -1.9,
        'M_min': 1e6,
        'M_max': 1e10,
    }

    total_particles = 10000
    target_stream_particles = int(total_particles * fstream)
    stream_integration_time = 8.0
    stream_n_steps = 1000
    min_stream_particles = 1
    background_scale_radius = 20.0
    background_r_max = 300.0
    satellite_scale_radius = 50.0
    projection_range = 100.0

    host_potential_type = 'NFW'
    host_params = {
        'logM': 12.0,
        'Rs': 25.0,
        'a': 1.0, 'b': 1.0, 'c': 1.0,
        'dirx': 0.0, 'diry': 0.0, 'dirz': 1.0,
        'x_origin': 0.0, 'y_origin': 0.0, 'z_origin': 0.0,
    }

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
    halo.satellites.M_max = 1e10

    # Sample satellites

    GM_sun = 1.32712e11
    kpc_to_km = 3.0857e16
    M_host = 10.0 ** host_params['logM']
    Rs_host = host_params['Rs']
    r_vir = 200.0

    satellite_masses = []
    satellite_positions = []
    satellite_velocities = []

    total_sampled_mass = 0
    sat_scale_radius = satellite_scale_radius
    sat_r_max = background_r_max
    r_max_density = hernquist_density(sat_scale_radius, sat_scale_radius)

    while total_sampled_mass < satellite_total_mass:
        mass = halo.satellites.sample(1)[0]
        satellite_masses.append(mass)
        total_sampled_mass += mass

        while True:
            r = rng.uniform(0, sat_r_max)
            density = hernquist_density(r, sat_scale_radius)
            if rng.uniform(0, r_max_density) < density:
                break

        theta = np.arccos(rng.uniform(-1, 1))
        phi = rng.uniform(0, 2 * np.pi)

        position = np.array([
            r * np.sin(theta) * np.cos(phi),
            r * np.sin(theta) * np.sin(phi),
            r * np.cos(theta)
        ])
        satellite_positions.append(position)

        escape_velocity = v_esc(r, M_host, Rs_host)
        v_mag = rng.uniform(0, escape_velocity)

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
        'masses': satellite_masses,
        'logM': np.log10(satellite_masses),
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
    total_mass = np.sum(sat_data['masses'])
    particles_allocated = {}

    for i in range(n_satellites):
        mass_fraction = sat_data['masses'][i] / total_mass
        n_particles_target = int(np.round(mass_fraction * target_stream_particles))

        if n_particles_target < min_stream_particles:
            continue

        particles_allocated[i] = n_particles_target

        satellite_params = {
            'logM': sat_data['logM'][i],
            'Rs': 100.0
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

    # Generate background
    t_start = time.time()

    scale_radius = background_scale_radius

    total_stream_particles_generated = len(halo_particles_streams['positions'])
    n_bg_target = int(total_stream_particles_generated * (1 - fstream) / fstream)
    total_particles_target = total_stream_particles_generated + n_bg_target

    if total_particles_target < total_particles * 0.9:
        warnings.warn(f"Total particles target ({total_particles_target}) below 90% of target ({total_particles})")

    positions_bg = []
    r_max = background_r_max
    r_max_density = hernquist_density(scale_radius, scale_radius)

    for _ in range(n_bg_target):
        while True:
            r = rng.uniform(0, r_max)
            theta = np.arccos(rng.uniform(-1, 1))
            phi = rng.uniform(0, 2 * np.pi)

            density = hernquist_density(r, scale_radius)
            if rng.uniform(0, r_max_density) < density:
                x = r * np.sin(theta) * np.cos(phi)
                y = r * np.sin(theta) * np.sin(phi)
                z = r * np.cos(theta)
                positions_bg.append([x, y, z])
                break

    positions_bg = np.array(positions_bg)
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

    import os
    output_dir = 'tests/outputs/full_pipeline_example'
    os.makedirs(output_dir, exist_ok=True)

    # Figure 1: Satellite mass distribution histogram (log-spaced bins)
    fig, ax = plt.subplots(figsize=(4, 3))
    masses_list = sat_data['masses']
    log_bins = np.logspace(np.log10(masses_list.min()), np.log10(masses_list.max()), 30)
    counts, edges = np.histogram(masses_list, bins=log_bins)
    bin_centers = np.sqrt(edges[:-1] * edges[1:])
    bin_widths = edges[1:] - edges[:-1]
    density = counts / np.log10(edges[1:] / edges[:-1])

    x_step = np.concatenate([edges[:-1], edges[-1:]])
    y_step = np.concatenate([density, [density[-1]]])
    ax.step(x_step, y_step, where='post', linewidth=2, color='C0')

    ax.set_xlabel(r'Satellite Mass ($M_\odot$)')
    ax.set_ylabel(r'$dN/d\log M$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(mass_function_params['M_min'], mass_function_params['M_max'])
    info_text = f'$N_{{\\rm sat}}={n_satellites}$\n$\\alpha={mass_function_params["alpha"]}$'
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
    masses_allocated = [masses_list[i] for i in allocated_indices]
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
