#!/usr/bin/env python3
"""Mock halo generation test."""

import dataclasses
import json
import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

from streamhalo import MockHalo
from streamhalo.potentials import tidal_radius
from streamhalo.sampling import (
    make_broken_powerlaw_position_sampler,
    uniform_velocity_sampler,
)


# ---------------------------------------------------------------------------
# Default host-potential component parameters (Milky Way-like, 4-component)
# ---------------------------------------------------------------------------
_ORIGIN = {'x_origin': 0.0, 'y_origin': 0.0, 'z_origin': 0.0,
           'dirx': 0.0, 'diry': 0.0, 'dirz': 1.0}

_DEFAULT_HOST_PARAMS: List[Dict[str, Any]] = [
    {'logM': float(np.log10(4.7717e10)), 'Rs': 2.6,       'Hs': 0.3,   **_ORIGIN},  # disk
    {'logM': float(np.log10(5.00e9)),    'Rs': 1.0,                     **_ORIGIN},  # bulge
    {'logM': float(np.log10(1.8142e9)),  'Rs': 0.0688867,               **_ORIGIN},  # nucleus
    {'logM': float(np.log10(5.5427e11)), 'Rs': 15.626,
     'a': 1.0, 'b': 1.0, 'c': 1.0,                                      **_ORIGIN},  # NFW halo
]


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalar types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------
@dataclass
class MockHaloConfig:
    """All parameters for the mock halo generation test.

    Can be constructed directly or loaded from a JSON file::

        cfg = MockHaloConfig()                       # all defaults
        cfg = MockHaloConfig.from_json('my.json')    # override from file
        cfg.to_json('saved.json')                    # serialise to file
    """

    # --- particle budget ---
    f_stream: float = 0.5
    """Fraction of total particles assigned to tidal streams."""
    total_particles: int = 100_000
    """Target total particle count (streams + background)."""
    min_stream_particles: int = 100
    """Minimum particle allocation for a satellite to be included."""

    # --- satellite population ---
    satellite_stellar_mass_target: float = 1e9
    """Cumulative stellar-mass target (M☉) used to stop satellite sampling."""
    mass_function: str = 'powerlaw'
    """Halo mass function type: 'powerlaw', 'truncated_powerlaw', or 'schechter'."""
    alpha_mf: float = -1.9
    """Power-law slope of the halo mass function (dN/dM ∝ M^alpha)."""
    M_min: float = 1e7
    """Minimum satellite halo mass (M☉)."""
    M_max: float = 3e10
    """Maximum satellite halo mass (M☉)."""

    # --- stream integration ---
    integration_time: float = 4.0
    """Stream orbital integration time (Gyr)."""
    dt: float = 0.5
    """Integration time step (Myr)."""

    # --- spatial distribution ---
    r_min: float = 2.0
    """Inner radial bound for satellite and stream position sampling (kpc)."""
    r_max_sat: float = 100.0
    """Outer radial bound for satellite position sampling (kpc)."""
    r_max_bg: float = 200.0
    """Outer radial bound for background particle sampling (kpc)."""
    alpha_inner: float = -3.0
    """Broken power-law inner slope for radial number density n(r) ∝ r^alpha."""
    alpha_outer: float = -3.0
    """Broken power-law outer slope."""

    # --- plummer scale ---
    plummer_tidal_fraction: float = 0.1
    """Satellite Plummer scale radius as a fraction of the King tidal radius."""
    nfw_component_index: int = 3
    """Index into host_params of the NFW component used for tidal radius calculation."""

    # --- visualization ---
    projection_range: float = 200.0
    """Half-width of the XY projection plot (kpc)."""

    # --- host potential ---
    host_potential_type: str = 'Composite:MiyamotoNagai,Hernquist,Hernquist,NFW'
    """Potential type string passed to MockHalo (plain or 'Composite:A,B,...')."""
    host_params: List[Dict[str, Any]] = field(
        default_factory=lambda: _DEFAULT_HOST_PARAMS
    )
    """List of parameter dicts for each potential component."""

    # --- derived quantities (computed, never stored in JSON) ---
    @property
    def target_stream_particles(self) -> int:
        """Target number of stream particles (total_particles * f_stream)."""
        return int(self.total_particles * self.f_stream)

    @property
    def n_steps(self) -> int:
        """Number of integration steps (integration_time [Gyr] / dt [Myr])."""
        return int(round(self.integration_time * 1000.0 / self.dt))

    # --- I/O ---
    @classmethod
    def from_json(cls, path: str) -> 'MockHaloConfig':
        """Load config from a JSON file, using defaults for any missing keys.

        Unknown keys in the JSON are silently ignored so that partial override
        files (containing only the parameters you want to change) work cleanly.
        """
        with open(path) as f:
            data = json.load(f)
        known = {fld.name for fld in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_json(self, path: str) -> None:
        """Serialise the full config to a JSON file."""
        with open(path, 'w') as f:
            json.dump(dataclasses.asdict(self), f, indent=2, cls=_NumpyEncoder)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def dndlogm(data, bins):
    """Return (bin_centers, dN/d log M) for log-spaced bins."""
    counts, edges = np.histogram(data, bins=bins)
    return np.sqrt(edges[:-1] * edges[1:]), counts / np.log10(edges[1:] / edges[:-1])


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
def test_mock_halo_generation(cfg: MockHaloConfig = None):
    """End-to-end test: sample satellites, generate streams + background, plot, cache."""

    cfg = cfg or MockHaloConfig()

    rng = np.random.default_rng(42)

    # NFW component used for tidal-radius calculation
    nfw = cfg.host_params[cfg.nfw_component_index]

    halo = MockHalo(
        cfg.host_potential_type, cfg.host_params,
        rng=rng,
        mass_function=cfg.mass_function,
        alpha=cfg.alpha_mf,
        M_min=cfg.M_min,
        M_max=cfg.M_max,
    )

    halo.sample_satellites(
        stellar_mass_target=cfg.satellite_stellar_mass_target,
        position_sampler=make_broken_powerlaw_position_sampler(
            r_min=cfg.r_min, r_max=cfg.r_max_sat,
            alpha_inner=cfg.alpha_inner, alpha_outer=cfg.alpha_outer,
        ),
        velocity_sampler=uniform_velocity_sampler,
    )

    halo.generate_streams(
        target_particles=cfg.target_stream_particles,
        min_particles=cfg.min_stream_particles,
        integration_time=cfg.integration_time,
        n_steps=cfg.n_steps,
        plummer_scale=lambda r_sat, M_sat: (
            cfg.plummer_tidal_fraction
            * tidal_radius(r_sat, M_sat, nfw['logM'], nfw['Rs'])
        ),
    )

    os.makedirs('tests/outputs', exist_ok=True)
    halo.save('tests/outputs/mock_halo_cache.npz')

    n_stream = len(halo.stream_positions)
    n_bg = int(n_stream * (1 - cfg.f_stream) / cfg.f_stream)
    if n_stream + n_bg < cfg.total_particles * 0.9:
        warnings.warn(
            f"Total particles ({n_stream + n_bg}) below 90% of target ({cfg.total_particles})"
        )

    halo.generate_background(
        n_particles=n_bg,
        position_sampler=make_broken_powerlaw_position_sampler(
            r_min=cfg.r_min, r_max=cfg.r_max_bg,
            alpha_inner=cfg.alpha_inner, alpha_outer=cfg.alpha_outer,
        ),
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

    # --- plot 1: mass distribution ---
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

    # --- plot 2: XY projection ---
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(combined_pos[~stream_mask, 0], combined_pos[~stream_mask, 1],
               c='gray', s=1, alpha=0.3, label=f'Background ({n_bg_actual:,})')
    ax.scatter(combined_pos[stream_mask, 0], combined_pos[stream_mask, 1],
               c=combined_idx[stream_mask], cmap='tab20', s=1, alpha=0.6,
               label=f'Streams ({n_stream_actual:,})')
    ax.set_xlabel('X (kpc)')
    ax.set_ylabel('Y (kpc)')
    ax.set_xlim(-cfg.projection_range, cfg.projection_range)
    ax.set_ylim(-cfg.projection_range, cfg.projection_range)
    ax.set_aspect('equal')
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/mock_halo_xy_projection.png', dpi=150, bbox_inches='tight')
    plt.close()

    # --- plot 3: radial distribution ---
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

    # --- assertions ---
    allocated = sorted(halo.particles_per_satellite)
    correlation = np.corrcoef(
        np.log10([stellar_masses[i] for i in allocated]),
        [halo.particles_per_satellite[i] for i in allocated],
    )[0, 1]

    assert n_stream_actual > 0
    assert n_bg_actual > 0
    assert abs(fraction_actual - cfg.f_stream) < 0.1
    assert correlation > 0.4


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run mock halo generation test.')
    parser.add_argument(
        '--config', metavar='PATH',
        help='JSON config file to load parameters from (missing keys fall back to defaults).',
    )
    parser.add_argument(
        '--save-config', metavar='PATH',
        help='Write the effective config (after loading --config) to a JSON file and exit.',
    )
    args = parser.parse_args()

    cfg = MockHaloConfig.from_json(args.config) if args.config else MockHaloConfig()

    if args.save_config:
        cfg.to_json(args.save_config)
        print(f'Config saved to {args.save_config}')
    else:
        test_mock_halo_generation(cfg)
