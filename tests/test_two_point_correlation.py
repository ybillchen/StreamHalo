#!/usr/bin/env python3
"""Two-point angular correlation function of StreamHalo streams vs BJ05."""

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist

from streamhalo import MockHalo
from streamhalo.potentials import tidal_radius
from streamhalo.sampling import (
    make_broken_powerlaw_position_sampler,
    uniform_velocity_sampler,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GALAXIA_DATA   = '/home/ybchen/Downloads/galaxia-0.7.2/GalaxiaData'
BJ_HALO        = 'halo02'
SHORT_IDX_PATH = 'tests/outputs/bj05_comparison/bj05_short_stream_indices.npy'

OBSERVER    = np.array([8.0, 0.0, 0.0])  # kpc — solar galactocentric position
SIGMA_ANGLE = 10.0                        # deg — shuffle width for random catalogue


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def xyz_to_lb(positions, observer=OBSERVER):
    """Galactocentric Cartesian → sky (l, b) in degrees as seen from observer."""
    dx = positions - observer
    r  = np.linalg.norm(dx, axis=1)
    l  = np.rad2deg(np.arctan2(dx[:, 1], dx[:, 0])) % 360.0
    b  = np.rad2deg(np.arcsin(np.clip(dx[:, 2] / r, -1.0, 1.0)))
    return l, b


def coords_to_xyz(l, b):
    """(l, b) in degrees → unit vectors."""
    l_r, b_r = np.deg2rad(l), np.deg2rad(b)
    return np.column_stack([
        np.cos(b_r) * np.cos(l_r),
        np.cos(b_r) * np.sin(l_r),
        np.sin(b_r),
    ])


def chord_to_angle_deg(chord):
    """Euclidean chord distance between unit vectors → degrees."""
    return 2.0 * np.rad2deg(np.arcsin(np.clip(chord / 2.0, 0.0, 1.0)))


def angular_shuffle(l, b, sigma_angle_deg, n_random, rng):
    """Shuffled random catalogue: resample data and rotate each point by
    θ ~ N(0, sigma_angle) about a random great-circle axis (Rodrigues formula).
    Preserves large-scale distribution while erasing clustering at small scales.
    """
    idx  = rng.choice(len(l), n_random, replace=True)
    xyz  = coords_to_xyz(l[idx], b[idx])

    rand_vecs = rng.standard_normal((n_random, 3))
    dots      = np.einsum('ij,ij->i', rand_vecs, xyz)[:, None]
    axes      = rand_vecs - dots * xyz
    axes     /= np.linalg.norm(axes, axis=1, keepdims=True)

    theta = rng.normal(0.0, np.deg2rad(sigma_angle_deg), n_random)
    cos_t, sin_t = np.cos(theta)[:, None], np.sin(theta)[:, None]
    xyz_r = xyz * cos_t + np.cross(axes, xyz) * sin_t  # axis ⊥ n → axis·n = 0

    l_rand = np.rad2deg(np.arctan2(xyz_r[:, 1], xyz_r[:, 0])) % 360.0
    b_rand = np.rad2deg(np.arcsin(np.clip(xyz_r[:, 2], -1.0, 1.0)))
    return l_rand, b_rand


# ---------------------------------------------------------------------------
# Landy-Szalay estimator
# ---------------------------------------------------------------------------

def landy_szalay(l_data, b_data, theta_bins, n_random=50_000, rng=None,
                 sigma_angle=None):
    """Landy-Szalay (1993) estimator: w(θ) = (DD/n_DD) / (RR/n_RR) - 1.

    DD and RR are computed on subsamples of 2000 points and rescaled.
    If sigma_angle is given the random catalogue is built via angular_shuffle;
    otherwise a uniform sphere is used.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_data = len(l_data)
    if n_data < 10:
        return np.zeros(len(theta_bins) - 1), np.sqrt(theta_bins[:-1] * theta_bins[1:])

    xyz_data  = coords_to_xyz(l_data, b_data)
    n_sub_dd  = min(n_data, 2000)
    idx_d     = rng.choice(n_data, n_sub_dd, replace=False)
    scale_dd  = (n_data / n_sub_dd) ** 2 if n_data > n_sub_dd else 1.0
    DD, _     = np.histogram(chord_to_angle_deg(pdist(xyz_data[idx_d])), bins=theta_bins)
    DD        = DD * scale_dd

    if sigma_angle is not None:
        l_rand, b_rand = angular_shuffle(l_data, b_data, sigma_angle, n_random, rng)
    else:
        l_rand = rng.uniform(0.0, 360.0, n_random)
        b_rand = np.rad2deg(np.arcsin(rng.uniform(-1.0, 1.0, n_random)))
    xyz_rand  = coords_to_xyz(l_rand, b_rand)
    n_sub_rr  = min(2000, n_random)
    idx_r     = rng.choice(n_random, n_sub_rr, replace=False)
    RR, _     = np.histogram(chord_to_angle_deg(pdist(xyz_rand[idx_r])), bins=theta_bins)
    RR        = RR * (n_random / n_sub_rr) ** 2

    n_DD = n_data * (n_data - 1) / 2.0
    n_RR = n_random * (n_random - 1) / 2.0
    with np.errstate(divide='ignore', invalid='ignore'):
        w = np.nan_to_num((DD / n_DD) / (RR / n_RR) - 1.0,
                          nan=0.0, posinf=0.0, neginf=0.0)
    return w, np.sqrt(theta_bins[:-1] * theta_bins[1:])


def compute_w_with_errors(l, b, theta_bins, n_random=50_000,
                          n_bootstrap=30, rng=None, sigma_angle=None):
    """w(θ) from the full catalogue + bootstrap standard errors."""
    if rng is None:
        rng = np.random.default_rng()

    N = len(l)
    w_main, theta_c = landy_szalay(l, b, theta_bins, n_random=n_random,
                                   rng=rng, sigma_angle=sigma_angle)
    if n_bootstrap < 2:
        return w_main, np.zeros(len(theta_bins) - 1), theta_c

    w_boots = np.empty((n_bootstrap, len(theta_bins) - 1))
    for i in range(n_bootstrap):
        idx = rng.choice(N, N, replace=True)
        w_boots[i], _ = landy_szalay(l[idx], b[idx], theta_bins,
                                     n_random=n_random // 2, rng=rng,
                                     sigma_angle=sigma_angle)
    return w_main, np.std(w_boots, axis=0), theta_c


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _build_mock_halo(rng):
    """Fallback: build a MockHalo matching the test_mock_halo_generation setup."""
    _orig = {'x_origin': 0.0, 'y_origin': 0.0, 'z_origin': 0.0,
             'dirx': 0.0, 'diry': 0.0, 'dirz': 1.0}
    host_potential_type = 'Composite:MiyamotoNagai,Hernquist,Hernquist,NFW'
    host_params = [
        {'logM': float(np.log10(4.7717e10)), 'Rs': 2.6, 'Hs': 0.3, **_orig},
        {'logM': float(np.log10(5.00e9)),    'Rs': 1.0,             **_orig},
        {'logM': float(np.log10(1.8142e9)),  'Rs': 0.0688867,       **_orig},
        {'logM': float(np.log10(5.5427e11)), 'Rs': 15.626,
         'a': 1.0, 'b': 1.0, 'c': 1.0,     **_orig},
    ]
    nfw = host_params[3]
    halo = MockHalo(host_potential_type, host_params, rng=rng,
                    mass_function='powerlaw', alpha=-1.9, M_min=1e7, M_max=3e10)
    halo.sample_satellites(
        stellar_mass_target=1e9,
        position_sampler=make_broken_powerlaw_position_sampler(
            r_min=2.0, r_max=100.0, alpha_inner=-3, alpha_outer=-3),
        velocity_sampler=uniform_velocity_sampler,
    )
    halo.generate_streams(
        target_particles=5000, min_particles=100,
        integration_time=4.0, n_steps=8000,
        plummer_scale=lambda r_sat, M_sat: (
            0.1 * tidal_radius(r_sat, M_sat, nfw['logM'], nfw['Rs'])),
    )
    return halo


def _load_stream_positions():
    """Load from cache (tests/outputs/mock_halo_cache.npz) or regenerate."""
    cache_path = 'tests/outputs/mock_halo_cache.npz'
    if os.path.exists(cache_path):
        d = np.load(cache_path)
        return d['stream_positions'], d['stream_index']
    rng = np.random.default_rng(42)
    halo = _build_mock_halo(rng)
    os.makedirs('tests/outputs', exist_ok=True)
    halo.save(cache_path)
    return halo.stream_positions, halo.stream_index


def _load_bj05_positions(n_match=None, rng=None):
    """Load BJ05 positions, excluding short streams, with mass-weighted subsampling.
    Returns (positions, index) or (None, None) if data is unavailable.
    """
    try:
        import ebf
    except ImportError:
        print("  [BJ05] ebf not available — skipping.")
        return None, None

    filenames_path = f'{GALAXIA_DATA}/nbody1/filenames/{BJ_HALO}.txt'
    if not os.path.exists(filenames_path):
        print(f"  [BJ05] not found: {filenames_path}")
        return None, None

    with open(filenames_path) as f:
        lines = f.read().strip().splitlines()
    base_dir  = GALAXIA_DATA + '/' + lines[0].strip()
    sat_files = [ln.strip() for ln in lines[2:2 + int(lines[1].split()[0])]]

    if not os.path.exists(base_dir):
        print(f"  [BJ05] particle data not found: {base_dir}")
        return None, None

    all_pos, all_mass, all_idx = [], [], []
    for i, fname in enumerate(sat_files):
        d = ebf.read(base_dir + fname, '/')
        all_pos.append(np.array(d['pos3']))
        all_mass.append(np.array(d['mass']))
        all_idx.append(np.full(len(d['pos3']), i, dtype=int))

    positions = np.concatenate(all_pos,  axis=0)
    masses    = np.concatenate(all_mass)
    index     = np.concatenate(all_idx)

    if os.path.exists(SHORT_IDX_PATH):
        short = set(np.load(SHORT_IDX_PATH).tolist())
        keep  = ~np.isin(index, list(short))
        positions, masses, index = positions[keep], masses[keep], index[keep]
        print(f"  [BJ05] Excluded {len(short)} short-stream progenitors; "
              f"{len(positions):,} particles remain.")

    if n_match is not None and n_match < len(positions):
        if rng is None:
            rng = np.random.default_rng(0)
        weights = masses / masses.sum()
        idx = rng.choice(len(positions), n_match, replace=False, p=weights)
        positions, masses, index = positions[idx], masses[idx], index[idx]
        print(f"  [BJ05] Mass-weighted subsample: {len(positions):,} particles.")

    return positions, index


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def test_two_point_correlation():
    """w(θ) for StreamHalo streams vs BJ05 (halo02) and a uniform-random null.

    Assertions
    ----------
    - Null |w| < 1.0 in all bins.
    - StreamHalo w > 0 in at least one bin below 10°.
    - StreamHalo exceeds null in ≥ half the bins below 10°.
    - BJ05 (if available): w > 0 below 10° and same sign as StreamHalo in ≥ half those bins.
    """
    output_dir = 'tests/outputs/two_point_correlation'
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(0)

    theta_bins = np.logspace(np.log10(0.3), 2.0, 15)

    stream_pos, stream_idx = _load_stream_positions()
    n_particles = len(stream_pos)
    l_stream, b_stream = xyz_to_lb(stream_pos)

    print(f"\nComputing w(θ) for {n_particles:,} StreamHalo particles ...")
    w_stream, w_err, theta_c = compute_w_with_errors(
        l_stream, b_stream, theta_bins,
        n_random=50_000, n_bootstrap=30, rng=rng, sigma_angle=SIGMA_ANGLE)

    print("\nLoading BJ05 data ...")
    bj05_pos, bj05_idx = _load_bj05_positions(n_match=n_particles, rng=rng)
    has_bj05 = bj05_pos is not None
    if has_bj05:
        l_bj05, b_bj05 = xyz_to_lb(bj05_pos)
        print(f"Computing w(θ) for {len(bj05_pos):,} BJ05 particles ...")
        w_bj05, w_bj05_err, _ = compute_w_with_errors(
            l_bj05, b_bj05, theta_bins,
            n_random=50_000, n_bootstrap=30, rng=rng, sigma_angle=SIGMA_ANGLE)
    else:
        w_bj05 = w_bj05_err = None

    print("Computing w(θ) for uniform random null ...")
    l_null = rng.uniform(0.0, 360.0, n_particles)
    b_null = np.rad2deg(np.arcsin(rng.uniform(-1.0, 1.0, n_particles)))
    w_null, w_null_err, _ = compute_w_with_errors(
        l_null, b_null, theta_bins,
        n_random=50_000, n_bootstrap=20, rng=rng, sigma_angle=SIGMA_ANGLE)

    # --- print table ---
    bj_hdr = f"{'w_bj05':>10} {'±':>2} {'w_bj_err':>9}" if has_bj05 else ""
    print(f"\n{'θ_lo':>6} {'θ_hi':>6} {'θ_c':>6}  "
          f"{'w_stream':>10} {'±':>2} {'w_err':>8}  {bj_hdr}  {'w_null':>8}")
    print('-' * (60 + (26 if has_bj05 else 0)))
    for k in range(len(theta_c)):
        bj = f"{w_bj05[k]:10.4f}    {w_bj05_err[k]:9.4f}  " if has_bj05 else ""
        print(f"{theta_bins[k]:6.2f} {theta_bins[k+1]:6.2f} {theta_c[k]:6.2f}  "
              f"{w_stream[k]:10.4f}    {w_err[k]:8.4f}  {bj}{w_null[k]:8.4f}")

    # --- plot 1: w(θ) comparison ---
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.fill_between(theta_c, w_stream - w_err, w_stream + w_err, alpha=0.25, color='C0')
    ax.semilogx(theta_c, w_stream, 'o-', color='C0', linewidth=2,
                label=f'StreamHalo ({n_particles:,})')
    if has_bj05:
        ax.fill_between(theta_c, w_bj05 - w_bj05_err, w_bj05 + w_bj05_err,
                        alpha=0.25, color='C1')
        ax.semilogx(theta_c, w_bj05, 's-', color='C1', linewidth=2,
                    label=f'BJ05 {BJ_HALO} ({len(bj05_pos):,})')
    ax.fill_between(theta_c, w_null - w_null_err, w_null + w_null_err,
                    alpha=0.15, color='gray')
    ax.semilogx(theta_c, w_null, 'D--', color='gray', linewidth=1.2,
                markersize=4, label='Uniform random (null)')
    ax.axhline(0, color='k', linewidth=0.8, linestyle=':')
    ax.set_xlabel(r'Angular separation $\theta$ (deg)')
    ax.set_ylabel(r'$w(\theta)$')
    ax.set_title('Two-Point Angular Correlation Function')
    ax.legend(fontsize=9)
    ax.set_xlim(theta_bins[0], theta_bins[-1])
    plt.tight_layout()
    plt.savefig(f'{output_dir}/w_theta_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # --- plot 2: sky projection ---
    if has_bj05:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].scatter(l_stream, b_stream, c=stream_idx, cmap='tab20',
                        s=1, alpha=0.4, rasterized=True)
        axes[0].set_title(f'StreamHalo ({n_particles:,})')
        axes[1].scatter(l_bj05, b_bj05, c=bj05_idx % 20, cmap='tab20',
                        s=1, alpha=0.4, rasterized=True)
        axes[1].set_title(f'BJ05 {BJ_HALO} ({len(bj05_pos):,})')
        for ax in axes:
            ax.set_xlabel('$l$ (deg)')
            ax.set_ylabel('$b$ (deg)')
            ax.set_xlim(0, 360)
            ax.set_ylim(-90, 90)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/sky_projection_comparison.png',
                    dpi=150, bbox_inches='tight')
    else:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.scatter(l_stream, b_stream, c=stream_idx, cmap='tab20',
                   s=1, alpha=0.5, rasterized=True)
        ax.set_xlabel('$l$ (deg)')
        ax.set_ylabel('$b$ (deg)')
        ax.set_title('Sky projection')
        ax.set_xlim(0, 360)
        ax.set_ylim(-90, 90)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/sky_projection.png', dpi=150, bbox_inches='tight')
    plt.close()

    # --- plot 3: per-satellite ---
    large_sats = [s for s in np.unique(stream_idx) if (stream_idx == s).sum() > 200]
    if large_sats:
        cmap20 = plt.get_cmap('tab20')
        fig, ax = plt.subplots(figsize=(5, 4))
        for s in large_sats:
            mask = stream_idx == s
            ws, _, _ = compute_w_with_errors(
                *xyz_to_lb(stream_pos[mask]), theta_bins,
                n_random=20_000, n_bootstrap=0, rng=rng, sigma_angle=SIGMA_ANGLE)
            ax.semilogx(theta_c, ws, color=cmap20(s % 20), alpha=0.6, linewidth=1)
        ax.semilogx(theta_c, w_stream, 'k-', linewidth=2, label='All StreamHalo')
        if has_bj05:
            ax.semilogx(theta_c, w_bj05, '--', color='C1', linewidth=2,
                        label=f'All BJ05 ({BJ_HALO})')
        ax.axhline(0, color='k', linewidth=0.8, linestyle=':')
        ax.set_xlabel(r'$\theta$ (deg)')
        ax.set_ylabel(r'$w(\theta)$')
        ax.set_title(f'Per-satellite  (N={len(large_sats)})')
        ax.legend(fontsize=9)
        ax.set_xlim(theta_bins[0], theta_bins[-1])
        plt.tight_layout()
        plt.savefig(f'{output_dir}/w_theta_per_satellite.png',
                    dpi=150, bbox_inches='tight')
        plt.close()

    # --- assertions ---
    small = theta_c < 10.0
    assert np.all(np.abs(w_null) < 1.0), f"Null |w| >= 1.0: {w_null}"
    w_small = w_stream[small]
    assert np.any(w_small > 0), "StreamHalo w not positive below 10°."
    assert np.sum(w_small > w_null[small]) >= len(w_small) // 2, \
        "StreamHalo not more clustered than null in most bins below 10°."
    if has_bj05:
        w_bj05_small = w_bj05[small]
        assert np.any(w_bj05_small > 0), "BJ05 w not positive below 10°."
        assert np.sum(np.sign(w_small) == np.sign(w_bj05_small)) >= len(w_small) // 2, \
            "StreamHalo and BJ05 disagree in sign in most bins below 10°."

    print("\nAll assertions passed.")
    print(f"Outputs written to {output_dir}/")


if __name__ == '__main__':
    test_two_point_correlation()
