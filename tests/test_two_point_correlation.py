#!/usr/bin/env python3
"""Two-point angular correlation function of StreamHalo streams vs BJ05."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist

from tests.plot_style import apply_style, style_ax, legend as _legend
apply_style()

from streamhalo import MockHalo
from streamhalo.potentials import tidal_radius
from streamhalo.sampling import (
    make_broken_powerlaw_position_sampler,
    uniform_velocity_sampler,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_GALAXIA_CANDIDATES = [
    '/home/ybchen/Downloads/galaxia-0.7.2/GalaxiaData',   # Linux
    '/Users/ybchen/Downloads/galaxia-0.7.2/GalaxiaData',  # macOS
]
GALAXIA_DATA = next((p for p in _GALAXIA_CANDIDATES if os.path.isdir(p)),
                    _GALAXIA_CANDIDATES[0])
BJ_HALO      = 'halo02'
R_HALF_MAX   = 10.0  # kpc — exclude progenitors whose stream half-mass radius is below this
R_GC_MIN     = 10.0  # kpc — exclude progenitors whose CM is closer than this to the GC

OBSERVER    = np.array([8.0, 0.0, 0.0])  # kpc — solar galactocentric position
SIGMA_ANGLE = 10.0                        # deg — shuffle width for random catalogue

# Shared estimator parameters — imported by sweep_subsample_size.py for consistency
N_SUB      = 3000   # subsample size per draw
N_RESAMPLE = 6      # number of independent draws
N_RANDOM   = 10_000 # random catalogue size for RR


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

def compute_rr(l, b, theta_bins, n_random=10_000, rng=None, sigma_angle=None):
    """Compute the RR histogram and normalisation from a random catalogue.

    Returns (RR_hist, n_RR) suitable for passing to landy_szalay.
    Separating this from landy_szalay allows reuse across many DD evaluations.
    """
    if rng is None:
        rng = np.random.default_rng()
    if sigma_angle is not None:
        l_rand, b_rand = angular_shuffle(l, b, sigma_angle, n_random, rng)
    else:
        l_rand = rng.uniform(0.0, 360.0, n_random)
        b_rand = np.rad2deg(np.arcsin(rng.uniform(-1.0, 1.0, n_random)))
    xyz_rand = coords_to_xyz(l_rand, b_rand)
    RR, _    = np.histogram(chord_to_angle_deg(pdist(xyz_rand)), bins=theta_bins)
    return RR, n_random * (n_random - 1) / 2.0


def landy_szalay(l_data, b_data, theta_bins, n_random=10_000, rng=None,
                 sigma_angle=None, rr_hist=None, n_rr=None):
    """Landy-Szalay (1993) estimator: w(θ) = (DD/n_DD) / (RR/n_RR) - 1.

    Expects pre-subsampled data. If rr_hist and n_rr are provided the RR step
    is skipped (use compute_rr to precompute and share across many calls).
    """
    if rng is None:
        rng = np.random.default_rng()

    n_data = len(l_data)
    if n_data < 10:
        return np.zeros(len(theta_bins) - 1), np.sqrt(theta_bins[:-1] * theta_bins[1:])

    xyz_data = coords_to_xyz(l_data, b_data)
    DD, _    = np.histogram(chord_to_angle_deg(pdist(xyz_data)), bins=theta_bins)

    if rr_hist is None or n_rr is None:
        rr_hist, n_rr = compute_rr(l_data, b_data, theta_bins, n_random, rng, sigma_angle)

    n_DD = n_data * (n_data - 1) / 2.0
    with np.errstate(divide='ignore', invalid='ignore'):
        w = np.nan_to_num((DD / n_DD) / (rr_hist / n_rr) - 1.0,
                          nan=0.0, posinf=0.0, neginf=0.0)
    return w, np.sqrt(theta_bins[:-1] * theta_bins[1:])


def compute_w_with_errors(l, b, theta_bins, n_random=10_000,
                          n_resample=30, n_sub=3000, rng=None, sigma_angle=None):
    """w(θ) via repeated subsampling: median across draws = main value, std = error.

    Each of n_resample draws takes n_sub points without replacement from the
    full catalogue and computes w(θ) independently.
    """
    if rng is None:
        rng = np.random.default_rng()

    N = len(l)
    n_sub   = min(n_sub, N)
    theta_c = np.sqrt(theta_bins[:-1] * theta_bins[1:])

    if n_resample < 2:
        w, _ = landy_szalay(l, b, theta_bins, n_random=n_random,
                            rng=rng, sigma_angle=sigma_angle)
        return w, np.zeros(len(theta_bins) - 1), theta_c

    rr_hist, n_rr = compute_rr(l, b, theta_bins, n_random, rng, sigma_angle)

    w_subs = np.empty((n_resample, len(theta_bins) - 1))
    for i in range(n_resample):
        idx = rng.choice(N, n_sub, replace=False)
        w_subs[i], _ = landy_szalay(l[idx], b[idx], theta_bins,
                                    rr_hist=rr_hist, n_rr=n_rr, rng=rng)
    return np.median(w_subs, axis=0), np.std(w_subs, axis=0), theta_c


# ---------------------------------------------------------------------------
# GP-based estimator
# ---------------------------------------------------------------------------

def compute_w_gp(l, b, theta_eval, n_random=10_000, n_resample=30, n_sub=3000,
                 n_bins_fine=100, rng=None, sigma_angle=None):
    """GP-based w(θ): fit GP to pair density per subsample, combine via law of total variance.

    For each of n_resample draws of n_sub points:
      1. Bin all pairwise angular separations into n_bins_fine log-spaced bins
      2. Fit a GP to the normalised pair density
      3. w_i(θ) = μ_DD,i(θ) / μ_RR(θ) − 1  with  σ²_w,i = (σ_DD,i / μ_RR)²  (delta method)

    Combines draws:
      w_mean = mean_i(w_i)
      w_std  = sqrt( mean_i(σ²_w,i) + var_i(w_i) )   [law of total variance]

    Returns (w_mean, w_std, theta_eval).
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel

    if rng is None:
        rng = np.random.default_rng()
    N = len(l)

    # Fine bins shared by all draws
    theta_fine  = np.logspace(np.log10(theta_eval.min()), np.log10(theta_eval.max()),
                               n_bins_fine + 1)
    theta_c_fine = np.sqrt(theta_fine[:-1] * theta_fine[1:])
    X_fit  = np.log10(theta_c_fine).reshape(-1, 1)
    X_eval = np.log10(theta_eval).reshape(-1, 1)

    # --- Build RR once and fit GP ---
    if sigma_angle is not None:
        l_rand, b_rand = angular_shuffle(l, b, sigma_angle, n_random, rng)
    else:
        l_rand = rng.uniform(0.0, 360.0, n_random)
        b_rand = np.rad2deg(np.arcsin(rng.uniform(-1.0, 1.0, n_random)))
    RR_hist, _ = np.histogram(chord_to_angle_deg(pdist(coords_to_xyz(l_rand, b_rand))),
                               bins=theta_fine)
    n_rr    = n_random * (n_random - 1) / 2.0
    rr_norm = RR_hist / n_rr
    noise_rr = np.maximum(RR_hist, 1) / n_rr ** 2

    gp_rr = GaussianProcessRegressor(
        kernel=ConstantKernel(1.0) * Matern(length_scale=1.0, length_scale_bounds=(0.3, 10.0), nu=1.5),
        alpha=noise_rr, normalize_y=True, n_restarts_optimizer=3)
    gp_rr.fit(X_fit, rr_norm)
    mu_rr, _ = gp_rr.predict(X_eval, return_std=True)
    mu_rr = np.maximum(mu_rr, 1e-12)

    # Optimise DD kernel hyperparameters once on the first draw; reuse thereafter
    fitted_kernel = None
    w_means = np.empty((n_resample, len(theta_eval)))
    w_vars  = np.empty((n_resample, len(theta_eval)))

    for i in range(n_resample):
        idx = rng.choice(N, n_sub, replace=False)
        DD_hist, _ = np.histogram(
            chord_to_angle_deg(pdist(coords_to_xyz(l[idx], b[idx]))), bins=theta_fine)
        n_dd    = n_sub * (n_sub - 1) / 2.0
        dd_norm  = DD_hist / n_dd
        noise_dd = np.maximum(DD_hist, 1) / n_dd ** 2

        kernel = (fitted_kernel if fitted_kernel is not None
                  else ConstantKernel(1.0) * Matern(length_scale=1.0, length_scale_bounds=(0.3, 10.0), nu=1.5))
        gp_dd = GaussianProcessRegressor(
            kernel=kernel, alpha=noise_dd, normalize_y=True,
            n_restarts_optimizer=(3 if i == 0 else 0))
        gp_dd.fit(X_fit, dd_norm)
        if i == 0:
            fitted_kernel = gp_dd.kernel_

        mu_dd, sigma_dd = gp_dd.predict(X_eval, return_std=True)
        w_means[i] = mu_dd / mu_rr - 1.0
        w_vars[i]  = (sigma_dd / mu_rr) ** 2

    w_mean = w_means.mean(axis=0)
    w_std  = np.sqrt(w_vars.mean(axis=0) + w_means.var(axis=0))
    return w_mean, w_std, theta_eval


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _short_stream_indices(positions, index,
                          r_half_max=R_HALF_MAX, r_gc_min=R_GC_MIN):
    """Return progenitor indices whose streams are spatially compact and not too close-in."""
    short = []
    for prog in np.unique(index):
        mask = index == prog
        r_cm = positions[mask].mean(axis=0)
        r_half = np.median(np.linalg.norm(positions[mask] - r_cm, axis=1))
        if r_half < r_half_max and np.linalg.norm(r_cm) > r_gc_min:
            short.append(prog)
    return np.array(short, dtype=int)


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
    """Load stream positions from cache (or regenerate), then append background.

    Background is drawn from the same broken-power-law sampler as test_mock_halo
    (r_min=2, r_max=200 kpc) with n_bg = n_stream * (1-f_stream)/f_stream.
    Returns (positions, index) where index >= 0 for stream particles, -1 for background.
    """
    f_stream = 0.8
    cache_path = 'tests/outputs/mock_halo_cache.npz'
    rng_bg = np.random.default_rng(0)

    if os.path.exists(cache_path):
        d = np.load(cache_path)
        stream_pos = d['stream_positions']
        stream_idx = d['stream_index']
    else:
        rng = np.random.default_rng(42)
        halo = _build_mock_halo(rng)
        os.makedirs('tests/outputs', exist_ok=True)
        halo.save(cache_path)
        stream_pos = halo.stream_positions
        stream_idx = halo.stream_index

    n_bg = int(len(stream_pos) * (1 - f_stream) / f_stream)
    bg_sampler = make_broken_powerlaw_position_sampler(r_min=2.0, r_max=200.0,
                                                       alpha_inner=-3, alpha_outer=-3)
    bg_pos = np.array([bg_sampler(rng_bg) for _ in range(n_bg)])
    bg_idx = np.full(n_bg, -1, dtype=int)

    positions = np.vstack([stream_pos, bg_pos])
    index     = np.concatenate([stream_idx, bg_idx])
    return positions, index


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

    short = _short_stream_indices(positions, index)
    if len(short):
        keep = ~np.isin(index, short)
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

    theta_bins = np.logspace(np.log10(0.1), 2.0, 21)

    stream_pos, stream_idx = _load_stream_positions()
    n_particles = len(stream_pos)
    l_stream, b_stream = xyz_to_lb(stream_pos)

    print(f"\nComputing w(θ) for {n_particles:,} StreamHalo particles ...")
    w_stream, w_err, theta_c = compute_w_with_errors(
        l_stream, b_stream, theta_bins,
        n_random=N_RANDOM, n_resample=N_RESAMPLE, rng=rng, sigma_angle=SIGMA_ANGLE)

    print("\nLoading BJ05 data ...")
    bj05_pos, bj05_idx = _load_bj05_positions(n_match=n_particles, rng=rng)
    has_bj05 = bj05_pos is not None
    if has_bj05:
        l_bj05, b_bj05 = xyz_to_lb(bj05_pos)
        print(f"Computing w(θ) for {len(bj05_pos):,} BJ05 particles ...")
        w_bj05, w_bj05_err, _ = compute_w_with_errors(
            l_bj05, b_bj05, theta_bins,
            n_random=N_RANDOM, n_resample=N_RESAMPLE, rng=rng, sigma_angle=SIGMA_ANGLE)
    else:
        w_bj05 = w_bj05_err = None

    print("Computing w(θ) for uniform random null ...")
    l_null = rng.uniform(0.0, 360.0, n_particles)
    b_null = np.rad2deg(np.arcsin(rng.uniform(-1.0, 1.0, n_particles)))
    w_null, w_null_err, _ = compute_w_with_errors(
        l_null, b_null, theta_bins,
        n_random=N_RANDOM, n_resample=N_RESAMPLE, rng=rng, sigma_angle=SIGMA_ANGLE)

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
    ax.semilogx(theta_c, w_stream, 'o-', color='C0', linewidth=2, label='StreamHalo')
    if has_bj05:
        ax.fill_between(theta_c, w_bj05 - w_bj05_err, w_bj05 + w_bj05_err,
                        alpha=0.25, color='C1')
        ax.semilogx(theta_c, w_bj05, 's-', color='C1', linewidth=2,
                    label=f'BJ05 {BJ_HALO}')
    ax.fill_between(theta_c, w_null - w_null_err, w_null + w_null_err,
                    alpha=0.15, color='gray')
    ax.semilogx(theta_c, w_null, 'D--', color='gray', linewidth=1.2,
                markersize=4, label='Null')
    ax.axhline(0, color='k', linewidth=0.8, linestyle=':')
    style_ax(ax, xlabel=r'Angular separation $\theta$ (deg)', ylabel=r'$w(\theta)$')
    _legend(ax, loc='lower left')
    ax.text(0.03, 0.97, f'$n_{{\\rm sub}}={N_SUB}$',
            transform=ax.transAxes, fontsize=10, va='top', ha='left')
    ax.set_xlim(0.1, 100)
    ax.set_ylim(-0.5, 0.5)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/w_theta_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # --- plot 2: sky projection ---
    stream_mask = stream_idx >= 0

    def _sky_scatter(ax, l, b, idx, mask):
        ax.scatter(l[~mask], b[~mask], c='gray', s=1, alpha=0.2,
                   rasterized=True, label='Background')
        ax.scatter(l[mask], b[mask], c=idx[mask] % 20, cmap='tab20',
                   s=1, alpha=0.5, rasterized=True, label='Streams')

    if has_bj05:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        _sky_scatter(axes[0], l_stream, b_stream, stream_idx, stream_mask)
        axes[0].set_title(f'StreamHalo ({n_particles:,})', fontsize=12)
        axes[1].scatter(l_bj05, b_bj05, c=bj05_idx % 20, cmap='tab20',
                        s=1, alpha=0.4, rasterized=True)
        axes[1].set_title(f'BJ05 {BJ_HALO} ({len(bj05_pos):,})', fontsize=12)
        for ax in axes:
            style_ax(ax, xlabel='$l$ (deg)', ylabel='$b$ (deg)')
            ax.set_xlim(0, 360)
            ax.set_ylim(-90, 90)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/sky_projection_comparison.png',
                    dpi=150, bbox_inches='tight')
    else:
        fig, ax = plt.subplots(figsize=(8, 4))
        _sky_scatter(ax, l_stream, b_stream, stream_idx, stream_mask)
        style_ax(ax, xlabel='$l$ (deg)', ylabel='$b$ (deg)')
        ax.set_xlim(0, 360)
        ax.set_ylim(-90, 90)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/sky_projection.png', dpi=150, bbox_inches='tight')
    plt.close()

    # --- plot 3: per-satellite ---
    large_sats = [s for s in np.unique(stream_idx) if s >= 0 and (stream_idx == s).sum() > 200]
    if large_sats:
        cmap20 = plt.get_cmap('tab20')
        fig, ax = plt.subplots(figsize=(5, 4))
        for s in large_sats:
            mask = stream_idx == s
            ws, _ = landy_szalay(
                *xyz_to_lb(stream_pos[mask]), theta_bins,
                n_random=20_000, rng=rng, sigma_angle=SIGMA_ANGLE)
            ax.semilogx(theta_c, ws, color=cmap20(s % 20), alpha=0.6, linewidth=1)
        ax.semilogx(theta_c, w_stream, 'k-', linewidth=2, label='All StreamHalo')
        if has_bj05:
            ax.semilogx(theta_c, w_bj05, '--', color='C1', linewidth=2,
                        label=f'All BJ05 ({BJ_HALO})')
        ax.axhline(0, color='k', linewidth=0.8, linestyle=':')
        style_ax(ax, xlabel=r'$\theta$ (deg)', ylabel=r'$w(\theta)$')
        ax.set_title(f'Per-satellite  (N={len(large_sats)})', fontsize=12)
        _legend(ax)
        ax.set_xlim(0.1, 100)
        ax.set_ylim(-0.5, 0.5)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/w_theta_per_satellite.png',
                    dpi=150, bbox_inches='tight')
        plt.close()

    # --- assertions ---
    small = theta_c < 10.0
    assert np.all(np.abs(w_null) < 1.0), f"Null |w| >= 1.0: {w_null}"
    w_small = w_stream[small]
    assert np.any(w_small > 0), "StreamHalo w not positive below 10°."
    assert np.sum(w_small > w_null[small]) >= len(w_small) // 4, \
        "StreamHalo not more clustered than null in any bins below 10°."
    if has_bj05:
        w_bj05_small = w_bj05[small]
        assert np.any(w_bj05_small > 0), "BJ05 w not positive below 10°."
        assert np.sum(np.sign(w_small) == np.sign(w_bj05_small)) >= len(w_small) // 2, \
            "StreamHalo and BJ05 disagree in sign in most bins below 10°."

    print("\nAll assertions passed.")
    print(f"Outputs written to {output_dir}/")


if __name__ == '__main__':
    test_two_point_correlation()
