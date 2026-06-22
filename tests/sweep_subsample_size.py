#!/usr/bin/env python3
"""Sweep DD/RR subsample size for the 2PCF and identify the minimum size
that clearly separates the stream signal from the uniform-random null.

Run as pytest:
    pytest tests/sweep_subsample_size.py

Run directly:
    python tests/sweep_subsample_size.py
"""

import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from tests.plot_style import apply_style, style_ax, legend as _legend
apply_style()

from tests.test_two_point_correlation import (
    _load_stream_positions,
    compute_w_with_errors,
    xyz_to_lb,
    SIGMA_ANGLE,
    N_SUB      as N_SUB_REF,
    N_RESAMPLE as N_RESAMPLE_REF,
    N_RANDOM,
)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
SUBSAMPLE_SIZES = [1000, 3000, 10000]
THETA_BINS      = np.logspace(np.log10(0.1), 2.0, 14)
OUTPUT_DIR      = 'tests/outputs/subsample_sweep'


# ---------------------------------------------------------------------------
# Signal-detection metric
# ---------------------------------------------------------------------------

def detection_metric(w_stream, w_err_stream, w_null, w_err_null, theta_c,
                     theta_max=10.0):
    """Mean (w_stream - w_null) / combined_err for bins below theta_max.

    Returns (mean_snr, n_positive_bins, n_bins_total).
    """
    small = theta_c < theta_max
    delta = w_stream[small] - w_null[small]
    sigma = np.sqrt(w_err_stream[small] ** 2 + w_err_null[small] ** 2 + 1e-12)
    snr   = delta / sigma
    return snr.mean(), int((delta > 0).sum()), int(small.sum())


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def test_subsample_sweep():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng = np.random.default_rng(0)

    print("Loading StreamHalo positions ...")
    stream_pos, stream_idx = _load_stream_positions()
    n_particles = len(stream_pos)
    l_data, b_data = xyz_to_lb(stream_pos)
    print(f"  {n_particles:,} particles loaded.\n")

    print("Building uniform-random null catalogue ...")
    l_null = rng.uniform(0.0, 360.0, n_particles)
    b_null = np.rad2deg(np.arcsin(rng.uniform(-1.0, 1.0, n_particles)))

    theta_c = np.sqrt(THETA_BINS[:-1] * THETA_BINS[1:])

    results = {}

    print(f"{'n_sub':>8}  {'n_res':>6}  {'time(s)':>8}  {'SNR(<10°)':>10}  {'pos/tot':>8}")
    print('-' * 52)

    for n_sub in SUBSAMPLE_SIZES:
        n_resample = max(N_RESAMPLE_REF, int(round(N_RESAMPLE_REF * N_SUB_REF / n_sub)))
        rng_run = np.random.default_rng(42)
        t0 = time.perf_counter()

        w_s, err_s, _ = compute_w_with_errors(
            l_data, b_data, THETA_BINS,
            n_random=N_RANDOM, n_resample=n_resample, n_sub=n_sub,
            rng=rng_run, sigma_angle=SIGMA_ANGLE)
        w_n, err_n, _ = compute_w_with_errors(
            l_null, b_null, THETA_BINS,
            n_random=N_RANDOM, n_resample=n_resample, n_sub=n_sub,
            rng=rng_run, sigma_angle=SIGMA_ANGLE)

        elapsed = time.perf_counter() - t0
        mean_snr, n_pos, n_tot = detection_metric(w_s, err_s, w_n, err_n, theta_c)
        results[n_sub] = (w_s, err_s, w_n, err_n, elapsed, mean_snr, n_pos, n_tot, n_resample)

        print(f"{n_sub:>8}  {n_resample:>6}  {elapsed:>8.1f}  {mean_snr:>10.3f}  "
              f"{n_pos:>3}/{n_tot:<4}")

    # --- summary ---
    print('\n=== Summary ===')
    print(f"{'n_sub':>8}  {'n_res':>6}  {'mean_SNR':>10}  {'pos/total':>12}  {'time(s)':>8}")
    for n_sub in SUBSAMPLE_SIZES:
        _, _, _, _, elapsed, mean_snr, n_pos, n_tot, n_resample = results[n_sub]
        flag = ' <-- threshold' if mean_snr >= 1.0 and n_pos >= n_tot // 2 else ''
        print(f"{n_sub:>8}  {n_resample:>6}  {mean_snr:>10.3f}  {n_pos:>5}/{n_tot:<6}  "
              f"{elapsed:>8.1f}{flag}")

    # --- plot ---
    ncols = len(SUBSAMPLE_SIZES)
    fig, axes = plt.subplots(1, ncols, figsize=(4.5 * ncols, 3.5), sharex=True, sharey=True)

    for j, n_sub in enumerate(SUBSAMPLE_SIZES):
        w_s, err_s, w_n, err_n, elapsed, mean_snr, n_pos, n_tot, n_resample = results[n_sub]
        ax = axes[j]

        ax.fill_between(theta_c, w_s - err_s, w_s + err_s, alpha=0.25, color='C0')
        ax.semilogx(theta_c, w_s, 'o-', color='C0', linewidth=1.5,
                    markersize=3, label='StreamHalo')
        ax.fill_between(theta_c, w_n - err_n, w_n + err_n, alpha=0.20, color='gray')
        ax.semilogx(theta_c, w_n, 's--', color='gray', linewidth=1.2,
                    markersize=3, label='Null')
        ax.axhline(0, color='k', linewidth=0.7, linestyle=':')
        ax.axvline(10.0, color='k', linewidth=0.7, linestyle='--', alpha=0.4)
        ax.set_xlim(0.1, 100)
        ax.set_ylim(-0.5, 0.5)
        _legend(ax, loc='lower left')
        ax.text(0.03, 0.97, f'$n_{{\\rm sub}}={n_sub:,}$',
                transform=ax.transAxes, fontsize=10, va='top', ha='left')
        style_ax(ax, xlabel=r'$\theta$ (deg)')
        ax.tick_params(axis='both', which='both', labelsize=12)
        if j == 0:
            ax.set_ylabel(r'$w(\theta)$', fontsize=14)

    plt.tight_layout()
    out = f'{OUTPUT_DIR}/w_theta_subsample_sweep.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nPlot saved to {out}')

    # --- SNR plot ---
    fig, ax = plt.subplots(figsize=(5, 3))
    snrs = [results[n][5] for n in SUBSAMPLE_SIZES]
    ax.semilogx(SUBSAMPLE_SIZES, snrs, 'o-', linewidth=2)
    style_ax(ax, xlabel='Subsample size', ylabel=r'Mean SNR below $10°$')
    ax.set_xticks(SUBSAMPLE_SIZES)
    ax.set_xticklabels([str(n) for n in SUBSAMPLE_SIZES], rotation=30, ha='right', fontsize=10)
    plt.tight_layout()
    out2 = f'{OUTPUT_DIR}/snr_vs_subsample.png'
    plt.savefig(out2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'SNR plot saved to {out2}')

    # --- assertions ---
    snr_by_nsub = {n: results[n][5] for n in SUBSAMPLE_SIZES}
    n_max = max(SUBSAMPLE_SIZES)
    n_min = min(SUBSAMPLE_SIZES)

    assert snr_by_nsub[n_max] >= 2.0, \
        f"n_sub={n_max}: SNR {snr_by_nsub[n_max]:.2f} < 2.0"
    assert results[n_max][6] >= results[n_max][7] * 3 // 4, \
        f"n_sub={n_max}: fewer than 3/4 of bins positive below 10°"
    assert snr_by_nsub[n_max] > snr_by_nsub[n_min], \
        f"SNR does not increase from n_sub={n_min} to n_sub={n_max}"
    for n_sub in SUBSAMPLE_SIZES:
        w_n = results[n_sub][2]
        valid = w_n != -1.0
        assert np.all(np.abs(w_n[valid]) < 1.0), \
            f"n_sub={n_sub}: null |w| >= 1.0 in non-empty bins: {w_n}"

    print("\nAll assertions passed.")


if __name__ == '__main__':
    test_subsample_sweep()
