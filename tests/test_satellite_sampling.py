#!/usr/bin/env python3
"""Satellite mass-function sampling test."""

import os

import matplotlib.pyplot as plt
import numpy as np

from streamhalo.satellites import SatellitePopulation


def test_satellite_sampling():
    """Power-law mass-function sampling matches the analytical prediction."""

    rng = np.random.default_rng(42)
    output_dir = 'tests/outputs/satellite_sampling'
    os.makedirs(output_dir, exist_ok=True)

    sat_pop = SatellitePopulation(
        mass_function='powerlaw', alpha=-1.9, M_min=1e7, M_max=1e11, rng=rng,
    )

    n_sats = 20
    masses = sat_pop.sample(n_sats)
    logM = sat_pop.sample_log10(n_sats)

    large_sample = sat_pop.sample(10000)
    stellar_masses_sample = sat_pop.stellar_mass(large_sample)

    fig, ax = plt.subplots(figsize=(5, 3))

    log_bins_h = np.logspace(np.log10(sat_pop.M_min), np.log10(sat_pop.M_max), 50)
    counts_h, edges_h = np.histogram(large_sample, bins=log_bins_h)
    density_h = counts_h / np.log10(edges_h[1:] / edges_h[:-1])
    bin_centers_h = np.sqrt(edges_h[:-1] * edges_h[1:])
    ax.step(bin_centers_h, density_h, where='mid', linewidth=2, label='Halo mass', color='C0')

    log_bins_s = np.logspace(np.log10(stellar_masses_sample.min()),
                              np.log10(stellar_masses_sample.max()), 50)
    counts_s, edges_s = np.histogram(stellar_masses_sample, bins=log_bins_s)
    density_s = counts_s / np.log10(edges_s[1:] / edges_s[:-1])
    bin_centers_s = np.sqrt(edges_s[:-1] * edges_s[1:])
    ax.step(bin_centers_s, density_s, where='mid', linewidth=2,
            label='Stellar mass (Behroozi+13)', color='C1')

    M_range = np.logspace(np.log10(sat_pop.M_min), np.log10(sat_pop.M_max), 200)
    expected_unnorm = M_range ** (sat_pop.alpha + 1)
    expected_at_edges = np.interp(bin_centers_h, M_range, expected_unnorm)
    expected = expected_unnorm / expected_at_edges.sum() * density_h.sum()
    ax.plot(M_range, expected, 'r--', linewidth=2, label='Expected power law')

    expected_at_bins = np.interp(bin_centers_h, M_range, expected)
    correlation = np.corrcoef(density_h, expected_at_bins)[0, 1]
    assert correlation > 0.8, f"Power-law distribution mismatch: correlation = {correlation:.3f}"

    ax.set_xlabel(r'Mass ($M_\odot$)')
    ax.set_ylabel(r'$dN/d\log M$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(stellar_masses_sample.min(), sat_pop.M_max)

    mean_log = np.log10(large_sample.mean())
    median_log = np.log10(np.median(large_sample))
    info_text = (f'$\\alpha={sat_pop.alpha}$\n'
                 f'$\\langle M_h \\rangle = 10^{{{mean_log:.1f}}}$\n'
                 f'Median $M_h = 10^{{{median_log:.1f}}}$\n'
                 f'$N_{{\\rm sat}}={len(large_sample)}$')
    ax.text(0.05, 0.05, info_text, transform=ax.transAxes, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=9)
    ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/satellite_mass_function.png', dpi=150, bbox_inches='tight')
    plt.close()

    assert len(masses) == n_sats
    assert np.all((masses >= sat_pop.M_min) & (masses <= sat_pop.M_max))
    assert len(logM) == n_sats
    assert len(large_sample) == 10000


if __name__ == '__main__':
    test_satellite_sampling()
