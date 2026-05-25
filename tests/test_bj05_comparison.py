"""Compare StreamHalo radial profile against Bullock & Johnston 2005 via Galaxia N-body data."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pytest

from streamhalo.potentials import v_esc as _v_esc

GALAXIA_DATA = '/home/ybchen/Downloads/galaxia-0.7.2/GalaxiaData'
BJ_HALO = 'halo02'

_origin = {'x_origin': 0.0, 'y_origin': 0.0, 'z_origin': 0.0,
           'dirx': 0.0, 'diry': 0.0, 'dirz': 1.0}
HOST_POTENTIAL_TYPE = 'Composite:MiyamotoNagai,Hernquist,Hernquist,NFW'
HOST_PARAMS = [
    {'logM': np.log10(4.7717e10), 'Rs': 2.6, 'Hs': 0.3, **_origin},   # disk
    {'logM': np.log10(5e9),       'Rs': 1.0,             **_origin},   # bulge
    {'logM': np.log10(1.8142e9),  'Rs': 0.0688867,       **_origin},   # nucleus
    {'logM': np.log10(5.5427e11), 'Rs': 15.626,                        # halo
     'a': 1.0, 'b': 1.0, 'c': 1.0, **_origin},
]


def v_esc(r, host_potential_type=HOST_POTENTIAL_TYPE, host_params=HOST_PARAMS, r_vir=200.0):
    return _v_esc(r, host_potential_type, host_params, r_vir=r_vir)


def load_bj05_positions(halo=BJ_HALO):
    """Load all particle positions (kpc) and progenitor index from a B&J 2005 halo model."""
    ebf = pytest.importorskip('ebf')

    filenames_path = f'{GALAXIA_DATA}/nbody1/filenames/{halo}.txt'
    if not os.path.exists(filenames_path):
        pytest.skip(f'B&J 2005 filenames not found: {filenames_path}')

    with open(filenames_path) as f:
        lines = f.read().strip().splitlines()

    base_dir = GALAXIA_DATA + '/' + lines[0].strip()
    n_files = int(lines[1].split()[0])
    sat_files = [l.strip() for l in lines[2:2 + n_files]]

    if not os.path.exists(base_dir):
        pytest.skip(f'B&J 2005 particle data not found: {base_dir}')

    all_pos, all_vel, all_mass, all_idx, sat_masses = [], [], [], [], []
    for i, fname in enumerate(sat_files):
        data = ebf.read(base_dir + fname, '/')
        pos = np.array(data['pos3'])
        vel = np.array(data['vel3'])
        mass = np.array(data['mass'])
        all_pos.append(pos)
        all_vel.append(vel)
        all_mass.append(mass)
        all_idx.append(np.full(len(pos), i, dtype=int))
        sat_masses.append(mass.sum())

    return (np.concatenate(all_pos, axis=0), np.concatenate(all_vel, axis=0),
            np.concatenate(all_mass), np.concatenate(all_idx), np.array(sat_masses))


def nfw_potential(r, log_M_vir=np.log10(5.5427e11), Rs=15.626, r_vir=200.0):
    """NFW potential in (km/s)^2; r, Rs, r_vir in kpc."""
    G = 4.302e-6  # (km/s)^2 kpc / M_sun
    M_vir = 10.0 ** log_M_vir
    c = r_vir / Rs
    f_c = np.log(1.0 + c) - c / (1.0 + c)
    return -G * M_vir * np.log(1.0 + r / Rs) / (r * f_c)


def lz_energy(positions, velocities, log_M_vir=np.log10(5.5427e11), Rs=15.626, r_vir=200.0):
    """Return Lz (kpc km/s) and total energy (km/s)^2."""
    x, y = positions[:, 0], positions[:, 1]
    vx, vy, vz = velocities[:, 0], velocities[:, 1], velocities[:, 2]
    r = np.linalg.norm(positions, axis=1)
    Lz = x * vy - y * vx
    E = 0.5 * (vx ** 2 + vy ** 2 + vz ** 2) + nfw_potential(r, log_M_vir, Rs, r_vir)
    return Lz, E


def self_binding_energy(positions, velocities, masses):
    """
    Return specific KE (CM frame) and specific self-grav PE for a set of particles.
    PE estimated as -G * M_tot / (2 * r_half), where r_half is the 3D half-mass radius.
    Both returned in (km/s)^2 per unit mass.
    Bound if KE_spec + PE_spec < 0.
    """
    G = 4.302e-6  # (km/s)^2 kpc / M_sun
    M_tot = masses.sum()
    r_cm = (masses[:, None] * positions).sum(axis=0) / M_tot
    v_cm = (masses[:, None] * velocities).sum(axis=0) / M_tot
    dv = velocities - v_cm
    KE_spec = (masses * 0.5 * np.sum(dv ** 2, axis=1)).sum() / M_tot
    dr = np.linalg.norm(positions - r_cm, axis=1)
    r_half = np.median(dr)
    PE_spec = -G * M_tot / (2.0 * r_half)
    return KE_spec, PE_spec


def radial_profile(positions, r_min=10.0, r_max=1000.0, n_bins=40):
    """Compute dN/d log r histogram."""
    radii = np.sqrt(np.sum(positions ** 2, axis=1))
    log_bins = np.logspace(np.log10(r_min), np.log10(r_max), n_bins + 1)
    counts, edges = np.histogram(radii, bins=log_bins)
    density = counts / np.log10(edges[1:] / edges[:-1])
    bin_centers = np.sqrt(edges[:-1] * edges[1:])
    return bin_centers, density


def test_bj05_comparison():
    """Compare StreamHalo stream radial distribution with B&J 2005 halo model."""
    output_dir = 'tests/outputs/bj05_comparison'
    os.makedirs(output_dir, exist_ok=True)

    # --- Load B&J 2005 data ---
    bj_positions, bj_velocities, bj_masses, bj_index, bj_sat_masses = load_bj05_positions()

    # --- Exclude short streams ---
    short_idx_path = 'tests/outputs/bj05_comparison/bj05_short_stream_indices.npy'
    if os.path.exists(short_idx_path):
        short_indices = set(np.load(short_idx_path).tolist())
        keep = ~np.isin(bj_index, list(short_indices))
        bj_positions = bj_positions[keep]
        bj_velocities = bj_velocities[keep]
        bj_masses = bj_masses[keep]
        bj_index = bj_index[keep]

    # --- Load StreamHalo results from test_mock_halo (needed for mass threshold) ---
    cache_path = 'tests/outputs/mock_halo_cache.npz'
    if not os.path.exists(cache_path):
        pytest.skip(f'Mock halo cache not found: {cache_path}. Run test_mock_halo first.')
    cache = np.load(cache_path)
    stream_positions = cache['stream_positions']
    stream_velocities = cache['stream_velocities']
    stream_index = cache['stream_index']
    sh_stellar_masses = cache['satellite_stellar_masses']
    sh_halo_masses = cache['satellite_halo_masses']

    # --- Keep only BJ05 progenitors above the mass equivalent of 100 StreamHalo particles ---
    min_particles_equiv = 100
    sh_mass_per_particle = sh_stellar_masses.sum() / len(stream_positions)
    min_mass_threshold = min_particles_equiv * sh_mass_per_particle
    bj_prog_masses = np.array([bj_masses[bj_index == p].sum()
                                for p in range(len(bj_sat_masses))])
    keep_progs = np.where(bj_prog_masses > min_mass_threshold)[0]
    keep_mask = np.isin(bj_index, keep_progs)
    bj_positions = bj_positions[keep_mask]
    bj_velocities = bj_velocities[keep_mask]
    bj_masses = bj_masses[keep_mask]
    bj_index = bj_index[keep_mask]
    bj_sat_masses = bj_sat_masses[keep_progs]
    print(f"\nStreamHalo mass/particle = {sh_mass_per_particle:.2e} M_sun  "
          f"=> BJ05 threshold = {min_mass_threshold:.2e} M_sun")
    print(f"BJ05: kept {len(keep_progs)} progenitors with mass >{min_mass_threshold:.2e} M_sun "
          f"({keep_mask.sum():,} / {len(keep_mask):,} particles)")

    # --- BJ05 self-binding analysis ---
    print(f"\n{'Prog':>5}  {'N':>7}  {'KE_spec':>12}  {'PE_spec':>12}  {'E_spec':>12}  {'Status':>8}")
    print('-' * 65)
    n_bound, n_unbound = 0, 0
    for prog in np.unique(bj_index):
        mask = bj_index == prog
        ke, pe = self_binding_energy(bj_positions[mask], bj_velocities[mask], bj_masses[mask])
        e_spec = ke + pe
        status = 'BOUND' if e_spec < 0 else 'UNBOUND'
        if e_spec < 0:
            n_bound += 1
        else:
            n_unbound += 1
        print(f'{prog:>5}  {mask.sum():>7}  {ke:>12.1f}  {pe:>12.1f}  {e_spec:>12.1f}  {status:>8}')
    print(f"\nBound: {n_bound}  Unbound: {n_unbound}  (out of {n_bound + n_unbound} progenitors)")

    # --- Subsample B&J05 so that r>10 kpc count matches StreamHalo ---
    r_match = 10.0
    bj_radii_all = np.sqrt(np.sum(bj_positions ** 2, axis=1))
    n_sh_r10 = int(np.sum(np.linalg.norm(stream_positions, axis=1) > r_match))
    n_bj_r10 = int(np.sum(bj_radii_all > r_match))
    fraction = n_sh_r10 / n_bj_r10
    n_show = int(round(fraction * len(bj_positions)))
    rng_proj = np.random.default_rng(0)
    bj_weights = bj_masses / bj_masses.sum()
    bj_sample = rng_proj.choice(len(bj_positions), size=n_show, replace=False, p=bj_weights)
    bj_show = bj_positions[bj_sample]
    bj_index_show = bj_index[bj_sample]

    # --- Radial profiles ---
    r_min, r_max = 10.0, 1000.0
    bj_centers, bj_density = radial_profile(bj_show, r_min, r_max)
    sh_centers, sh_density = radial_profile(stream_positions, r_min, r_max)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(4, 3))

    ax.plot(bj_centers, bj_density, linewidth=2, color='C1', label=f'BJ05 ({BJ_HALO})')
    ax.plot(sh_centers, sh_density, linewidth=2, color='C0', label=f'StreamHalo')

    ax.set_xlabel('Radius (kpc)')
    ax.set_ylabel(r'$dN/d\log r$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(r_min, r_max)
    ax.legend(loc='lower left')
    ax.text(0.95, 0.95,
            f'BJ05: {n_show:,} particles\nStreamHalo: {len(stream_positions):,} particles',
            transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=8)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/radial_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # --- XY projection comparison ---
    proj = 200.0

    cmap = plt.get_cmap('tab20')

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    for prog in np.unique(bj_index_show):
        mask = bj_index_show == prog
        axes[0].scatter(bj_show[mask, 0], bj_show[mask, 1],
                        s=1, alpha=0.5, color=cmap(prog % 20), rasterized=True)
    axes[0].set_title(f'BJ05 ({BJ_HALO})')
    axes[0].text(0.95, 0.95, f'$N = {n_show:,}$',
                 transform=axes[0].transAxes, verticalalignment='top',
                 horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=9)

    for prog in np.unique(stream_index):
        mask = stream_index == prog
        axes[1].scatter(stream_positions[mask, 0], stream_positions[mask, 1],
                        s=1, alpha=0.5, color=cmap(prog % 20), rasterized=True)
    axes[1].set_title('StreamHalo')
    axes[1].text(0.95, 0.95, f'$N = {len(stream_positions):,}$',
                 transform=axes[1].transAxes, verticalalignment='top',
                 horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=9)

    for ax in axes:
        ax.set_xlim(-proj, proj)
        ax.set_ylim(-proj, proj)
        ax.set_aspect('equal')
        ax.set_xlabel('$x$ (kpc)')

    axes[0].set_ylabel('$y$ (kpc)')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/xy_projection_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # --- XZ projection comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    for prog in np.unique(bj_index_show):
        mask = bj_index_show == prog
        axes[0].scatter(bj_show[mask, 0], bj_show[mask, 2],
                        s=1, alpha=0.5, color=cmap(prog % 20), rasterized=True)
    axes[0].set_title(f'BJ05 ({BJ_HALO})')
    axes[0].text(0.95, 0.95, f'$N = {n_show:,}$',
                 transform=axes[0].transAxes, verticalalignment='top',
                 horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=9)

    for prog in np.unique(stream_index):
        mask = stream_index == prog
        axes[1].scatter(stream_positions[mask, 0], stream_positions[mask, 2],
                        s=1, alpha=0.5, color=cmap(prog % 20), rasterized=True)
    axes[1].set_title('StreamHalo')
    axes[1].text(0.95, 0.95, f'$N = {len(stream_positions):,}$',
                 transform=axes[1].transAxes, verticalalignment='top',
                 horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=9)

    for ax in axes:
        ax.set_xlim(-proj, proj)
        ax.set_ylim(-proj, proj)
        ax.set_aspect('equal')
        ax.set_xlabel('$x$ (kpc)')

    axes[0].set_ylabel('$z$ (kpc)')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/xz_projection_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # --- Satellite mass function comparison ---
    active_idx = np.unique(stream_index[stream_index >= 0])
    sh_stellar_active = sh_stellar_masses[active_idx]
    sh_halo_active = sh_halo_masses[active_idx]

    all_masses = np.concatenate([bj_sat_masses, sh_stellar_active, sh_halo_active])
    log_m_min = np.log10(all_masses.min())
    log_m_max = np.log10(all_masses.max())
    mass_bins = np.logspace(log_m_min, log_m_max, 25)

    def dndlogm(masses, bins):
        counts, edges = np.histogram(masses, bins=bins)
        return counts / np.log10(edges[1:] / edges[:-1])

    bin_centers = np.sqrt(mass_bins[:-1] * mass_bins[1:])

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.step(bin_centers, dndlogm(bj_sat_masses, mass_bins), where='mid', linewidth=2,
            color='C1', label=f'BJ05 stellar ({len(bj_sat_masses)} sats)')
    ax.step(bin_centers, dndlogm(sh_stellar_active, mass_bins), where='mid', linewidth=2,
            color='C0', label=f'StreamHalo stellar ({len(sh_stellar_active)} sats)')
    ax.step(bin_centers, dndlogm(sh_halo_active, mass_bins), where='mid', linewidth=2,
            color='C0', linestyle='--', label=f'StreamHalo halo ({len(sh_halo_active)} sats)')
    ax.set_xlabel(r'Mass ($M_\odot$)')
    ax.set_ylabel(r'$dN/d\log M$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/mass_function_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # --- Satellite total mass vs median particle mass (BJ05) ---
    bj_progs_unique = np.unique(bj_index)
    bj_sat_total = np.array([bj_masses[bj_index == p].sum() for p in bj_progs_unique])
    bj_sat_median_m = np.array([np.median(bj_masses[bj_index == p]) for p in bj_progs_unique])

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.scatter(bj_sat_total, bj_sat_median_m, s=20, alpha=0.8, color='C1')
    ax.set_xlabel(r'Satellite total mass ($M_\odot$)')
    ax.set_ylabel(r'Median particle mass ($M_\odot$)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(f'BJ05 ({BJ_HALO}) — {len(bj_progs_unique)} satellites')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/bj05_sat_vs_particle_mass.png', dpi=150, bbox_inches='tight')
    plt.close()

    # --- Lz vs E comparison ---
    bj_vel_show = bj_velocities[bj_sample]
    bj_Lz, bj_E = lz_energy(bj_show, bj_vel_show)
    sh_Lz, sh_E = lz_energy(stream_positions, stream_velocities)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)

    E_min = min(bj_E.min(), sh_E.min())
    Lz_lim = 1e4

    for prog in np.unique(bj_index_show):
        mask = bj_index_show == prog
        axes[0].scatter(bj_Lz[mask], bj_E[mask],
                        s=1, alpha=0.4, color=cmap(prog % 20), rasterized=True)
    axes[0].set_title(f'BJ05 ({BJ_HALO})')
    axes[0].set_xlabel(r'$L_z$ (kpc km/s)')
    axes[0].set_ylabel(r'$E$ (km/s)$^2$')
    axes[0].set_xlim(-Lz_lim, Lz_lim)
    axes[0].set_ylim(E_min, 0)

    for prog in np.unique(stream_index):
        mask = stream_index == prog
        axes[1].scatter(sh_Lz[mask], sh_E[mask],
                        s=1, alpha=0.4, color=cmap(prog % 20), rasterized=True)
    axes[1].set_title('StreamHalo')
    axes[1].set_xlabel(r'$L_z$ (kpc km/s)')
    axes[1].set_xlim(-Lz_lim, Lz_lim)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/lz_energy_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # --- |V| vs galactocentric distance ---
    bj_r_show = np.linalg.norm(bj_show, axis=1)
    bj_v_show = np.linalg.norm(bj_vel_show, axis=1)
    sh_r = np.linalg.norm(stream_positions, axis=1)
    sh_v = np.linalg.norm(stream_velocities, axis=1)

    r_curve = np.logspace(np.log10(1.0), np.log10(1000.0), 300)
    v_esc_curve = np.array([v_esc(r) for r in r_curve])

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)

    for prog in np.unique(bj_index_show):
        mask = bj_index_show == prog
        axes[0].scatter(bj_r_show[mask], bj_v_show[mask],
                        s=1, alpha=0.4, color=cmap(prog % 20), rasterized=True)
    axes[0].plot(r_curve, v_esc_curve, 'k--', linewidth=1.5, label=r'$v_{\rm esc}$')
    axes[0].set_title(f'BJ05 ({BJ_HALO})')
    axes[0].set_xlabel('Galactocentric distance (kpc)')
    axes[0].set_ylabel(r'$|V|$ (km/s)')
    axes[0].set_xscale('log')
    axes[0].set_xlim(1e0, 1e3)
    axes[0].set_ylim(0, 700)
    axes[0].legend(fontsize=8)

    for prog in np.unique(stream_index):
        mask = stream_index == prog
        axes[1].scatter(sh_r[mask], sh_v[mask],
                        s=1, alpha=0.4, color=cmap(prog % 20), rasterized=True)
    axes[1].plot(r_curve, v_esc_curve, 'k--', linewidth=1.5, label=r'$v_{\rm esc}$')
    axes[1].set_title('StreamHalo')
    axes[1].set_xlabel('Galactocentric distance (kpc)')
    axes[1].set_xscale('log')
    axes[1].set_xlim(1e0, 1e3)
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/speed_vs_radius_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # --- Vr vs r comparison ---
    bj_Vr = np.sum(bj_show * bj_vel_show, axis=1) / bj_r_show
    sh_Vr = np.sum(stream_positions * stream_velocities, axis=1) / sh_r

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)

    for prog in np.unique(bj_index_show):
        mask = bj_index_show == prog
        axes[0].scatter(bj_r_show[mask], bj_Vr[mask],
                        s=1, alpha=0.4, color=cmap(prog % 20), rasterized=True)
    axes[0].axhline(0, color='k', linewidth=0.8, linestyle='--')
    axes[0].set_title(f'BJ05 ({BJ_HALO})')
    axes[0].set_xlabel('Galactocentric distance (kpc)')
    axes[0].set_ylabel(r'$V_r$ (km/s)')
    axes[0].set_xlim(0, 250)
    axes[0].set_ylim(-600, 600)

    for prog in np.unique(stream_index):
        mask = stream_index == prog
        axes[1].scatter(sh_r[mask], sh_Vr[mask],
                        s=1, alpha=0.4, color=cmap(prog % 20), rasterized=True)
    axes[1].axhline(0, color='k', linewidth=0.8, linestyle='--')
    axes[1].set_title('StreamHalo')
    axes[1].set_xlabel('Galactocentric distance (kpc)')
    axes[1].set_xlim(0, 250)
    axes[1].set_ylim(-600, 600)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/vr_vs_radius_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # --- Vr vs Vt comparison ---
    bj_Vt = np.sqrt(np.maximum(np.sum(bj_vel_show**2, axis=1) - bj_Vr**2, 0.0))
    sh_Vt = np.sqrt(np.maximum(np.sum(stream_velocities**2, axis=1) - sh_Vr**2, 0.0))

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    for prog in np.unique(bj_index_show):
        mask = bj_index_show == prog
        axes[0].scatter(bj_Vr[mask], bj_Vt[mask],
                        s=1, alpha=0.4, color=cmap(prog % 20), rasterized=True)
    axes[0].set_title(f'BJ05 ({BJ_HALO})')
    axes[0].set_xlabel(r'$V_r$ (km/s)')
    axes[0].set_ylabel(r'$V_t$ (km/s)')
    axes[0].set_xlim(-700, 700)
    axes[0].set_ylim(0, 700)

    for prog in np.unique(stream_index):
        mask = stream_index == prog
        axes[1].scatter(sh_Vr[mask], sh_Vt[mask],
                        s=1, alpha=0.4, color=cmap(prog % 20), rasterized=True)
    axes[1].set_title('StreamHalo')
    axes[1].set_xlabel(r'$V_r$ (km/s)')
    axes[1].set_xlim(-700, 700)
    axes[1].set_ylim(0, 700)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/vr_vt_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    test_bj05_comparison()
