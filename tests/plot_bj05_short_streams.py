"""Plot BJ05 short streams (half-mass radius < 10 kpc w.r.t. CM)."""

import os
import sys

import ebf
import matplotlib.pyplot as plt
import numpy as np

GALAXIA_DATA = '/Users/ybchen/Downloads/galaxia-0.7.2/GalaxiaData'
BJ_HALO = 'halo07'
R_HALF_MAX = 10.0  # kpc
OUTPUT_DIR = 'tests/outputs/bj05_comparison'


def load_bj05(halo=BJ_HALO):
    filenames_path = f'{GALAXIA_DATA}/nbody1/filenames/{halo}.txt'
    if not os.path.exists(filenames_path):
        sys.exit(f'Filenames not found: {filenames_path}')

    with open(filenames_path) as f:
        lines = f.read().strip().splitlines()

    base_dir = GALAXIA_DATA + '/' + lines[0].strip()
    n_files = int(lines[1].split()[0])
    sat_files = [l.strip() for l in lines[2:2 + n_files]]

    all_pos, all_idx = [], []
    for i, fname in enumerate(sat_files):
        data = ebf.read(base_dir + fname, '/')
        pos = np.array(data['pos3'])
        all_pos.append(pos)
        all_idx.append(np.full(len(pos), i, dtype=int))

    return np.concatenate(all_pos, axis=0), np.concatenate(all_idx)


def half_mass_radius(positions):
    """Median distance from center of mass (proxy for 3D half-mass radius)."""
    r_cm = positions.mean(axis=0)
    return np.median(np.linalg.norm(positions - r_cm, axis=1)), r_cm


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f'Loading BJ05 {BJ_HALO}...')
    positions, index = load_bj05()

    R_GC_MIN = 10.0  # kpc

    short_progs = []
    for prog in np.unique(index):
        mask = index == prog
        r_half, r_cm = half_mass_radius(positions[mask])
        r_gc = np.linalg.norm(r_cm)
        if r_half < R_HALF_MAX and r_gc > R_GC_MIN:
            short_progs.append((prog, r_half, r_gc, r_cm, mask))

    selected_indices = [p[0] for p in short_progs]
    print(f'Short streams (r_half < {R_HALF_MAX} kpc, r_gc > {R_GC_MIN} kpc): '
          f'{len(short_progs)} / {len(np.unique(index))}')
    print(f'Progenitor indices: {selected_indices}')

    np.save(f'{OUTPUT_DIR}/bj05_short_stream_indices.npy', np.array(selected_indices))

    cmap = plt.get_cmap('tab20')
    fig, ax = plt.subplots(figsize=(6, 6))

    for prog, r_half, r_gc, r_cm, mask in short_progs:
        pos = positions[mask]
        ax.scatter(pos[:, 0], pos[:, 1],
                   s=2, alpha=0.6, color=cmap(prog % 20), rasterized=True)
        ax.annotate(f'{prog}', xy=(r_cm[0], r_cm[1]), fontsize=6,
                    color=cmap(prog % 20), ha='center')

    ax.set_xlabel('$x$ (kpc)')
    ax.set_ylabel('$y$ (kpc)')
    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)
    ax.set_aspect('equal')
    ax.set_title(f'BJ05 {BJ_HALO}: short streams ($r_{{1/2}} < {R_HALF_MAX}$ kpc, $N={len(short_progs)}$)')

    outpath = f'{OUTPUT_DIR}/bj05_short_streams_xy.png'
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {outpath}')


if __name__ == '__main__':
    main()
