#!/usr/bin/env python3
"""Plot stream #9928 (highest median galactocentric radius) at t=4 Gyr.

Run:
    conda run -n StreaMAX python tests/plot_stream9928.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from StreaMAX.StreaMAX import generate_stream as streamax_generate_stream

_origin = {'x_origin': 0.0, 'y_origin': 0.0, 'z_origin': 0.0,
           'dirx': 0.0, 'diry': 0.0, 'dirz': 1.0}
HOST_POTENTIAL_TYPE = 'Composite:MiyamotoNagai,Hernquist,Hernquist,NFW'
HOST_PARAMS = [
    {'logM': np.log10(4.7717e10), 'Rs': 2.6, 'Hs': 0.3, **_origin},
    {'logM': np.log10(5e9),       'Rs': 1.0,             **_origin},
    {'logM': np.log10(1.8142e9),  'Rs': 0.0688867,       **_origin},
    {'logM': np.log10(5.5427e11), 'Rs': 15.626,
     'a': 1.0, 'b': 1.0, 'c': 1.0, **_origin},
]

# Stream #5247: highest median galactocentric radius with >100 stars (~114 kpc), t_int=4 Gyr
SATELLITE_IC = np.array([-61.7990, 48.3170, 42.6823,     # kpc
                          118.2035, 57.1942, -146.4335], dtype=float)  # km/s
SATELLITE_PARAMS = {'logM': 10.359, 'Rs': 2.5080}
ALPHA = 0.5
M_F_SAT = 8.0
T_INT = 4.0  # Gyr

N_STEPS_STREAMAX = 3999
N_PARTICLES = 4000
PROJ = 300.0  # kpc


def main():
    output_dir = 'tests/outputs/stream5247'
    os.makedirs(output_dir, exist_ok=True)

    print('Generating stream...')
    _, xv_sat, xv_stream, _ = streamax_generate_stream(
        SATELLITE_IC,
        HOST_POTENTIAL_TYPE,
        HOST_PARAMS,
        'Plummer',
        SATELLITE_PARAMS,
        T_INT,
        ALPHA,
        N_STEPS_STREAMAX,
        N_PARTICLES,
        False,
        m_f_sat=M_F_SAT,
        type_method='Chen2025',
    )

    pos = np.array(xv_stream[:, :3])
    sat = np.array(xv_sat[-1, :3])

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), facecolor='black')
    fig.subplots_adjust(wspace=0.05)

    panels = [
        (axes[0], pos[:, 0], pos[:, 1], sat[0], sat[1], 'x (kpc)', 'y (kpc)', 'XY'),
        (axes[1], pos[:, 0], pos[:, 2], sat[0], sat[2], 'x (kpc)', 'z (kpc)', 'XZ'),
    ]

    for ax, px, py, sx, sy, xl, yl, title in panels:
        ax.set_facecolor('black')
        ax.scatter(px, py, s=1, c='cyan', alpha=0.5, rasterized=True)
        ax.plot(sx, sy, 'o', color='yellow', markersize=6, zorder=5, label='Satellite')
        ax.set_xlim(-PROJ, PROJ)
        ax.set_ylim(-PROJ, PROJ)
        ax.set_aspect('equal')
        ax.set_xlabel(xl, color='white')
        ax.set_ylabel(yl, color='white')
        ax.set_title(title, color='white', fontsize=10)
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('white')

    axes[0].legend(loc='upper right', fontsize=8,
                   labelcolor='white', facecolor='black', edgecolor='white')
    axes[1].yaxis.set_label_position('right')
    axes[1].yaxis.tick_right()

    r_med = np.median(np.linalg.norm(pos, axis=1))
    fig.suptitle(f'Stream #5247 — t = {T_INT} Gyr  |  median r = {r_med:.1f} kpc',
                 color='white', fontsize=11, y=1.01)

    out_path = f'{output_dir}/stream5247_snapshot.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='black')
    plt.close()
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
