#!/usr/bin/env python3
"""JAX parallelization test."""

import multiprocessing
import os
import sys
import threading
import time

import matplotlib.pyplot as plt
import numpy as np
import psutil
from pathlib import Path

from streamhalo import MockHalo


def v_esc(r, M_host, Rs):
    """Calculate escape velocity at radius r for NFW halo."""
    GM_sun = 1.32712e11
    kpc_to_km = 3.0857e16
    x = r / Rs
    M_enclosed = M_host * (np.log(1.0 + x) - x / (1.0 + x))
    r_km = r * kpc_to_km
    v_esc = np.sqrt(2.0 * GM_sun * M_enclosed / r_km)
    return v_esc


def mon_exec(tgt_f, int_val=0.2):
    """Monitor CPU usage during function execution."""
    meas = []
    proc = psutil.Process()
    mon = True
    st = time.time()

    def mon_cpu():
        nonlocal meas
        while mon:
            try:
                cp = proc.cpu_percent(interval=int_val)
                nt = proc.num_threads()
                el = time.time() - st
                meas.append({
                    'elapsed': round(el, 1),
                    'cpu': round(cp, 1),
                    'threads': nt
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break

    mt = threading.Thread(target=mon_cpu, daemon=True)
    mt.start()

    tgt_f()

    mon = False
    mt.join(timeout=1)
    wt = time.time() - st

    return meas, wt


def gen_streams(n_sats=10, n_part=1000, n_steps=1000):
    """Generate tidal streams for benchmarking. Returns particle positions."""
    rng = np.random.default_rng(42)

    int_time = 8.0

    h_type = 'NFW'
    h_params = {
        'logM': 12.0,
        'Rs': 25.0,
        'a': 1.0, 'b': 1.0, 'c': 1.0,
        'dirx': 0.0, 'diry': 0.0, 'dirz': 1.0,
        'x_origin': 0.0, 'y_origin': 0.0, 'z_origin': 0.0,
    }

    halo = MockHalo(
        host_potential_type=h_type,
        host_params=h_params,
        n_satellites=1,
        mass_function='powerlaw',
        n_stream_particles=n_part,
        rng=rng
    )

    M_h = 10.0 ** h_params['logM']
    Rs_h = h_params['Rs']

    tot_p = 0
    all_pos = []
    for i in range(n_sats):
        # Create satellite initial condition
        r = rng.uniform(10, 100)
        theta = np.arccos(rng.uniform(-1, 1))
        phi = rng.uniform(0, 2 * np.pi)

        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)

        v_e = v_esc(r, M_h, Rs_h)
        v_m = rng.uniform(0, v_e)

        v_t = np.arccos(rng.uniform(-1, 1))
        v_p = rng.uniform(0, 2 * np.pi)

        vx = v_m * np.sin(v_t) * np.cos(v_p)
        vy = v_m * np.sin(v_t) * np.sin(v_p)
        vz = v_m * np.cos(v_t)

        sat_ic = np.array([x, y, z, vx, vy, vz])

        t_s, xv_s, xv_str, xhi_str = halo.add_stream(
            sat_ic,
            'Plummer',
            {'logM': 8.0, 'Rs': 50.0},
            int_time,
            n_steps=n_steps,
            unroll=False,
            n_particles=n_part
        )
        tot_p += len(xv_str)
        all_pos.append(xv_str[:, :3])

    if all_pos:
        all_pos = np.vstack(all_pos)
    else:
        all_pos = np.array([])

    return tot_p, all_pos


def test_jax_par():
    """Benchmark JAX parallelization with varying stream sizes."""
    output_dir = Path("tests/outputs/parallelization_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    cpu_count = multiprocessing.cpu_count()
    y_max = cpu_count * 100

    n_steps = 1000
    nsat = 10
    np_vals = [1000, 3000, 10000]

    res = {}

    for np_v in np_vals:
        def tf():
            return gen_streams(
                n_sats=nsat,
                n_part=np_v,
                n_steps=n_steps
            )

        measurements, wall_time = mon_exec(tf, int_val=0.1)
        tp, pos = tf()

        if measurements:
            cpu_vals = np.array([x['cpu'] for x in measurements])
            peak_cpu = np.max(cpu_vals)
            median_cpu = np.median(cpu_vals)
            mean_cpu = np.mean(cpu_vals)
        else:
            peak_cpu = median_cpu = mean_cpu = 0

        res[np_v] = {
            'measurements': measurements,
            'wall_time': wall_time,
            'peak_cpu': peak_cpu,
            'median_cpu': median_cpu,
            'mean_cpu': mean_cpu
        }

    fig, axes = plt.subplots(3, 1, figsize=(4, 5), sharex=True)

    max_time = max(res[np_v]['wall_time'] for np_v in np_vals) * 1.05

    for i, np_v in enumerate(np_vals):
        data = res[np_v]
        elapsed = np.array([x['elapsed'] for x in data['measurements']])
        cpu_vals = np.array([x['cpu'] for x in data['measurements']])

        ax = axes[i]
        ax.plot(elapsed, cpu_vals, linewidth=2, color='C0')
        ax.axhline(y=data['median_cpu'], color='g', linestyle=':', alpha=0.7, linewidth=1)

        exponent = int(np.log10(np_v))
        mantissa = int(np_v / (10 ** exponent))
        info_text = f"$n_{{\\rm particles}} = {mantissa} \\times 10^{{{exponent}}}$\nDuration: {data['wall_time']:.2f} s"
        ax.text(0.95, 0.92, info_text, transform=ax.transAxes, verticalalignment='top',
                horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=9)

        ax.set_ylabel('CPU Usage (%)')
        ax.set_xlim(left=0, right=max_time)
        ax.set_ylim(bottom=0, top=y_max)

    axes[2].set_xlabel('Time (s)')

    plt.tight_layout()
    plt.savefig(output_dir / 'parallelization_results_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    assert len(res) == len(np_vals), "Not all n_particles values tested"
    for np_v, d in res.items():
        assert len(d['measurements']) > 0, f"No measurements for n_particles={np_v}"
        assert d['peak_cpu'] >= 0, f"Invalid peak CPU for n_particles={np_v}"


if __name__ == '__main__':
    test_jax_parallelization()
