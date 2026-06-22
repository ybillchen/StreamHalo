#!/usr/bin/env python3
"""JAX parallelization test."""

import multiprocessing
import os
import sys
import threading
import time

import jax
import matplotlib.pyplot as plt
import numpy as np
import psutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests.plot_style import apply_style, style_ax, legend as _legend
apply_style()

try:
    import pynvml
    _PYNVML_AVAILABLE = True
except ImportError:
    _PYNVML_AVAILABLE = False

from streamhalo import MockHalo


def detect_gpu():
    """Return (is_gpu, gpu_handle_or_None) based on JAX devices."""
    devices = jax.devices()
    is_gpu = any('cuda' in str(d).lower() or 'gpu' in str(d).lower() for d in devices)
    if is_gpu and _PYNVML_AVAILABLE:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return True, handle
    return is_gpu, None


def mon_exec(tgt_f, int_val=0.2, gpu_handle=None):
    """Monitor CPU (and optionally GPU) usage during function execution."""
    meas = []
    proc = psutil.Process()
    mon = True
    st = time.time()

    def mon_worker():
        nonlocal meas
        while mon:
            try:
                cp = proc.cpu_percent(interval=int_val)
                nt = proc.num_threads()
                el = time.time() - st
                entry = {
                    'elapsed': round(el, 1),
                    'cpu': round(cp, 1),
                    'threads': nt,
                }
                if gpu_handle is not None:
                    util = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
                    entry['gpu_util'] = util.gpu
                    entry['gpu_mem_pct'] = round(mem.used / mem.total * 100, 1)
                meas.append(entry)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break

    mt = threading.Thread(target=mon_worker, daemon=True)
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

    def position_sampler(rng):
        r = rng.uniform(10, 100)
        theta = np.arccos(rng.uniform(-1, 1))
        phi = rng.uniform(0, 2 * np.pi)
        return np.array([
            r * np.sin(theta) * np.cos(phi),
            r * np.sin(theta) * np.sin(phi),
            r * np.cos(theta),
        ])

    def velocity_sampler(rng, r, v_esc_r):
        v_m = rng.uniform(0, v_esc_r)
        v_t = np.arccos(rng.uniform(-1, 1))
        v_p = rng.uniform(0, 2 * np.pi)
        return np.array([
            v_m * np.sin(v_t) * np.cos(v_p),
            v_m * np.sin(v_t) * np.sin(v_p),
            v_m * np.cos(v_t),
        ])

    halo = MockHalo(
        host_potential_type=h_type,
        host_params=h_params,
        rng=rng,
        mass_function='powerlaw',
    )

    halo.sample_satellites(
        n_satellites=n_sats,
        position_sampler=position_sampler,
        velocity_sampler=velocity_sampler,
    )

    halo.generate_streams(
        target_particles=n_sats * n_part,
        integration_time=int_time,
        n_steps=n_steps,
        plummer_scale=50.0,
    )

    tot_p = len(halo.stream_positions)
    all_pos = halo.stream_positions

    return tot_p, all_pos


def _make_cpu_plot(res, np_vals, cpu_count, output_dir):
    fig, axes = plt.subplots(len(np_vals), 1, figsize=(4, 5), sharex=True)
    if len(np_vals) == 1:
        axes = [axes]

    y_max = cpu_count * 100
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
        info_text = (
            f"$n_{{\\rm particles}} = {mantissa} \\times 10^{{{exponent}}}$\n"
            f"Duration: {data['wall_time']:.2f} s"
        )
        ax.text(0.95, 0.92, info_text, transform=ax.transAxes, va='top', ha='right', fontsize=10)
        style_ax(ax, ylabel='CPU Usage (%)')
        ax.set_xlim(left=0, right=max_time)
        ax.set_ylim(bottom=0, top=y_max)

    axes[-1].set_xlabel('Time (s)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'parallelization_results_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


def _make_gpu_plot(res, np_vals, output_dir):
    fig, axes = plt.subplots(len(np_vals), 1, figsize=(4, 5), sharex=True)
    if len(np_vals) == 1:
        axes = [axes]

    max_time = max(res[np_v]['wall_time'] for np_v in np_vals) * 1.05

    for i, np_v in enumerate(np_vals):
        data = res[np_v]
        elapsed = np.array([x['elapsed'] for x in data['measurements']])
        gpu_util = np.array([x.get('gpu_util', 0) for x in data['measurements']])
        gpu_mem = np.array([x.get('gpu_mem_pct', 0) for x in data['measurements']])

        ax = axes[i]
        ax.plot(elapsed, gpu_util, linewidth=2, color='C1', label='GPU util %')
        ax.plot(elapsed, gpu_mem, linewidth=2, color='C2', linestyle='--', label='GPU mem %')
        ax.axhline(y=np.median(gpu_util), color='C1', linestyle=':', alpha=0.7, linewidth=1)

        exponent = int(np.log10(np_v))
        mantissa = int(np_v / (10 ** exponent))
        info_text = (
            f"$n_{{\\rm particles}} = {mantissa} \\times 10^{{{exponent}}}$\n"
            f"Duration: {data['wall_time']:.2f} s"
        )
        ax.text(0.95, 0.92, info_text, transform=ax.transAxes, va='top', ha='right', fontsize=10)
        style_ax(ax, ylabel='GPU Usage (%)')
        ax.set_xlim(left=0, right=max_time)
        ax.set_ylim(bottom=0, top=105)
        if i == 0:
            _legend(ax, loc='upper left')

    axes[-1].set_xlabel('Time (s)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'parallelization_gpu_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


def test_jax_par():
    """Benchmark JAX parallelization with varying stream sizes."""
    output_dir = Path("tests/outputs/parallelization_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    is_gpu, gpu_handle = detect_gpu()

    cpu_count = multiprocessing.cpu_count()
    n_steps = 1000
    nsat = 10
    base_np_vals = [1000, 3000, 10000]
    np_vals = [v * 100 for v in base_np_vals] if is_gpu else base_np_vals

    res = {}

    for np_v in np_vals:
        def tf(np_v=np_v):
            return gen_streams(n_sats=nsat, n_part=np_v, n_steps=n_steps)

        measurements, wall_time = mon_exec(tf, int_val=0.1, gpu_handle=gpu_handle)
        tf()  # second call to capture final particle count (monitoring call already ran it)

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
            'mean_cpu': mean_cpu,
        }

    _make_cpu_plot(res, np_vals, cpu_count, output_dir)

    if is_gpu and gpu_handle is not None:
        _make_gpu_plot(res, np_vals, output_dir)

    assert len(res) == len(np_vals), "Not all n_particles values tested"
    for np_v, d in res.items():
        assert len(d['measurements']) > 0, f"No measurements for n_particles={np_v}"
        assert d['peak_cpu'] >= 0, f"Invalid peak CPU for n_particles={np_v}"


if __name__ == '__main__':
    test_jax_par()
