"""Composite-potential utilities (escape velocity, tidal radius, ...)."""

import numpy as np

from StreaMAX.potentials import (
    NFW_potential, Hernquist_potential, MiyamotoNagai_potential,
)
from StreaMAX.utils import prepare_params as stremax_prepare_params
from StreaMAX.constants import KPCGYR_TO_KMS

_POT_FNS = {
    'NFW': NFW_potential,
    'Hernquist': Hernquist_potential,
    'MiyamotoNagai': MiyamotoNagai_potential,
}


def composite_phi(r, host_potential_type, host_params):
    """Composite potential at radius r (evaluated along the x-axis)."""
    comp_types = host_potential_type.split(':')[1].split(',')
    return sum(
        float(_POT_FNS[t](float(r), 0.0, 0.0, stremax_prepare_params(p)))
        for t, p in zip(comp_types, host_params)
    )


def v_esc(r, host_potential_type, host_params, r_vir=200.0):
    """Escape velocity (km/s) with virial radius as zero-point: v_esc²=2*(Phi(r_vir)-Phi(r))."""
    phi_in = composite_phi(r, host_potential_type, host_params)
    phi_out = composite_phi(r_vir, host_potential_type, host_params)
    return float(np.sqrt(2.0 * (phi_out - phi_in)) * KPCGYR_TO_KMS)


def tidal_radius(r_sat, M_sat, logM_host, Rs_host, r_vir=200.0):
    """King tidal radius: r_t = r_sat * (M_sat / (3 * M_host(<r_sat)))^(1/3) for an NFW host."""
    M_host = 10.0 ** logM_host
    c = r_vir / Rs_host
    f_c = np.log(1.0 + c) - c / (1.0 + c)
    x = r_sat / Rs_host
    M_enc = M_host * (np.log(1.0 + x) - x / (1.0 + x)) / f_c
    return r_sat * (M_sat / (3.0 * M_enc)) ** (1.0 / 3.0)
