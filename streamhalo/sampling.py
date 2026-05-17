"""Position and velocity sampling utilities."""

import numpy as np


def sample_r_broken_powerlaw(rng, r_min=1.0, r_max=300.0, r_break=100.0,
                             alpha_inner=-3, alpha_outer=-4):
    """Sample one galactocentric radius from broken power-law n(r) ~ r^alpha."""
    ei = 2 + alpha_inner   # exponent of dN/dr in inner region
    eo = 2 + alpha_outer   # exponent of dN/dr in outer region
    ratio = r_break ** (ei - eo)  # continuity normalisation B/A

    def _piece(r_lo, r_hi, exp, scale=1.0):
        if exp == -1:
            return scale * np.log(r_hi / r_lo)
        return scale * (r_hi**(exp+1) - r_lo**(exp+1)) / (exp + 1)

    I1 = _piece(r_min, r_break, ei)
    I2 = _piece(r_break, r_max, eo, scale=ratio)
    total = I1 + I2

    u = rng.uniform(0, 1)
    if u * total <= I1:
        if ei == -1:
            return r_min * np.exp(u * total)
        return (r_min**(ei+1) + u * total * (ei+1)) ** (1.0/(ei+1))
    u_adj = (u * total - I1) / ratio
    if eo == -1:
        return r_break * np.exp(u_adj)
    return (r_break**(eo+1) + u_adj * (eo+1)) ** (1.0/(eo+1))


def sample_isotropic_direction(rng):
    """Return a 3D unit vector sampled isotropically on the sphere."""
    theta = np.arccos(rng.uniform(-1, 1))
    phi = rng.uniform(0, 2 * np.pi)
    return np.array([np.sin(theta) * np.cos(phi),
                     np.sin(theta) * np.sin(phi),
                     np.cos(theta)])


def _df_hernquist(E):
    """Hernquist (1990) isotropic DF; E in [-1, 0] (dimensionless)."""
    q = np.sqrt(-E)
    return (1.0 / (8.0 * np.sqrt(2.0) * np.pi**3)
            * (1.0 - q*q)**(-2.5)
            * (3.0*np.arcsin(q) + q*np.sqrt(1.0 - q*q)*(1.0 - 2.0*q*q)*(8.0*q**4 - 8.0*q*q - 3.0)))


def sample_v_hernquist(rng, v_esc_val, phi_scale, n_grid=1000):
    """Sample speed from the Hernquist isotropic DF via numerical CDF inversion.

    phi_scale = |Phi(r_ref)| sets the global energy normalisation (use r_min as ref).
    """
    q_max = np.sqrt(0.5 * v_esc_val**2 / phi_scale)  # dimensionless momentum at v=0

    q_grid = np.linspace(1e-6, q_max * (1.0 - 1e-9), n_grid)
    pdf_q = np.maximum(np.array([_df_hernquist(-q**2) for q in q_grid]) * 2.0 * q_grid, 0.0)

    cdf = np.cumsum(pdf_q)
    cdf /= cdf[-1]

    q_s = np.interp(rng.uniform(0.0, 1.0), cdf, q_grid)
    return float(np.sqrt(v_esc_val**2 - 2.0 * q_s**2 * phi_scale))


def make_broken_powerlaw_position_sampler(r_min, r_max, r_break=100.0,
                                          alpha_inner=-3, alpha_outer=-3):
    """Return a callable rng -> (x, y, z) sampling 3D positions isotropically with broken-power-law radius."""
    def _sampler(rng):
        r = sample_r_broken_powerlaw(rng, r_min, r_max, r_break, alpha_inner, alpha_outer)
        return r * sample_isotropic_direction(rng)
    return _sampler


def uniform_velocity_sampler(rng, r, v_esc_r):
    """Sample velocity uniformly in [0, v_esc] with isotropic direction."""
    return rng.uniform(0.0, v_esc_r) * sample_isotropic_direction(rng)


def make_hernquist_velocity_sampler(phi_scale):
    """Return a velocity sampler (rng, r, v_esc) -> v that uses the Hernquist isotropic DF."""
    def _sampler(rng, r, v_esc_r):
        return sample_v_hernquist(rng, v_esc_r, phi_scale) * sample_isotropic_direction(rng)
    return _sampler
