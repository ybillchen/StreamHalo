"""Satellite galaxy generation and mass function sampling."""

import numpy as np
from scipy.special import gamma


class SatellitePopulation:
    """Generate satellite population from mass functions."""

    VALID_MASS_FUNCTIONS = ['powerlaw', 'truncated_powerlaw']

    def __init__(self, mass_function='powerlaw', alpha=-1.9, M_min=1e7, M_max=1e10, rng=None):
        """Initialize satellite population generator."""
        if mass_function not in self.VALID_MASS_FUNCTIONS:
            raise ValueError(
                f"Invalid mass function '{mass_function}'. "
                f"Must be one of: {self.VALID_MASS_FUNCTIONS}"
            )

        self.mass_function = mass_function
        self.alpha = alpha
        self.M_min = M_min
        self.M_max = M_max
        self.rng = rng if rng is not None else np.random.default_rng()

        if self.alpha >= 0:
            raise ValueError(f"Power-law index alpha must be negative, got {self.alpha}")

    def _powerlaw_normalization(self):
        """Compute normalization for power-law mass function."""
        if abs(self.alpha + 1) < 1e-6:
            return 1.0 / np.log(self.M_max / self.M_min)
        else:
            exponent = self.alpha + 1
            return exponent / (self.M_max**exponent - self.M_min**exponent)

    def sample(self, n_satellites):
        """Sample satellite masses from mass function."""
        if self.alpha >= 0:
            raise ValueError("Power-law index must be negative")

        u = self.rng.uniform(0, 1, size=n_satellites)
        exponent = self.alpha + 1

        if abs(exponent) < 1e-6:
            masses = self.M_min * np.exp(u * np.log(self.M_max / self.M_min))
        else:
            M_min_pow = self.M_min**exponent
            M_max_pow = self.M_max**exponent
            masses = (M_min_pow + u * (M_max_pow - M_min_pow))**(1.0 / exponent)

        return masses

    def sample_log10(self, n_satellites):
        """Sample masses and return in log10 format."""
        masses = self.sample(n_satellites)
        return np.log10(masses)

    def generate(self, n_satellites, orbital_parameters=None):
        """Generate satellite properties: masses, positions, velocities."""
        masses = self.sample(n_satellites)
        logM = np.log10(masses)

        # Set orbital parameters
        if orbital_parameters is None:
            orbital_parameters = {}

        velocity_scale = orbital_parameters.get('velocity_scale', 100.0)
        r_min, r_max = orbital_parameters.get('radial_range', (20.0, 150.0))

        radii = self.rng.uniform(r_min, r_max, size=n_satellites)
        thetas = np.arccos(self.rng.uniform(-1, 1, size=n_satellites))
        phis = self.rng.uniform(0, 2 * np.pi, size=n_satellites)
        positions = np.array([
            radii * np.sin(thetas) * np.cos(phis),
            radii * np.sin(thetas) * np.sin(phis),
            radii * np.cos(thetas)
        ]).T

        v_mag = self.rng.normal(0, velocity_scale, size=n_satellites)
        v_thetas = np.arccos(self.rng.uniform(-1, 1, size=n_satellites))
        v_phis = self.rng.uniform(0, 2 * np.pi, size=n_satellites)

        velocities = np.array([
            v_mag * np.sin(v_thetas) * np.cos(v_phis),
            v_mag * np.sin(v_thetas) * np.sin(v_phis),
            v_mag * np.cos(v_thetas)
        ]).T

        return {
            'masses': masses,
            'logM': logM,
            'positions': positions,
            'velocities': velocities,
            'initial_conditions': np.hstack([positions, velocities]),
        }
