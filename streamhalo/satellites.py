"""Satellite galaxy generation and mass function sampling."""

import numpy as np


class SatellitePopulation:
    """Generate satellite population from mass functions."""

    VALID_MASS_FUNCTIONS = ['powerlaw', 'truncated_powerlaw', 'schechter']
    VALID_SHMR = ['constant', 'behroozi13']

    def __init__(self, mass_function='powerlaw', alpha=-1.9, M_min=1e7, M_max=1e11,
                 M_star=10**10.5, shmr='behroozi13', stellar_mass_fraction=0.01, rng=None):
        """Initialize satellite population generator."""
        if mass_function not in self.VALID_MASS_FUNCTIONS:
            raise ValueError(
                f"Invalid mass function '{mass_function}'. "
                f"Must be one of: {self.VALID_MASS_FUNCTIONS}"
            )
        if shmr not in self.VALID_SHMR:
            raise ValueError(
                f"Invalid SHMR '{shmr}'. Must be one of: {self.VALID_SHMR}"
            )

        self.mass_function = mass_function
        self.alpha = alpha
        self.M_min = M_min
        self.M_max = M_max
        self.M_star = M_star
        self.shmr = shmr
        self.stellar_mass_fraction = stellar_mass_fraction  # used only when shmr='constant'
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

    def _sample_powerlaw(self, n):
        """Sample from truncated power law on [M_min, M_max]."""
        u = self.rng.uniform(0, 1, size=n)
        exponent = self.alpha + 1
        if abs(exponent) < 1e-6:
            return self.M_min * np.exp(u * np.log(self.M_max / self.M_min))
        M_min_pow = self.M_min**exponent
        M_max_pow = self.M_max**exponent
        return (M_min_pow + u * (M_max_pow - M_min_pow))**(1.0 / exponent)

    def _sample_schechter(self, n):
        """Rejection sampling for Schechter function using power-law proposal."""
        # Proposal: power law M^alpha; weight: exp(-M/M_star)
        # Maximum weight is at M_min (since weight is decreasing)
        max_weight = np.exp(-self.M_min / self.M_star)
        collected = []
        while sum(len(a) for a in collected) < n:
            n_needed = n - sum(len(a) for a in collected)
            batch = self._sample_powerlaw(max(n_needed * 10, 100))
            accept = self.rng.uniform(0, 1, size=len(batch)) < (
                np.exp(-batch / self.M_star) / max_weight
            )
            collected.append(batch[accept])
        return np.concatenate(collected)[:n]

    def sample(self, n_satellites):
        """Sample satellite masses from mass function."""
        if self.mass_function in ('powerlaw', 'truncated_powerlaw'):
            return self._sample_powerlaw(n_satellites)
        elif self.mass_function == 'schechter':
            return self._sample_schechter(n_satellites)

    def sample_log10(self, n_satellites):
        """Sample masses and return in log10 format."""
        return np.log10(self.sample(n_satellites))

    def _stellar_mass_behroozi13(self, halo_mass):
        """Behroozi+2013 SHMR at z=0 (ApJ 770, 57, Eq. 3)."""
        log_M1 = 11.514
        log_eps = -1.777
        alpha_b = -1.412
        delta = 3.508
        gamma = 0.316

        M1 = 10.0 ** log_M1

        def f(x):
            term1 = -np.log10(10.0 ** (alpha_b * x) + 1.0)
            term2 = (delta * np.log10(1.0 + np.exp(x)) ** gamma
                     / (1.0 + np.exp(-10.0 ** x)))
            return term1 + term2

        x = np.log10(np.asarray(halo_mass, dtype=float) / M1)
        log_Mstar = log_eps + log_M1 + f(x) - f(0.0)
        return 10.0 ** log_Mstar

    def stellar_mass(self, halo_mass):
        """Convert halo mass to stellar mass using the configured SHMR."""
        if self.shmr == 'constant':
            return np.asarray(halo_mass, dtype=float) * self.stellar_mass_fraction
        elif self.shmr == 'behroozi13':
            return self._stellar_mass_behroozi13(halo_mass)

    def generate(self, n_satellites, orbital_parameters=None):
        """Generate satellite properties: masses, positions, velocities."""
        halo_masses = self.sample(n_satellites)
        stellar_masses = self.stellar_mass(halo_masses)
        logM_halo = np.log10(halo_masses)

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
            'halo_masses': halo_masses,
            'stellar_masses': stellar_masses,
            'logM_halo': logM_halo,
            'positions': positions,
            'velocities': velocities,
            'initial_conditions': np.hstack([positions, velocities]),
        }
