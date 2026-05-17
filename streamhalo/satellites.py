"""Satellite galaxy generation and mass function sampling."""

import numpy as np


class SatellitePopulation:
    """Sample satellite halo masses and convert to stellar masses via SHMR."""

    VALID_MASS_FUNCTIONS = ['powerlaw', 'truncated_powerlaw', 'schechter']
    VALID_SHMR = ['constant', 'behroozi13']

    def __init__(self, mass_function='powerlaw', alpha=-1.9, M_min=1e7, M_max=1e11,
                 M_star=10**10.5, shmr='behroozi13', stellar_mass_fraction=0.01, rng=None):
        if mass_function not in self.VALID_MASS_FUNCTIONS:
            raise ValueError(
                f"Invalid mass function '{mass_function}'. "
                f"Must be one of: {self.VALID_MASS_FUNCTIONS}"
            )
        if shmr not in self.VALID_SHMR:
            raise ValueError(f"Invalid SHMR '{shmr}'. Must be one of: {self.VALID_SHMR}")
        if alpha >= 0:
            raise ValueError(f"Power-law index alpha must be negative, got {alpha}")

        self.mass_function = mass_function
        self.alpha = alpha
        self.M_min = M_min
        self.M_max = M_max
        self.M_star = M_star
        self.shmr = shmr
        self.stellar_mass_fraction = stellar_mass_fraction  # used only when shmr='constant'
        self.rng = rng if rng is not None else np.random.default_rng()

    def _sample_powerlaw(self, n):
        u = self.rng.uniform(0, 1, size=n)
        exponent = self.alpha + 1
        if abs(exponent) < 1e-6:
            return self.M_min * np.exp(u * np.log(self.M_max / self.M_min))
        M_min_pow = self.M_min**exponent
        M_max_pow = self.M_max**exponent
        return (M_min_pow + u * (M_max_pow - M_min_pow))**(1.0 / exponent)

    def _sample_schechter(self, n):
        """Rejection sampling for Schechter function using a power-law proposal."""
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
        """Sample n halo masses from the configured mass function."""
        if self.mass_function in ('powerlaw', 'truncated_powerlaw'):
            return self._sample_powerlaw(n_satellites)
        return self._sample_schechter(n_satellites)

    def sample_log10(self, n_satellites):
        return np.log10(self.sample(n_satellites))

    def sample_until_stellar_mass(self, target):
        """Sample halo masses one at a time until cumulative stellar mass >= target."""
        halo_masses = []
        total = 0.0
        while total < target:
            m = self.sample(1)[0]
            halo_masses.append(m)
            total += float(self.stellar_mass(m))
        return np.array(halo_masses)

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
        return 10.0 ** (log_eps + log_M1 + f(x) - f(0.0))

    def stellar_mass(self, halo_mass):
        """Convert halo mass to stellar mass using the configured SHMR."""
        if self.shmr == 'constant':
            return np.asarray(halo_mass, dtype=float) * self.stellar_mass_fraction
        return self._stellar_mass_behroozi13(halo_mass)
